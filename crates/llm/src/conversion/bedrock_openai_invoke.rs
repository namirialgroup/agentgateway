//! OpenAI-compatible InvokeModel transport for Bedrock Runtime
//! (`InvokeModel` / `InvokeModelWithResponseStream` with an OpenAI Chat
//! Completions JSON body) — the native transport for OpenWeight models whose
//! Converse surface is deficient. First target: Moonshot Kimi K3.1
//! (`global.moonshotai.kimi-k3.1`), whose Converse API rejects `temperature`,
//! cannot replay multi-turn reasoning content, and carries no explicit
//! caching, while the OpenAI surface accepts the full sampling envelope
//! (`temperature`, `reasoning_effort`, tools) and reports per-class token
//! details.
//!
//! LIVE-WIRE CONTRACT (probed against real AWS eu-west-1 K3.1, 2026-11-12):
//!  - `InvokeModel` returns the OpenAI `chat.completion` JSON object directly;
//!  - `InvokeModelWithResponseStream` emits AWS EventStream `chunk` frames
//!    whose base64 `bytes` payload is ONE RAW OpenAI `chat.completion.chunk`
//!    JSON per frame (no `data:` prefix, no embedded SSE framing), ending
//!    with a final `choices:[]` chunk that carries `usage` and
//!    `amazon-bedrock-invocationMetrics`;
//!  - usage carries `prompt_tokens_details.cached_tokens` (cache read),
//!    `prompt_tokens_details.cache_write_tokens`, and
//!    `completion_tokens_details.reasoning_tokens` — mapped by the existing
//!    typed OpenAI usage (`types::completions::Usage`).
//!
//! This module never selects Converse: a backend configured with the
//! `OpenAIInvoke` transport is served exclusively through the Runtime
//! InvokeModel surface (fail-closed by construction — see
//! `bedrock::Provider::supported_chat_formats`/`get_path_for_route`).

use agent_core::strng;
use agent_http::Body;
use bytes::Bytes;
use serde_json::json;

use crate::{AIError, parse};

/// Render the OpenAI Chat Completions body for Bedrock Runtime InvokeModel.
///
/// Adjustments applied to the rendered OpenAI body (never the client's
/// request):
///  1. Multi-turn reasoning hygiene: `reasoning_content` is stripped from ALL
///     prior-turn messages. Reasoning streams of third-party models are
///     provider-bound artifacts; replaying them on a fresh attempt (in
///     particular after a failover from another provider) is the exact
///     incompatibility class the Converse path hits ("fails multi-turn
///     requests that carry earlier reasoning"). The CURRENT turn's assistant
///     output is produced by the model itself, so dropping history-side
///     reasoning never degrades the active response.
///  2. Streaming usage guarantee: `stream_options.include_usage = true` is
///     injected when the body requests `stream: true` (K3.1 emits the usage
///     tail chunks only under `include_usage`; accounting requires it).
///     `stream_options` is dropped when the request is not streaming (the
///     field is invalid without `stream`).
///
/// `temperature`, `top_p`, `tools`, `tool_choice`, `reasoning_effort` and all
/// other sampling fields pass through VERBATIM — the OpenAI surface accepts
/// them natively (probe-verified); no mapping, no stripping.
pub fn prepare_invoke_body(body: &[u8]) -> Result<Vec<u8>, AIError> {
	let mut value: serde_json::Value =
		serde_json::from_slice(body).map_err(AIError::RequestMarshal)?;
	let obj = value.as_object_mut().ok_or_else(|| {
		AIError::UnsupportedConversion(strng::literal!(
			"bedrock OpenAIInvoke transport requires a JSON object body"
		))
	})?;

	let streaming = obj.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

	// 1. Strip provider-bound reasoning artifacts from conversation history.
	if let Some(messages) = obj.get_mut("messages").and_then(|m| m.as_array_mut()) {
		for message in messages.iter_mut() {
			let is_assistant = message.get("role").and_then(|r| r.as_str()) == Some("assistant");
			if !is_assistant {
				continue;
			}
			if let Some(message_obj) = message.as_object_mut() {
				message_obj.remove("reasoning_content");
			}
		}
	}

	// 2. Streaming usage guarantee (include_usage), never on non-streaming.
	match (streaming, obj.get_mut("stream_options")) {
		(true, None) => {
			obj.insert(
				"stream_options".to_string(),
				json!({ "include_usage": true }),
			);
		},
		(true, Some(options)) => {
			if let Some(options_obj) = options.as_object_mut() {
				options_obj.insert("include_usage".to_string(), json!(true));
			}
		},
		(false, Some(_)) => {
			// stream_options without stream is invalid on the OpenAI surface.
			obj.remove("stream_options");
		},
		(false, None) => {},
	}

	serde_json::to_vec(&value).map_err(AIError::RequestMarshal)
}

/// Bedrock Runtime streaming: each `InvokeModelWithResponseStream` frame
/// carries one raw OpenAI `chat.completion.chunk` JSON, re-emitted here as
/// OpenAI-format SSE (`data: {...}` lines, no `event:` names — the wire
/// format OpenAI-compat client translation expects). The terminal
/// `amazon-bedrock-invocationMetrics` frame is followed by `data: [DONE]`.
///
/// The caller composes the result with the standard OpenAI upstream stream
/// translations (`completions::passthrough_stream` for Completions input,
/// `completions::from_messages::translate_stream` for Messages input,
/// `openai_compat::to_responses::translate_stream` for Responses input), so
/// usage capture (cached/cache-write/reasoning token classes) rides the
/// proven OpenAI-compat code path byte-for-byte.
pub fn transcode_stream_to_sse(b: Body, buffer_limit: usize) -> Body {
	parse::aws_sse::transform_raw(b, buffer_limit, openai_invoke_stream_frame)
}

/// Decode one `InvokeModelWithResponseStream` EventStream frame into
/// `(event-name, payload)` SSE pairs for the OpenAI-format client stream.
/// OpenAI SSE carries NO event names — the payload travels as a bare
/// `data:` line, so the name is always empty (`encode_sse_event("")`).
fn openai_invoke_stream_frame(m: parse::aws_sse::Message) -> Vec<(String, Bytes)> {
	use base64::Engine as _;

	let get_header = |name: &str| -> Option<String> {
		m.headers()
			.iter()
			.find(|h| h.name().as_str() == name)
			.and_then(|h| h.value().as_string().ok())
			.map(|s| s.as_str().to_owned())
	};

	if get_header(":message-type").as_deref() == Some("exception") {
		let exception_type = get_header(":exception-type").unwrap_or_else(|| "unknown".to_owned());
		let error_message = String::from_utf8_lossy(m.payload());
		// OpenAI-format error object: surfaced as a data event so the
		// client-side OpenAI error handling sees it (never a silent cut).
		let body = json!({
			"error": {
				"type": "upstream_error",
				"code": exception_type,
				"message": error_message,
			},
		});
		return vec![("".to_string(), Bytes::from(body.to_string()))];
	}

	// Stream-terminating metadata frame (":event-type: metadata"): carries
	// invocation metrics outside the OpenAI contract — dropped.
	if get_header(":event-type").as_deref() == Some("metadata") {
		return vec![];
	}

	let event_type = get_header(":event-type");
	let payload = m.payload();
	let chunk: Option<Bytes> = (|| {
		let v: serde_json::Value = serde_json::from_slice(payload).ok()?;
		// 1. LIVE wire shape: :event-type "chunk" + {"bytes": "<base64 raw
		//    OpenAI chunk JSON>"} (probe-verified on real K3.1 traffic
		//    2026-11-12 — identical envelope to the native Messages frames).
		if let Some(encoded) = v.get("bytes").and_then(|b| b.as_str()) {
			return base64::engine::general_purpose::STANDARD
				.decode(encoded)
				.ok()
				.map(Bytes::from);
		}
		// 2. Doc/SDK shape (defensive): {"chunk": {"bytes": "<base64>"}}.
		if let Some(encoded) = v
			.get("chunk")
			.and_then(|chunk| chunk.get("bytes"))
			.and_then(|b| b.as_str())
		{
			return base64::engine::general_purpose::STANDARD
				.decode(encoded)
				.ok()
				.map(Bytes::from);
		}
		// 3. Direct OpenAI chunk (defensive): an `object` field marks a bare
		//    chat.completion.chunk document.
		if v.get("object").and_then(|t| t.as_str()).is_some() {
			return Some(Bytes::copy_from_slice(payload));
		}
		None
	})();

	let Some(chunk) = chunk else {
		let body = json!({
			"error": {
				"type": "upstream_error",
				"code": "frame_decode",
				"message": format!(
					"unrecognized InvokeModelWithResponseStream frame (event-type: {})",
					event_type.as_deref().unwrap_or("<none>")),
			},
		});
		return vec![("".to_string(), Bytes::from(body.to_string()))];
	};

	// The terminal chunk carries `amazon-bedrock-invocationMetrics`; after it
	// the OpenAI SSE contract requires the `data: [DONE]` terminator (AWS does
	// not send one itself).
	if chunk
		.as_ref()
		.windows("amazon-bedrock-invocationMetrics".len())
		.any(|w| w == "amazon-bedrock-invocationMetrics".as_bytes())
	{
		return vec![
			("".to_string(), chunk),
			("".to_string(), Bytes::from_static(b"[DONE]")),
		];
	}

	vec![("".to_string(), chunk)]
}

#[cfg(test)]
mod tests {
	use super::*;
	use serde_json::Value;

	#[test]
	fn strips_reasoning_content_from_assistant_history_only() {
		let body = json!({
			"model": "global.moonshotai.kimi-k3.1",
			"messages": [
				{"role": "system", "content": "be brief"},
				{"role": "user", "content": "hi", "reasoning_content": "user-side stray"},
				{"role": "assistant", "content": "answer", "reasoning_content": "prior-turn thinking"},
				{"role": "assistant", "content": [
					{"type": "text", "text": "with reasoning"},
					{"reasoning_content": "nested block"}
				]}
			],
			"temperature": 0.2,
			"reasoning_effort": "low"
		});
		let prepared = prepare_invoke_body(&serde_json::to_vec(&body).unwrap()).unwrap();
		let out: Value = serde_json::from_slice(&prepared).unwrap();

		// Assistant history-side reasoning removed (top-level field).
		assert_eq!(
			out["messages"][2],
			json!({"role": "assistant", "content": "answer"})
		);
		// User messages keep their fields (even stray ones — only assistant
		// turns carry provider-bound reasoning).
		assert_eq!(
			out["messages"][1]["reasoning_content"],
			json!("user-side stray")
		);
		// Sampling envelope passes through verbatim.
		assert_eq!(out["temperature"], json!(0.2));
		assert_eq!(out["reasoning_effort"], json!("low"));
		// Non-streaming: no stream_options injected.
		assert!(out.get("stream_options").is_none());
	}

	#[test]
	fn injects_include_usage_for_streaming_requests() {
		let body = json!({
			"model": "global.moonshotai.kimi-k3.1",
			"messages": [{"role": "user", "content": "hi"}],
			"stream": true,
			"max_tokens": 16
		});
		let prepared = prepare_invoke_body(&serde_json::to_vec(&body).unwrap()).unwrap();
		let out: Value = serde_json::from_slice(&prepared).unwrap();
		assert_eq!(out["stream_options"]["include_usage"], json!(true));
		assert_eq!(out["stream"], json!(true));
		assert_eq!(out["max_tokens"], json!(16));
	}

	#[test]
	fn forces_include_usage_true_over_client_false_on_streaming() {
		let body = json!({
			"model": "global.moonshotai.kimi-k3.1",
			"messages": [{"role": "user", "content": "hi"}],
			"stream": true,
			"stream_options": {"include_usage": false}
		});
		let prepared = prepare_invoke_body(&serde_json::to_vec(&body).unwrap()).unwrap();
		let out: Value = serde_json::from_slice(&prepared).unwrap();
		assert_eq!(out["stream_options"]["include_usage"], json!(true));
	}

	#[test]
	fn drops_stream_options_without_stream() {
		let body = json!({
			"model": "global.moonshotai.kimi-k3.1",
			"messages": [{"role": "user", "content": "hi"}],
			"stream_options": {"include_usage": true}
		});
		let prepared = prepare_invoke_body(&serde_json::to_vec(&body).unwrap()).unwrap();
		let out: Value = serde_json::from_slice(&prepared).unwrap();
		assert!(out.get("stream_options").is_none());
	}

	#[test]
	fn rejects_non_object_body() {
		assert!(prepare_invoke_body(b"[1,2,3]").is_err());
	}

	fn frame(headers: Vec<(&str, &str)>, payload: Bytes) -> crate::parse::aws_sse::Message {
		use aws_smithy_types::event_stream::{Header, HeaderValue};
		let hv: Vec<Header> = headers
			.into_iter()
			.map(|(name, value)| {
				Header::new(
					name.to_string(),
					HeaderValue::String(value.to_string().into()),
				)
			})
			.collect();
		crate::parse::aws_sse::Message::new_from_parts(hv, payload)
	}

	#[test]
	fn stream_frame_decodes_live_wire_shape_and_appends_done_on_metrics() {
		use base64::Engine as _;

		let chunk_json =
			br#"{"choices":[{"delta":{"content":"OK"}}],"object":"chat.completion.chunk"}"#;
		let chunk_frame = frame(
			vec![
				(":message-type", "event"),
				(":event-type", "chunk"),
				(":content-type", "application/json"),
			],
			Bytes::from(
				json!({"bytes": base64::engine::general_purpose::STANDARD.encode(chunk_json)}).to_string(),
			),
		);
		let events = openai_invoke_stream_frame(chunk_frame);
		assert_eq!(events.len(), 1);
		assert_eq!(events[0].0, "");
		assert_eq!(&events[0].1[..], &chunk_json[..]);

		let usage_json = br#"{"choices":[],"usage":{"prompt_tokens":90},"amazon-bedrock-invocationMetrics":{"inputTokenCount":90}}"#;
		let metrics_frame = frame(
			vec![(":event-type", "chunk")],
			Bytes::from(
				json!({"bytes": base64::engine::general_purpose::STANDARD.encode(usage_json)}).to_string(),
			),
		);
		let events = openai_invoke_stream_frame(metrics_frame);
		assert_eq!(events.len(), 2);
		assert_eq!(&events[1].1[..], b"[DONE]");
	}

	#[test]
	fn stream_frame_exception_becomes_openai_error_object() {
		let message = frame(
			vec![
				(":message-type", "exception"),
				(":exception-type", "AccessDeniedException"),
			],
			Bytes::from_static(b"Model access denied"),
		);
		let events = openai_invoke_stream_frame(message);
		assert_eq!(events.len(), 1);
		let v: Value = serde_json::from_slice(&events[0].1).unwrap();
		assert_eq!(v["error"]["code"], json!("AccessDeniedException"));
		assert_eq!(v["error"]["type"], json!("upstream_error"));
	}

	#[test]
	fn stream_frame_metadata_is_dropped_and_unknown_surfaces_error() {
		let metadata = frame(vec![(":event-type", "metadata")], Bytes::from_static(b"{}"));
		assert!(openai_invoke_stream_frame(metadata).is_empty());

		let unrecognized = frame(
			vec![(":event-type", "chunk")],
			Bytes::from_static(b"not-json"),
		);
		let events = openai_invoke_stream_frame(unrecognized);
		let v: Value = serde_json::from_slice(&events[0].1).unwrap();
		assert!(
			v["error"]["message"]
				.as_str()
				.unwrap()
				.contains("unrecognized")
		);
	}
}
