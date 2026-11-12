//! OpenAIInvoke Bedrock transport end-to-end tests (WS2, 2026-11-12).
//!
//! Pins the wire contract of the `OpenAIInvoke` transport against a mocked
//! Bedrock Runtime:
//!  - requests ride `/model/{model}/invoke[-with-response-stream]` with the
//!    OpenAI Chat Completions body, SigV4-signed for `bedrock`;
//!  - history-side `reasoning_content` is stripped, `temperature` /
//!    `reasoning_effort` / tools pass through verbatim, streaming requests get
//!    `stream_options.include_usage = true`;
//!  - the EventStream response (raw OpenAI chunk JSON per frame) is re-framed
//!    as OpenAI SSE with the `data: [DONE]` terminator;
//!  - client Anthropic Messages requests are served through the
//!    completions translation (Fireworks-parity path).
//!
//! The live wire shape (EventStream `chunk` frames with base64-wrapped raw
//! OpenAI chunk JSON, usage tail chunks with cached/cache-write/reasoning
//! token details) was probe-verified against real AWS eu-west-1 K3.1 traffic
//! and is pinned by the unit fixtures in
//! `crates/llm/src/conversion/bedrock_openai_invoke.rs`; this suite exercises
//! the full gateway (routing, SigV4, streaming pipeline) over sockets.

use axum::http::StatusCode;
use serde_json::{Value, json};
use wiremock::matchers;
use wiremock::{Mock, MockServer, ResponseTemplate};

use crate::common::gateway::AgentGateway;
use crate::common::prelude::*;

const MODEL: &str = "global.moonshotai.kimi-k3.1";
const REGION: &str = "eu-west-1";
const AWS_TEST_ACCESS_KEY: &str = "AKIDOPENAIINVOKETEST";
const AWS_TEST_SECRET_KEY: &str = "openai-invoke-test-secret";

fn invoke_config(bedrock: &str) -> String {
	format!(
		r#"
config: {{}}
frontendPolicies:
  http:
    maxBufferSize: 10485760
binds:
- port: $PORT
  listeners:
  - name: llm
    protocol: HTTP
    routes:
    - name: kimi-k31-openaiinvoke
      matches:
      - path:
          pathPrefix: /v1
      policies:
        ai:
          routes:
            /v1/chat/completions: completions
            /v1/messages: messages
            /v1/responses: responses
      backends:
      - ai:
          name: bedrock-kimi-k31
          provider:
            bedrock:
              model: {MODEL}
              region: {REGION}
              transport: openAIInvoke
          hostOverride: {bedrock}
          # empty prefix: keeps the Bedrock default-path rewrite (invoke paths) active
          # despite the host override (test-only; Production uses the real Bedrock host).
          pathPrefix: ""
          policies:
            backendAuth:
              aws:
                accessKeyId: {AWS_TEST_ACCESS_KEY}
                secretAccessKey: {AWS_TEST_SECRET_KEY}
                region: {REGION}
                serviceName: bedrock
"#
	)
}

fn chat_body() -> Value {
	json!({
		"model": MODEL,
		"max_tokens": 128,
		"temperature": 0.2,
		"reasoning_effort": "low",
		"messages": [
			{"role": "user", "content": "What is the weather in Milan?"},
			{"role": "assistant", "content": "checking", "tool_calls": [
				{"id": "call_1", "type": "function",
				 "function": {"name": "get_weather", "arguments": "{\"city\":\"Milan\"}"}}],
				"reasoning_content": "prior-turn reasoning that must NOT be replayed"},
			{"role": "tool", "tool_call_id": "call_1", "content": "18C sunny"}
		],
		"tools": [{"type": "function", "function": {
			"name": "get_weather", "description": "Get weather",
			"parameters": {"type": "object", "properties": {"city": {"type": "string"}},
				"required": ["city"]}}}]
	})
}

async fn send(gw: &AgentGateway, path: &str, body: Value) -> (StatusCode, Value) {
	let resp = gw
		.send_request_json(&format!("http://localhost{path}"), body)
		.await;
	let status = resp.status();
	let bytes = resp.into_body().collect().await.unwrap().to_bytes();
	let parsed = serde_json::from_slice(&bytes).unwrap_or_else(|err| {
		panic!(
			"status {status}: response is not JSON ({err}): {}",
			String::from_utf8_lossy(&bytes)
		)
	});
	(status, parsed)
}

fn stream_chunk(delta: Value, finish: Value) -> Value {
	json!({"id": "chatcmpl-invoke-1", "object": "chat.completion.chunk", "created": 1,
		"model": MODEL,
		"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
}

/// One AWS EventStream frame whose payload is the RAW OpenAI chunk JSON
/// (the decoder's direct-JSON defensive shape; the live base64 `bytes`
/// wrapper is pinned by the unit fixtures in agent-llm).
fn event_stream_frame(payload: &Value) -> Vec<u8> {
	use aws_smithy_eventstream::frame::write_message_to;
	use aws_smithy_types::event_stream::{Header, HeaderValue, Message};
	let headers = vec![
		Header::new(
			":message-type".to_string(),
			HeaderValue::String("event".into()),
		),
		Header::new(
			":event-type".to_string(),
			HeaderValue::String("chunk".into()),
		),
		Header::new(
			":content-type".to_string(),
			HeaderValue::String("application/json".into()),
		),
	];
	let mut buf = Vec::new();
	write_message_to(
		&Message::new_from_parts(headers, bytes::Bytes::from(payload.to_string())),
		&mut buf,
	)
	.expect("frame encodes");
	buf
}

fn event_stream_response(frames: Vec<Value>) -> ResponseTemplate {
	let mut body = Vec::new();
	for frame in frames {
		body.extend_from_slice(&event_stream_frame(&frame));
	}
	ResponseTemplate::new(200).set_body_raw(body, "application/vnd.amazon.eventstream")
}

fn assert_sigv4_signed_for_bedrock(req: &wiremock::Request) {
	let auth = req
		.headers
		.get("authorization")
		.and_then(|v| v.to_str().ok())
		.expect("authorization header");
	assert!(
		auth.starts_with(&format!(
			"AWS4-HMAC-SHA256 Credential={AWS_TEST_ACCESS_KEY}/"
		)),
		"{auth}"
	);
	assert!(
		auth.contains(&format!("/{REGION}/bedrock/aws4_request")),
		"wrong signing scope: {auth}"
	);
}

#[tokio::test]
async fn openai_invoke_non_stream_strips_history_reasoning_and_signs_sigv4() {
	let mock = MockServer::start().await;
	let upstream = mock.address().to_string();
	Mock::given(matchers::method("POST"))
		.and(matchers::path(format!("/model/{MODEL}/invoke")))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({
			"id": "chatcmpl-invoke-ns", "object": "chat.completion", "created": 1,
			"model": MODEL,
			"choices": [{"index": 0, "finish_reason": "stop",
				"message": {"role": "assistant", "content": "18C and sunny"}}],
			"usage": {"prompt_tokens": 90, "completion_tokens": 12, "total_tokens": 102,
				"prompt_tokens_details": {"cached_tokens": 60, "cache_write_tokens": 0},
				"completion_tokens_details": {"reasoning_tokens": 4}}
		})))
		.mount(&mock)
		.await;

	let gw = AgentGateway::new(invoke_config(&upstream)).await.unwrap();
	let (status, body) = send(&gw, "/v1/chat/completions", chat_body()).await;
	assert_eq!(status, StatusCode::OK, "{body}");
	// OpenAI-shaped passthrough for OpenAI clients, usage intact.
	assert_eq!(
		body["choices"][0]["message"]["content"],
		json!("18C and sunny")
	);
	assert_eq!(
		body["usage"]["prompt_tokens_details"]["cached_tokens"],
		json!(60)
	);
	assert_eq!(
		body["usage"]["completion_tokens_details"]["reasoning_tokens"],
		json!(4)
	);

	let received = mock.received_requests().await.unwrap();
	assert_eq!(received.len(), 1);
	let upstream_req = &received[0];
	// InvokeModel path for the global profile id.
	assert_eq!(upstream_req.url.path(), format!("/model/{MODEL}/invoke"));
	// SigV4 for the bedrock service.
	assert_sigv4_signed_for_bedrock(upstream_req);

	let sent: Value = serde_json::from_slice(upstream_req.body.as_slice()).unwrap();
	// Sampling + effort + tools pass through VERBATIM (Converse rejects
	// temperature; the OpenAI surface accepts it).
	assert_eq!(sent["temperature"], json!(0.2));
	assert_eq!(sent["reasoning_effort"], json!("low"));
	assert_eq!(sent["tools"][0]["function"]["name"], json!("get_weather"));
	// Multi-turn hygiene: the assistant history message arrives WITHOUT its
	// reasoning_content; everything else survives.
	assert_eq!(
		sent["messages"][1],
		json!({"role": "assistant", "content": "checking", "tool_calls": [
			{"id": "call_1", "type": "function",
			 "function": {"name": "get_weather", "arguments": "{\"city\":\"Milan\"}"}}]})
	);
	assert!(sent["messages"][1].get("reasoning_content").is_none());
	// Non-streaming: no stream_options injected.
	assert!(sent.get("stream_options").is_none());
}

#[tokio::test]
async fn openai_invoke_stream_reframes_eventstream_to_sse_with_done() {
	let mock = MockServer::start().await;
	let upstream = mock.address().to_string();
	Mock::given(matchers::method("POST"))
		.and(matchers::path(format!(
			"/model/{MODEL}/invoke-with-response-stream"
		)))
		.respond_with(event_stream_response(vec![
			stream_chunk(json!({"role": "assistant", "content": ""}), Value::Null),
			stream_chunk(json!({"content": "HE"}), Value::Null),
			stream_chunk(json!({"content": "Y"}), json!("stop")),
			// Terminal chunk: choices [] + usage + invocation metrics.
			json!({"id": "chatcmpl-invoke-1", "object": "chat.completion.chunk", "created": 1,
				"model": MODEL, "choices": [],
				"usage": {"prompt_tokens": 90, "completion_tokens": 24, "total_tokens": 114,
					"prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
					"completion_tokens_details": {"reasoning_tokens": 8}},
				"amazon-bedrock-invocationMetrics": {"inputTokenCount": 90, "outputTokenCount": 24}}),
		]))
		.mount(&mock)
		.await;

	let gw = AgentGateway::new(invoke_config(&upstream)).await.unwrap();
	let mut body = json!(chat_body());
	body["stream"] = json!(true);
	let resp = gw
		.send_request_json("http://localhost/v1/chat/completions", body)
		.await;
	assert_eq!(resp.status(), StatusCode::OK);
	let bytes = resp.into_body().collect().await.unwrap().to_bytes();
	let raw = String::from_utf8_lossy(&bytes);
	// OpenAI SSE framing with the [DONE] terminator (AWS never sends one).
	assert!(raw.starts_with("data: {"), "{raw}");
	assert!(raw.contains("data: [DONE]"), "{raw}");
	assert!(raw.contains("\"reasoning_tokens\":8"), "{raw}");
	// Exactly one [DONE], at the very end.
	assert_eq!(raw.matches("data: [DONE]").count(), 1);
	assert!(raw.trim_end().ends_with("data: [DONE]"), "{raw}");

	let received = mock.received_requests().await.unwrap();
	assert_eq!(received.len(), 1);
	let upstream_req = &received[0];
	assert_eq!(
		upstream_req.url.path(),
		format!("/model/{MODEL}/invoke-with-response-stream")
	);
	let sent: Value = serde_json::from_slice(upstream_req.body.as_slice()).unwrap();
	// Streaming usage guarantee injected.
	assert_eq!(sent["stream"], json!(true));
	assert_eq!(sent["stream_options"]["include_usage"], json!(true));
	// History reasoning still stripped on the streaming path.
	assert!(sent["messages"][1].get("reasoning_content").is_none());
}

#[tokio::test]
async fn openai_invoke_serves_anthropic_messages_clients_through_completions_translation() {
	let mock = MockServer::start().await;
	let upstream = mock.address().to_string();
	Mock::given(matchers::method("POST"))
		.and(matchers::path(format!("/model/{MODEL}/invoke")))
		.respond_with(ResponseTemplate::new(200).set_body_json(json!({
			"id": "chatcmpl-invoke-msg", "object": "chat.completion", "created": 1,
			"model": MODEL,
			"choices": [{"index": 0, "finish_reason": "stop",
				"message": {"role": "assistant", "content": "MSG-OK",
					"reasoning_content": "short thinking"}}],
			"usage": {"prompt_tokens": 90, "completion_tokens": 30, "total_tokens": 120}
		})))
		.mount(&mock)
		.await;

	let gw = AgentGateway::new(invoke_config(&upstream)).await.unwrap();
	let (status, body) = send(
		&gw,
		"/v1/messages",
		json!({
			"model": MODEL,
			"max_tokens": 200,
			"messages": [{"role": "user", "content": "Say exactly: MSG-OK"}]
		}),
	)
	.await;
	assert_eq!(status, StatusCode::OK, "{body}");
	// Anthropic Messages response shape for the Anthropic client.
	assert_eq!(body["type"], json!("message"));
	assert_eq!(body["stop_reason"], json!("end_turn"));
	let text = body["content"]
		.as_array()
		.unwrap()
		.iter()
		.find(|b| b["type"] == "text")
		.expect("text content block")
		.clone();
	assert_eq!(text["text"], json!("MSG-OK"));

	// The upstream body was the OpenAI conversion of the Anthropic request.
	let received = mock.received_requests().await.unwrap();
	assert_eq!(received.len(), 1);
	let sent: Value = serde_json::from_slice(received[0].body.as_slice()).unwrap();
	assert_eq!(sent["max_completion_tokens"], json!(200));
	let roles: Vec<&str> = sent["messages"]
		.as_array()
		.unwrap()
		.iter()
		.filter_map(|m| m["role"].as_str())
		.collect();
	assert_eq!(roles, vec!["user"]);
}
