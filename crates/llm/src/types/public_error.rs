use std::borrow::Cow;
use std::time::Duration;

use agent_http::{Body, RawBody};
use bytes::{Bytes, BytesMut};
use http::{HeaderMap, StatusCode};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tokio_sse_codec::{Event, Frame, SseDecoder};
use tokio_util::codec::BytesCodec;

use crate::{AIError, InputFormat};

pub fn public_status(status: StatusCode) -> StatusCode {
	match status {
		StatusCode::UNAUTHORIZED
		| StatusCode::FORBIDDEN
		| StatusCode::INTERNAL_SERVER_ERROR
		| StatusCode::NOT_IMPLEMENTED => StatusCode::BAD_GATEWAY,
		StatusCode::REQUEST_TIMEOUT => StatusCode::GATEWAY_TIMEOUT,
		_ => status,
	}
}

pub fn sanitize_headers(headers: &mut HeaderMap) {
	let internal: Vec<_> = headers
		.keys()
		.filter(|name| {
			let name = name.as_str();
			!(matches!(
				name,
				"content-type"
					| "cache-control"
					| "vary"
					| "retry-after"
					| "retry-after-ms"
					| "x-request-id"
					| "x-synapse-request-id"
					| "x-synapse-client-request-id"
			) || name.starts_with("access-control-")
				|| name.starts_with("x-ratelimit-")
				|| name.starts_with("ratelimit-")
				|| name.starts_with("anthropic-ratelimit-"))
		})
		.cloned()
		.collect();
	for name in internal {
		headers.remove(name);
	}
}

pub fn strip_internal_response_headers(headers: &mut HeaderMap) {
	let internal: Vec<_> = headers
		.keys()
		.filter(|name| {
			let name = name.as_str();
			(matches!(name, "fireworks-prompt-tokens" | "traceresponse")
				|| name.starts_with("x-amzn-")
				|| name.starts_with("x-aws-")
				|| name.starts_with("x-envoy-")
				|| name.starts_with("x-fireworks-")
				|| name.starts_with("x-provider-")
				|| name.starts_with("x-debug-"))
				|| (name.starts_with("x-synapse-")
					&& !matches!(name, "x-synapse-request-id" | "x-synapse-client-request-id"))
		})
		.cloned()
		.collect();
	for name in internal {
		headers.remove(name);
	}
}

pub fn upstream_error(format: InputFormat, status: StatusCode, translated: &Bytes) -> Bytes {
	let message = if translated
		.windows(b"capability_rejected: prompt_too_long".len())
		.any(|window| window == b"capability_rejected: prompt_too_long")
	{
		"capability_rejected: prompt_too_long Input exceeds the model context window."
	} else {
		upstream_message(format, status)
	};
	synthesize(format, status, message)
}

pub fn local_error(format: InputFormat, status: StatusCode, error: &AIError) -> Bytes {
	let message = match error {
		AIError::RequestParsing(..) => "The request body is not valid for this endpoint.",
		AIError::MissingField(_) => "The request is missing a required field.",
		AIError::ModelNotFound | AIError::MessageNotFound => {
			"The requested model or message is not available."
		},
		AIError::RequestTooLarge => "The request body exceeds the gateway size limit.",
		AIError::UnsupportedModel
		| AIError::UnsupportedContent
		| AIError::UnsupportedConversion(_)
		| AIError::StreamingUnsupported => "The request uses a feature unsupported by this endpoint.",
		AIError::UnsupportedEncoding(_) => "The request content encoding is not supported.",
		_ => "The gateway could not process the LLM response.",
	};
	synthesize(format, status, message)
}

fn upstream_message(format: InputFormat, status: StatusCode) -> &'static str {
	if format == InputFormat::CountTokens && status == StatusCode::BAD_REQUEST {
		return "Token counting is not supported for this model or request.";
	}
	match status {
		StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => {
			"The upstream provider rejected the request as invalid."
		},
		StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
			"The gateway could not authenticate to the upstream provider."
		},
		StatusCode::NOT_FOUND => "The requested upstream resource is not available.",
		StatusCode::REQUEST_TIMEOUT => "The upstream provider timed out.",
		StatusCode::CONFLICT => "The upstream provider rejected the request due to a conflict.",
		StatusCode::PAYLOAD_TOO_LARGE => "The request exceeds the upstream provider's size limit.",
		StatusCode::TOO_MANY_REQUESTS => "The upstream provider rate limit was exceeded.",
		StatusCode::BAD_GATEWAY | StatusCode::SERVICE_UNAVAILABLE | StatusCode::GATEWAY_TIMEOUT => {
			"The upstream provider is temporarily unavailable."
		},
		_ if status.is_server_error() => "The upstream provider failed to process the request.",
		_ => "The upstream provider returned an error.",
	}
}

fn error_type(status: StatusCode) -> &'static str {
	match public_status(status) {
		StatusCode::BAD_REQUEST | StatusCode::UNPROCESSABLE_ENTITY => "invalid_request_error",
		StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => "authentication_error",
		StatusCode::NOT_FOUND => "not_found_error",
		StatusCode::TOO_MANY_REQUESTS => "rate_limit_error",
		StatusCode::SERVICE_UNAVAILABLE => "overloaded_error",
		_ => "api_error",
	}
}

fn synthesize(format: InputFormat, status: StatusCode, message: &str) -> Bytes {
	let status = public_status(status);
	let body = match format {
		InputFormat::Messages | InputFormat::CountTokens => json!({
			"type": "error",
			"error": {"type": error_type(status), "message": message}
		}),
		InputFormat::Gemini | InputFormat::GeminiCountTokens => json!({
			"error": {
				"code": status.as_u16(),
				"message": message,
				"status": error_type(status)
			}
		}),
		_ => json!({
			"error": {
				"type": error_type(status),
				"message": message,
				"param": null,
				"code": null
			}
		}),
	};
	Bytes::from(serde_json::to_vec(&body).expect("public error JSON must serialize"))
}

pub fn sanitize_stream(format: InputFormat, body: Body, buffer_limit: usize) -> Body {
	let decoder = SseDecoder::<Bytes>::with_max_size(buffer_limit);
	let encoder = BytesCodec::new();
	let sanitized = crate::parse::transform::parser(body, decoder, encoder, move |event| {
		let mut events = Vec::new();
		match event {
			crate::parse::transform::TransformEvent::Item(frame) => match frame {
				Frame::Event(Event { id, name, data }) => {
					let data = sanitize_stream_event(format, &name, data);
					events.push(encode_event(id, name, data));
				},
				Frame::Comment(comment) => events.push(encode_comment(comment)),
				Frame::Retry(duration) => events.push(encode_retry(duration)),
			},
			crate::parse::transform::TransformEvent::Error => {
				let data = synthesize_stream_error(format, None, StatusCode::INTERNAL_SERVER_ERROR);
				events.push(crate::parse::encode_sse_event("error", data));
			},
			crate::parse::transform::TransformEvent::Eof => {},
		}
		events
	});
	sanitized.transform_stream(|body| {
		let stream = futures_util::stream::unfold(body, |mut body| async move {
			body.frame().await.map(|result| {
				let frame = result.map(|frame| match frame.into_trailers() {
					Ok(trailers) => {
						let mut trailers = trailers;
						sanitize_headers(&mut trailers);
						http_body::Frame::trailers(trailers)
					},
					Err(frame) => frame,
				});
				(frame, body)
			})
		});
		RawBody::new(http_body_util::StreamBody::new(stream))
	})
}

fn sanitize_stream_event(format: InputFormat, name: &str, data: Bytes) -> Bytes {
	let value = serde_json::from_slice::<Value>(&data).ok();
	let is_error = name == "error"
		|| value.as_ref().is_some_and(|value| {
			value.get("type").and_then(Value::as_str) == Some("error")
				|| value.get("type").and_then(Value::as_str) == Some("response.failed")
				|| (value.get("error").is_some()
					&& value.get("choices").is_none()
					&& value.get("type").is_none())
		});
	if !is_error {
		return data;
	}
	if format == InputFormat::Responses
		&& value
			.as_ref()
			.and_then(|value| value.get("type"))
			.and_then(Value::as_str)
			== Some("response.failed")
	{
		let mut value = value.expect("response.failed was parsed");
		value["response"]["error"] = json!({
			"code": null,
			"message": upstream_message(format, StatusCode::INTERNAL_SERVER_ERROR)
		});
		return Bytes::from(serde_json::to_vec(&value).expect("public stream error must serialize"));
	}
	synthesize_stream_error(format, value.as_ref(), StatusCode::INTERNAL_SERVER_ERROR)
}

fn synthesize_stream_error(
	format: InputFormat,
	original: Option<&Value>,
	status: StatusCode,
) -> Bytes {
	if format == InputFormat::Responses {
		let sequence_number = original
			.and_then(|value| value.get("sequence_number"))
			.cloned()
			.unwrap_or_else(|| json!(0));
		let body = json!({
			"type": "error",
			"code": null,
			"message": upstream_message(format, status),
			"param": null,
			"sequence_number": sequence_number
		});
		return Bytes::from(serde_json::to_vec(&body).expect("public stream error must serialize"));
	}
	synthesize(format, status, upstream_message(format, status))
}

fn encode_event(id: Option<Cow<'static, str>>, name: Cow<'static, str>, data: Bytes) -> Bytes {
	let mut out = BytesMut::new();
	if let Some(id) = id {
		out.extend_from_slice(b"id: ");
		out.extend_from_slice(id.as_bytes());
		out.extend_from_slice(b"\n");
	}
	if name.as_ref() != "message" {
		out.extend_from_slice(b"event: ");
		out.extend_from_slice(name.as_bytes());
		out.extend_from_slice(b"\n");
	}
	for line in data.split(|byte| *byte == b'\n') {
		out.extend_from_slice(b"data: ");
		out.extend_from_slice(line);
		out.extend_from_slice(b"\n");
	}
	out.extend_from_slice(b"\n");
	out.freeze()
}

fn encode_comment(comment: Bytes) -> Bytes {
	let mut out = BytesMut::new();
	for line in comment.split(|byte| *byte == b'\n') {
		out.extend_from_slice(b": ");
		out.extend_from_slice(line);
		out.extend_from_slice(b"\n");
	}
	out.extend_from_slice(b"\n");
	out.freeze()
}

fn encode_retry(duration: Duration) -> Bytes {
	Bytes::from(format!("retry: {}\n\n", duration.as_millis()))
}

#[cfg(test)]
mod tests {
	use http_body_util::BodyExt;

	use super::*;

	const SENSITIVE: &str = "User: arn:aws:sts::123456789012:assumed-role/prod-agentgateway-bedrock-irsa/session is not authorized to perform: bedrock:CountTokens on bedrock-runtime.eu-west-1.amazonaws.com; expected type MessagesRequest, got invalid type";

	#[test]
	fn buffered_errors_do_not_echo_provider_or_parser_details() {
		for format in [
			InputFormat::Messages,
			InputFormat::Completions,
			InputFormat::Responses,
			InputFormat::CountTokens,
		] {
			let translated = Bytes::from(format!(
				r#"{{"type":"error","error":{{"type":"api_error","message":"{SENSITIVE}"}}}}"#
			));
			let body = upstream_error(format, StatusCode::FORBIDDEN, &translated);
			let text = String::from_utf8_lossy(&body);
			for marker in [
				"123456789012",
				"assumed-role",
				"bedrock:CountTokens",
				"bedrock-runtime",
				"invalid type",
			] {
				assert!(!text.contains(marker), "{format:?} leaked {marker}: {text}");
			}
			let value: Value = serde_json::from_slice(&body).unwrap();
			if matches!(format, InputFormat::Messages | InputFormat::CountTokens) {
				assert_eq!(value["type"], "error");
				assert_eq!(value["error"]["type"], "api_error");
			} else {
				assert_eq!(value["error"]["type"], "api_error");
			}
		}
	}

	#[test]
	fn context_overflow_keeps_only_the_safe_capability_diagnostic() {
		let translated = Bytes::from(format!(
			r#"{{"error":{{"message":"capability_rejected: prompt_too_long {SENSITIVE}"}}}}"#
		));
		let body = upstream_error(InputFormat::Messages, StatusCode::BAD_REQUEST, &translated);
		let value: Value = serde_json::from_slice(&body).unwrap();
		assert_eq!(
			value["error"]["message"],
			"capability_rejected: prompt_too_long Input exceeds the model context window."
		);
		assert!(!String::from_utf8_lossy(&body).contains("assumed-role"));
	}

	#[test]
	fn response_headers_remove_provider_usage_and_trace_headers_on_success_and_error() {
		for sanitize in [
			strip_internal_response_headers as fn(&mut HeaderMap),
			sanitize_headers as fn(&mut HeaderMap),
		] {
			let mut headers = HeaderMap::new();
			headers.insert("fireworks-prompt-tokens", "123".parse().unwrap());
			headers.insert("traceresponse", "00-private-trace-id-01".parse().unwrap());
			headers.insert("x-request-id", "public-request-id".parse().unwrap());
			headers.insert("x-ratelimit-remaining", "9".parse().unwrap());
			headers.insert("retry-after", "3".parse().unwrap());
			sanitize(&mut headers);
			assert!(headers.get("fireworks-prompt-tokens").is_none());
			assert!(headers.get("traceresponse").is_none());
			assert_eq!(headers["x-request-id"], "public-request-id");
			assert_eq!(headers["x-ratelimit-remaining"], "9");
			assert_eq!(headers["retry-after"], "3");
		}
	}

	#[tokio::test]
	async fn stream_replaces_only_error_events_and_sanitizes_trailers() {
		let input = format!(
			": keepalive\n\nevent: message_start\nid: evt-1\ndata: {{\"type\":\"message_start\",\"message\":{{\"id\":\"msg_1\"}}}}\n\nevent: error\ndata: {{\"type\":\"error\",\"error\":{{\"type\":\"api_error\",\"message\":\"{SENSITIVE}\"}}}}\n\nretry: 2500\n\n"
		);
		let mut trailers = HeaderMap::new();
		trailers.insert("x-amzn-requestid", "private-provider-id".parse().unwrap());
		trailers.insert("fireworks-prompt-tokens", "123".parse().unwrap());
		trailers.insert("traceresponse", "00-private-trace-id-01".parse().unwrap());
		trailers.insert("x-synapse-request-id", "public-synapse-id".parse().unwrap());
		let frames = vec![
			Ok::<_, std::io::Error>(http_body::Frame::data(Bytes::from(input))),
			Ok(http_body::Frame::trailers(trailers)),
		];
		let body = sanitize_stream(
			InputFormat::Messages,
			Body::new(http_body_util::StreamBody::new(futures_util::stream::iter(
				frames,
			))),
			64 * 1024,
		);
		let collected = body.collect().await.unwrap();
		let trailers = collected.trailers().expect("trailers should be preserved");
		assert!(trailers.get("x-amzn-requestid").is_none());
		assert!(trailers.get("fireworks-prompt-tokens").is_none());
		assert!(trailers.get("traceresponse").is_none());
		assert_eq!(trailers["x-synapse-request-id"], "public-synapse-id");
		let output = collected.to_bytes();
		let text = String::from_utf8_lossy(&output);
		assert!(text.contains("event: message_start"));
		assert!(text.contains("id: evt-1"));
		assert!(text.contains("retry: 2500"));
		assert!(text.contains(": keepalive"));
		assert!(text.contains("The upstream provider failed to process the request."));
		assert!(!text.contains("123456789012"));
		assert!(!text.contains("assumed-role"));
		assert!(!text.contains("invalid type"));
	}

	#[tokio::test]
	async fn responses_failed_event_preserves_envelope_but_not_error_details() {
		let input = format!(
			"event: response.failed\ndata: {{\"type\":\"response.failed\",\"sequence_number\":9,\"response\":{{\"id\":\"resp_1\",\"status\":\"failed\",\"error\":{{\"code\":\"internal\",\"message\":\"{SENSITIVE}\"}}}}}}\n\n"
		);
		let body = sanitize_stream(InputFormat::Responses, Body::from(input), 64 * 1024);
		let output = body.collect().await.unwrap().to_bytes();
		let text = String::from_utf8_lossy(&output);
		assert!(text.contains("\"sequence_number\":9"));
		assert!(text.contains("\"id\":\"resp_1\""));
		assert!(!text.contains("assumed-role"));
		assert!(!text.contains("bedrock:CountTokens"));
	}
}
