//! Provider failover for LLM backends with ordered priority groups.
//!
//! These tests run the full gateway over real sockets against mock providers. They mirror the
//! production `kimi-k3-backend` and compare two policy sets:
//!   - `Profile::Incident`: the backend and policies live on 2026-10-25 (Fireworks -> Cloudflare ->
//!     OpenRouter, retry without codes, no provider deadline).
//!   - `Profile::Hardened`: Fireworks -> Amazon Bedrock (Global CRIS, OpenAI Chat Completions on
//!     `bedrock-runtime`, SigV4) with the Kimi-scoped retry, buffering, deadline and idle timeout.
//!     Cloudflare and OpenRouter are not configured; their mocks run only to prove they are never
//!     contacted.
//!
//! Durations are scaled down: 1s here stands for the 180s Fireworks deadline and the 120s stream
//! idle timeout, and 5s for the 600s route request timeout used in production.

use std::net::SocketAddr;

use agentgateway::test_helpers::oteltracemock;
use bytes::Bytes;
use futures_util::StreamExt;
use opentelemetry_proto::tonic::collector::trace::v1::{
	ExportTraceServiceRequest, ExportTraceServiceResponse,
};
use tokio::sync::mpsc;
use wiremock::matchers;

use crate::common::gateway::AgentGateway;
use crate::common::prelude::*;

const FIREWORKS_PATH: &str = "/inference/v1/chat/completions";
const CLOUDFLARE_PATH: &str = "/client/v4/accounts/test/ai/v1/chat/completions";
const OPENROUTER_PATH: &str = "/api/v1/chat/completions";
/// OpenAI-compatible Chat Completions on `bedrock-runtime` (not Converse, not `bedrock-mantle`).
const BEDROCK_PATH: &str = "/openai/v1/chat/completions";
const BEDROCK_MODEL: &str = "global.moonshotai.kimi-k3";
const BEDROCK_REGION: &str = "eu-west-1";
// Test-only static credentials. Production signs with the pod's IRSA credentials.
const AWS_TEST_ACCESS_KEY: &str = "AKIDKIMIFAILOVERTEST";
const AWS_TEST_SECRET_KEY: &str = "kimi-failover-test-secret";
const FIREWORKS_TEST_KEY: &str = "fireworks-test-key";
const FRONTEND_MAX_BUFFER: usize = 10 * 1024 * 1024;
const COMPLETION: &[u8] = include_bytes!("../../../llm/src/tests/response/completions/basic.json");

/// Bedrock request transformation, identical to the `bedrock-kimi-k3` provider in synapse-infra
/// `kimi-k3-backend.yaml`:
///   - drops the Fireworks-only cache/metrics fields (Bedrock uses implicit prompt caching);
///   - maps `reasoning_effort` exactly like the Fireworks provider so both providers behave the
///     same for a given request (Bedrock accepts low/high/max; it rejects `minimal`);
///   - fills `content: ""` on assistant tool-call turns, as for every Kimi K3 upstream.
const BEDROCK_TRANSFORM: &str = r#"json(request.body).with(body, toJson(
  body.filterKeys(key, !(key in ["prompt_cache_key", "prompt_cache_isolation_key", "perf_metrics_in_response"]))
  .merge({"messages": body.messages.map(message,
    message.role == "assistant" && has(message.tool_calls)
    ? message.merge({"content": default(message.content, "")})
    : message)})
  .merge(has(body.reasoning_effort)
    ? {"reasoning_effort": body.reasoning_effort == "low"
      ? "low"
      : (body.reasoning_effort == "medium" || body.reasoning_effort == "high")
        ? "high"
        : "max"}
    : {})))"#;

/// Fireworks request transformation and cache-key header expression, verbatim from the
/// `fireworks-kimi-k3` provider in synapse-infra `kimi-k3-backend.yaml`. The tests inject the
/// trusted `x-synapse-principal-id` header the way gatekeeper ext_authz does in production.
const FIREWORKS_TRANSFORM: &str = r#"has(json(request.body).reasoning_effort) ? toJson(json(request.body).merge({"reasoning_effort":
  json(request.body).reasoning_effort == "low"
  ? "low"
  : (json(request.body).reasoning_effort == "medium"
    || json(request.body).reasoning_effort == "high")
    ? "high"
    : "max",
  "prompt_cache_key": coalesce(apiKey.user_id,
    request.headers["x-synapse-principal-id"]),
  "prompt_cache_isolation_key": coalesce(apiKey.user_id,
    request.headers["x-synapse-principal-id"]),
  "perf_metrics_in_response": true,
  "messages": json(request.body).messages.map(message,
  message.role == "assistant" && has(message.tool_calls)
  ? message.merge({"content": default(message.content, "")})
  : message)}))
: toJson(json(request.body).merge({
  "prompt_cache_key": coalesce(apiKey.user_id,
    request.headers["x-synapse-principal-id"]),
  "prompt_cache_isolation_key": coalesce(apiKey.user_id,
    request.headers["x-synapse-principal-id"]),
  "perf_metrics_in_response": true,
  "messages":
  json(request.body).messages.map(message,
  message.role == "assistant" && has(message.tool_calls)
  ? message.merge({"content": default(message.content, "")})
  : message)}))"#;
const FIREWORKS_CACHE_KEY_HEADER: &str = r#"coalesce(apiKey.user_id,
  request.headers["x-synapse-principal-id"])"#;
const PRINCIPAL: &str = "principal-1";

#[derive(Clone, Copy, PartialEq)]
enum Profile {
	/// Live policies on 2026-10-25: retry without codes, `response.code` health condition,
	/// Fireworks -> Cloudflare -> OpenRouter, no provider deadline.
	Incident,
	/// Kimi-scoped retry on 429/502/503/504 with a single retry, request buffering opt-in for
	/// replaying large prompts, Fireworks deadline, stream idle timeout, Fireworks -> Bedrock.
	Hardened,
	/// `Hardened` without the `buffer.request` opt-in: the shape every other retry route has.
	HardenedWithoutBufferOptIn,
}

struct Providers {
	fireworks: String,
	cloudflare: String,
	bedrock: String,
	openrouter: String,
}

fn indent(text: &str, spaces: usize) -> String {
	let pad = " ".repeat(spaces);
	text
		.lines()
		.map(|l| format!("{pad}{l}"))
		.collect::<Vec<_>>()
		.join("\n")
}

fn gateway_config(p: &Providers, profile: Profile, tracing: Option<SocketAddr>) -> String {
	let tracing = tracing
		.map(|addr| {
			format!(
				r#"
  tracing:
    host: {addr}
    randomSampling: true"#
			)
		})
		.unwrap_or_default();
	// Production evaluates `json(request.body)` before routing (extract-model and the retry
	// precondition), so the request body is always fully buffered when the retry loop starts.
	let precondition =
		r#"'!(default(json(request.body).model, "") in ["zai-org/GLM-OCR", "dots-ocr"])'"#;
	let (route_policies, groups) = match profile {
		Profile::Incident => (
			format!(
				r#"
        retry:
          attempts: 2
          backoff: 10ms
          codes: []
          precondition: {precondition}
        timeout:
          requestTimeout: 600s"#
			),
			format!(
				r#"
          - providers:
            - name: fireworks-kimi-k3
              provider:
                openAI:
                  model: accounts/fireworks/models/kimi-k3
              hostOverride: {fireworks}
              pathOverride: {FIREWORKS_PATH}
          - providers:
            - name: cloudflare-kimi-k3
              provider:
                openAI:
                  model: moonshotai/kimi-k3
              hostOverride: {cloudflare}
              pathOverride: {CLOUDFLARE_PATH}
          - providers:
            - name: openrouter-kimi-k3
              provider:
                openAI:
                  model: moonshotai/kimi-k3
              hostOverride: {openrouter}
              pathOverride: {OPENROUTER_PATH}"#,
				fireworks = p.fireworks,
				cloudflare = p.cloudflare,
				openrouter = p.openrouter,
			),
		),
		Profile::Hardened | Profile::HardenedWithoutBufferOptIn => (
			format!(
				r#"{buffer}
        retry:
          attempts: 1
          backoff: 10ms
          codes: [429, 502, 503, 504]
          precondition: {precondition}
        timeout:
          requestTimeout: 5s
          responseIdleTimeout: 1s"#,
				buffer = if profile == Profile::Hardened {
					format!(
						r#"
        buffer:
          request:
            maxBytes: {FRONTEND_MAX_BUFFER}
            failureMode: failClosed"#
					)
				} else {
					String::new()
				}
			),
			format!(
				r#"
          - providers:
            - name: fireworks-kimi-k3
              provider:
                openAI:
                  model: accounts/fireworks/models/kimi-k3
              hostOverride: {fireworks}
              pathOverride: {FIREWORKS_PATH}
              policies:
                http:
                  requestTimeout: 1s
                backendAuth:
                  key: {FIREWORKS_TEST_KEY}
                transformations:
                  request:
                    set:
                      x-session-affinity: |
{key_header}
                      x-prompt-cache-isolation-key: |
{key_header}
                    body: |
{fireworks_transform}
          - providers:
            - name: bedrock-kimi-k3
              provider:
                openAI:
                  model: {BEDROCK_MODEL}
              hostOverride: {bedrock}
              pathOverride: {BEDROCK_PATH}
              policies:
                backendAuth:
                  aws:
                    accessKeyId: {AWS_TEST_ACCESS_KEY}
                    secretAccessKey: {AWS_TEST_SECRET_KEY}
                    region: {BEDROCK_REGION}
                    serviceName: bedrock
                transformations:
                  request:
                    body: |
{bedrock_transform}"#,
				fireworks = p.fireworks,
				bedrock = p.bedrock,
				fireworks_transform = indent(FIREWORKS_TRANSFORM, 22),
				key_header = indent(FIREWORKS_CACHE_KEY_HEADER, 24),
				bedrock_transform = indent(BEDROCK_TRANSFORM, 22),
			),
		),
	};
	format!(
		r#"
config: {{}}
frontendPolicies:
  http:
    maxBufferSize: {FRONTEND_MAX_BUFFER}{tracing}
binds:
- port: $PORT
  listeners:
  - name: llm
    protocol: HTTP
    routes:
    - name: kimi-k3
      matches:
      - path:
          pathPrefix: /v1
      policies:
        requestHeaderModifier:
          set:
            x-synapse-principal-id: {PRINCIPAL}{route_policies}
      backends:
      - ai:
          groups:{groups}
        policies:
          health:
            unhealthyExpression: "response.code >= 500 || response.code == 429"
            eviction:
              consecutiveFailures: 1
              duration: 30s
"#
	)
}

async fn provider(path: &str, response: ResponseTemplate) -> MockServer {
	let mock = MockServer::start().await;
	Mock::given(matchers::method("POST"))
		.and(matchers::path(path))
		.respond_with(response)
		.mount(&mock)
		.await;
	mock
}

fn ok() -> ResponseTemplate {
	ResponseTemplate::new(200).set_body_raw(COMPLETION, "application/json")
}

/// Non-streaming Bedrock Chat Completions response with implicit-cache usage fields
/// (`prompt_tokens_details.cached_tokens` / `cache_write_tokens`, as documented by AWS).
fn bedrock_ok() -> ResponseTemplate {
	ResponseTemplate::new(200).set_body_json(json!({
		"id": "chatcmpl-bedrock-1",
		"object": "chat.completion",
		"created": 1,
		"model": "moonshotai.kimi-k3",
		"choices": [{"index": 0, "finish_reason": "stop", "message": {
			"role": "assistant", "content": "done", "reasoning_content": "thinking"}}],
		"usage": {"prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240,
			"prompt_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 150}},
	}))
}

fn failure(code: u16) -> ResponseTemplate {
	ResponseTemplate::new(code).set_body_raw(
		r#"{"error":{"message":"Service Overloaded"}}"#,
		"application/json",
	)
}

async fn received(mock: &MockServer) -> Vec<wiremock::Request> {
	mock.received_requests().await.unwrap_or_default()
}

/// A provider that accepts connections and resets them before sending a response.
async fn resetting_provider() -> (String, Arc<AtomicUsize>) {
	let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
	let addr = listener.local_addr().unwrap();
	let accepted = Arc::new(AtomicUsize::new(0));
	let count = accepted.clone();
	tokio::spawn(async move {
		while let Ok((mut stream, _)) = listener.accept().await {
			count.fetch_add(1, Ordering::SeqCst);
			let mut buf = [0u8; 1024];
			let _ = stream.read(&mut buf).await;
			// SO_LINGER(0) turns the close into a TCP RST; a zero timeout never blocks.
			#[allow(deprecated)]
			let _ = stream.set_linger(Some(Duration::ZERO));
			drop(stream);
		}
	});
	(addr.to_string(), accepted)
}

/// A provider that reads the request and never answers; reports when the gateway closes the
/// connection.
async fn silent_provider() -> (String, Arc<AtomicUsize>, mpsc::UnboundedReceiver<()>) {
	let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
	let addr = listener.local_addr().unwrap();
	let accepted = Arc::new(AtomicUsize::new(0));
	let count = accepted.clone();
	let (closed_tx, closed_rx) = mpsc::unbounded_channel();
	tokio::spawn(async move {
		while let Ok((mut stream, _)) = listener.accept().await {
			count.fetch_add(1, Ordering::SeqCst);
			let closed_tx = closed_tx.clone();
			tokio::spawn(async move {
				let mut buf = vec![0u8; 64 * 1024];
				loop {
					match stream.read(&mut buf).await {
						Ok(0) | Err(_) => {
							let _ = closed_tx.send(());
							return;
						},
						Ok(_) => {},
					}
				}
			});
		}
	});
	(addr.to_string(), accepted, closed_rx)
}

/// A provider that streams `chunks` SSE events `interval` apart, then optionally stalls forever.
async fn streaming_provider(chunks: usize, interval: Duration, stall: bool) -> String {
	let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
	let addr = listener.local_addr().unwrap();
	tokio::spawn(async move {
		while let Ok((stream, _)) = listener.accept().await {
			tokio::spawn(async move {
				let service = service_fn(
					move |_req: http::Request<hyper::body::Incoming>| async move {
						let events = futures_util::stream::unfold(0usize, move |i| async move {
							if i > 0 {
								tokio::time::sleep(interval).await;
							}
							if i < chunks {
								let event = format!(
									"data: {{\"id\":\"c\",\"object\":\"chat.completion.chunk\",\"created\":1,\"model\":\"m\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"t{i}\"}},\"finish_reason\":null}}]}}\n\n"
								);
								Some((Ok::<_, Infallible>(Frame::data(Bytes::from(event))), i + 1))
							} else if stall {
								futures_util::future::pending::<()>().await;
								None
							} else if i == chunks {
								Some((
									Ok(Frame::data(Bytes::from_static(b"data: [DONE]\n\n"))),
									i + 1,
								))
							} else {
								None
							}
						});
						Ok::<_, Infallible>(
							http::Response::builder()
								.header("content-type", "text/event-stream")
								.body(StreamBody::new(events))
								.unwrap(),
						)
					},
				);
				let _ = hyper::server::conn::http1::Builder::new()
					.serve_connection(TokioIo::new(stream), service)
					.await;
			});
		}
	});
	addr.to_string()
}

/// Bedrock Chat Completions stream: reasoning, a tool call split across deltas, and the final
/// usage chunk requested by `stream_options.include_usage`.
fn bedrock_tool_call_stream() -> ResponseTemplate {
	let chunk = |delta: Value, finish: Value| {
		json!({"id": "chatcmpl-bedrock-2", "object": "chat.completion.chunk", "created": 1,
			"model": "moonshotai.kimi-k3",
			"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
	};
	let events = [
		chunk(
			json!({"role": "assistant", "reasoning_content": "need the file"}),
			Value::Null,
		),
		chunk(
			json!({"tool_calls": [{"index": 0, "id": "call_9", "type": "function",
			"function": {"name": "read_file", "arguments": ""}}]}),
			Value::Null,
		),
		chunk(
			json!({"tool_calls": [{"index": 0, "function": {"arguments": "{\"path\":"}}]}),
			Value::Null,
		),
		chunk(
			json!({"tool_calls": [{"index": 0, "function": {"arguments": "\"README.md\"}"}}]}),
			Value::Null,
		),
		chunk(json!({}), json!("tool_calls")),
		json!({"id": "chatcmpl-bedrock-2", "object": "chat.completion.chunk", "created": 1,
			"model": "moonshotai.kimi-k3", "choices": [],
			"usage": {"prompt_tokens": 1200, "completion_tokens": 40, "total_tokens": 1240,
				"prompt_tokens_details": {"cached_tokens": 1000, "cache_write_tokens": 0}}}),
	];
	let mut body = String::new();
	for e in events {
		body.push_str(&format!("data: {e}\n\n"));
	}
	body.push_str("data: [DONE]\n\n");
	ResponseTemplate::new(200).set_body_raw(body, "text/event-stream")
}

/// Synthetic agent prompt of `size` bytes (production median ~50k tokens, max ~267k).
fn prompt_of(size: usize, tag: &str) -> Value {
	json!({
		"model": "moonshotai.kimi-k3",
		"max_tokens": 1024,
		"messages": [
			{"role": "system", "content": "x".repeat(size)},
			{"role": "user", "content": format!("request {tag}")},
		],
	})
}

/// Larger than the 64 KiB replay cap.
fn large_request(tag: &str) -> Value {
	prompt_of(200 * 1024, tag)
}

fn small_request(tag: &str) -> Value {
	json!({
		"model": "moonshotai.kimi-k3",
		"messages": [{"role": "user", "content": format!("request {tag}")}],
	})
}

fn tool_request() -> Value {
	json!({
		"model": "moonshotai.kimi-k3",
		"tools": [{"type": "function", "function": {
			"name": "read_file",
			"description": "Read a file",
			"parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
		}}],
		"messages": [
			{"role": "system", "content": "x".repeat(100 * 1024)},
			{"role": "developer", "content": "Prefer tools."},
			{"role": "user", "content": "open README"},
			{"role": "assistant", "reasoning_content": "I should read it.", "tool_calls": [{"id": "call_1", "type": "function",
				"function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}}]},
			{"role": "tool", "tool_call_id": "call_1", "content": "hello"},
			{"role": "assistant", "content": "It says hello.", "reasoning_content": "Summarize."},
			{"role": "user", "content": "and now?"},
		],
	})
}

fn unique_path() -> String {
	format!("/v1/chat/completions/{:x}", rand::rng().random::<u64>())
}

struct Outcome {
	status: StatusCode,
	body: Bytes,
	path: String,
}

async fn send(gw: &AgentGateway, body: Value) -> Outcome {
	let path = unique_path();
	let resp = gw
		.send_request_json(&format!("http://localhost{path}"), body)
		.await;
	let status = resp.status();
	let body = resp.into_body().collect().await.unwrap().to_bytes();
	Outcome { status, body, path }
}

async fn access_log(path: &str) -> Value {
	agent_core::telemetry::testing::eventually_find(&[("scope", "request"), ("http.path", path)])
		.await
		.expect("access log for request")
}

fn sent_body(req: &wiremock::Request) -> Value {
	serde_json::from_slice(&req.body).unwrap()
}

fn upstream_model(req: &wiremock::Request) -> String {
	sent_body(req)["model"].as_str().unwrap().to_string()
}

fn header<'a>(req: &'a wiremock::Request, name: &str) -> Option<&'a str> {
	req.headers.get(name).and_then(|v| v.to_str().ok())
}

/// Recomputes the SigV4 signature over the request exactly as Bedrock received it and checks it
/// matches. This proves the signature covers the final (transformed, replayed) body and URL, the
/// signing scope is `eu-west-1/bedrock`, and no other provider's credential was forwarded.
fn assert_sigv4_signed_for_bedrock(req: &wiremock::Request) {
	use aws_credential_types::Credentials;
	use aws_sigv4::http_request::{SignableBody, SignableRequest, SigningSettings, sign};
	use aws_sigv4::sign::v4::SigningParams;

	let auth = header(req, "authorization").expect("authorization header");
	assert!(
		auth.starts_with(&format!(
			"AWS4-HMAC-SHA256 Credential={AWS_TEST_ACCESS_KEY}/"
		)),
		"{auth}"
	);
	assert!(
		auth.contains(&format!("/{BEDROCK_REGION}/bedrock/aws4_request")),
		"{auth}"
	);
	assert!(!auth.contains(FIREWORKS_TEST_KEY));
	let signed_headers: Vec<&str> = auth
		.split("SignedHeaders=")
		.nth(1)
		.and_then(|s| s.split(',').next())
		.expect("signed headers")
		.split(';')
		.collect();
	let amz_date = header(req, "x-amz-date").expect("x-amz-date");
	let time: std::time::SystemTime =
		chrono::NaiveDateTime::parse_from_str(amz_date, "%Y%m%dT%H%M%SZ")
			.unwrap()
			.and_utc()
			.into();
	let identity =
		Credentials::new(AWS_TEST_ACCESS_KEY, AWS_TEST_SECRET_KEY, None, None, "test").into();
	let params = SigningParams::builder()
		.identity(&identity)
		.region(BEDROCK_REGION)
		.name("bedrock")
		.time(time)
		.settings(SigningSettings::default())
		.build()
		.unwrap()
		.into();
	let headers: Vec<(&str, &str)> = signed_headers
		.iter()
		.map(|name| (*name, header(req, name).expect("signed header present")))
		.collect();
	let uri = format!(
		"https://{}{}",
		header(req, "host").expect("host"),
		req.url.path()
	);
	let signable = SignableRequest::new(
		"POST",
		uri,
		headers.into_iter(),
		SignableBody::Bytes(&req.body),
	)
	.unwrap();
	let (_, signature) = sign(signable, &params).unwrap().into_parts();
	assert!(
		auth.ends_with(&format!("Signature={signature}")),
		"signature does not cover the received request: {auth}"
	);
}

/// What Bedrock must and must not receive after a failover from Fireworks.
fn assert_bedrock_request(req: &wiremock::Request) {
	assert_eq!(req.url.path(), BEDROCK_PATH);
	assert_sigv4_signed_for_bedrock(req);
	let body = sent_body(req);
	assert_eq!(body["model"], BEDROCK_MODEL);
	for field in [
		"prompt_cache_key",
		"prompt_cache_isolation_key",
		"perf_metrics_in_response",
	] {
		assert!(body.get(field).is_none(), "{field} leaked to Bedrock");
	}
	for name in ["x-session-affinity", "x-prompt-cache-isolation-key"] {
		assert!(header(req, name).is_none(), "{name} leaked to Bedrock");
	}
}

struct Setup {
	gw: AgentGateway,
	fireworks: MockServer,
	cloudflare: MockServer,
	bedrock: MockServer,
	openrouter: MockServer,
}

impl Setup {
	/// Fallback provider of the profile: Cloudflare in the incident shape, Bedrock when hardened.
	fn fallback(&self, profile: Profile) -> &MockServer {
		match profile {
			Profile::Incident => &self.cloudflare,
			_ => &self.bedrock,
		}
	}

	/// Cloudflare and OpenRouter are not approved automatic fallbacks for Kimi K3.
	async fn assert_no_unapproved_provider(&self) {
		assert_eq!(
			received(&self.cloudflare).await.len(),
			0,
			"Cloudflare contacted"
		);
		assert_eq!(
			received(&self.openrouter).await.len(),
			0,
			"OpenRouter contacted"
		);
	}
}

/// Mocks for every provider except Fireworks. `bedrock` answers with `bedrock_response`.
struct Others {
	cloudflare: MockServer,
	bedrock: MockServer,
	openrouter: MockServer,
}

impl Others {
	async fn start(bedrock_response: ResponseTemplate) -> Self {
		Self {
			cloudflare: provider(CLOUDFLARE_PATH, ok()).await,
			bedrock: provider(BEDROCK_PATH, bedrock_response).await,
			openrouter: provider(OPENROUTER_PATH, ok()).await,
		}
	}

	fn providers(&self, fireworks: String) -> Providers {
		Providers {
			fireworks,
			cloudflare: self.cloudflare.address().to_string(),
			bedrock: self.bedrock.address().to_string(),
			openrouter: self.openrouter.address().to_string(),
		}
	}

	async fn assert_no_unapproved_provider(&self) {
		assert_eq!(
			received(&self.cloudflare).await.len(),
			0,
			"Cloudflare contacted"
		);
		assert_eq!(
			received(&self.openrouter).await.len(),
			0,
			"OpenRouter contacted"
		);
	}
}

async fn setup(profile: Profile, fireworks: ResponseTemplate, fallback: ResponseTemplate) -> Setup {
	let fireworks = provider(FIREWORKS_PATH, fireworks).await;
	let (cloudflare, bedrock) = match profile {
		Profile::Incident => (
			provider(CLOUDFLARE_PATH, fallback).await,
			provider(BEDROCK_PATH, bedrock_ok()).await,
		),
		_ => (
			provider(CLOUDFLARE_PATH, ok()).await,
			provider(BEDROCK_PATH, fallback).await,
		),
	};
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks: fireworks.address().to_string(),
		cloudflare: cloudflare.address().to_string(),
		bedrock: bedrock.address().to_string(),
		openrouter: openrouter.address().to_string(),
	};
	let gw = AgentGateway::new(gateway_config(&providers, profile, None))
		.await
		.unwrap();
	Setup {
		gw,
		fireworks,
		cloudflare,
		bedrock,
		openrouter,
	}
}

// ---------------------------------------------------------------------------------------------
// Incident reproduction: the backend and policies deployed on 2026-10-25.
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn incident_503_reaches_client_and_only_this_replica_evicts() {
	let s = setup(Profile::Incident, failure(503), ok()).await;
	let cloudflare = s.fallback(Profile::Incident);

	// The first request is not retried (no `codes`), so the client sees Fireworks' 503.
	let first = send(&s.gw, large_request("a")).await;
	assert_eq!(first.status, StatusCode::SERVICE_UNAVAILABLE);
	assert_eq!(received(&s.fireworks).await.len(), 1);
	assert_eq!(received(cloudflare).await.len(), 0);

	// The 503 evicted Fireworks in this process only, so the next request uses Cloudflare.
	let second = send(&s.gw, large_request("b")).await;
	assert_eq!(second.status, StatusCode::OK);
	assert_eq!(received(&s.fireworks).await.len(), 1);
	assert_eq!(received(cloudflare).await.len(), 1);
}

#[tokio::test]
async fn incident_hung_provider_is_never_evicted() {
	let s = setup(
		Profile::Incident,
		ok().set_delay(Duration::from_secs(30)),
		ok(),
	)
	.await;
	// The client gives up (OpenHands timeout / edge idle timeout, scaled to 1s).
	let path = unique_path();
	let res = tokio::time::timeout(
		Duration::from_secs(1),
		s.gw
			.send_request_json(&format!("http://localhost{path}"), large_request("hang")),
	)
	.await;
	assert!(res.is_err(), "the hung provider never answers");
	tokio::time::sleep(Duration::from_millis(200)).await;

	// No status means no eviction, so the next request goes back to the hung provider.
	let path = unique_path();
	let res = tokio::time::timeout(
		Duration::from_secs(1),
		s.gw
			.send_request_json(&format!("http://localhost{path}"), large_request("hang2")),
	)
	.await;
	assert!(res.is_err());
	assert_eq!(received(s.fallback(Profile::Incident)).await.len(), 0);
}

#[tokio::test]
async fn incident_eviction_already_reaches_openrouter() {
	// Each 503 evicts its provider for 30s on this replica, so when Fireworks and Cloudflare both
	// fail, the next request is sent to the third priority group (OpenRouter) even without
	// retries.
	let s = setup(Profile::Incident, failure(503), failure(503)).await;
	assert_eq!(
		send(&s.gw, small_request("x")).await.status,
		StatusCode::SERVICE_UNAVAILABLE
	);
	assert_eq!(
		send(&s.gw, small_request("y")).await.status,
		StatusCode::SERVICE_UNAVAILABLE
	);
	assert_eq!(send(&s.gw, small_request("z")).await.status, StatusCode::OK);
	assert_eq!(received(&s.openrouter).await.len(), 1);
}

#[tokio::test]
async fn incident_three_replicas_expose_one_failure_per_replica() {
	let fireworks = provider(FIREWORKS_PATH, failure(503)).await;
	let others = Others::start(bedrock_ok()).await;
	let providers = others.providers(fireworks.address().to_string());
	let mut replicas = Vec::new();
	for _ in 0..3 {
		replicas.push(
			AgentGateway::new(gateway_config(&providers, Profile::Incident, None))
				.await
				.unwrap(),
		);
	}
	let mut failures = 0;
	for round in 0..3 {
		for (i, gw) in replicas.iter().enumerate() {
			let res = send(gw, large_request(&format!("r{round}-g{i}"))).await;
			if res.status != StatusCode::OK {
				failures += 1;
			}
		}
	}
	assert_eq!(failures, 3);
	assert_eq!(received(&fireworks).await.len(), 3);
	assert_eq!(received(&others.cloudflare).await.len(), 6);
}

// ---------------------------------------------------------------------------------------------
// Hardened policies: Fireworks -> Bedrock.
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn hardened_200_does_not_fail_over() {
	let s = setup(Profile::Hardened, ok(), bedrock_ok()).await;
	let res = send(&s.gw, large_request("ok")).await;
	assert_eq!(res.status, StatusCode::OK);
	let fw = received(&s.fireworks).await;
	assert_eq!(fw.len(), 1);
	assert_eq!(upstream_model(&fw[0]), "accounts/fireworks/models/kimi-k3");
	// The Fireworks-specific transformation and credential apply on the Fireworks attempt.
	assert_eq!(sent_body(&fw[0])["prompt_cache_key"], PRINCIPAL);
	assert_eq!(header(&fw[0], "x-session-affinity"), Some(PRINCIPAL));
	assert_eq!(
		header(&fw[0], "authorization"),
		Some(format!("Bearer {FIREWORKS_TEST_KEY}").as_str())
	);
	assert_eq!(received(&s.bedrock).await.len(), 0);
	s.assert_no_unapproved_provider().await;
	let log = access_log(&res.path).await;
	assert!(log.get("retry.attempt").is_none(), "{log:#}");
	assert_eq!(log["endpoint"], s.fireworks.address().to_string());
}

#[rstest::rstest]
#[case::service_unavailable(503)]
#[case::bad_gateway(502)]
#[case::gateway_timeout(504)]
#[case::rate_limited(429)]
#[tokio::test]
async fn hardened_retryable_status_fails_over_to_bedrock(#[case] code: u16) {
	let s = setup(Profile::Hardened, failure(code), bedrock_ok()).await;
	let res = send(&s.gw, large_request("fo")).await;
	assert_eq!(
		res.status,
		StatusCode::OK,
		"{}",
		String::from_utf8_lossy(&res.body)
	);

	let fw = received(&s.fireworks).await;
	let br = received(&s.bedrock).await;
	assert_eq!(fw.len(), 1);
	assert_eq!(br.len(), 1);
	assert_bedrock_request(&br[0]);
	// The replayed prompt is the full original prompt, not a truncated replay buffer.
	assert_eq!(
		sent_body(&br[0])["messages"][0]["content"]
			.as_str()
			.unwrap()
			.len(),
		200 * 1024
	);
	s.assert_no_unapproved_provider().await;
	// The Bedrock response reaches the client unchanged.
	let body: Value = serde_json::from_slice(&res.body).unwrap();
	assert_eq!(body["choices"][0]["message"]["content"], "done");

	// One downstream request -> one usage record, attributed to the provider that served it.
	let log = access_log(&res.path).await;
	assert_eq!(log["retry.attempt"], 1, "{log:#}");
	assert_eq!(log["endpoint"], s.bedrock.address().to_string());
	assert_eq!(
		log["retry.reasons"],
		format!("{}={code}", s.fireworks.address())
	);
	assert_eq!(log["gen_ai.request.model"], BEDROCK_MODEL, "{log:#}");
	assert_eq!(log["gen_ai.usage.input_tokens"], 1200, "{log:#}");
	assert_eq!(log["gen_ai.usage.output_tokens"], 40, "{log:#}");
	assert_eq!(log["gen_ai.usage.cache_read.input_tokens"], 1000, "{log:#}");
	assert_eq!(
		log["gen_ai.usage.cache_creation.input_tokens"], 150,
		"{log:#}"
	);

	// Fireworks stays evicted for this replica: the next request goes straight to Bedrock.
	let next = send(&s.gw, large_request("next")).await;
	assert_eq!(next.status, StatusCode::OK);
	assert_eq!(received(&s.fireworks).await.len(), 1);
	assert_eq!(received(&s.bedrock).await.len(), 2);
}

#[rstest::rstest]
#[case::kib_200(200 * 1024)]
#[case::mib_1(1024 * 1024)]
#[case::mib_8(8 * 1024 * 1024)]
#[tokio::test]
async fn hardened_large_prompts_replay_to_bedrock_in_full(#[case] size: usize) {
	let s = setup(Profile::Hardened, failure(503), bedrock_ok()).await;
	let res = send(&s.gw, prompt_of(size, "big")).await;
	assert_eq!(res.status, StatusCode::OK, "{size}");
	let br = received(&s.bedrock).await;
	assert_eq!(br.len(), 1);
	assert_bedrock_request(&br[0]);
	assert_eq!(
		sent_body(&br[0])["messages"][0]["content"]
			.as_str()
			.unwrap()
			.len(),
		size
	);
}

#[tokio::test]
async fn prompt_over_buffer_limit_is_rejected_as_today_and_never_replayed() {
	// A body larger than the 10 MiB frontend limit cannot be parsed for LLM routing, so it is
	// rejected with 413 before any provider is contacted, exactly as with the incident-day
	// policies. With `failureMode: failClosed` the buffer policy rejects it with the same 413;
	// `failOpen` would turn it into a misleading 503 (looks transient, invites client retries).
	// No provider ever gets a truncated prompt.
	let huge = || prompt_of(FRONTEND_MAX_BUFFER + 1024 * 1024, "huge");
	let today = setup(Profile::Incident, ok(), ok()).await;
	let hardened = setup(Profile::Hardened, ok(), bedrock_ok()).await;
	let before = send(&today.gw, huge()).await;
	let after = send(&hardened.gw, huge()).await;
	assert_eq!(after.status, before.status);
	assert_eq!(after.status, StatusCode::PAYLOAD_TOO_LARGE);
	for s in [&today, &hardened] {
		assert_eq!(received(&s.fireworks).await.len(), 0);
		assert_eq!(received(&s.bedrock).await.len(), 0);
		s.assert_no_unapproved_provider().await;
	}
}

#[tokio::test]
async fn large_prompt_replay_requires_buffer_opt_in() {
	// Routes without a `buffer.request` policy keep the 64 KiB replay cap, so this fix does not
	// change retry behavior for other routes (Bedrock, Claude, GLM) that have retry policies.
	let s = setup(
		Profile::HardenedWithoutBufferOptIn,
		failure(503),
		bedrock_ok(),
	)
	.await;
	let large = send(&s.gw, large_request("large")).await;
	assert_eq!(large.status, StatusCode::SERVICE_UNAVAILABLE);
	assert_eq!(received(&s.bedrock).await.len(), 0);
	// Small prompts were always replayable; the 503 evicted Fireworks, so this one goes to
	// Bedrock directly.
	let small = send(&s.gw, small_request("small")).await;
	assert_eq!(small.status, StatusCode::OK);
	assert_eq!(received(&s.fireworks).await.len(), 1);
	assert_eq!(received(&s.bedrock).await.len(), 1);
}

#[tokio::test]
async fn hardened_connection_reset_fails_over() {
	let (fireworks, accepted) = resetting_provider().await;
	let others = Others::start(bedrock_ok()).await;
	let gw = AgentGateway::new(gateway_config(
		&others.providers(fireworks.clone()),
		Profile::Hardened,
		None,
	))
	.await
	.unwrap();

	let res = send(&gw, large_request("reset")).await;
	assert_eq!(res.status, StatusCode::OK);
	assert!(accepted.load(Ordering::SeqCst) >= 1);
	let br = received(&others.bedrock).await;
	assert_eq!(br.len(), 1);
	assert_bedrock_request(&br[0]);
	let log = access_log(&res.path).await;
	assert_eq!(
		log["retry.reasons"],
		format!("{fireworks}=upstream_call_failed"),
		"{log:#}"
	);

	// The reset evicted Fireworks before the retry, and it stays evicted.
	let before = accepted.load(Ordering::SeqCst);
	let next = send(&gw, large_request("reset2")).await;
	assert_eq!(next.status, StatusCode::OK);
	assert_eq!(accepted.load(Ordering::SeqCst), before);
	assert_eq!(received(&others.bedrock).await.len(), 2);
	others.assert_no_unapproved_provider().await;
}

#[tokio::test]
async fn hardened_hang_before_first_byte_recovers_within_deadline() {
	let s = setup(
		Profile::Hardened,
		ok().set_delay(Duration::from_secs(30)),
		bedrock_ok(),
	)
	.await;
	let start = std::time::Instant::now();
	let res = send(&s.gw, large_request("hang")).await;
	let elapsed = start.elapsed();
	assert_eq!(res.status, StatusCode::OK);
	// Fireworks deadline (1s) + backoff + Bedrock, well before the client's own timeout.
	assert!(elapsed < Duration::from_secs(3), "took {elapsed:?}");
	assert!(elapsed >= Duration::from_secs(1), "took {elapsed:?}");
	let br = received(&s.bedrock).await;
	assert_eq!(br.len(), 1);
	assert_bedrock_request(&br[0]);
	let log = access_log(&res.path).await;
	assert_eq!(
		log["retry.reasons"],
		format!("{}=upstream_timeout", s.fireworks.address()),
		"{log:#}"
	);

	// The timeout evicted Fireworks: no second 1s wait for the next request.
	let start = std::time::Instant::now();
	let next = send(&s.gw, large_request("hang2")).await;
	assert_eq!(next.status, StatusCode::OK);
	assert!(start.elapsed() < Duration::from_millis(900));
	assert_eq!(received(&s.fireworks).await.len(), 1);
	s.assert_no_unapproved_provider().await;
}

#[tokio::test]
async fn hardened_long_stream_is_not_cut_by_provider_deadline() {
	// 10 chunks 300ms apart = ~3s total, longer than the 1s Fireworks deadline and with gaps
	// shorter than the 1s idle timeout. The deadline only bounds the wait for response headers.
	let fireworks = streaming_provider(10, Duration::from_millis(300), false).await;
	let others = Others::start(bedrock_ok()).await;
	let gw = AgentGateway::new(gateway_config(
		&others.providers(fireworks),
		Profile::Hardened,
		None,
	))
	.await
	.unwrap();
	let mut body = large_request("stream");
	body["stream"] = json!(true);
	let res = send(&gw, body).await;
	assert_eq!(res.status, StatusCode::OK);
	let text = String::from_utf8_lossy(&res.body);
	for i in 0..10 {
		assert!(
			text.contains(&format!("\"t{i}\"")),
			"missing chunk {i}: {text}"
		);
	}
	assert!(text.contains("[DONE]"), "{text}");
	assert_eq!(received(&others.bedrock).await.len(), 0);
}

#[tokio::test]
async fn hardened_stream_stall_after_first_token_is_terminated() {
	let fireworks = streaming_provider(1, Duration::from_millis(10), true).await;
	let others = Others::start(bedrock_ok()).await;
	let gw = AgentGateway::new(gateway_config(
		&others.providers(fireworks),
		Profile::Hardened,
		None,
	))
	.await
	.unwrap();
	let mut body = large_request("stall");
	body["stream"] = json!(true);
	let path = unique_path();
	let start = std::time::Instant::now();
	let resp = gw
		.send_request_json(&format!("http://localhost{path}"), body)
		.await;
	assert_eq!(resp.status(), StatusCode::OK);
	let mut stream = resp.into_body().into_data_stream();
	let first = stream
		.next()
		.await
		.expect("first chunk")
		.expect("first chunk ok");
	assert!(String::from_utf8_lossy(&first).contains("\"t0\""));
	// The idle timeout ends the body with an error instead of holding it open until the edge
	// idle timeout. Headers were already sent, so there is no retry on another provider.
	let rest = tokio::time::timeout(Duration::from_secs(5), async {
		let mut errored = false;
		while let Some(frame) = stream.next().await {
			if frame.is_err() {
				errored = true;
				break;
			}
		}
		errored
	})
	.await
	.expect("stalled stream must be terminated by the idle timeout");
	assert!(rest, "stalled stream should end with an error");
	assert!(start.elapsed() < Duration::from_secs(3));
	assert_eq!(received(&others.bedrock).await.len(), 0);
}

#[tokio::test]
async fn hardened_streaming_tool_call_from_bedrock_after_failover() {
	let s = setup(Profile::Hardened, failure(503), bedrock_tool_call_stream()).await;
	let mut req = tool_request();
	req["stream"] = json!(true);
	let res = send(&s.gw, req).await;
	assert_eq!(res.status, StatusCode::OK);
	let br = received(&s.bedrock).await;
	assert_eq!(br.len(), 1);
	assert_bedrock_request(&br[0]);
	// The gateway asks for the final usage chunk so the attempt can be accounted.
	assert_eq!(sent_body(&br[0])["stream_options"]["include_usage"], true);

	// The client can reconstruct the streamed tool call from the deltas.
	let text = String::from_utf8_lossy(&res.body);
	let mut name = String::new();
	let mut arguments = String::new();
	let mut finish = None;
	for line in text.lines().filter_map(|l| l.strip_prefix("data: ")) {
		if line == "[DONE]" {
			continue;
		}
		let event: Value = serde_json::from_str(line).unwrap();
		for choice in event["choices"].as_array().into_iter().flatten() {
			for call in choice["delta"]["tool_calls"]
				.as_array()
				.into_iter()
				.flatten()
			{
				name.push_str(call["function"]["name"].as_str().unwrap_or_default());
				arguments.push_str(call["function"]["arguments"].as_str().unwrap_or_default());
			}
			if let Some(f) = choice["finish_reason"].as_str() {
				finish = Some(f.to_string());
			}
		}
	}
	assert_eq!(name, "read_file");
	assert_eq!(
		serde_json::from_str::<Value>(&arguments).unwrap(),
		json!({"path": "README.md"})
	);
	assert_eq!(finish.as_deref(), Some("tool_calls"));
	assert!(text.contains("need the file"), "reasoning stream lost");

	let log = access_log(&res.path).await;
	assert_eq!(log["endpoint"], s.bedrock.address().to_string());
	assert_eq!(log["gen_ai.usage.input_tokens"], 1200, "{log:#}");
	assert_eq!(log["gen_ai.usage.output_tokens"], 40, "{log:#}");
	assert_eq!(log["gen_ai.usage.cache_read.input_tokens"], 1000, "{log:#}");
	s.assert_no_unapproved_provider().await;
}

#[rstest::rstest]
#[case::bad_request(400)]
#[case::unauthorized(401)]
#[case::forbidden(403)]
#[tokio::test]
async fn hardened_client_errors_are_not_retried_or_failed_over(#[case] code: u16) {
	let s = setup(Profile::Hardened, failure(code), bedrock_ok()).await;
	let res = send(&s.gw, large_request("4xx")).await;
	assert_eq!(res.status.as_u16(), code);
	assert_eq!(received(&s.fireworks).await.len(), 1);
	assert_eq!(received(&s.bedrock).await.len(), 0);
	// Fireworks is not evicted by a client error.
	let _ = send(&s.gw, large_request("4xx-2")).await;
	assert_eq!(received(&s.fireworks).await.len(), 2);
	assert_eq!(received(&s.bedrock).await.len(), 0);
}

#[rstest::rstest]
#[case::throttled(429)]
#[case::server_error(500)]
#[case::unavailable(503)]
#[case::validation(400)]
#[case::access_denied(403)]
#[tokio::test]
async fn hardened_bedrock_failure_after_failover_is_bounded(#[case] code: u16) {
	// Fireworks sheds the request and Bedrock fails too: the client gets Bedrock's own status
	// and error body after exactly two upstream attempts. There is no third provider.
	let bedrock_error = ResponseTemplate::new(code).set_body_raw(
		r#"{"error":{"message":"bedrock failure","type":"ThrottlingException"}}"#,
		"application/json",
	);
	let s = setup(Profile::Hardened, failure(503), bedrock_error).await;
	let start = std::time::Instant::now();
	let res = send(&s.gw, large_request("both")).await;
	assert_eq!(res.status.as_u16(), code);
	assert!(
		String::from_utf8_lossy(&res.body).contains("bedrock failure"),
		"{}",
		String::from_utf8_lossy(&res.body)
	);
	assert!(start.elapsed() < Duration::from_secs(2));
	assert_eq!(received(&s.fireworks).await.len(), 1);
	assert_eq!(received(&s.bedrock).await.len(), 1);
	s.assert_no_unapproved_provider().await;
	let log = access_log(&res.path).await;
	assert_eq!(log["endpoint"], s.bedrock.address().to_string(), "{log:#}");
	assert_eq!(
		log["retry.reasons"],
		format!("{}=503", s.fireworks.address())
	);
}

#[tokio::test]
async fn hardened_bedrock_hang_after_failover_is_bounded_by_route_timeout() {
	// Fireworks sheds the request and Bedrock never answers: the route request timeout ends the
	// request with a gateway timeout (5s here, 600s in production) instead of hanging.
	let fireworks = provider(FIREWORKS_PATH, failure(503)).await;
	let (bedrock, accepted, mut closed) = silent_provider().await;
	let cloudflare = provider(CLOUDFLARE_PATH, ok()).await;
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks: fireworks.address().to_string(),
		cloudflare: cloudflare.address().to_string(),
		bedrock,
		openrouter: openrouter.address().to_string(),
	};
	let gw = AgentGateway::new(gateway_config(&providers, Profile::Hardened, None))
		.await
		.unwrap();
	let start = std::time::Instant::now();
	let res = send(&gw, large_request("bedrock-hang")).await;
	let elapsed = start.elapsed();
	assert_eq!(res.status, StatusCode::GATEWAY_TIMEOUT);
	assert!(
		elapsed >= Duration::from_secs(5) && elapsed < Duration::from_secs(7),
		"{elapsed:?}"
	);
	assert_eq!(accepted.load(Ordering::SeqCst), 1);
	tokio::time::timeout(Duration::from_secs(2), closed.recv())
		.await
		.expect("Bedrock connection closed at the route timeout");
	assert_eq!(received(&cloudflare).await.len(), 0);
	assert_eq!(received(&openrouter).await.len(), 0);
}

#[tokio::test]
async fn hardened_all_providers_down_is_bounded() {
	let s = setup(Profile::Hardened, failure(503), failure(503)).await;
	let requests = 10;
	for i in 0..requests {
		let res = send(&s.gw, large_request(&format!("down-{i}"))).await;
		assert_eq!(res.status, StatusCode::SERVICE_UNAVAILABLE);
	}
	let fw = received(&s.fireworks).await.len();
	let br = received(&s.bedrock).await.len();
	// At most one retry per request; never more than 2x the downstream request count.
	assert!(fw + br <= 2 * requests, "fireworks={fw} bedrock={br}");
	assert!(fw >= 1 && br >= 1, "fireworks={fw} bedrock={br}");
	// Neither Cloudflare nor OpenRouter is configured, so neither is ever contacted.
	s.assert_no_unapproved_provider().await;
}

#[tokio::test]
async fn hardened_three_replicas_keep_independent_health_but_hide_failures() {
	let fireworks = provider(FIREWORKS_PATH, failure(503)).await;
	let others = Others::start(bedrock_ok()).await;
	let providers = others.providers(fireworks.address().to_string());
	let mut replicas = Vec::new();
	for _ in 0..3 {
		replicas.push(
			AgentGateway::new(gateway_config(&providers, Profile::Hardened, None))
				.await
				.unwrap(),
		);
	}
	for round in 0..3 {
		for (i, gw) in replicas.iter().enumerate() {
			let res = send(gw, large_request(&format!("r{round}-g{i}"))).await;
			assert_eq!(res.status, StatusCode::OK, "replica {i} round {round}");
		}
	}
	// Each replica learns about the failing provider on its own (one Fireworks attempt each),
	// but the in-request failover means no client sees the failure.
	assert_eq!(received(&fireworks).await.len(), 3);
	assert_eq!(received(&others.bedrock).await.len(), 9);
	others.assert_no_unapproved_provider().await;
}

#[tokio::test]
async fn hardened_client_disconnect_cancels_upstream_without_retry() {
	let (fireworks, accepted, mut closed) = silent_provider().await;
	let others = Others::start(bedrock_ok()).await;
	let gw = AgentGateway::new(gateway_config(
		&others.providers(fireworks),
		Profile::Hardened,
		None,
	))
	.await
	.unwrap();
	let path = unique_path();
	// The client aborts after 300ms, before the 1s Fireworks deadline.
	let res = tokio::time::timeout(
		Duration::from_millis(300),
		gw.send_request_json(&format!("http://localhost{path}"), large_request("cancel")),
	)
	.await;
	assert!(res.is_err());
	// The gateway closes its upstream connection promptly.
	tokio::time::timeout(Duration::from_secs(2), closed.recv())
		.await
		.expect("upstream connection should be closed after the client disconnects");
	assert_eq!(accepted.load(Ordering::SeqCst), 1);
	tokio::time::sleep(Duration::from_millis(1500)).await;
	// No retry or failover is started for a request nobody is waiting for.
	assert_eq!(received(&others.bedrock).await.len(), 0);
	assert_eq!(accepted.load(Ordering::SeqCst), 1);
	// A cancellation is not a provider failure: Fireworks is not evicted.
	let _ = tokio::time::timeout(
		Duration::from_millis(300),
		gw.send_request_json(
			&format!("http://localhost{}", unique_path()),
			large_request("again"),
		),
	)
	.await;
	assert_eq!(accepted.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn hardened_client_disconnect_during_bedrock_attempt_closes_upstream() {
	let fireworks = provider(FIREWORKS_PATH, failure(503)).await;
	let (bedrock, accepted, mut closed) = silent_provider().await;
	let cloudflare = provider(CLOUDFLARE_PATH, ok()).await;
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks: fireworks.address().to_string(),
		cloudflare: cloudflare.address().to_string(),
		bedrock,
		openrouter: openrouter.address().to_string(),
	};
	let gw = AgentGateway::new(gateway_config(&providers, Profile::Hardened, None))
		.await
		.unwrap();
	let res = tokio::time::timeout(
		Duration::from_millis(500),
		gw.send_request_json(
			&format!("http://localhost{}", unique_path()),
			large_request("cancel-bedrock"),
		),
	)
	.await;
	assert!(res.is_err());
	assert_eq!(accepted.load(Ordering::SeqCst), 1);
	tokio::time::timeout(Duration::from_secs(2), closed.recv())
		.await
		.expect("Bedrock connection should be closed after the client disconnects");
	tokio::time::sleep(Duration::from_millis(500)).await;
	assert_eq!(accepted.load(Ordering::SeqCst), 1);
	assert_eq!(received(&cloudflare).await.len(), 0);
	assert_eq!(received(&openrouter).await.len(), 0);
}

#[tokio::test]
async fn hardened_tool_call_and_reasoning_history_are_equivalent_after_failover() {
	let s = setup(Profile::Hardened, failure(503), bedrock_ok()).await;
	let res = send(&s.gw, tool_request()).await;
	assert_eq!(res.status, StatusCode::OK);
	let fw = sent_body(&received(&s.fireworks).await[0]);
	let br_req = &received(&s.bedrock).await[0];
	assert_bedrock_request(br_req);
	let br = sent_body(br_req);
	// Tools, system/developer/user turns, tool calls, tool results and prior reasoning are
	// forwarded unchanged; only the provider model id and the Fireworks-only fields differ, and
	// assistant tool-call turns get `content: ""`.
	assert_eq!(fw["tools"], br["tools"]);
	let roles: Vec<&str> = br["messages"]
		.as_array()
		.unwrap()
		.iter()
		.map(|m| m["role"].as_str().unwrap())
		.collect();
	assert_eq!(
		roles,
		[
			"system",
			"developer",
			"user",
			"assistant",
			"tool",
			"assistant",
			"user"
		]
	);
	assert_eq!(
		br["messages"][3]["tool_calls"],
		fw["messages"][3]["tool_calls"]
	);
	assert_eq!(br["messages"][3]["content"], "");
	assert_eq!(fw["messages"][3]["content"], "");
	assert_eq!(br["messages"][3]["reasoning_content"], "I should read it.");
	assert_eq!(br["messages"][4], fw["messages"][4]);
	assert_eq!(br["messages"][5], fw["messages"][5]);
}

#[rstest::rstest]
#[case::low("low", "low")]
#[case::medium("medium", "high")]
#[case::high("high", "high")]
#[case::xhigh("xhigh", "max")]
#[case::max("max", "max")]
#[case::minimal("minimal", "max")]
#[tokio::test]
async fn hardened_reasoning_effort_matches_on_both_providers(
	#[case] requested: &str,
	#[case] expected: &str,
) {
	// Bedrock receives the same normalized effort as Fireworks would have for this request, and
	// never a value Bedrock rejects (`minimal`).
	let s = setup(Profile::Hardened, failure(503), bedrock_ok()).await;
	let mut req = large_request("effort");
	req["reasoning_effort"] = json!(requested);
	let res = send(&s.gw, req).await;
	assert_eq!(res.status, StatusCode::OK);
	let fw = sent_body(&received(&s.fireworks).await[0]);
	let br = sent_body(&received(&s.bedrock).await[0]);
	assert_eq!(fw["reasoning_effort"], expected);
	assert_eq!(br["reasoning_effort"], expected);
}

#[tokio::test]
async fn hardened_request_without_reasoning_effort_keeps_bedrock_default() {
	let s = setup(Profile::Hardened, failure(503), bedrock_ok()).await;
	let res = send(&s.gw, large_request("no-effort")).await;
	assert_eq!(res.status, StatusCode::OK);
	let fw = sent_body(&received(&s.fireworks).await[0]);
	let br = sent_body(&received(&s.bedrock).await[0]);
	assert!(fw.get("reasoning_effort").is_none(), "{fw}");
	assert!(br.get("reasoning_effort").is_none(), "{br}");
}

#[tokio::test]
async fn hardened_never_routes_to_cloudflare_or_openrouter() {
	// Privacy gate: neither is an approved automatic fallback for Kimi K3. Exercise every failure
	// mode, including Bedrock failing too, and assert neither is ever contacted.
	for (fw, br) in [
		(failure(503), failure(503)),
		(failure(429), failure(502)),
		(failure(502), failure(429)),
		(ok().set_delay(Duration::from_secs(30)), failure(503)),
		(failure(504), ok().set_delay(Duration::from_secs(30))),
	] {
		let s = setup(Profile::Hardened, fw, br).await;
		for i in 0..3 {
			let _ = send(&s.gw, small_request(&format!("p{i}"))).await;
		}
		s.assert_no_unapproved_provider().await;
	}
}

// ---------------------------------------------------------------------------------------------
// Trace evidence: retry and provider selection as recorded in spans.
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct SpanRecord {
	trace_id: Vec<u8>,
	name: String,
	server: bool,
	attributes: std::collections::HashMap<String, String>,
}

struct TraceHandler(mpsc::UnboundedSender<SpanRecord>);

#[async_trait::async_trait]
impl oteltracemock::Handler for TraceHandler {
	async fn export(
		&mut self,
		request: &ExportTraceServiceRequest,
	) -> Result<ExportTraceServiceResponse, tonic::Status> {
		use opentelemetry_proto::tonic::common::v1::any_value::Value as V;
		for span in request
			.resource_spans
			.iter()
			.flat_map(|r| &r.scope_spans)
			.flat_map(|s| &s.spans)
		{
			let attributes = span
				.attributes
				.iter()
				.filter_map(|kv| {
					let v = match kv.value.as_ref()?.value.as_ref()? {
						V::StringValue(s) => s.clone(),
						V::IntValue(i) => i.to_string(),
						V::BoolValue(b) => b.to_string(),
						V::DoubleValue(d) => d.to_string(),
						_ => return None,
					};
					Some((kv.key.clone(), v))
				})
				.collect();
			let _ = self.0.send(SpanRecord {
				trace_id: span.trace_id.clone(),
				name: span.name.clone(),
				server: span.kind == opentelemetry_proto::tonic::trace::v1::span::SpanKind::Server as i32,
				attributes,
			});
		}
		oteltracemock::ok_response()
	}
}

/// `host:port` the upstream span dialed.
fn upstream_target(span: &SpanRecord) -> Option<String> {
	span.attributes.get("endpoint").cloned()
}

#[tokio::test]
async fn hardened_failover_is_visible_in_traces() {
	let (tx, mut rx) = mpsc::unbounded_channel();
	let otel = oteltracemock::OtelTraceMock::new(move || TraceHandler(tx.clone()))
		.spawn()
		.await;
	let fireworks = provider(FIREWORKS_PATH, failure(503)).await;
	let others = Others::start(bedrock_ok()).await;
	let gw = AgentGateway::new(gateway_config(
		&others.providers(fireworks.address().to_string()),
		Profile::Hardened,
		Some(otel.address),
	))
	.await
	.unwrap();
	let res = send(&gw, large_request("trace")).await;
	assert_eq!(res.status, StatusCode::OK);

	let is_upstream = |s: &SpanRecord| !s.server && s.attributes.contains_key("http.status");
	let mut spans: Vec<SpanRecord> = Vec::new();
	let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
	loop {
		let complete = spans.iter().find(|s| s.server).is_some_and(|root| {
			spans
				.iter()
				.filter(|s| is_upstream(s) && s.trace_id == root.trace_id)
				.count()
				>= 2
		});
		if complete {
			break;
		}
		match tokio::time::timeout_at(deadline, rx.recv()).await {
			Ok(Some(span)) => spans.push(span),
			_ => panic!(
				"request and upstream spans not exported: {:?}",
				spans
					.iter()
					.map(|s| (&s.name, s.server, s.attributes.get("http.status")))
					.collect::<Vec<_>>()
			),
		}
	}
	let root = spans.iter().find(|s| s.server).unwrap();
	assert_eq!(
		root.attributes.get("retry.attempt").map(String::as_str),
		Some("1")
	);
	assert_eq!(
		root.attributes.get("retry.reasons"),
		Some(&format!("{}=503", fireworks.address()))
	);
	assert_eq!(
		root.attributes.get("endpoint"),
		Some(&others.bedrock.address().to_string())
	);
	assert_eq!(
		root
			.attributes
			.get("gen_ai.request.model")
			.map(String::as_str),
		Some(BEDROCK_MODEL)
	);
	let upstream: Vec<_> = spans
		.iter()
		.filter(|s| is_upstream(s) && s.trace_id == root.trace_id)
		.map(|s| (upstream_target(s), s.attributes.get("http.status").cloned()))
		.collect();
	assert!(
		upstream.contains(&(
			Some(fireworks.address().to_string()),
			Some("503".to_string())
		)),
		"{upstream:?}"
	);
	assert!(
		upstream.contains(&(
			Some(others.bedrock.address().to_string()),
			Some("200".to_string())
		)),
		"{upstream:?}"
	);
}
