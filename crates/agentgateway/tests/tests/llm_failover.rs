//! Provider failover for LLM backends with ordered priority groups.
//!
//! These tests run the full gateway over real sockets against mock providers. They mirror the
//! production `kimi-k3-backend` shape (Fireworks -> Cloudflare [-> OpenRouter]) and compare the
//! policy set deployed during the 2026-10-25 incident (`Profile::Incident`) with the hardened one
//! (`Profile::Hardened`). Durations are scaled down: 1s here stands for the 180s Fireworks
//! deadline and the 120s stream idle timeout used in production.

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
const COMPLETION: &[u8] = include_bytes!("../../../llm/src/tests/response/completions/basic.json");

#[derive(Clone, Copy, PartialEq)]
enum Profile {
	/// Live policies on 2026-10-25: retry without codes, `response.code` health condition,
	/// OpenRouter as third group, no provider deadline.
	Incident,
	/// Kimi-scoped retry on 429/502/503/504 with a single retry, request buffering opt-in for
	/// replaying large prompts, Fireworks deadline, stream idle timeout, and no OpenRouter group.
	Hardened,
	/// `Hardened` without the `buffer.request` opt-in: the shape every other retry route has.
	HardenedWithoutBufferOptIn,
}

struct Providers {
	fireworks: String,
	cloudflare: String,
	openrouter: String,
}

fn gateway_config(p: &Providers, profile: Profile, tracing: Option<SocketAddr>) -> String {
	let tracing = tracing
		.map(|addr| {
			format!(
				r#"
frontendPolicies:
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
	let (route_policies, fireworks_policies, openrouter_group) = match profile {
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
			String::new(),
			format!(
				r#"
          - providers:
            - name: openrouter-kimi-k3
              provider:
                openAI:
                  model: moonshotai/kimi-k3
              hostOverride: {}
              pathOverride: {OPENROUTER_PATH}"#,
				p.openrouter
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
          requestTimeout: 600s
          responseIdleTimeout: 1s"#,
				buffer = if profile == Profile::Hardened {
					r#"
        buffer:
          request:
            maxBytes: 10485760
            failureMode: failOpen"#
				} else {
					""
				}
			),
			r#"
              policies:
                http:
                  requestTimeout: 1s"#
				.to_string(),
			String::new(),
		),
	};
	format!(
		r#"
config: {{}}{tracing}
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
      policies:{route_policies}
      backends:
      - ai:
          groups:
          - providers:
            - name: fireworks-kimi-k3
              provider:
                openAI:
                  model: accounts/fireworks/models/kimi-k3
              hostOverride: {fireworks}
              pathOverride: {FIREWORKS_PATH}{fireworks_policies}
          - providers:
            - name: cloudflare-kimi-k3
              provider:
                openAI:
                  model: moonshotai/kimi-k3
              hostOverride: {cloudflare}
              pathOverride: {CLOUDFLARE_PATH}
              policies:
                transformations:
                  request:
                    body: |
                      toJson(json(request.body).merge({{"messages":
                        json(request.body).messages.map(message,
                        message.role == "assistant" && has(message.tool_calls)
                        ? message.merge({{"content": default(message.content, "")}})
                        : message)}}))
{openrouter_group}
        policies:
          health:
            unhealthyExpression: "response.code >= 500 || response.code == 429"
            eviction:
              consecutiveFailures: 1
              duration: 30s
"#,
		fireworks = p.fireworks,
		cloudflare = p.cloudflare,
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

/// Synthetic agent prompt larger than the 64 KiB replay cap (production median ~50k tokens).
fn large_request(tag: &str) -> Value {
	json!({
		"model": "moonshotai.kimi-k3",
		"max_tokens": 1024,
		"messages": [
			{"role": "system", "content": "x".repeat(200 * 1024)},
			{"role": "user", "content": format!("request {tag}")},
		],
	})
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
			{"role": "user", "content": "open README"},
			{"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
				"function": {"name": "read_file", "arguments": "{\"path\":\"README.md\"}"}}]},
			{"role": "tool", "tool_call_id": "call_1", "content": "hello"},
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

fn upstream_model(req: &wiremock::Request) -> String {
	let body: Value = serde_json::from_slice(&req.body).unwrap();
	body["model"].as_str().unwrap().to_string()
}

struct Setup {
	gw: AgentGateway,
	fireworks: MockServer,
	cloudflare: MockServer,
	openrouter: MockServer,
}

async fn setup(
	profile: Profile,
	fireworks: ResponseTemplate,
	cloudflare: ResponseTemplate,
) -> Setup {
	let fireworks = provider(FIREWORKS_PATH, fireworks).await;
	let cloudflare = provider(CLOUDFLARE_PATH, cloudflare).await;
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks: fireworks.address().to_string(),
		cloudflare: cloudflare.address().to_string(),
		openrouter: openrouter.address().to_string(),
	};
	let gw = AgentGateway::new(gateway_config(&providers, profile, None))
		.await
		.unwrap();
	Setup {
		gw,
		fireworks,
		cloudflare,
		openrouter,
	}
}

// ---------------------------------------------------------------------------------------------
// Incident reproduction: the policies deployed on 2026-10-25.
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn incident_503_reaches_client_and_only_this_replica_evicts() {
	let s = setup(Profile::Incident, failure(503), ok()).await;

	// The first request is not retried (no `codes`), so the client sees Fireworks' 503.
	let first = send(&s.gw, large_request("a")).await;
	assert_eq!(first.status, StatusCode::SERVICE_UNAVAILABLE);
	assert_eq!(received(&s.fireworks).await.len(), 1);
	assert_eq!(received(&s.cloudflare).await.len(), 0);

	// The 503 evicted Fireworks in this process only, so the next request uses Cloudflare.
	let second = send(&s.gw, large_request("b")).await;
	assert_eq!(second.status, StatusCode::OK);
	assert_eq!(received(&s.fireworks).await.len(), 1);
	assert_eq!(received(&s.cloudflare).await.len(), 1);
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
	assert_eq!(received(&s.cloudflare).await.len(), 0);
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

// ---------------------------------------------------------------------------------------------
// Hardened policies.
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn hardened_200_does_not_fail_over() {
	let s = setup(Profile::Hardened, ok(), ok()).await;
	let res = send(&s.gw, large_request("ok")).await;
	assert_eq!(res.status, StatusCode::OK);
	let fw = received(&s.fireworks).await;
	assert_eq!(fw.len(), 1);
	assert_eq!(upstream_model(&fw[0]), "accounts/fireworks/models/kimi-k3");
	assert_eq!(received(&s.cloudflare).await.len(), 0);
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
async fn hardened_retryable_status_fails_over_to_cloudflare(#[case] code: u16) {
	let s = setup(Profile::Hardened, failure(code), ok()).await;
	let res = send(&s.gw, large_request("fo")).await;
	assert_eq!(
		res.status,
		StatusCode::OK,
		"{}",
		String::from_utf8_lossy(&res.body)
	);

	let fw = received(&s.fireworks).await;
	let cf = received(&s.cloudflare).await;
	assert_eq!(fw.len(), 1);
	assert_eq!(cf.len(), 1);
	assert_eq!(upstream_model(&cf[0]), "moonshotai/kimi-k3");
	// The replayed prompt is the full original prompt, not a truncated replay buffer.
	let sent: Value = serde_json::from_slice(&cf[0].body).unwrap();
	assert_eq!(
		sent["messages"][0]["content"].as_str().unwrap().len(),
		200 * 1024
	);
	assert_eq!(received(&s.openrouter).await.len(), 0);

	// One downstream request -> one usage record, attributed to the provider that served it.
	let log = access_log(&res.path).await;
	assert_eq!(log["retry.attempt"], 1, "{log:#}");
	assert_eq!(log["endpoint"], s.cloudflare.address().to_string());
	assert_eq!(
		log["retry.reasons"],
		format!("{}={code}", s.fireworks.address())
	);
	assert_eq!(log["gen_ai.usage.input_tokens"], 17);
	assert_eq!(log["gen_ai.usage.output_tokens"], 23);

	// Fireworks stays evicted for this replica: the next request goes straight to Cloudflare.
	let next = send(&s.gw, large_request("next")).await;
	assert_eq!(next.status, StatusCode::OK);
	assert_eq!(received(&s.fireworks).await.len(), 1);
	assert_eq!(received(&s.cloudflare).await.len(), 2);
}

#[tokio::test]
async fn large_prompt_replay_requires_buffer_opt_in() {
	// Routes without a `buffer.request` policy keep the 64 KiB replay cap, so this fix does not
	// change retry behavior for other routes (Bedrock, Claude, GLM) that have retry policies.
	let s = setup(Profile::HardenedWithoutBufferOptIn, failure(503), ok()).await;
	let large = send(&s.gw, large_request("large")).await;
	assert_eq!(large.status, StatusCode::SERVICE_UNAVAILABLE);
	assert_eq!(received(&s.cloudflare).await.len(), 0);
	// Small prompts were always replayable; the 503 evicted Fireworks, so this one goes to
	// Cloudflare directly.
	let small = send(&s.gw, small_request("small")).await;
	assert_eq!(small.status, StatusCode::OK);
	assert_eq!(received(&s.fireworks).await.len(), 1);
}

#[tokio::test]
async fn hardened_connection_reset_fails_over() {
	let (fireworks, accepted) = resetting_provider().await;
	let cloudflare = provider(CLOUDFLARE_PATH, ok()).await;
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks: fireworks.clone(),
		cloudflare: cloudflare.address().to_string(),
		openrouter: openrouter.address().to_string(),
	};
	let gw = AgentGateway::new(gateway_config(&providers, Profile::Hardened, None))
		.await
		.unwrap();

	let res = send(&gw, large_request("reset")).await;
	assert_eq!(res.status, StatusCode::OK);
	assert!(accepted.load(Ordering::SeqCst) >= 1);
	assert_eq!(received(&cloudflare).await.len(), 1);
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
	assert_eq!(received(&cloudflare).await.len(), 2);
}

#[tokio::test]
async fn hardened_hang_before_first_byte_recovers_within_deadline() {
	let s = setup(
		Profile::Hardened,
		ok().set_delay(Duration::from_secs(30)),
		ok(),
	)
	.await;
	let start = std::time::Instant::now();
	let res = send(&s.gw, large_request("hang")).await;
	let elapsed = start.elapsed();
	assert_eq!(res.status, StatusCode::OK);
	// Fireworks deadline (1s) + backoff + Cloudflare, well before the client's own timeout.
	assert!(elapsed < Duration::from_secs(3), "took {elapsed:?}");
	assert!(elapsed >= Duration::from_secs(1), "took {elapsed:?}");
	assert_eq!(received(&s.cloudflare).await.len(), 1);
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
}

#[tokio::test]
async fn hardened_long_stream_is_not_cut_by_provider_deadline() {
	// 10 chunks 300ms apart = ~3s total, longer than the 1s Fireworks deadline and with gaps
	// shorter than the 1s idle timeout. The deadline only bounds the wait for response headers.
	let fireworks = streaming_provider(10, Duration::from_millis(300), false).await;
	let cloudflare = provider(CLOUDFLARE_PATH, ok()).await;
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks,
		cloudflare: cloudflare.address().to_string(),
		openrouter: openrouter.address().to_string(),
	};
	let gw = AgentGateway::new(gateway_config(&providers, Profile::Hardened, None))
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
	assert_eq!(received(&cloudflare).await.len(), 0);
}

#[tokio::test]
async fn hardened_stream_stall_after_first_token_is_terminated() {
	let fireworks = streaming_provider(1, Duration::from_millis(10), true).await;
	let cloudflare = provider(CLOUDFLARE_PATH, ok()).await;
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks,
		cloudflare: cloudflare.address().to_string(),
		openrouter: openrouter.address().to_string(),
	};
	let gw = AgentGateway::new(gateway_config(&providers, Profile::Hardened, None))
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
	assert_eq!(received(&cloudflare).await.len(), 0);
}

#[rstest::rstest]
#[case::bad_request(400)]
#[case::unauthorized(401)]
#[case::forbidden(403)]
#[tokio::test]
async fn hardened_client_errors_are_not_retried_or_failed_over(#[case] code: u16) {
	let s = setup(Profile::Hardened, failure(code), ok()).await;
	let res = send(&s.gw, large_request("4xx")).await;
	assert_eq!(res.status.as_u16(), code);
	assert_eq!(received(&s.fireworks).await.len(), 1);
	assert_eq!(received(&s.cloudflare).await.len(), 0);
	// Fireworks is not evicted by a client error.
	let _ = send(&s.gw, large_request("4xx-2")).await;
	assert_eq!(received(&s.fireworks).await.len(), 2);
	assert_eq!(received(&s.cloudflare).await.len(), 0);
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
	let cf = received(&s.cloudflare).await.len();
	// At most one retry per request; never more than 2x the downstream request count.
	assert!(fw + cf <= 2 * requests, "fireworks={fw} cloudflare={cf}");
	assert!(fw >= 1 && cf >= 1, "fireworks={fw} cloudflare={cf}");
	// OpenRouter is not configured, so it is never contacted.
	assert_eq!(received(&s.openrouter).await.len(), 0);
}

#[tokio::test]
async fn hardened_three_replicas_keep_independent_health_but_hide_failures() {
	let fireworks = provider(FIREWORKS_PATH, failure(503)).await;
	let cloudflare = provider(CLOUDFLARE_PATH, ok()).await;
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks: fireworks.address().to_string(),
		cloudflare: cloudflare.address().to_string(),
		openrouter: openrouter.address().to_string(),
	};
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
	assert_eq!(received(&cloudflare).await.len(), 9);
}

#[tokio::test]
async fn incident_three_replicas_expose_one_failure_per_replica() {
	let fireworks = provider(FIREWORKS_PATH, failure(503)).await;
	let cloudflare = provider(CLOUDFLARE_PATH, ok()).await;
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks: fireworks.address().to_string(),
		cloudflare: cloudflare.address().to_string(),
		openrouter: openrouter.address().to_string(),
	};
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
	assert_eq!(received(&cloudflare).await.len(), 6);
}

#[tokio::test]
async fn hardened_client_disconnect_cancels_upstream_without_retry() {
	let (fireworks, accepted, mut closed) = silent_provider().await;
	let cloudflare = provider(CLOUDFLARE_PATH, ok()).await;
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks,
		cloudflare: cloudflare.address().to_string(),
		openrouter: openrouter.address().to_string(),
	};
	let gw = AgentGateway::new(gateway_config(&providers, Profile::Hardened, None))
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
	assert_eq!(received(&cloudflare).await.len(), 0);
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
async fn hardened_tool_call_request_is_equivalent_after_failover() {
	let s = setup(Profile::Hardened, failure(503), ok()).await;
	let res = send(&s.gw, tool_request()).await;
	assert_eq!(res.status, StatusCode::OK);
	let fw: Value = serde_json::from_slice(&received(&s.fireworks).await[0].body).unwrap();
	let cf: Value = serde_json::from_slice(&received(&s.cloudflare).await[0].body).unwrap();
	// Tools and tool turns are forwarded unchanged; only the provider model id differs, and the
	// Cloudflare validator workaround fills the missing assistant `content`.
	assert_eq!(fw["tools"], cf["tools"]);
	assert_eq!(
		cf["messages"][2]["tool_calls"],
		fw["messages"][2]["tool_calls"]
	);
	assert_eq!(cf["messages"][2]["content"], "");
	assert_eq!(cf["messages"][3], fw["messages"][3]);
	assert_eq!(cf["model"], "moonshotai/kimi-k3");
}

#[tokio::test]
async fn hardened_never_routes_to_openrouter() {
	// Privacy gate: OpenRouter is not an approved processor for this traffic, so it is not part
	// of the hardened backend. Exercise every failure mode and assert it is never contacted.
	for (fw, cf) in [
		(failure(503), failure(503)),
		(failure(429), failure(502)),
		(ok().set_delay(Duration::from_secs(30)), failure(503)),
	] {
		let s = setup(Profile::Hardened, fw, cf).await;
		for i in 0..3 {
			let _ = send(&s.gw, small_request(&format!("p{i}"))).await;
		}
		assert_eq!(received(&s.openrouter).await.len(), 0);
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
	let cloudflare = provider(CLOUDFLARE_PATH, ok()).await;
	let openrouter = provider(OPENROUTER_PATH, ok()).await;
	let providers = Providers {
		fireworks: fireworks.address().to_string(),
		cloudflare: cloudflare.address().to_string(),
		openrouter: openrouter.address().to_string(),
	};
	let gw = AgentGateway::new(gateway_config(
		&providers,
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
		Some(&cloudflare.address().to_string())
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
			Some(cloudflare.address().to_string()),
			Some("200".to_string())
		)),
		"{upstream:?}"
	);
}
