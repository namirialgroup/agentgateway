//! Dynamic Tool Search execution-mode policy — versioned HTTPS artifact with
//! bounded background refresh (2026-10-20 design §C2, the interim
//! distribution ahead of an xDS-native policy resource).
//!
//! Contract:
//! - The artifact is fetched in the BACKGROUND only; request handling never
//!   performs I/O for mode resolution and never depends synchronously on the
//!   Control Plane. Last-known-good is kept on fetch/parse failures; a
//!   deployment without `AGENTGATEWAY_TOOL_SEARCH_POLICY_URL` behaves exactly
//!   like the env-only guard (back-compat).
//! - FAIL-CLOSED COLD START: when the policy URL IS configured but no valid
//!   snapshot is available (first boot, restart, or every fetch failing
//!   before one ever succeeded — e.g. a token-rotation crossover), native
//!   Tool Search is DISABLED for every model: the governed policy may be
//!   stricter than the env allowlist, so falling back to env-only would be
//!   an escalation, not a safe default. Requests keep being served through
//!   the eager/translation downgrade. Env-only behavior applies ONLY when
//!   dynamic distribution is not configured at all.
//! - TWO-KEY RULE for native: the env allowlist
//!   (`AGENTGATEWAY_TOOL_SEARCH_MODELS`, the deployment safety ceiling
//!   generated from runtime-verified capabilities) must ALSO name the model.
//!   The artifact may restrict (`eager`/`off`) or confirm, but never
//!   manufactures native support. Modes are resolved per model alias at
//!   request time against the actually selected attempt's route (the
//!   per-attempt format/provider guards remain the enforcement of record).
//! - `off` preserves ordinary tools: it forces the eager downgrade (and for a
//!   native-allowlisted model additionally disables native serving), it never
//!   blocks the model itself — mode resolution is orthogonal to Gatekeeper
//!   admission, budgets and per-user model authorization.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwapOption;
use serde::Deserialize;

/// Hard bounds: a policy artifact larger than this, or with more entries, is
/// rejected and the last-known-good (or env-only behavior) is kept.
const MAX_POLICY_BYTES: usize = 256 * 1024;
const MAX_POLICY_MODELS: usize = 4096;
const DEFAULT_REFRESH_SECS: u64 = 60;
const FETCH_TIMEOUT: Duration = Duration::from_secs(5);

/// Modes mirrored from the control-plane domain
/// (`packages/llm-governance-core/src/domain/toolSearch.ts`). `gateway` is
/// NOT accepted here: unimplemented capability is never advertised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "lowercase")]
pub enum PolicyMode {
	Auto,
	Native,
	Eager,
	Off,
}

impl PolicyMode {
	fn parse_str(value: &str) -> Option<PolicyMode> {
		match value {
			"auto" => Some(PolicyMode::Auto),
			"native" => Some(PolicyMode::Native),
			"eager" => Some(PolicyMode::Eager),
			"off" => Some(PolicyMode::Off),
			_ => None,
		}
	}
}

/// A validated policy snapshot. Parsing is strict and deterministic: unknown
/// fields, unknown modes, missing revision, or out-of-bounds size/entry count
/// are all rejected (fail-closed to the previous snapshot).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSearchPolicyDoc {
	pub revision: String,
	pub default_mode: PolicyMode,
	pub models: BTreeMap<String, PolicyMode>,
}

impl ToolSearchPolicyDoc {
	pub fn parse(bytes: &[u8]) -> Result<ToolSearchPolicyDoc, String> {
		if bytes.len() > MAX_POLICY_BYTES {
			return Err(format!(
				"policy artifact exceeds {} bytes",
				MAX_POLICY_BYTES
			));
		}
		#[derive(Deserialize)]
		#[serde(deny_unknown_fields)]
		struct Raw {
			revision: String,
			#[serde(default = "default_mode")]
			default_mode: String,
			#[serde(default)]
			models: BTreeMap<String, String>,
		}
		let raw: Raw = serde_json::from_slice(bytes)
			.map_err(|error| format!("policy artifact is not valid JSON: {error}"))?;
		if raw.revision.trim().is_empty() {
			return Err("policy artifact has an empty revision".to_string());
		}
		let default_mode = PolicyMode::parse_str(&raw.default_mode)
			.ok_or_else(|| format!("invalid default_mode: {}", raw.default_mode))?;
		if raw.models.len() > MAX_POLICY_MODELS {
			return Err(format!(
				"policy artifact lists more than {} models",
				MAX_POLICY_MODELS
			));
		}
		let mut models = BTreeMap::new();
		for (alias, mode) in raw.models {
			let mode =
				PolicyMode::parse_str(&mode).ok_or_else(|| format!("invalid mode for {alias}: {mode}"))?;
			models.insert(alias, mode);
		}
		Ok(ToolSearchPolicyDoc {
			revision: raw.revision,
			default_mode,
			models,
		})
	}

	fn mode_for(&self, model: &str) -> PolicyMode {
		self.models.get(model).copied().unwrap_or(self.default_mode)
	}
}

fn default_mode() -> String {
	"auto".to_string()
}

static POLICY: once_cell::sync::Lazy<ArcSwapOption<ToolSearchPolicyDoc>> =
	once_cell::sync::Lazy::new(ArcSwapOption::empty);

/// Whether a dynamic policy source is configured for this process (set once,
/// synchronously, by `spawn_policy_refresher` BEFORE any request can be
/// served). Process-local by design: a restart must not widen a governed
/// deployment for even one request while the first fetch is pending.
static POLICY_SOURCE_CONFIGURED: std::sync::atomic::AtomicBool =
	std::sync::atomic::AtomicBool::new(false);

/// Install a validated snapshot (background refresher and tests). The swap is
/// atomic: concurrent requests see either the previous or the new snapshot,
/// never a torn one.
pub fn install_policy(doc: ToolSearchPolicyDoc) {
	POLICY.store(Some(Arc::new(doc)));
}

/// Drop the snapshot AND mark the source unconfigured (tests / shutdown):
/// resets the module to the exact not-configured baseline.
pub fn clear_policy() {
	POLICY.store(None);
	POLICY_SOURCE_CONFIGURED.store(false, std::sync::atomic::Ordering::Release);
}

/// Mark a dynamic policy source as configured (called by the refresher
/// bootstrap before the first fetch is even attempted).
pub fn set_policy_source_configured() {
	POLICY_SOURCE_CONFIGURED.store(true, std::sync::atomic::Ordering::Release);
}

/// Currently active snapshot, if one was ever fetched and validated.
pub fn current_policy() -> Option<Arc<ToolSearchPolicyDoc>> {
	POLICY.load_full()
}

/// The request-time decision integrating the env bootstrap ceiling and the
/// dynamic artifact. Pure: no I/O, no locks beyond the atomic snapshot load.
pub struct ToolSearchDecision {
	pub allow_native: bool,
	/// Sanitized reason for telemetry (static strings, never model or tool
	/// names).
	pub reason: &'static str,
	/// Revision of the policy snapshot this decision was resolved against
	/// (`None` when no artifact is loaded — env-only when the source is not
	/// configured, fail-closed when it is).
	pub policy_revision: Option<String>,
}

/// Resolve whether native Tool Search may be served for `model`.
/// `env_allows_native` is the deployment allowlist result (the safety
/// ceiling); the artifact can only restrict or confirm, never grant.
pub fn tool_search_decision(env_allows_native: bool, model: &str) -> ToolSearchDecision {
	let Some(policy) = current_policy() else {
		// No artifact ever loaded. If a dynamic policy source is configured,
		// this is a cold start / pre-first-success window (fresh pod, or
		// every fetch so far failed, e.g. a token-rotation crossover): the
		// governed policy may be STRICTER than the env allowlist, so
		// env-only would be an escalation. Deny native for every model and
		// keep serving through the eager/translation downgrade until a
		// valid revision arrives. Exact env-only behavior applies only when
		// dynamic distribution is not configured at all.
		if POLICY_SOURCE_CONFIGURED.load(std::sync::atomic::Ordering::Acquire) {
			return ToolSearchDecision {
				allow_native: false,
				reason: "policy-source-unavailable",
				policy_revision: None,
			};
		}
		// Not configured: exact env-only behavior (back-compat).
		return ToolSearchDecision {
			allow_native: env_allows_native,
			reason: if env_allows_native {
				"env-allowlist"
			} else {
				"env-allowlist-miss"
			},
			policy_revision: None,
		};
	};
	let revision = Some(policy.revision.clone());
	match policy.mode_for(model) {
		PolicyMode::Off => ToolSearchDecision {
			allow_native: false,
			reason: "policy-off",
			policy_revision: revision,
		},
		PolicyMode::Eager => ToolSearchDecision {
			allow_native: false,
			reason: "policy-eager",
			policy_revision: revision,
		},
		PolicyMode::Native => {
			if env_allows_native {
				ToolSearchDecision {
					allow_native: true,
					reason: "policy-native-env-confirmed",
					policy_revision: revision,
				}
			} else {
				// Two-key rule: the artifact alone cannot bootstrap native.
				ToolSearchDecision {
					allow_native: false,
					reason: "policy-native-without-env-bootstrap",
					policy_revision: revision,
				}
			}
		},
		PolicyMode::Auto => ToolSearchDecision {
			allow_native: env_allows_native,
			reason: "policy-auto-env",
			policy_revision: revision,
		},
	}
}

/// Fetch + validate one snapshot. Exposed separately from the refresher loop
/// so tests and operational runbooks can drive a single refresh
/// deterministically. On any failure the current snapshot is left untouched
/// (last-known-good) and the error is returned for logging.
///
/// `AGENTGATEWAY_TOOL_SEARCH_POLICY_TOKEN` (optional): sent as
/// `Authorization: Bearer <token>` when the artifact endpoint requires the
/// control plane's machine token. The token lives in the deployment's secret
/// pipeline, never in the policy URL itself.
pub async fn refresh_policy_once(url: &str) -> Result<ToolSearchPolicyDoc, String> {
	let client = reqwest::Client::builder()
		.timeout(FETCH_TIMEOUT)
		.build()
		.map_err(|error| format!("policy http client: {error}"))?;
	let mut request = client.get(url);
	if let Ok(token) = std::env::var("AGENTGATEWAY_TOOL_SEARCH_POLICY_TOKEN")
		&& !token.trim().is_empty()
	{
		request = request.bearer_auth(token);
	}
	let response = request
		.send()
		.await
		.map_err(|error| format!("policy fetch failed: {error}"))?;
	if !response.status().is_success() {
		return Err(format!("policy fetch returned HTTP {}", response.status()));
	}
	let bytes = response
		.bytes()
		.await
		.map_err(|error| format!("policy body read failed: {error}"))?;
	let doc = ToolSearchPolicyDoc::parse(&bytes)?;
	install_policy(doc.clone());
	Ok(doc)
}

/// Spawn the background refresher (bounded, jittered). No-op when
/// `AGENTGATEWAY_TOOL_SEARCH_POLICY_URL` is unset — the deployment keeps
/// env-only behavior with zero overhead.
pub fn spawn_policy_refresher() {
	let Ok(url) = std::env::var("AGENTGATEWAY_TOOL_SEARCH_POLICY_URL") else {
		return;
	};
	if url.trim().is_empty() {
		return;
	}
	// Mark the source BEFORE spawning (synchronous with server startup):
	// from this instant, "no snapshot yet" resolves fail-closed for native
	// instead of env-only — a governed deployment must never widen while
	// the very first fetch is still pending.
	set_policy_source_configured();
	let refresh_secs = std::env::var("AGENTGATEWAY_TOOL_SEARCH_POLICY_REFRESH_SECS")
		.ok()
		.and_then(|value| value.trim().parse::<u64>().ok())
		.filter(|secs| *secs > 0)
		.unwrap_or(DEFAULT_REFRESH_SECS);
	tokio::spawn(async move {
		loop {
			match refresh_policy_once(&url).await {
				Ok(doc) => {
					tracing::info!(
						tool_search.policy_revision = %doc.revision,
						tool_search.policy_models = doc.models.len(),
						"tool search policy refreshed"
					);
				},
				Err(error) => {
					// Keep last-known-good; if none was ever loaded, stay
					// fail-closed (native disabled) — never widen. Requests
					// keep flowing through the translation downgrade.
					tracing::warn!(
						"tool search policy refresh kept last-known-good (or stays fail-closed until first success): {error}"
					);
				},
			}
			// ±20% jitter so many replicas do not synchronize on the CP.
			let jitter = (refresh_secs as f64 * 0.2) as u64;
			let delay = refresh_secs + rand_jitter(jitter);
			tokio::time::sleep(Duration::from_secs(delay.max(1))).await;
		}
	});
}

fn rand_jitter(max: u64) -> u64 {
	if max == 0 {
		return 0;
	}
	// Cheap jitter without pulling a rand dependency: time-based.
	let nanos = std::time::SystemTime::now()
		.duration_since(std::time::UNIX_EPOCH)
		.map(|duration| duration.subsec_nanos() as u64)
		.unwrap_or(0);
	nanos % (max + 1)
}

#[cfg(test)]
mod tests {
	use super::*;

	fn doc(json: &str) -> ToolSearchPolicyDoc {
		ToolSearchPolicyDoc::parse(json.as_bytes()).expect("valid doc")
	}

	#[test]
	fn parses_a_strict_policy_document() {
		let parsed = doc(
			r#"{"revision":"rev-1","default_mode":"eager","models":{"claude-sonnet-5-5":"native","glm-5p3":"off"}}"#,
		);
		assert_eq!(parsed.revision, "rev-1");
		assert_eq!(parsed.default_mode, PolicyMode::Eager);
		assert_eq!(parsed.models.len(), 2);
		assert_eq!(parsed.mode_for("claude-sonnet-5-5"), PolicyMode::Native);
		assert_eq!(parsed.mode_for("glm-5p3"), PolicyMode::Off);
		// Unknown model falls back to the default.
		assert_eq!(parsed.mode_for("other"), PolicyMode::Eager);
	}

	#[test]
	fn rejects_unknown_fields_modes_and_shapes() {
		assert!(ToolSearchPolicyDoc::parse(br#"{"revision":"r","default_mode":"gateway"}"#).is_err());
		assert!(ToolSearchPolicyDoc::parse(br#"{"revision":"r","models":{"m":"gateway"}}"#).is_err());
		assert!(ToolSearchPolicyDoc::parse(br#"{"revision":"r","extra":true}"#).is_err());
		assert!(ToolSearchPolicyDoc::parse(br#"{"default_mode":"auto"}"#).is_err());
		assert!(ToolSearchPolicyDoc::parse(br#"{"revision":"  ","default_mode":"auto"}"#).is_err());
		assert!(ToolSearchPolicyDoc::parse(br#"not json"#).is_err());
	}

	#[test]
	fn rejects_oversized_documents() {
		let mut big = String::from(r#"{"revision":"r","models":{"m":"auto""#);
		while big.len() <= MAX_POLICY_BYTES {
			big.push_str(&format!(",\"m{}\":\"auto\"", big.len()));
		}
		big.push_str("}}");
		assert!(ToolSearchPolicyDoc::parse(big.as_bytes()).is_err());
	}

	#[test]
	fn parses_generated_control_plane_artifact_when_provided() {
		// Bridge check for the CP->gateway vertical slice: point
		// TOOLSEARCH_POLICY_FIXTURE at an artifact emitted by
		// scripts/llm-gateway/generate-toolsearch-policy.ts and the REAL
		// parser must accept it. Skipped when the variable is unset (normal
		// CI); wired into the release runbook.
		let Ok(path) = std::env::var("TOOLSEARCH_POLICY_FIXTURE") else {
			return;
		};
		let bytes =
			std::fs::read(&path).unwrap_or_else(|error| panic!("fixture {path} unreadable: {error}"));
		let doc = ToolSearchPolicyDoc::parse(&bytes)
			.unwrap_or_else(|error| panic!("generated artifact must parse: {error}"));
		assert!(!doc.revision.trim().is_empty());
		assert!(!doc.models.is_empty());
	}

	#[tokio::test]
	async fn decision_matrix_env_and_artifact_two_key() {
		// Both policy-mutating tests share the crate's global test lock so
		// they never observe each other's snapshots.
		let _guard = crate::config::lock_env_for_tests_async().await;
		install_policy(doc(
			r#"{"revision":"rev-1","default_mode":"auto","models":{"sonnet":"native","glm":"eager","kimi":"off"}}"#,
		));
		// Native requires BOTH artifact and env bootstrap.
		assert!(tool_search_decision(true, "sonnet").allow_native);
		assert!(!tool_search_decision(false, "sonnet").allow_native);
		// Artifact restrictions apply even when env would allow.
		assert!(!tool_search_decision(true, "glm").allow_native);
		assert!(!tool_search_decision(true, "kimi").allow_native);
		// Auto defers to the env ceiling.
		assert!(tool_search_decision(true, "other").allow_native);
		assert!(!tool_search_decision(false, "other").allow_native);
		clear_policy();
		// Without an artifact (and not configured): exact env-only behavior.
		assert!(tool_search_decision(true, "sonnet").allow_native);
		assert!(!tool_search_decision(false, "sonnet").allow_native);
	}

	#[tokio::test]
	async fn configured_source_without_snapshot_fails_closed() {
		let _guard = crate::config::lock_env_for_tests_async().await;
		clear_policy();
		set_policy_source_configured();
		// Cold start / restart / every-fetch-failed-so-far window: even an
		// env-allowlisted model must NOT get native — the governed policy may
		// be stricter than the allowlist, so env-only would be an escalation.
		let decision = tool_search_decision(true, "claude-sonnet-5-5");
		assert!(!decision.allow_native);
		assert_eq!(decision.reason, "policy-source-unavailable");
		assert!(decision.policy_revision.is_none());
		assert!(!tool_search_decision(false, "other").allow_native);
		clear_policy();
		// Not configured at all: exact env-only back-compat.
		let decision = tool_search_decision(true, "claude-sonnet-5-5");
		assert!(decision.allow_native);
		assert_eq!(decision.reason, "env-allowlist");
	}

	#[tokio::test]
	async fn rotation_crossover_stays_fail_closed_then_recovers() {
		use std::net::SocketAddr;
		use std::sync::Arc;
		use std::sync::atomic::{AtomicBool, Ordering};
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let _guard = crate::config::lock_env_for_tests_async().await;
		clear_policy();
		let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
			.await
			.unwrap();
		let addr = listener.local_addr().unwrap();
		// Credential-rotation crossover shape: the endpoint answers 401
		// (old token vs new route) until the rotation completes, then serves
		// the valid artifact again.
		let serving_valid = Arc::new(AtomicBool::new(false));
		let flag = serving_valid.clone();
		let body_ok = r#"{"revision":"v1","default_mode":"eager","models":{"sonnet":"native"}}"#;
		tokio::spawn(async move {
			loop {
				let Ok((mut socket, _)) = listener.accept().await else {
					break;
				};
				let ok = flag.load(Ordering::SeqCst);
				let mut buf = [0u8; 4096];
				let _ = socket.read(&mut buf).await;
				let response = if ok {
					format!(
						"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
						body_ok.len(),
						body_ok
					)
				} else {
					"HTTP/1.1 401 Unauthorized\r\nContent-Length: 0\r\nConnection: close\r\n\r\n".to_string()
				};
				let _ = socket.write_all(response.as_bytes()).await;
				let _ = socket.shutdown().await;
			}
		});
		let url = format!("http://{addr}/policy.json");

		// A restarted process with the source configured: every fetch 401s.
		set_policy_source_configured();
		assert!(refresh_policy_once(&url).await.is_err());
		let decision = tool_search_decision(true, "sonnet");
		assert!(
			!decision.allow_native,
			"crossover must not widen to env-only"
		);
		assert_eq!(decision.reason, "policy-source-unavailable");

		// Rotation completes: one good fetch restores governed resolution
		// (no restart involved).
		serving_valid.store(true, Ordering::SeqCst);
		let doc = refresh_policy_once(&url)
			.await
			.expect("valid fetch after rotation");
		assert_eq!(doc.revision, "v1");
		let decision = tool_search_decision(true, "sonnet");
		assert!(decision.allow_native);
		assert_eq!(decision.reason, "policy-native-env-confirmed");
		assert_eq!(decision.policy_revision.as_deref(), Some("v1"));
		clear_policy();
	}

	#[tokio::test]
	async fn refresher_swaps_snapshots_and_keeps_last_known_good() {
		use std::net::SocketAddr;
		use std::sync::atomic::{AtomicBool, Ordering};
		use tokio::io::{AsyncReadExt, AsyncWriteExt};
		let _guard = crate::config::lock_env_for_tests_async().await;
		clear_policy();
		let body_ok = r#"{"revision":"v1","default_mode":"eager","models":{"sonnet":"native"}}"#;
		let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
			.await
			.unwrap();
		let addr = listener.local_addr().unwrap();
		let serve_ok = Arc::new(AtomicBool::new(true));
		let serve_ok_task = serve_ok.clone();
		tokio::spawn(async move {
			loop {
				let Ok((mut socket, _)) = listener.accept().await else {
					break;
				};
				let ok = serve_ok_task.load(Ordering::SeqCst);
				let mut buf = [0u8; 4096];
				let _ = socket.read(&mut buf).await;
				let response = if ok {
					format!(
						"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
						body_ok.len(),
						body_ok
					)
				} else {
					"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
						.to_string()
				};
				let _ = socket.write_all(response.as_bytes()).await;
				let _ = socket.shutdown().await;
			}
		});
		let url = format!("http://{addr}/policy.json");

		// First refresh: snapshot installs and the decision uses it.
		let first = refresh_policy_once(&url).await.expect("first refresh");
		assert_eq!(first.revision, "v1");
		assert!(tool_search_decision(true, "sonnet").allow_native);
		assert!(!tool_search_decision(true, "glm").allow_native);

		// Upstream breaks: refresh fails, last-known-good stays active —
		// the mode change never requires a restart, and a broken artifact
		// source never degrades serving.
		serve_ok.store(false, Ordering::SeqCst);
		assert!(refresh_policy_once(&url).await.is_err());
		assert_eq!(current_policy().unwrap().revision, "v1");
		assert!(tool_search_decision(true, "sonnet").allow_native);

		clear_policy();
	}

	/// P3 demonstration (2026-10-20): the full no-restart policy lifecycle
	/// against REAL control-plane builder output — revisions generated from
	/// the prod catalog dump by
	/// apps/control-plane/scratch/toolsearch-p3-demo-generate.mts through the
	/// same buildToolSearchPolicyArtifact the publish action runs. Gated on
	/// TOOLSEARCH_DEMO_DIR so normal CI skips it; the release runbook drives
	/// it. Covers the operator's 10-step acceptance list: initial artifact,
	/// auto->explicit-eager refresh, per-model-only change, invalid artifact
	/// (last-known-good), unavailable endpoint, rollback restore, and native
	/// models unaffected throughout.
	#[tokio::test]
	async fn policy_refresh_demo_with_real_control_plane_artifacts() {
		use std::net::SocketAddr;
		use std::sync::atomic::{AtomicBool, Ordering};
		use tokio::io::{AsyncReadExt, AsyncWriteExt};

		let Ok(dir) = std::env::var("TOOLSEARCH_DEMO_DIR") else {
			return;
		};
		let _guard = crate::config::lock_env_for_tests_async().await;
		let previous_allowlist = std::env::var_os("AGENTGATEWAY_TOOL_SEARCH_MODELS");
		unsafe {
			// Mirrors the prod deployment allowlist (the safety ceiling).
			std::env::set_var(
				"AGENTGATEWAY_TOOL_SEARCH_MODELS",
				"claude-fable-5,claude-opus-5-5,claude-sonnet-5-5",
			);
		}
		clear_policy();

		let read = |name: &str| std::fs::read(format!("{dir}/{name}")).expect("demo artifact readable");
		let rev_a = read("rev-a.json");
		let rev_b = read("rev-b.json");
		let rev_c = read("rev-c.json");
		let invalid = read("invalid.json");
		let rollback = read("rev-d-rollback.json");

		// Swappable artifact endpoint: `body` is what the control plane is
		// currently serving; `serve_ok=false` simulates the endpoint being
		// down entirely (connection up, HTTP 500).
		let body: Arc<std::sync::Mutex<Vec<u8>>> = Arc::new(std::sync::Mutex::new(rev_a.clone()));
		let serve_ok = Arc::new(AtomicBool::new(true));
		let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
			.await
			.unwrap();
		let addr = listener.local_addr().unwrap();
		{
			let body = body.clone();
			let serve_ok = serve_ok.clone();
			tokio::spawn(async move {
				loop {
					let Ok((mut socket, _)) = listener.accept().await else {
						break;
					};
					let body = body.clone();
					let serve_ok = serve_ok.clone();
					tokio::spawn(async move {
						let mut buf = [0u8; 4096];
						let _ = socket.read(&mut buf).await;
						let response = if serve_ok.load(Ordering::SeqCst) {
							let payload = body.lock().unwrap().clone();
							format!(
								"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
								payload.len()
							)
							.into_bytes()
							.into_iter()
							.chain(payload)
							.collect::<Vec<u8>>()
						} else {
							"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
								.as_bytes()
								.to_vec()
						};
						let _ = socket.write_all(&response).await;
						let _ = socket.shutdown().await;
					});
				}
			});
		}
		let url = format!("http://{addr}/api/internal/llm-gateway/toolsearch-policy");

		// Steps 1-2: start with the current validated artifact. The CP
		// builder resolved g53's `auto` to its effective mode `eager` (no
		// runtime-verified native capability on its route), so the gateway
		// serves eager compatibility under the artifact's explicit entry. A
		// model NOT yet listed in the artifact (e.g. a newly enabled route
		// before the next publish) falls back to default auto -> env ceiling.
		refresh_policy_once(&url).await.expect("rev-a installs");
		assert_eq!(
			current_policy().unwrap().revision,
			"tsrev-demo-a-2026-10-20"
		);
		let g53 = tool_search_decision(false, "claude-gateway-fireworks-route-g53");
		assert!(!g53.allow_native);
		assert_eq!(g53.reason, "policy-eager");
		assert_eq!(
			g53.policy_revision.as_deref(),
			Some("tsrev-demo-a-2026-10-20")
		);
		let unlisted = tool_search_decision(false, "claude-gateway-fireworks-route-g52");
		assert!(!unlisted.allow_native);
		assert_eq!(unlisted.reason, "policy-auto-env");

		// Step 10 (checked at every stage): verified native models keep native.
		let sonnet = tool_search_decision(true, "claude-sonnet-5-5");
		assert!(sonnet.allow_native);
		assert_eq!(sonnet.reason, "policy-native-env-confirmed");

		// Steps 3-5: publish g53 explicitly eager; the refresh swaps the
		// snapshot in place — no restart — and the decision reports the new
		// revision with the explicit-eager reason.
		*body.lock().unwrap() = rev_b.clone();
		refresh_policy_once(&url).await.expect("rev-b installs");
		let g53 = tool_search_decision(false, "claude-gateway-fireworks-route-g53");
		assert!(!g53.allow_native);
		assert_eq!(g53.reason, "policy-eager");
		assert_eq!(
			g53.policy_revision.as_deref(),
			Some("tsrev-demo-b-2026-10-20")
		);

		// Step 6: change ONLY the Flash route (g53f -> off); g53 is untouched.
		*body.lock().unwrap() = rev_c.clone();
		refresh_policy_once(&url).await.expect("rev-c installs");
		let flash = tool_search_decision(false, "claude-gateway-fireworks-route-g53f");
		assert!(!flash.allow_native);
		assert_eq!(flash.reason, "policy-off");
		let g53 = tool_search_decision(false, "claude-gateway-fireworks-route-g53");
		assert_eq!(
			g53.reason, "policy-eager",
			"g53 must be unchanged by the flash edit"
		);
		assert!(tool_search_decision(true, "claude-sonnet-5-5").allow_native);

		// Step 7: an invalid artifact NEVER installs; last-known-good keeps
		// serving rev-c semantics.
		*body.lock().unwrap() = invalid.clone();
		assert!(refresh_policy_once(&url).await.is_err());
		assert_eq!(
			current_policy().unwrap().revision,
			"tsrev-demo-c-2026-10-20"
		);
		assert_eq!(
			tool_search_decision(false, "claude-gateway-fireworks-route-g53f").reason,
			"policy-off"
		);

		// Step 8: endpoint unavailable entirely — same last-known-good.
		serve_ok.store(false, Ordering::SeqCst);
		assert!(refresh_policy_once(&url).await.is_err());
		assert_eq!(
			current_policy().unwrap().revision,
			"tsrev-demo-c-2026-10-20"
		);
		serve_ok.store(true, Ordering::SeqCst);

		// Step 9: rollback re-publishes the exact prior approved bytes; the
		// gateway applies them on the next refresh — still no restart.
		*body.lock().unwrap() = rollback;
		refresh_policy_once(&url).await.expect("rollback installs");
		assert_eq!(
			current_policy().unwrap().revision,
			"tsrev-demo-b-2026-10-20"
		);
		assert_eq!(
			tool_search_decision(false, "claude-gateway-fireworks-route-g53").reason,
			"policy-eager"
		);
		// Step 10, final: native models unaffected end-to-end.
		let sonnet = tool_search_decision(true, "claude-sonnet-5-5");
		assert!(sonnet.allow_native);
		assert_eq!(
			sonnet.policy_revision.as_deref(),
			Some("tsrev-demo-b-2026-10-20")
		);

		clear_policy();
		match previous_allowlist {
			Some(value) => unsafe {
				std::env::set_var("AGENTGATEWAY_TOOL_SEARCH_MODELS", value);
			},
			None => unsafe {
				std::env::remove_var("AGENTGATEWAY_TOOL_SEARCH_MODELS");
			},
		}
	}
}
