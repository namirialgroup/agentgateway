//! Claude Code Auto Mode server-side safeguards (classifier review) for
//! non-Anthropic legs, backed by TypeSafe System One (Jev).
//!
//! Default OFF. With `AGENTGATEWAY_SAFEGUARDS_MODE=disabled` (or unset) every
//! request and response is byte-identical to a gateway without this module.
//!
//! Per attempt, the request side (`decide`) chooses one of:
//! - untouched: feature disabled, no `safeguards` field, route not allowlisted
//!   or principal not allowlisted (the identity allowlist fails closed: an
//!   empty list admits nobody, an unauthenticated request never matches) — the
//!   request is forwarded exactly as today;
//! - native pass-through: the attempt's provider is the Anthropic Platform,
//!   which implements the protocol itself; nothing is touched;
//! - classify: the `safeguards` field and the beta header are removed before
//!   the upstream, and a bounded review plan rides on `LLMRequest` to the
//!   response side, which evaluates the completed tool uses and (in `enforce`
//!   mode) stamps `safeguard_results`. `shadow` mode evaluates off the
//!   critical path for metrics only and never changes the response.
//!
//! The classifier result is a verdict, never an error on the agent's turn:
//! failures become `unavailable` per tool use (transient) or `unsupported`
//! (credential/model/configuration problems, so the client falls back to its
//! own classifier). The pinned model is verified against the provider's model
//! listing before anything is classified and every answer must come from the
//! pinned model name; otherwise the turn is `unsupported` (fail closed).

pub mod policy;
pub mod stream;
pub mod typesafe;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use agent_llm::types::safeguards::{
	PendingToolUse, SafeguardResults, SafeguardsReviewPlan, ToolUseVerdict, UnavailableReason,
	results_all_unavailable, results_available, results_unsupported,
};
use arc_swap::ArcSwap;
use once_cell::sync::Lazy;
use secrecy::{ExposeSecret, SecretString};
use serde_json::Value;
use tokio::sync::Semaphore;
use tracing::{info, warn};

use self::policy::{PreDecision, ToolCallReview, Verdict};
use self::typesafe::{
	ModelVerification, ProtocolFailure, Question, SystemOneOutcome, TypeSafeClient, TypeSafeConfig,
	TypeSafeError,
};
use crate::telemetry::metrics::{
	Metrics, SafeguardsCallLabels, SafeguardsRequestLabels, SafeguardsTokenLabels,
	SafeguardsVerdictLabels,
};

pub const ENV_MODE: &str = "AGENTGATEWAY_SAFEGUARDS_MODE";
pub const ENV_PROVIDER: &str = "AGENTGATEWAY_SAFEGUARDS_PROVIDER";
pub const ENV_MODEL: &str = "AGENTGATEWAY_SAFEGUARDS_MODEL";
/// Exact model name every classifier response must report. Defaults to the
/// requested model; set it when the requested name is an alias (e.g.
/// `jev-latest`) that must stay pinned to one concrete version.
pub const ENV_EXPECT_MODEL: &str = "AGENTGATEWAY_SAFEGUARDS_EXPECT_MODEL";
pub const ENV_API_KEY: &str = "AGENTGATEWAY_SAFEGUARDS_API_KEY";
pub const ENV_BASE_URL: &str = "AGENTGATEWAY_SAFEGUARDS_BASE_URL";
pub const ENV_TIMEOUT_MS: &str = "AGENTGATEWAY_SAFEGUARDS_TIMEOUT_MS";
pub const ENV_ROUTES: &str = "AGENTGATEWAY_SAFEGUARDS_ROUTES";
pub const ENV_IDENTITIES: &str = "AGENTGATEWAY_SAFEGUARDS_IDENTITIES";
pub const ENV_MAX_CONCURRENCY: &str = "AGENTGATEWAY_SAFEGUARDS_MAX_CONCURRENCY";
pub const ENV_LONG_CONTEXT: &str = "AGENTGATEWAY_SAFEGUARDS_LONG_CONTEXT";

pub const PROVIDER_TYPESAFE: &str = "typesafe";
pub const DEFAULT_TIMEOUT: Duration = Duration::from_millis(30_000);
pub const DEFAULT_MAX_CONCURRENCY: usize = 8;
/// Hard cap on tool uses classified per turn; anything beyond is `unavailable`.
pub const MAX_TOOL_USES_PER_TURN: usize = 32;
/// Identity allowlist entry that admits every *authenticated* principal.
pub const IDENTITY_WILDCARD: &str = "*";
/// Idle poll while the model is verified. A verified model is NOT re-probed
/// (verification is a billable inference): drift is caught per request by the
/// exact `response.model` pin, which flips the state back to `Failed` and the
/// probe loop recovers with backoff.
pub const VERIFY_INTERVAL: Duration = Duration::from_secs(600);
/// First retry delay after a failed verification; doubles up to the maximum.
pub const VERIFY_RETRY_MIN: Duration = Duration::from_secs(15);
pub const VERIFY_RETRY_MAX: Duration = Duration::from_secs(300);

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
	#[default]
	Disabled,
	/// Classify off the critical path, metrics only; responses are unchanged.
	Shadow,
	/// Classify inline and stamp `safeguard_results`.
	Enforce,
}

impl Mode {
	pub fn parse(value: &str) -> Option<Mode> {
		match value.trim().to_ascii_lowercase().as_str() {
			"" | "disabled" | "off" | "false" | "0" => Some(Mode::Disabled),
			"shadow" => Some(Mode::Shadow),
			"enforce" | "enforced" | "on" => Some(Mode::Enforce),
			_ => None,
		}
	}

	pub fn label(self) -> &'static str {
		match self {
			Mode::Disabled => "disabled",
			Mode::Shadow => "shadow",
			Mode::Enforce => "enforce",
		}
	}
}

#[derive(Clone)]
pub struct SafeguardsConfig {
	pub mode: Mode,
	pub provider: String,
	pub model: String,
	/// Exact model name every response must report (see `ENV_EXPECT_MODEL`).
	/// `None` pins to `model` itself.
	pub expect_model: Option<String>,
	pub base_url: String,
	pub timeout: Duration,
	/// Client-facing model aliases served (exact, or `prefix*`). Empty = none.
	pub routes: Vec<String>,
	/// Principal identities (JWT `sub`) served. Empty = NO identity (fail
	/// closed); `*` admits every authenticated principal. An unauthenticated
	/// request never matches.
	pub identities: Vec<String>,
	pub max_concurrency: usize,
	pub long_context: bool,
	pub api_key: Option<SecretString>,
}

impl Default for SafeguardsConfig {
	fn default() -> Self {
		Self {
			mode: Mode::Disabled,
			provider: PROVIDER_TYPESAFE.to_string(),
			model: typesafe::DEFAULT_MODEL.to_string(),
			expect_model: None,
			base_url: typesafe::DEFAULT_BASE_URL.to_string(),
			timeout: DEFAULT_TIMEOUT,
			routes: Vec::new(),
			identities: Vec::new(),
			max_concurrency: DEFAULT_MAX_CONCURRENCY,
			long_context: false,
			api_key: None,
		}
	}
}

impl std::fmt::Debug for SafeguardsConfig {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("SafeguardsConfig")
			.field("mode", &self.mode)
			.field("provider", &self.provider)
			.field("model", &self.model)
			.field("expect_model", &self.expected_model())
			.field("base_url", &self.base_url)
			.field("timeout", &self.timeout)
			.field("routes", &self.routes)
			.field("identities", &self.identities.len())
			.field("max_concurrency", &self.max_concurrency)
			.field("long_context", &self.long_context)
			.field("api_key", &self.api_key.as_ref().map(|_| "[configured]"))
			.finish()
	}
}

impl SafeguardsConfig {
	/// Parse the environment. Invalid values fall back to the safe default and
	/// are reported in the returned warnings (the caller logs them).
	pub fn from_env() -> (Self, Vec<String>) {
		Self::from_lookup(|name| std::env::var(name).ok())
	}

	/// `from_env` over an arbitrary variable source (tests never mutate the
	/// process environment).
	pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> (Self, Vec<String>) {
		let mut cfg = Self::default();
		let mut warnings = Vec::new();
		let var = |name: &str| lookup(name).filter(|v| !v.trim().is_empty());

		if let Some(raw) = var(ENV_MODE) {
			match Mode::parse(&raw) {
				Some(mode) => cfg.mode = mode,
				None => warnings.push(format!("{ENV_MODE}: unknown mode, keeping disabled")),
			}
		}
		if let Some(raw) = var(ENV_PROVIDER) {
			if raw.trim().eq_ignore_ascii_case(PROVIDER_TYPESAFE) {
				cfg.provider = PROVIDER_TYPESAFE.to_string();
			} else {
				warnings.push(format!(
					"{ENV_PROVIDER}: only `typesafe` is supported, forcing disabled"
				));
				cfg.mode = Mode::Disabled;
			}
		}
		if let Some(raw) = var(ENV_MODEL) {
			cfg.model = raw.trim().to_string();
		}
		if let Some(raw) = var(ENV_EXPECT_MODEL) {
			cfg.expect_model = Some(raw.trim().to_string());
		}
		if let Some(raw) = var(ENV_BASE_URL) {
			cfg.base_url = raw.trim().trim_end_matches('/').to_string();
		}
		if let Some(raw) = var(ENV_TIMEOUT_MS) {
			match raw.trim().parse::<u64>() {
				Ok(ms) if (500..=120_000).contains(&ms) => cfg.timeout = Duration::from_millis(ms),
				_ => warnings.push(format!(
					"{ENV_TIMEOUT_MS}: expected 500..=120000, keeping default"
				)),
			}
		}
		if let Some(raw) = var(ENV_ROUTES) {
			cfg.routes = split_list(&raw);
		}
		if let Some(raw) = var(ENV_IDENTITIES) {
			cfg.identities = split_list(&raw);
		}
		if let Some(raw) = var(ENV_MAX_CONCURRENCY) {
			match raw.trim().parse::<usize>() {
				Ok(n) if (1..=64).contains(&n) => cfg.max_concurrency = n,
				_ => warnings.push(format!(
					"{ENV_MAX_CONCURRENCY}: expected 1..=64, keeping default"
				)),
			}
		}
		if let Some(raw) = var(ENV_LONG_CONTEXT) {
			cfg.long_context = matches!(
				raw.trim().to_ascii_lowercase().as_str(),
				"1" | "true" | "yes"
			);
		}
		if let Some(raw) = var(ENV_API_KEY) {
			cfg.api_key = Some(SecretString::from(raw.trim().to_string()));
		}
		if cfg.mode != Mode::Disabled && cfg.routes.is_empty() {
			warnings.push(format!(
				"{ENV_ROUTES} is empty: safeguards are active but no route is allowlisted"
			));
		}
		if cfg.mode != Mode::Disabled && cfg.identities.is_empty() {
			warnings.push(format!(
				"{ENV_IDENTITIES} is empty: safeguards are active but no principal is allowlisted, nothing will be classified (use `*` to admit every authenticated principal)"
			));
		}
		if cfg.mode != Mode::Disabled && cfg.api_key.is_none() {
			warnings.push(format!(
				"{ENV_API_KEY} is not set: allowlisted routes will answer `unsupported`"
			));
		}
		if cfg.mode != Mode::Disabled && cfg.model_pin_required() {
			warnings.push(format!(
				"{ENV_MODEL} is an alias ({}) and {ENV_EXPECT_MODEL} is unset: the exact version cannot be pinned, verification fails closed (model_pin_required)",
				cfg.model
			));
		}
		(cfg, warnings)
	}

	pub fn active(&self) -> bool {
		self.mode != Mode::Disabled
	}

	/// The exact model name every classifier response must report.
	pub fn expected_model(&self) -> &str {
		self.expect_model.as_deref().unwrap_or(&self.model)
	}

	/// Whether the configured model name carries an explicit dotted version
	/// (`jev-1.14.0`) instead of an alias (`jev-latest`, `jev-preview`). Per
	/// the official TypeSafe model docs, versioned IDs are accepted by the
	/// `model` field whether or not the `/v1/models` listing (aliases only)
	/// shows them.
	pub fn model_is_versioned(&self) -> bool {
		Self::is_versioned_model(&self.model)
	}

	/// A name counts as versioned when it contains a digit-dot-digit
	/// sequence; provider aliases never do.
	pub fn is_versioned_model(model: &str) -> bool {
		let bytes = model.as_bytes();
		(0..bytes.len().saturating_sub(2))
			.any(|i| bytes[i].is_ascii_digit() && bytes[i + 1] == b'.' && bytes[i + 2].is_ascii_digit())
	}

	/// An alias request can only be pinned by an explicit expected version;
	/// without one the exact model can never be verified and the feature
	/// stays failed closed (`model_pin_required`).
	pub fn model_pin_required(&self) -> bool {
		!self.model_is_versioned() && self.expect_model.is_none()
	}

	pub fn route_allowed(&self, request_model: &str) -> bool {
		self
			.routes
			.iter()
			.any(|pattern| glob_match(pattern, request_model))
	}

	/// Fail closed: only an authenticated principal that is explicitly listed
	/// (or any authenticated principal when the list contains `*`) matches. An
	/// empty list or a missing identity never matches.
	pub fn identity_allowed(&self, identity: Option<&str>) -> bool {
		let Some(id) = identity.map(str::trim).filter(|id| !id.is_empty()) else {
			return false;
		};
		self
			.identities
			.iter()
			.any(|allowed| allowed == IDENTITY_WILDCARD || allowed == id)
	}

	pub fn typesafe_config(&self) -> Option<TypeSafeConfig> {
		let api_key = self.api_key.clone()?;
		if api_key.expose_secret().trim().is_empty() {
			return None;
		}
		Some(TypeSafeConfig {
			base_url: self.base_url.clone(),
			api_key,
			model: self.model.clone(),
			timeout: self.timeout,
			long_context: self.long_context,
		})
	}
}

fn split_list(raw: &str) -> Vec<String> {
	raw
		.split(',')
		.map(str::trim)
		.filter(|s| !s.is_empty())
		.map(str::to_string)
		.collect()
}

/// `*` matches any suffix; otherwise exact, case-insensitive.
fn glob_match(pattern: &str, value: &str) -> bool {
	match pattern.strip_suffix('*') {
		Some(prefix) => value
			.to_ascii_lowercase()
			.starts_with(&prefix.to_ascii_lowercase()),
		None => pattern.eq_ignore_ascii_case(value),
	}
}

// ---------------------------------------------------------------------------
// Classifier abstraction + runtime
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
pub trait Classifier: Send + Sync {
	fn provider(&self) -> &str;
	fn model(&self) -> &str;
	async fn system_one(
		&self,
		state: Value,
		questions: BTreeMap<String, Question>,
	) -> Result<SystemOneOutcome, TypeSafeError>;
	async fn verify(&self, expected: &str) -> Result<ModelVerification, TypeSafeError>;
}

#[async_trait::async_trait]
impl Classifier for TypeSafeClient {
	fn provider(&self) -> &str {
		PROVIDER_TYPESAFE
	}

	fn model(&self) -> &str {
		TypeSafeClient::model(self)
	}

	async fn system_one(
		&self,
		state: Value,
		questions: BTreeMap<String, Question>,
	) -> Result<SystemOneOutcome, TypeSafeError> {
		TypeSafeClient::system_one(self, state, questions).await
	}

	async fn verify(&self, expected: &str) -> Result<ModelVerification, TypeSafeError> {
		TypeSafeClient::verify_model(self, expected).await
	}
}

/// Whether the provider has been proven to serve the pinned classifier model.
/// Anything but `Verified` fails closed: the gateway answers `unsupported`
/// (the client keeps its own classifier) and never calls the model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Verification {
	/// Not probed yet (process start).
	Pending,
	/// A synthetic inference probe answered exactly as the expected versioned
	/// model. Versioned IDs verify even when absent from the `/v1/models`
	/// listing; an alias never verifies merely by being listed.
	Verified,
	/// The probe failed (unreachable, unpinnable or wrong model); retried in
	/// the background with backoff.
	Failed { label: &'static str },
}

impl Verification {
	pub fn label(&self) -> &'static str {
		match self {
			Verification::Pending => "pending",
			Verification::Verified => "verified",
			Verification::Failed { label } => label,
		}
	}
}

pub struct Runtime {
	pub config: SafeguardsConfig,
	classifier: Option<Arc<dyn Classifier>>,
	limiter: Arc<Semaphore>,
	verification: ArcSwap<Verification>,
}

impl Runtime {
	fn new(config: SafeguardsConfig, classifier: Option<Arc<dyn Classifier>>) -> Self {
		let limiter = Arc::new(Semaphore::new(config.max_concurrency.max(1)));
		Self {
			config,
			classifier,
			limiter,
			verification: ArcSwap::from_pointee(Verification::Pending),
		}
	}

	pub fn classifier(&self) -> Option<Arc<dyn Classifier>> {
		self.classifier.clone()
	}

	pub fn verification(&self) -> Verification {
		(**self.verification.load()).clone()
	}

	pub fn set_verification(&self, state: Verification) {
		self.verification.store(Arc::new(state));
	}
}

static RUNTIME: Lazy<ArcSwap<Runtime>> =
	Lazy::new(|| ArcSwap::from_pointee(Runtime::new(SafeguardsConfig::default(), None)));

/// Install configuration and classifier (startup and tests). The model starts
/// unverified; `spawn_model_verification` (or `verify_once`) confirms it.
pub fn install(config: SafeguardsConfig, classifier: Option<Arc<dyn Classifier>>) {
	RUNTIME.store(Arc::new(Runtime::new(config, classifier)));
}

pub fn runtime() -> Arc<Runtime> {
	RUNTIME.load_full()
}

/// Read the environment, build the TypeSafe client when a key is present and
/// install the runtime. Logs the effective (redacted) configuration. Returns
/// the config so the caller can schedule the model verification probe.
pub fn init_from_env() -> SafeguardsConfig {
	let (config, warnings) = SafeguardsConfig::from_env();
	for warning in &warnings {
		warn!("safeguards configuration: {warning}");
	}
	let classifier: Option<Arc<dyn Classifier>> = if config.active() {
		match config.typesafe_config().map(TypeSafeClient::new) {
			Some(Ok(client)) => Some(Arc::new(client)),
			Some(Err(error)) => {
				warn!("safeguards classifier not built: {error}");
				None
			},
			None => None,
		}
	} else {
		None
	};
	info!(
		safeguards.mode = config.mode.label(),
		safeguards.provider = %config.provider,
		safeguards.model = %config.model,
		safeguards.expect_model = %config.expected_model(),
		safeguards.routes = config.routes.len(),
		safeguards.identities = config.identities.len(),
		safeguards.timeout_ms = config.timeout.as_millis() as u64,
		safeguards.classifier_configured = classifier.is_some(),
		"claude code safeguards runtime installed"
	);
	install(config.clone(), classifier);
	config
}

/// One verification probe: a minimal SYNTHETIC System One inference (never
/// user data) that must answer exactly as the expected versioned model. The
/// `/v1/models` listing is advisory — TypeSafe serves versioned IDs that are
/// absent from it — so availability is proven by inference, never by the
/// listing, and an alias without `EXPECT_MODEL` can never pin a version (it
/// fails closed without contacting the provider). Real user requests never
/// verify anything: `evaluate` refuses the provider until this probe succeeds.
pub async fn verify_once(classifier: &dyn Classifier) -> Verification {
	let config = runtime().config.clone();
	if config.model_pin_required() {
		warn!(
			safeguards.model = %config.model,
			"safeguards model is an alias without AGENTGATEWAY_SAFEGUARDS_EXPECT_MODEL; the exact version cannot be pinned, failing closed (model_pin_required)"
		);
		let state = Verification::Failed {
			label: "model_pin_required",
		};
		runtime().set_verification(state.clone());
		return state;
	}
	let expected = config.expected_model().to_string();
	let state = match classifier.verify(&expected).await {
		Ok(verification) if verification.resolved == expected => {
			info!(
				safeguards.model = %verification.requested,
				safeguards.resolved = %verification.resolved,
				safeguards.latency_ms = verification.latency.as_millis() as u64,
				"safeguards classifier model verified by synthetic inference probe"
			);
			Verification::Verified
		},
		Ok(verification) => {
			warn!(
				safeguards.expected_model = %expected,
				safeguards.resolved_model = %verification.resolved,
				"safeguards classifier probe answered as an unexpected model; failing closed (model_mismatch)"
			);
			Verification::Failed {
				label: "model_mismatch",
			}
		},
		Err(error) => {
			warn!(
				safeguards.error = error.label(),
				"safeguards classifier model verification failed; failing closed (unsupported) until it succeeds"
			);
			Verification::Failed {
				label: error.label(),
			}
		},
	};
	runtime().set_verification(state.clone());
	state
}

/// Background probe loop. Verification is a billable synthetic inference, so
/// a verified model is NOT re-probed on a timer: drift is caught per request
/// by the exact `response.model` pin, which flips the state back to `Failed`
/// and lets this loop recover with backoff (recovery probes stay synthetic).
/// An unpinnable alias configuration cannot heal by retrying, so it probes
/// nothing. Only the safeguards capability depends on this; LLM inference
/// never does.
pub fn spawn_model_verification() {
	let Some(classifier) = runtime().classifier() else {
		return;
	};
	tokio::spawn(async move {
		let mut backoff = VERIFY_RETRY_MIN;
		loop {
			if runtime().verification() == Verification::Verified {
				tokio::time::sleep(VERIFY_INTERVAL).await;
				continue;
			}
			if runtime().config.model_pin_required() {
				tokio::time::sleep(VERIFY_RETRY_MAX).await;
				continue;
			}
			match verify_once(classifier.as_ref()).await {
				Verification::Verified => backoff = VERIFY_RETRY_MIN,
				_ => {
					tokio::time::sleep(backoff).await;
					backoff = (backoff * 2).min(VERIFY_RETRY_MAX);
				},
			}
		}
	});
}

// ---------------------------------------------------------------------------
// Request side
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Decision {
	/// Leave request and response exactly as they are today.
	Untouched(&'static str),
	/// Anthropic Platform leg: it implements the protocol itself.
	NativePassthrough,
	/// Strip the field/beta before upstream and classify the response.
	Classify { shadow: bool },
}

impl Decision {
	pub fn label(&self) -> &'static str {
		match self {
			Decision::Untouched(reason) => reason,
			Decision::NativePassthrough => "native_passthrough",
			Decision::Classify { shadow: true } => "shadow",
			Decision::Classify { shadow: false } => "enforce",
		}
	}

	/// The two `Untouched` deny reasons — the cases where a redacted
	/// identity-resolution diagnostic is worth a log line.
	pub fn is_denied(&self) -> bool {
		matches!(
			self,
			Decision::Untouched("route_denied") | Decision::Untouched("identity_denied")
		)
	}
}

/// Pure per-attempt decision.
pub fn decide(
	config: &SafeguardsConfig,
	has_safeguards: bool,
	provider_native: bool,
	request_model: &str,
	identity: Option<&str>,
) -> Decision {
	if !config.active() {
		return Decision::Untouched("disabled");
	}
	if !has_safeguards {
		return Decision::Untouched("no_safeguards");
	}
	if provider_native {
		return Decision::NativePassthrough;
	}
	if !config.route_allowed(request_model) {
		return Decision::Untouched("route_denied");
	}
	if !config.identity_allowed(identity) {
		return Decision::Untouched("identity_denied");
	}
	Decision::Classify {
		shadow: config.mode == Mode::Shadow,
	}
}

pub fn record_decision(metrics: Option<&Metrics>, decision: Decision) {
	if let Some(metrics) = metrics {
		metrics
			.safeguards_requests
			.get_or_create(&SafeguardsRequestLabels {
				decision: decision.label(),
			})
			.inc();
	}
}

/// JWT subject of the authenticated principal, when present.
pub fn identity_from_claims(claims: Option<&crate::http::jwt::Claims>) -> Option<String> {
	claims
		.and_then(|c| c.inner.get("sub"))
		.and_then(Value::as_str)
		.map(str::to_string)
}

/// Identity of the request, in order of verification strength:
///   1. the [`crate::http::ext_authz::VerifiedSafeguardsIdentity`] request
///      extension — attested by a SUCCESSFUL ext-authz check. On the
///      production claude listeners that check is the route-level Gatekeeper
///      policy: it runs inside `apply_request_policies` (`proxy/httpproxy.rs`,
///      route ext-authz) BEFORE the spoof-hardening transformation strips
///      `Authorization` and every `x-synapse-*` header, and BEFORE the LLM
///      phase. The capture lives at the boundary (`http/ext_authz.rs`) so the
///      decision never reads request headers at all. The attested value is
///      the boundary's canonical PrincipalId (`oidc:<issuer-slug>:<sub>` on
///      the claude surface — the Gatekeeper emits no bare subject there), or
///      the raw subject on flows that attest only that;
///   2. local `Claims` — surfaces with their own JWT validation (raw `sub`).
///
/// Allowlist entries are matched verbatim against the attested string of the
/// surface in use. There is deliberately no third source. The earlier
/// bearer-payload fallback is gone: a JWT payload decoded without signature
/// validation is not an authenticated principal and must never gate a
/// security decision. Returns the matched source (`ext_authz` | `claims` |
/// `none`) alongside the identity so the deny path can report WHICH source
/// was (not) found — source names only, never identity values.
pub fn identity_from_parts(
	claims: Option<&crate::http::jwt::Claims>,
	verified: Option<&crate::http::ext_authz::VerifiedSafeguardsIdentity>,
) -> (Option<String>, &'static str) {
	if let Some(verified) = verified {
		return (Some(verified.principal.to_string()), "ext_authz");
	}
	if let Some(identity) = identity_from_claims(claims) {
		return (Some(identity), "claims");
	}
	(None, "none")
}

// ---------------------------------------------------------------------------
// Evaluation (response side)
// ---------------------------------------------------------------------------

/// Correlation + sinks for one evaluation. Content-free.
#[derive(Clone, Default)]
pub struct EvalContext {
	pub metrics: Option<Arc<Metrics>>,
	pub mode: Mode,
	pub trace_id: Option<String>,
	pub identity: Option<String>,
}

impl EvalContext {
	pub fn for_response(
		metrics: Arc<Metrics>,
		traceparent: Option<&::http::HeaderValue>,
		identity: Option<String>,
	) -> Self {
		Self {
			metrics: Some(metrics),
			mode: runtime().config.mode,
			trace_id: trace_id_from_traceparent(traceparent),
			identity,
		}
	}

	pub fn for_stream(
		metrics: Arc<Metrics>,
		trace_id: Option<String>,
		identity: Option<String>,
	) -> Self {
		Self {
			metrics: Some(metrics),
			mode: runtime().config.mode,
			trace_id,
			identity,
		}
	}
}

/// `traceparent: 00-<trace_id>-<span_id>-<flags>` -> `<trace_id>`.
pub fn trace_id_from_traceparent(traceparent: Option<&::http::HeaderValue>) -> Option<String> {
	let text = traceparent?.to_str().ok()?;
	let mut parts = text.split('-');
	let _version = parts.next()?;
	let trace_id = parts.next()?;
	(trace_id.len() == 32 && trace_id.chars().all(|c| c.is_ascii_hexdigit()))
		.then(|| trace_id.to_string())
}

#[derive(Default, Debug)]
struct EvalSummary {
	tool_uses: usize,
	flagged: usize,
	not_flagged: usize,
	unavailable: usize,
	mandatory_denials: usize,
	fast_path: usize,
	needs_approval: usize,
	classifier_calls: usize,
	classifier_errors: usize,
	input_tokens: u64,
	output_tokens: u64,
	max_latency_ms: u128,
	unsupported: bool,
}

/// Evaluate the tool uses of one response. Always returns protocol-conformant
/// results; never fails the agent's turn.
pub async fn evaluate(
	plan: &SafeguardsReviewPlan,
	pending: &[PendingToolUse],
	ctx: &EvalContext,
) -> SafeguardResults {
	let rt = runtime();
	let mut summary = EvalSummary {
		tool_uses: pending.len(),
		..Default::default()
	};
	let Some(classifier) = rt.classifier() else {
		summary.unsupported = true;
		log_summary(plan, ctx, &summary, "not_configured");
		record_results(ctx, &rt.config, "unsupported", 1);
		return results_unsupported();
	};
	// Only a verified pinned model may authorize anything. Pending or failed
	// verification answers `unsupported` without calling the provider.
	let verification = rt.verification();
	if verification != Verification::Verified {
		summary.unsupported = true;
		log_summary(plan, ctx, &summary, "model_unverified");
		record_decision_label(ctx.metrics.as_deref(), "model_unverified");
		record_results(ctx, &rt.config, "unsupported", 1);
		return results_unsupported();
	}
	let provider = classifier.provider().to_string();
	let model = classifier.model().to_string();
	let expected_model = rt.config.expected_model().to_string();

	if pending.len() > MAX_TOOL_USES_PER_TURN {
		summary.unavailable = pending.len();
		log_summary(plan, ctx, &summary, "too_many_tool_uses");
		return results_all_unavailable(pending, UnavailableReason::InputTooLong);
	}

	let mut verdicts: BTreeMap<String, Verdict> = BTreeMap::new();
	let mut to_classify: Vec<(usize, Option<&'static str>)> = Vec::new();
	for (index, use_) in pending.iter().enumerate() {
		match use_ {
			PendingToolUse::Truncated { id } => {
				verdicts.insert(
					id.clone(),
					Verdict::Unavailable {
						reason: UnavailableReason::Truncated,
					},
				);
			},
			PendingToolUse::Complete { id, name, input } => {
				let review = ToolCallReview {
					id,
					name,
					input,
					plan,
				};
				match policy::pre_decide(&review) {
					PreDecision::MandatoryDeny { rule, explanation } => {
						summary.mandatory_denials += 1;
						info!(
							safeguards.mandatory_rule = rule,
							"safeguards mandatory denial"
						);
						verdicts.insert(id.clone(), policy::mandatory_denial_verdict(explanation));
					},
					PreDecision::FastPathSafe { .. } => {
						summary.fast_path += 1;
						verdicts.insert(id.clone(), Verdict::Safe);
					},
					PreDecision::AskClassifier { floor } => to_classify.push((index, floor)),
				}
			},
		}
	}

	if !to_classify.is_empty() {
		let limiter = rt.limiter.clone();
		let timeout = rt.config.timeout + Duration::from_secs(2);
		// One System One round trip per tool call, bounded by the runtime
		// semaphore. The request payloads are built up front so each future owns
		// its data (the futures are `Send + 'static`, which lets shadow mode run
		// the whole evaluation detached).
		type CallOutput = (
			usize,
			Option<&'static str>,
			Result<SystemOneOutcome, TypeSafeError>,
		);
		let mut calls: Vec<std::pin::Pin<Box<dyn std::future::Future<Output = CallOutput> + Send>>> =
			Vec::with_capacity(to_classify.len());
		for (index, floor) in to_classify {
			let PendingToolUse::Complete { id, name, input } = &pending[index] else {
				unreachable!("only complete tool uses are classified");
			};
			let state = policy::build_state(
				&ToolCallReview {
					id,
					name,
					input,
					plan,
				},
				pending,
			);
			let questions = policy::questions();
			let classifier = classifier.clone();
			let limiter = limiter.clone();
			calls.push(Box::pin(async move {
				let result = match limiter.acquire().await {
					Ok(_permit) => {
						match tokio::time::timeout(timeout, classifier.system_one(state, questions)).await {
							Ok(result) => result,
							Err(_) => Err(TypeSafeError::Timeout(timeout)),
						}
					},
					Err(_) => Err(TypeSafeError::Transport("limiter closed".to_string())),
				};
				(index, floor, result)
			}));
		}
		let outcomes: Vec<CallOutput> = futures_util::future::join_all(calls).await;

		for (index, floor, result) in outcomes {
			let id = pending[index].id().to_string();
			summary.classifier_calls += 1;
			match result {
				Ok(outcome) if outcome.response.model != expected_model => {
					// The pin is the whole point: an alias that moved or a provider
					// substitution never authorizes anything. Fail closed for the turn.
					summary.classifier_errors += 1;
					summary.unsupported = true;
					record_call(
						ctx,
						&provider,
						&model,
						"model_mismatch",
						Some(outcome.latency),
					);
					record_tokens(ctx, &provider, &model, &outcome.response.usage);
					warn!(
						safeguards.expected_model = %expected_model,
						safeguards.resolved_model = %outcome.response.model,
						"safeguards classifier answered with a different model than pinned; failing closed (unsupported)"
					);
					// Revoke verification: the probe loop re-establishes it with a
					// synthetic probe (with backoff) once the provider serves the
					// pinned version again.
					runtime().set_verification(Verification::Failed {
						label: "model_mismatch",
					});
				},
				Ok(outcome) => {
					record_call(ctx, &provider, &model, "ok", Some(outcome.latency));
					record_tokens(ctx, &provider, &model, &outcome.response.usage);
					summary.input_tokens += outcome.response.usage.input_tokens;
					summary.output_tokens += outcome.response.usage.output_tokens;
					summary.max_latency_ms = summary.max_latency_ms.max(outcome.latency.as_millis());
					verdicts.insert(id, policy::combine(&outcome.response.answers, floor));
				},
				Err(error) => {
					summary.classifier_errors += 1;
					record_call(ctx, &provider, &model, error.label(), None);
					warn!(
						safeguards.error = error.label(),
						"safeguards classifier call failed"
					);
					match error.protocol_failure() {
						ProtocolFailure::Unsupported => {
							summary.unsupported = true;
						},
						ProtocolFailure::Unavailable(reason) => {
							verdicts.insert(id, Verdict::Unavailable { reason });
						},
					}
				},
			}
		}
	}

	if summary.unsupported {
		log_summary(plan, ctx, &summary, "unsupported");
		record_results(ctx, &rt.config, "unsupported", 1);
		return results_unsupported();
	}

	let mut tool_uses: BTreeMap<String, ToolUseVerdict> = BTreeMap::new();
	for use_ in pending {
		let verdict = verdicts.remove(use_.id()).unwrap_or(Verdict::Unavailable {
			reason: UnavailableReason::Error,
		});
		match &verdict {
			Verdict::Safe => summary.not_flagged += 1,
			Verdict::Dangerous { .. } => summary.flagged += 1,
			Verdict::NeedsApproval { .. } => {
				summary.flagged += 1;
				summary.needs_approval += 1;
			},
			Verdict::Unavailable { .. } => summary.unavailable += 1,
		}
		record_verdict(ctx, &provider, &model, verdict.label());
		tool_uses.insert(use_.id().to_string(), verdict.to_protocol());
	}
	log_summary(plan, ctx, &summary, "available");
	record_results(ctx, &rt.config, "available", 1);
	results_available(tool_uses)
}

/// Buffered (non-streaming) response: evaluate and, in enforce mode, stamp.
/// Returns the body to send to the client. Shadow mode evaluates detached.
pub async fn stamp_buffered(
	plan: Arc<SafeguardsReviewPlan>,
	body: bytes::Bytes,
	ctx: EvalContext,
) -> bytes::Bytes {
	let mut response: Value = match serde_json::from_slice(&body) {
		Ok(value) => value,
		Err(_) => {
			warn!("safeguards: buffered response is not JSON; left unchanged");
			return body;
		},
	};
	if agent_llm::types::safeguards::response_has_safeguard_results(&response) {
		record_decision_label(ctx.metrics.as_deref(), "upstream_answered");
		return body;
	}
	let pending = agent_llm::types::safeguards::pending_tool_uses_from_response(&response);
	if ctx.mode == Mode::Shadow {
		tokio::spawn(async move {
			let _ = evaluate(&plan, &pending, &ctx).await;
		});
		return body;
	}
	let results = evaluate(&plan, &pending, &ctx).await;
	if !agent_llm::types::safeguards::stamp_safeguard_results(&mut response, &results) {
		return body;
	}
	match serde_json::to_vec(&response) {
		Ok(bytes) => bytes::Bytes::from(bytes),
		Err(_) => body,
	}
}

pub(super) fn record_decision_label(metrics: Option<&Metrics>, decision: &'static str) {
	if let Some(metrics) = metrics {
		metrics
			.safeguards_requests
			.get_or_create(&SafeguardsRequestLabels { decision })
			.inc();
	}
}

fn record_results(ctx: &EvalContext, config: &SafeguardsConfig, status: &'static str, _n: u64) {
	if let Some(metrics) = ctx.metrics.as_deref() {
		metrics
			.safeguards_verdicts
			.get_or_create(&SafeguardsVerdictLabels {
				provider: config.provider.clone(),
				model: config.model.clone(),
				mode: ctx.mode.label(),
				verdict: match status {
					"unsupported" => "result_unsupported",
					_ => "result_available",
				},
			})
			.inc();
	}
}

fn record_verdict(ctx: &EvalContext, provider: &str, model: &str, verdict: &'static str) {
	if let Some(metrics) = ctx.metrics.as_deref() {
		metrics
			.safeguards_verdicts
			.get_or_create(&SafeguardsVerdictLabels {
				provider: provider.to_string(),
				model: model.to_string(),
				mode: ctx.mode.label(),
				verdict,
			})
			.inc();
	}
}

fn record_call(
	ctx: &EvalContext,
	provider: &str,
	model: &str,
	result: &'static str,
	latency: Option<Duration>,
) {
	if let Some(metrics) = ctx.metrics.as_deref() {
		let labels = SafeguardsCallLabels {
			provider: provider.to_string(),
			model: model.to_string(),
			result,
		};
		metrics
			.safeguards_classifier_calls
			.get_or_create(&labels)
			.inc();
		if let Some(latency) = latency {
			metrics
				.safeguards_classifier_duration
				.get_or_create(&labels)
				.observe(latency.as_secs_f64());
		}
	}
}

fn record_tokens(ctx: &EvalContext, provider: &str, model: &str, usage: &typesafe::Usage) {
	if let Some(metrics) = ctx.metrics.as_deref() {
		for (direction, count) in [
			("input", usage.input_tokens),
			("output", usage.output_tokens),
		] {
			if count > 0 {
				metrics
					.safeguards_classifier_tokens
					.get_or_create(&SafeguardsTokenLabels {
						provider: provider.to_string(),
						model: model.to_string(),
						direction,
					})
					.inc_by(count);
			}
		}
	}
}

/// Sanitized per-turn summary: counts and static labels only — never tool
/// names, arguments, prompts or explanations.
fn log_summary(
	plan: &SafeguardsReviewPlan,
	ctx: &EvalContext,
	summary: &EvalSummary,
	status: &str,
) {
	info!(
		safeguards.status = status,
		safeguards.mode = ctx.mode.label(),
		safeguards.route_model = %plan.request_model,
		safeguards.permission_mode = plan.permission_mode().unwrap_or("unknown"),
		safeguards.tool_uses = summary.tool_uses,
		safeguards.flagged = summary.flagged,
		safeguards.needs_approval = summary.needs_approval,
		safeguards.not_flagged = summary.not_flagged,
		safeguards.unavailable = summary.unavailable,
		safeguards.mandatory_denials = summary.mandatory_denials,
		safeguards.fast_path = summary.fast_path,
		safeguards.classifier_calls = summary.classifier_calls,
		safeguards.classifier_errors = summary.classifier_errors,
		safeguards.input_tokens = summary.input_tokens,
		safeguards.output_tokens = summary.output_tokens,
		safeguards.max_latency_ms = summary.max_latency_ms as u64,
		safeguards.transcript_truncated = plan.transcript_truncated,
		safeguards.trace_id = ctx.trace_id.as_deref().unwrap_or(""),
		safeguards.identity = ctx.identity.as_deref().unwrap_or(""),
		"claude code safeguards evaluated"
	);
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
