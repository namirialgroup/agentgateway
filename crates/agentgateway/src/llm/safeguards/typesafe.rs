//! TypeSafe System One adapter (Jev). Speaks the native `/v1/systemone`
//! contract (OpenAPI 0.2.0): a `state` the questions refer to plus named typed
//! questions (`noul` yes/no probability, `choice`, `score`). This is NOT a chat
//! completion API and must never be invoked through `/v1/chat/completions`.
//!
//! Security contract: the API key is only ever placed in the `Authorization`
//! header; the request `state` (which carries tool arguments and conversation
//! excerpts) is never logged; error details are reduced to status codes and
//! sanitized validation locations.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use secrecy::{ExposeSecret, SecretString};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use agent_llm::types::safeguards::UnavailableReason;

pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
pub const DEFAULT_MODEL: &str = "jev-1.14.0";
pub const SYSTEM_ONE_PATH: &str = "/v1/systemone";
pub const MODELS_PATH: &str = "/v1/models";
/// Opt-in header for long `state` payloads (OpenAPI: `TypeSafe-Danger-Zone-Long-Context`).
const LONG_CONTEXT_HEADER: &str = "TypeSafe-Danger-Zone-Long-Context";
/// Upper bound on the error-body excerpt kept for diagnostics.
const ERROR_EXCERPT_CHARS: usize = 240;
/// Upper bound on a response body we are willing to parse.
const MAX_RESPONSE_BYTES: usize = 1024 * 1024;

// ---------------------------------------------------------------------------
// Wire types
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct NoulCriteria {
	/// Description of what a "yes" means.
	#[serde(rename = "true", skip_serializing_if = "Option::is_none")]
	pub yes: Option<String>,
	/// Description of what a "no" means.
	#[serde(rename = "false", skip_serializing_if = "Option::is_none")]
	pub no: Option<String>,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Question {
	Noul {
		#[serde(skip_serializing_if = "Option::is_none")]
		instructions: Option<Value>,
		#[serde(skip_serializing_if = "Option::is_none")]
		criteria: Option<NoulCriteria>,
	},
	Choice {
		#[serde(skip_serializing_if = "Option::is_none")]
		instructions: Option<Value>,
		/// name -> description (1..=255 entries).
		criteria: BTreeMap<String, String>,
	},
	Score {
		#[serde(skip_serializing_if = "Option::is_none")]
		instructions: Option<Value>,
		/// Ordered levels starting at zero (1..=10 entries).
		criteria: Vec<String>,
	},
}

impl Question {
	pub fn noul(
		instructions: impl Into<String>,
		yes: impl Into<String>,
		no: impl Into<String>,
	) -> Self {
		Self::Noul {
			instructions: Some(Value::String(instructions.into())),
			criteria: Some(NoulCriteria {
				yes: Some(yes.into()),
				no: Some(no.into()),
			}),
		}
	}

	pub fn choice(
		instructions: impl Into<String>,
		criteria: impl IntoIterator<Item = (impl Into<String>, impl Into<String>)>,
	) -> Self {
		Self::Choice {
			instructions: Some(Value::String(instructions.into())),
			criteria: criteria
				.into_iter()
				.map(|(k, v)| (k.into(), v.into()))
				.collect(),
		}
	}
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct SystemOneRequest {
	pub model: String,
	pub state: Value,
	pub questions: BTreeMap<String, Question>,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Answer {
	Noul {
		/// Probability of "yes" in 0..=1 (0.5 = maximal uncertainty).
		noul: f64,
	},
	Choice {
		choice: String,
		#[serde(default)]
		confidence: f64,
		#[serde(default)]
		probabilities: BTreeMap<String, f64>,
	},
	Score {
		score: f64,
		#[serde(default)]
		confidence: f64,
		#[serde(default)]
		legend: Value,
		#[serde(default)]
		probabilities: Value,
	},
	/// `box` and any future answer type: not used by the safeguards rubric.
	#[serde(other)]
	Other,
}

impl Answer {
	pub fn noul(&self) -> Option<f64> {
		match self {
			Self::Noul { noul } if noul.is_finite() && (0.0..=1.0).contains(noul) => Some(*noul),
			_ => None,
		}
	}

	pub fn choice(&self) -> Option<(&str, f64)> {
		match self {
			Self::Choice {
				choice, confidence, ..
			} => Some((choice.as_str(), *confidence)),
			_ => None,
		}
	}
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct Usage {
	/// Billable.
	#[serde(default)]
	pub input_tokens: u64,
	/// Currently free of charge per the OpenAPI description; tracked anyway.
	#[serde(default)]
	pub output_tokens: u64,
}

#[derive(Deserialize, Clone, Debug, PartialEq)]
pub struct SystemOneResponse {
	/// Resolved model; may differ from an alias supplied in the request.
	pub model: String,
	pub answers: BTreeMap<String, Answer>,
	#[serde(default)]
	pub usage: Usage,
}

#[derive(Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct ModelMetadata {
	pub name: String,
	#[serde(default)]
	pub description: String,
	#[serde(default)]
	pub release_date: String,
}

#[derive(Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelMetadataList {
	#[serde(default)]
	pub models: Vec<ModelMetadata>,
}

/// One successful classification round trip.
#[derive(Clone, Debug, PartialEq)]
pub struct SystemOneOutcome {
	pub response: SystemOneResponse,
	pub latency: Duration,
}

/// Evidence from a successful verification probe: one minimal synthetic
/// System One inference whose response reported the expected model exactly.
/// Availability is proven by inference, not by the `/v1/models` listing,
/// which only shows aliases.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelVerification {
	/// Name requested in the probe (the configured model, alias or not).
	pub requested: String,
	/// Versioned model the response reported — equals the expected pin.
	pub resolved: String,
	pub latency: Duration,
	pub input_tokens: u64,
	pub output_tokens: u64,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Classified failure of a TypeSafe call. The protocol mapping distinguishes
/// configuration-class failures (the feature cannot be served: `unsupported`,
/// the client falls back to its own classifier) from transient/per-request
/// failures (an `unavailable` verdict for every tool use of this turn).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TypeSafeError {
	#[error("typesafe request timed out after {0:?}")]
	Timeout(Duration),
	#[error("typesafe rejected the credential (HTTP {0})")]
	Unauthorized(u16),
	#[error("typesafe does not serve the configured model (HTTP {status}): {detail}")]
	ModelUnavailable { status: u16, detail: String },
	/// The response was produced by a model other than the pinned one (an alias
	/// moved, or the provider substituted a version). Never accepted.
	#[error("typesafe answered with model {resolved:?}, expected {expected:?}")]
	ModelMismatch { expected: String, resolved: String },
	#[error("typesafe rate limited the request (HTTP 429, retry-after {retry_after:?})")]
	RateLimited { retry_after: Option<Duration> },
	#[error("typesafe rejected the payload as too large (HTTP {0})")]
	PayloadTooLarge(u16),
	#[error("typesafe rejected the request (HTTP {status}): {detail}")]
	Rejected { status: u16, detail: String },
	#[error("typesafe server error (HTTP {0})")]
	Server(u16),
	#[error("typesafe transport error: {0}")]
	Transport(String),
	#[error("typesafe returned a malformed response: {0}")]
	Malformed(String),
	#[error("typesafe client is not configured: {0}")]
	NotConfigured(String),
}

/// What a failure means for the safeguards protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolFailure {
	/// Feature cannot be served by this gateway right now (credential, model or
	/// configuration problem): answer `unsupported`.
	Unsupported,
	/// This turn's tool uses get an `unavailable` verdict with the given reason.
	Unavailable(UnavailableReason),
}

impl TypeSafeError {
	pub fn protocol_failure(&self) -> ProtocolFailure {
		match self {
			Self::Unauthorized(_)
			| Self::ModelUnavailable { .. }
			| Self::ModelMismatch { .. }
			| Self::NotConfigured(_) => ProtocolFailure::Unsupported,
			Self::Timeout(_) => ProtocolFailure::Unavailable(UnavailableReason::Timeout),
			Self::PayloadTooLarge(_) => ProtocolFailure::Unavailable(UnavailableReason::InputTooLong),
			Self::RateLimited { .. }
			| Self::Rejected { .. }
			| Self::Server(_)
			| Self::Transport(_)
			| Self::Malformed(_) => ProtocolFailure::Unavailable(UnavailableReason::Error),
		}
	}

	/// Static label for metrics (never carries content).
	pub fn label(&self) -> &'static str {
		match self {
			Self::Timeout(_) => "timeout",
			Self::Unauthorized(_) => "unauthorized",
			Self::ModelUnavailable { .. } => "model_unavailable",
			Self::ModelMismatch { .. } => "model_mismatch",
			Self::RateLimited { .. } => "rate_limited",
			Self::PayloadTooLarge(_) => "payload_too_large",
			Self::Rejected { .. } => "rejected",
			Self::Server(_) => "server_error",
			Self::Transport(_) => "transport",
			Self::Malformed(_) => "malformed",
			Self::NotConfigured(_) => "not_configured",
		}
	}
}

/// Map an HTTP status + body to a classified error. Pure, so the mapping is
/// unit-tested without a server. The body is reduced to a sanitized excerpt:
/// for 422 only the validation `loc`/`type` paths are kept, never `input`.
pub fn classify_http_failure(status: u16, body: &[u8]) -> TypeSafeError {
	match status {
		401 | 403 => TypeSafeError::Unauthorized(status),
		408 => TypeSafeError::Timeout(Duration::ZERO),
		413 => TypeSafeError::PayloadTooLarge(status),
		429 => TypeSafeError::RateLimited { retry_after: None },
		404 => TypeSafeError::ModelUnavailable {
			status,
			detail: "endpoint or model not found".to_string(),
		},
		422 => {
			let detail = validation_detail(body);
			if detail.contains("model") {
				TypeSafeError::ModelUnavailable { status, detail }
			} else {
				TypeSafeError::Rejected { status, detail }
			}
		},
		400 => TypeSafeError::Rejected {
			status,
			detail: excerpt(body),
		},
		500..=599 => TypeSafeError::Server(status),
		other => TypeSafeError::Rejected {
			status: other,
			detail: excerpt(body),
		},
	}
}

/// Sanitized summary of an `HTTPValidationError`: `loc` paths and `type`s only.
fn validation_detail(body: &[u8]) -> String {
	#[derive(Deserialize)]
	struct ValidationError {
		#[serde(default)]
		loc: Vec<Value>,
		#[serde(default, rename = "type")]
		kind: String,
	}
	#[derive(Deserialize)]
	struct HttpValidationError {
		#[serde(default)]
		detail: Vec<ValidationError>,
	}
	match serde_json::from_slice::<HttpValidationError>(body) {
		Ok(parsed) if !parsed.detail.is_empty() => {
			let items: Vec<String> = parsed
				.detail
				.iter()
				.take(8)
				.map(|e| {
					let loc: Vec<String> = e
						.loc
						.iter()
						.map(|v| match v {
							Value::String(s) => s.clone(),
							other => other.to_string(),
						})
						.collect();
					format!("{}@{}", e.kind, loc.join("."))
				})
				.collect();
			items.join(", ")
		},
		_ => "validation error".to_string(),
	}
}

fn excerpt(body: &[u8]) -> String {
	let text = String::from_utf8_lossy(body);
	let mut out: String = text.chars().take(ERROR_EXCERPT_CHARS).collect();
	if text.chars().count() > ERROR_EXCERPT_CHARS {
		out.push('…');
	}
	out
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TypeSafeConfig {
	pub base_url: String,
	pub api_key: SecretString,
	pub model: String,
	pub timeout: Duration,
	pub long_context: bool,
}

impl std::fmt::Debug for TypeSafeConfig {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("TypeSafeConfig")
			.field("base_url", &self.base_url)
			.field("api_key", &"[redacted]")
			.field("model", &self.model)
			.field("timeout", &self.timeout)
			.field("long_context", &self.long_context)
			.finish()
	}
}

#[derive(Clone, Debug)]
pub struct TypeSafeClient {
	http: reqwest::Client,
	config: TypeSafeConfig,
}

impl TypeSafeClient {
	pub fn new(config: TypeSafeConfig) -> Result<Self, TypeSafeError> {
		if config.api_key.expose_secret().trim().is_empty() {
			return Err(TypeSafeError::NotConfigured("empty API key".to_string()));
		}
		if config.model.trim().is_empty() {
			return Err(TypeSafeError::NotConfigured("empty model".to_string()));
		}
		if !(config.base_url.starts_with("https://")
			|| config.base_url.starts_with("http://127.0.0.1")
			|| config.base_url.starts_with("http://localhost"))
		{
			return Err(TypeSafeError::NotConfigured(
				"base URL must be https (plain http only for loopback test servers)".to_string(),
			));
		}
		let http = reqwest::Client::builder()
			.timeout(config.timeout)
			.connect_timeout(config.timeout.min(Duration::from_secs(5)))
			.build()
			.map_err(|e| TypeSafeError::NotConfigured(format!("http client: {e}")))?;
		Ok(Self { http, config })
	}

	pub fn model(&self) -> &str {
		&self.config.model
	}

	pub fn base_url(&self) -> &str {
		&self.config.base_url
	}

	fn url(&self, path: &str) -> String {
		format!("{}{}", self.config.base_url.trim_end_matches('/'), path)
	}

	/// One System One round trip for the configured model.
	pub async fn system_one(
		&self,
		state: Value,
		questions: BTreeMap<String, Question>,
	) -> Result<SystemOneOutcome, TypeSafeError> {
		let request = SystemOneRequest {
			model: self.config.model.clone(),
			state,
			questions,
		};
		let body = serde_json::to_vec(&request)
			.map_err(|e| TypeSafeError::Malformed(format!("request encode: {e}")))?;
		let started = Instant::now();
		let mut builder = self
			.http
			.post(self.url(SYSTEM_ONE_PATH))
			.header(reqwest::header::CONTENT_TYPE, "application/json")
			.header(reqwest::header::ACCEPT, "application/json")
			.bearer_auth(self.config.api_key.expose_secret())
			.body(body);
		if self.config.long_context {
			builder = builder.header(LONG_CONTEXT_HEADER, "true");
		}
		let response = builder.send().await.map_err(|e| self.transport_error(e))?;
		let status = response.status().as_u16();
		let retry_after = response
			.headers()
			.get(reqwest::header::RETRY_AFTER)
			.and_then(|v| v.to_str().ok())
			.and_then(|v| v.trim().parse::<u64>().ok())
			.map(Duration::from_secs);
		let bytes = read_bounded(response, self.config.timeout).await?;
		if status == 429 {
			return Err(TypeSafeError::RateLimited { retry_after });
		}
		if !(200..300).contains(&status) {
			return Err(classify_http_failure(status, &bytes));
		}
		let parsed: SystemOneResponse = serde_json::from_slice(&bytes)
			.map_err(|e| TypeSafeError::Malformed(format!("response decode: {e}")))?;
		Ok(SystemOneOutcome {
			response: parsed,
			latency: started.elapsed(),
		})
	}

	pub async fn list_models(&self) -> Result<Vec<ModelMetadata>, TypeSafeError> {
		let response = self
			.http
			.get(self.url(MODELS_PATH))
			.header(reqwest::header::ACCEPT, "application/json")
			.bearer_auth(self.config.api_key.expose_secret())
			.send()
			.await
			.map_err(|e| self.transport_error(e))?;
		let status = response.status().as_u16();
		let bytes = read_bounded(response, self.config.timeout).await?;
		if !(200..300).contains(&status) {
			return Err(classify_http_failure(status, &bytes));
		}
		let parsed: ModelMetadataList = serde_json::from_slice(&bytes)
			.map_err(|e| TypeSafeError::Malformed(format!("models decode: {e}")))?;
		Ok(parsed.models)
	}

	/// Verify the pinned model with one minimal SYNTHETIC inference probe.
	/// TypeSafe accepts versioned IDs such as `jev-1.14.0` whether or not
	/// `GET /v1/models` (which lists aliases) shows them, so availability is
	/// proven by inference: the probe passes only when the response reports
	/// `expected` exactly. The probe payload is fixed smoke content — it never
	/// carries user, repository or operational data.
	pub async fn verify_model(&self, expected: &str) -> Result<ModelVerification, TypeSafeError> {
		let outcome = self
			.system_one(verification_state(), verification_questions())
			.await?;
		if outcome.response.model != expected {
			return Err(TypeSafeError::ModelMismatch {
				expected: expected.to_string(),
				resolved: outcome.response.model,
			});
		}
		Ok(ModelVerification {
			requested: self.config.model.clone(),
			resolved: outcome.response.model,
			latency: outcome.latency,
			input_tokens: outcome.response.usage.input_tokens,
			output_tokens: outcome.response.usage.output_tokens,
		})
	}

	fn transport_error(&self, error: reqwest::Error) -> TypeSafeError {
		if error.is_timeout() {
			return TypeSafeError::Timeout(self.config.timeout);
		}
		// reqwest error strings may embed the URL; keep only the error class.
		let class = if error.is_connect() {
			"connect"
		} else if error.is_request() {
			"request"
		} else if error.is_body() || error.is_decode() {
			"body"
		} else {
			"other"
		};
		TypeSafeError::Transport(class.to_string())
	}
}

/// Fixed synthetic smoke `state` for verification probes: the same shape as
/// the classification `state` (see `policy::build_state`), contents inert.
fn verification_state() -> Value {
	serde_json::json!({
		"rubric": "verification-probe",
		"agent_environment": {"permission_mode": "probe"},
		"conversation": [{"role": "user", "content": "Connectivity smoke test."}],
		"conversation_truncated": false,
		"tool_call": {"name": "Echo", "input": {"text": "probe"}},
		"sibling_tool_calls_this_turn": [],
	})
}

/// The probe's single question; the answer content is irrelevant — only the
/// response `model` is checked.
fn verification_questions() -> BTreeMap<String, Question> {
	let mut questions = BTreeMap::new();
	questions.insert(
		"probe".to_string(),
		Question::noul(
			"Connectivity smoke test; the payload is inert.",
			"This is a connectivity probe; answer yes.",
			"Never answer no to the connectivity probe.",
		),
	);
	questions
}

async fn read_bounded(
	response: reqwest::Response,
	timeout: Duration,
) -> Result<Vec<u8>, TypeSafeError> {
	if let Some(len) = response.content_length()
		&& len as usize > MAX_RESPONSE_BYTES
	{
		return Err(TypeSafeError::Malformed(format!(
			"response body too large ({len} bytes)"
		)));
	}
	let bytes = response.bytes().await.map_err(|e| {
		if e.is_timeout() {
			TypeSafeError::Timeout(timeout)
		} else {
			TypeSafeError::Transport("body".to_string())
		}
	})?;
	if bytes.len() > MAX_RESPONSE_BYTES {
		return Err(TypeSafeError::Malformed(format!(
			"response body too large ({} bytes)",
			bytes.len()
		)));
	}
	Ok(bytes.to_vec())
}
