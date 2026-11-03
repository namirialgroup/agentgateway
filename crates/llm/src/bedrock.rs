use agent_core::prelude::Strng;
use agent_core::strng;

use crate::apply;

#[derive(Debug, Clone)]
pub struct AwsRegion {
	pub region: String,
}

#[apply(schema_enum!)]
#[derive(Default)]
pub enum BedrockEndpointPreference {
	#[default]
	RuntimePreferred,
	MantlePreferred,
	MantleOnly,
	RuntimeOnly,
}

/// Chat transport for Anthropic Messages models on a Bedrock backend.
///
/// `Converse` (default) keeps the historical AWS Converse/ConverseStream
/// translation. `NativeMessages` serves `/v1/messages` traffic through
/// Bedrock Runtime `InvokeModel`/`InvokeModelWithResponseStream` with the
/// Anthropic Messages body passed through natively (including Tool Search
/// server tools, `defer_loading`, `tool_reference` blocks and client
/// `cache_control` TTLs, none of which Converse can represent). Runtime-only:
/// Mantle is a different service with different semantics and is never
/// selected by this transport.
#[apply(schema_enum!)]
#[derive(Default)]
pub enum BedrockTransport {
	#[default]
	Converse,
	NativeMessages,
}

impl BedrockTransport {
	pub fn as_str(&self) -> &'static str {
		match self {
			BedrockTransport::Converse => "converse",
			BedrockTransport::NativeMessages => "native-messages",
		}
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BedrockEndpoint {
	Runtime,
	Mantle,
}

// Mantle signs SigV4 under this name instead of the default "bedrock".
const MANTLE_SIGNING_SERVICE_NAME: &str = "bedrock-mantle";

/// Operator-approved cross-Region count_tokens destinations, as
/// comma-separated `base-model-id=region` entries (e.g.
/// `anthropic.claude-opus-5-5=us-east-1`). A card-listed Region is necessary
/// but NOT sufficient: count_tokens carries the full conversation payload,
/// so leaving the provider's Region requires this explicit per-model
/// authorization. Read per call (not cached) so tests and rolling config stay
/// independent.
const COUNT_TOKENS_APPROVED_REGIONS_ENV: &str = "AGENTGATEWAY_BEDROCK_COUNT_TOKENS_REGIONS";

/// CountTokens capability per Anthropic base model ID, from the AWS model
/// cards (verified 2026-11-03). Models absent from the table use the Runtime
/// CountTokens API surface (the historic resolution) — correct for Sonnet 4.6
/// and Haiku 4.5 and conservative for models whose cards were not verified.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CountTokensSurface {
	/// The Runtime CountTokens API serves this model (geo/global profiles).
	Runtime,
	/// bedrock-runtime CountTokens is denied for this model (CRIS-only
	/// launch); bedrock-mantle serves Anthropic-native count_tokens in
	/// exactly these commercial Regions (GovCloud partitions excluded — a
	/// commercial account cannot treat them as availability).
	Mantle { regions: &'static [&'static str] },
}

pub fn count_tokens_surface(base_model: &str) -> CountTokensSurface {
	match base_model {
		"anthropic.claude-opus-5-5" => CountTokensSurface::Mantle {
			regions: &["us-east-1", "ap-southeast-4"],
		},
		"anthropic.claude-sonnet-5-5" => CountTokensSurface::Mantle { regions: &[] },
		"anthropic.claude-haiku-5-5" => CountTokensSurface::Mantle { regions: &[] },
		"anthropic.claude-opus-4-8" => CountTokensSurface::Mantle {
			regions: &[
				"us-east-1",
				"eu-north-1",
				"eu-west-1",
				"ap-northeast-1",
				"ap-southeast-4",
			],
		},
		"anthropic.claude-sonnet-5" => CountTokensSurface::Mantle {
			regions: &["us-east-1", "eu-north-1", "eu-west-1", "ap-southeast-4"],
		},
		"anthropic.claude-opus-5" => CountTokensSurface::Mantle {
			regions: &["us-east-1", "eu-north-1", "eu-west-1", "ap-southeast-4"],
		},
		"anthropic.claude-fable-5" => CountTokensSurface::Mantle {
			regions: &["us-east-1"],
		},
		_ => CountTokensSurface::Runtime,
	}
}

/// Strip a cross-Region inference-profile prefix (`global.`/`eu.`/`us.`/…) to
/// the base model ID bedrock-mantle names; bare IDs pass through unchanged.
pub fn base_model_id(request_model: &str) -> &str {
	match request_model.split_once('.') {
		Some((_, base)) if base.starts_with("anthropic.") => base,
		_ => request_model,
	}
}

/// The resolved count_tokens destination for one request model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CountTokensResolution {
	/// Runtime CountTokens API (works only once `bedrock:CountTokens` is
	/// granted; models whose cards list Count tokens as supported there).
	Runtime,
	/// Anthropic-native count_tokens on bedrock-mantle in this Region
	/// (in-Region when the Region matches the card, or an explicitly approved
	/// cross-Region destination).
	Mantle { region: String },
	/// No supported surface in an approved Region: reject before dispatch —
	/// never send the payload to AWS, never return an estimate.
	Unsupported,
}

#[apply(schema!)]
#[cfg_attr(feature = "schema", schemars(rename = "BedrockProviderConfig"))]
pub struct Provider {
	/// Model ID to send to Bedrock, overriding the model in the client request.
	#[serde(default, rename = "model", skip_serializing_if = "Option::is_none")]
	pub model_override: Option<Strng>, // Optional: model override for Bedrock API path
	/// AWS region for the Bedrock endpoint.
	pub region: Strng, // Required: AWS region
	/// Identifier of the Bedrock guardrail to apply.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub guardrail_identifier: Option<Strng>,
	/// Version of the Bedrock guardrail to apply.
	#[serde(skip_serializing_if = "Option::is_none")]
	pub guardrail_version: Option<Strng>,
	/// Which endpoint to prefer (Runtime vs Mantle).
	#[serde(default)]
	pub endpoint_preference: BedrockEndpointPreference,
	/// Chat transport for Anthropic Messages models (`converse` by default;
	/// `native-messages` routes Messages traffic through Runtime InvokeModel).
	#[serde(default)]
	pub transport: BedrockTransport,
}

impl super::Provider for Provider {
	const NAME: Strng = strng::literal!("aws.bedrock");
}

impl Provider {
	pub fn is_anthropic_model(&self, request_model: &str) -> bool {
		request_model
			.to_ascii_lowercase()
			.contains("anthropic.claude")
	}

	/// Resolves which Bedrock endpoint (Runtime vs Mantle) serves the given route and model.
	pub fn resolve_endpoint(
		&self,
		route_type: super::RouteType,
		model_id: Option<&str>,
		catalog: crate::model_catalog::Catalog<'_>,
	) -> BedrockEndpoint {
		use super::RouteType as RT;
		match route_type {
			RT::Embeddings | RT::GeminiCountTokens | RT::Rerank | RT::Realtime => {
				BedrockEndpoint::Runtime
			},
			RT::Models => BedrockEndpoint::Mantle,
			RT::Detect | RT::Passthrough | RT::GenerateContent => BedrockEndpoint::Runtime,
			// count_tokens is model-capability-routed, NOT blanket-routed: only
			// models whose cards deny Runtime CountTokens AND have an approved
			// mantle Region resolve here (see resolve_count_tokens). The gateway
			// rejects `Unsupported` models before dispatch, so this arm is only
			// reached for dispatchable requests. Chat keeps its own resolution.
			RT::AnthropicTokenCount
				if model_id.is_some_and(|m| {
					matches!(
						self.resolve_count_tokens(m),
						CountTokensResolution::Mantle { .. }
					)
				}) =>
			{
				BedrockEndpoint::Mantle
			},
			// Chat and Runtime-capable count-tokens follow the model's endpoint:
			// Runtime's Converse / CountTokens APIs, or Mantle's native
			// OpenAI/Anthropic APIs.
			RT::Completions | RT::Messages | RT::Responses | RT::AnthropicTokenCount => {
				self.chat_endpoint(model_id, catalog)
			},
		}
	}

	/// Resolve the count_tokens destination for a request model (geo profile
	/// or base ID). In-Region mantle needs no approval; any cross-Region
	/// destination requires an explicit per-model operator approval in
	/// `AGENTGATEWAY_BEDROCK_COUNT_TOKENS_REGIONS` — the data-residency gate,
	/// because count_tokens ships the full conversation payload.
	pub fn resolve_count_tokens(&self, request_model: &str) -> CountTokensResolution {
		match count_tokens_surface(base_model_id(request_model)) {
			CountTokensSurface::Runtime => CountTokensResolution::Runtime,
			CountTokensSurface::Mantle { regions } => {
				// An explicit RuntimeOnly pin cannot reach the mantle surface.
				if matches!(
					self.endpoint_preference,
					BedrockEndpointPreference::RuntimeOnly
				) {
					return CountTokensResolution::Unsupported;
				}
				// In-Region first: no residency question, no approval needed.
				if regions.contains(&self.region.as_str()) {
					return CountTokensResolution::Mantle {
						region: self.region.as_str().to_string(),
					};
				}
				// Cross-Region only with an explicit, card-valid approval.
				if let Some(approved) = approved_count_tokens_region(base_model_id(request_model)) {
					if regions.contains(&approved.as_str()) {
						return CountTokensResolution::Mantle { region: approved };
					}
				}
				CountTokensResolution::Unsupported
			},
		}
	}

	fn chat_endpoint(
		&self,
		model_id: Option<&str>,
		catalog: crate::model_catalog::Catalog<'_>,
	) -> BedrockEndpoint {
		use BedrockEndpoint::{Mantle, Runtime};
		use BedrockEndpointPreference::*;

		// The native Messages transport is defined against Bedrock Runtime
		// InvokeModel APIs only. Never resolve it onto Mantle, whatever the
		// endpoint preference says — a misconfigured combination must stay on
		// the documented Runtime surface rather than silently changing service.
		if self.transport == BedrockTransport::NativeMessages {
			return Runtime;
		}

		use crate::model_catalog::tags;
		let has = |tag| model_id.is_some_and(|m| catalog.is_some_and(|c| c.model_has_tag(m, tag)));
		match self.endpoint_preference {
			RuntimeOnly => Runtime,
			MantleOnly => Mantle,
			// Inline guardrails require Runtime, even when the model only advertises Mantle.
			RuntimePreferred
				if self.guardrail_identifier.is_some() || self.guardrail_version.is_some() =>
			{
				Runtime
			},
			// Prefer Runtime; use Mantle only for models tagged Mantle but not Runtime.
			RuntimePreferred if has(tags::MANTLE) && !has(tags::RUNTIME) => Mantle,
			RuntimePreferred => Runtime,
			// Prefer Mantle; use Runtime only for models tagged Runtime but not Mantle.
			MantlePreferred if has(tags::RUNTIME) && !has(tags::MANTLE) => Runtime,
			MantlePreferred => Mantle,
		}
	}

	/// SigV4 signing-service override for the endpoint (`Some` for Mantle, else the default `bedrock`).
	pub fn signing_service_name(&self, endpoint: BedrockEndpoint) -> Option<&'static str> {
		match endpoint {
			BedrockEndpoint::Mantle => Some(MANTLE_SIGNING_SERVICE_NAME),
			BedrockEndpoint::Runtime => None,
		}
	}

	pub fn supported_chat_formats(
		&self,
		request_model: &str,
		catalog: crate::model_catalog::Catalog<'_>,
	) -> Vec<super::ChatFormat> {
		use super::ChatFormat;
		const NATIVE: [ChatFormat; 3] = [
			ChatFormat::OpenAICompletions,
			ChatFormat::AnthropicMessages,
			ChatFormat::OpenAIResponses,
		];
		match self.chat_endpoint(Some(request_model), catalog) {
			// all chat runtime models seem to support converse
			BedrockEndpoint::Runtime if self.transport == BedrockTransport::NativeMessages => {
				// Native Messages transport: Anthropic Messages requests pass
				// through natively over Runtime InvokeModel; OpenAI Responses
				// input still falls back to Converse (the translation table
				// has no Responses→Messages conversion, and failing closed to
				// the proven Converse translation beats advertising a gap).
				vec![ChatFormat::AnthropicMessages, ChatFormat::BedrockConverse]
			},
			BedrockEndpoint::Runtime => vec![ChatFormat::BedrockConverse],
			BedrockEndpoint::Mantle => {
				// Short circuit tag checks as we should just use the message endpoint
				if self.is_anthropic_model(request_model) {
					return vec![ChatFormat::AnthropicMessages];
				}
				if let Some(tags) = catalog.and_then(|c| c.get_model_tags(request_model)) {
					let declared: Vec<ChatFormat> = NATIVE
						.into_iter()
						.filter(|f| tags.contains(f.tag()))
						.collect();
					if !declared.is_empty() {
						return declared;
					}
				}
				// fallback for entries that havent gotten into loaded to catalog yet
				let basemodelname = request_model.to_ascii_lowercase();
				if basemodelname.contains("openai") || basemodelname.contains("grok") {
					vec![ChatFormat::OpenAICompletions, ChatFormat::OpenAIResponses]
				} else {
					vec![ChatFormat::OpenAICompletions]
				}
			},
		}
	}

	pub fn get_path_for_route(
		&self,
		route_type: super::RouteType,
		streaming: bool,
		model: &str,
		endpoint: BedrockEndpoint,
	) -> Strng {
		if matches!(endpoint, BedrockEndpoint::Mantle) {
			return match route_type {
				super::RouteType::Responses => strng::literal!("/v1/responses"),
				super::RouteType::Messages => strng::literal!("/anthropic/v1/messages"),
				super::RouteType::AnthropicTokenCount => {
					strng::literal!("/anthropic/v1/messages/count_tokens")
				},
				super::RouteType::Models => strng::literal!("/v1/models"),
				_ => strng::literal!("/v1/chat/completions"),
			};
		}

		const MODEL_SEGMENT: &percent_encoding::AsciiSet =
			&percent_encoding::CONTROLS.add(b'/').add(b'%');
		let model = percent_encoding::utf8_percent_encode(model, MODEL_SEGMENT);
		match route_type {
			super::RouteType::AnthropicTokenCount => strng::format!("/model/{model}/count-tokens"),
			super::RouteType::Embeddings => strng::format!("/model/{model}/invoke"),
			// Rerank uses the agent-runtime Rerank action (model goes in the body as an ARN).
			super::RouteType::Rerank => strng::literal!("/rerank"),
			// Native Messages transport: Anthropic Messages chat rides the
			// Runtime InvokeModel APIs. `upstream_route_type` is derived from
			// the selected chat translation, so RouteType::Messages here
			// guarantees the rendered body is the native Anthropic Messages
			// payload (never a Converse body).
			super::RouteType::Messages if self.transport == BedrockTransport::NativeMessages => {
				if streaming {
					strng::format!("/model/{model}/invoke-with-response-stream")
				} else {
					strng::format!("/model/{model}/invoke")
				}
			},
			_ if streaming => strng::format!("/model/{model}/converse-stream"),
			_ => strng::format!("/model/{model}/converse"),
		}
	}

	/// Region whose bedrock-mantle endpoint serves this provider: the
	/// `AGENTGATEWAY_BEDROCK_MANTLE_REGION` override when set, else the
	/// provider's own Region. Used for the mantle host and its SigV4 signing
	/// region.
	/// The bedrock-mantle host for a Region. Count-tokens callers pass the
	/// Region from `resolve_count_tokens` (which may be an approved
	/// cross-Region destination); every other mantle route uses the
	/// provider's own Region.
	pub fn mantle_host(&self, region: &str) -> Strng {
		strng::format!("bedrock-mantle.{}.api.aws", region)
	}

	pub fn get_host(&self, route_type: super::RouteType, endpoint: BedrockEndpoint) -> Strng {
		// Rerank always uses the agent-runtime host, independent of endpoint choice.
		if matches!(route_type, super::RouteType::Rerank) {
			return strng::format!("bedrock-agent-runtime.{}.amazonaws.com", self.region);
		}
		match endpoint {
			BedrockEndpoint::Mantle => self.mantle_host(self.region.as_str()),
			BedrockEndpoint::Runtime => {
				strng::format!("bedrock-runtime.{}.amazonaws.com", self.region)
			},
		}
	}
}

/// The operator-approved cross-Region count_tokens destination for a base
/// model ID, if any (see `COUNT_TOKENS_APPROVED_REGIONS_ENV`). Returns the
/// region string as configured; card validity is checked by the caller.
fn approved_count_tokens_region(base_model: &str) -> Option<String> {
	let approvals = std::env::var(COUNT_TOKENS_APPROVED_REGIONS_ENV).ok()?;
	approvals
		.split(',')
		.filter_map(|entry| entry.split_once('='))
		.find(|(model, _)| model.trim() == base_model)
		.map(|(_, region)| region.trim().to_string())
		.filter(|region| !region.is_empty())
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::{ChatFormat, RouteType};

	fn provider(pref: BedrockEndpointPreference) -> Provider {
		Provider {
			model_override: None,
			region: strng::new("us-east-1"),
			guardrail_identifier: None,
			guardrail_version: None,
			endpoint_preference: pref,
			transport: BedrockTransport::Converse,
		}
	}

	fn native_provider(pref: BedrockEndpointPreference) -> Provider {
		let mut p = provider(pref);
		p.transport = BedrockTransport::NativeMessages;
		p
	}

	/// Scoped guard for `AGENTGATEWAY_BEDROCK_COUNT_TOKENS_REGIONS`: sets the
	/// approvals under a shared mutex (parallel tests mutate this env, so
	/// every env-touching test must hold the guard for its whole body) and
	/// removes them on drop.
	static COUNT_TOKENS_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

	struct CountTokensRegionsGuard(std::sync::MutexGuard<'static, ()>);

	impl CountTokensRegionsGuard {
		fn set(value: &str) -> Self {
			let lock = COUNT_TOKENS_ENV_LOCK
				.lock()
				.unwrap_or_else(|poisoned| poisoned.into_inner());
			unsafe { std::env::set_var(COUNT_TOKENS_APPROVED_REGIONS_ENV, value) };
			Self(lock)
		}
	}

	impl Drop for CountTokensRegionsGuard {
		fn drop(&mut self) {
			unsafe { std::env::remove_var(COUNT_TOKENS_APPROVED_REGIONS_ENV) };
		}
	}

	fn provider_in(region: &str) -> Provider {
		let mut p = provider(BedrockEndpointPreference::RuntimePreferred);
		p.region = strng::new(region);
		p
	}

	#[test]
	fn count_tokens_resolution_follows_the_model_cards() {
		// Card-verified capabilities (2026-11-03), provider in eu-west-1,
		// no cross-Region approvals configured (pinned explicitly so a
		// parallel test's approvals cannot leak in).
		let _no_approvals = CountTokensRegionsGuard::set("");
		let eu = provider_in("eu-west-1");
		// Runtime-capable models keep the Runtime CountTokens API.
		for model in [
			"eu.anthropic.claude-sonnet-4-6",
			"global.anthropic.claude-sonnet-4-6",
			"eu.anthropic.claude-haiku-4-5-20251001-v1:0",
			"anthropic.claude-3-5-sonnet-20241022-v2:0",
		] {
			assert_eq!(
				eu.resolve_count_tokens(model),
				CountTokensResolution::Runtime,
				"model={model}"
			);
		}
		// Mantle-capable in eu-west-1: in-Region, no approval needed.
		for model in [
			"global.anthropic.claude-opus-4-8",
			"global.anthropic.claude-sonnet-5",
			"global.anthropic.claude-opus-5",
		] {
			assert_eq!(
				eu.resolve_count_tokens(model),
				CountTokensResolution::Mantle {
					region: "eu-west-1".to_string()
				},
				"model={model}"
			);
		}
		// Opus 5.5 has no eu-west-1 mantle: cross-Region-only, so without an
		// explicit approval it is unsupported — no default Global-to-US path.
		assert_eq!(
			eu.resolve_count_tokens("global.anthropic.claude-opus-5-5"),
			CountTokensResolution::Unsupported
		);
		// Sonnet/Haiku 5.5 have NO commercial mantle Region at all.
		for model in [
			"global.anthropic.claude-sonnet-5-5",
			"global.anthropic.claude-haiku-5-5",
		] {
			assert_eq!(
				eu.resolve_count_tokens(model),
				CountTokensResolution::Unsupported,
				"model={model}"
			);
		}
		// In the model's own mantle Region the same model resolves in-Region.
		assert_eq!(
			provider_in("us-east-1").resolve_count_tokens("global.anthropic.claude-opus-5-5"),
			CountTokensResolution::Mantle {
				region: "us-east-1".to_string()
			}
		);
	}

	#[test]
	fn count_tokens_cross_region_requires_explicit_card_valid_approval() {
		let _guard = CountTokensRegionsGuard::set(
			"anthropic.claude-opus-5-5=us-east-1,anthropic.claude-sonnet-5-5=us-east-1",
		);
		let eu = provider_in("eu-west-1");
		// Approved AND card-listed: allowed.
		assert_eq!(
			eu.resolve_count_tokens("global.anthropic.claude-opus-5-5"),
			CountTokensResolution::Mantle {
				region: "us-east-1".to_string()
			}
		);
		// Approved but NOT card-listed for the model: still unsupported — a
		// GovCloud-only model cannot be willed into a commercial Region, and
		// one model's approval never opens another's destination.
		assert_eq!(
			eu.resolve_count_tokens("global.anthropic.claude-sonnet-5-5"),
			CountTokensResolution::Unsupported
		);
		drop(_guard);
		// An approval naming a Region the card does not list is ignored.
		let _bad = CountTokensRegionsGuard::set("anthropic.claude-opus-5-5=eu-west-1");
		assert_eq!(
			eu.resolve_count_tokens("global.anthropic.claude-opus-5-5"),
			CountTokensResolution::Unsupported
		);
	}

	#[test]
	fn count_tokens_runtime_only_pin_on_mantle_model_is_unsupported() {
		let _no_approvals = CountTokensRegionsGuard::set("");
		let mut pinned = provider_in("us-east-1");
		pinned.endpoint_preference = BedrockEndpointPreference::RuntimeOnly;
		assert_eq!(
			pinned.resolve_count_tokens("global.anthropic.claude-opus-5-5"),
			CountTokensResolution::Unsupported
		);
		// Runtime-capable models are unaffected by the pin (Runtime is their
		// surface anyway).
		assert_eq!(
			pinned.resolve_count_tokens("eu.anthropic.claude-sonnet-4-6"),
			CountTokensResolution::Runtime
		);
	}

	#[test]
	fn count_tokens_endpoint_resolution_matches_the_resolution() {
		let _no_approvals = CountTokensRegionsGuard::set("");
		let eu = provider_in("eu-west-1");
		// Mantle-capable in-Region models resolve the mantle endpoint…
		assert_eq!(
			eu.resolve_endpoint(
				RouteType::AnthropicTokenCount,
				Some("global.anthropic.claude-sonnet-5"),
				None
			),
			BedrockEndpoint::Mantle
		);
		// …Runtime-capable and unsupported models keep the Runtime path (the
		// gateway rejects the unsupported ones before dispatch).
		for model in [
			"eu.anthropic.claude-sonnet-4-6",
			"global.anthropic.claude-opus-5-5",
			"global.anthropic.claude-sonnet-5-5",
		] {
			assert_eq!(
				eu.resolve_endpoint(RouteType::AnthropicTokenCount, Some(model), None),
				BedrockEndpoint::Runtime,
				"model={model}"
			);
		}
		// Count-tokens rides Mantle even when the chat transport is native:
		// the transport selector forces Runtime for chat only.
		let mut native = provider_in("us-east-1");
		native.transport = BedrockTransport::NativeMessages;
		assert_eq!(
			native.resolve_endpoint(
				RouteType::AnthropicTokenCount,
				Some("global.anthropic.claude-opus-5-5"),
				None
			),
			BedrockEndpoint::Mantle
		);
		assert_eq!(
			native.resolve_endpoint(
				RouteType::Messages,
				Some("global.anthropic.claude-opus-5-5"),
				None
			),
			BedrockEndpoint::Runtime
		);
		// Non-Anthropic models keep the Runtime CountTokens API.
		assert_eq!(
			eu.resolve_endpoint(
				RouteType::AnthropicTokenCount,
				Some("openai.gpt-6-luna"),
				None
			),
			BedrockEndpoint::Runtime
		);
	}

	#[test]
	fn mantle_host_uses_the_resolved_region() {
		let p = provider_in("us-east-1");
		assert_eq!(
			p.mantle_host("eu-west-1"),
			"bedrock-mantle.eu-west-1.api.aws"
		);
		assert_eq!(
			p.get_host(RouteType::Messages, BedrockEndpoint::Mantle),
			"bedrock-mantle.us-east-1.api.aws"
		);
		assert_eq!(
			p.get_host(RouteType::Messages, BedrockEndpoint::Runtime),
			"bedrock-runtime.us-east-1.amazonaws.com"
		);
	}

	#[test]
	fn resolve_endpoint_runtime_preferred_uses_catalog_tags() {
		use crate::model_catalog::{TestCatalog, tags};
		let mut p = provider(BedrockEndpointPreference::RuntimePreferred);
		let cat = TestCatalog::new([("openai.gpt-oss-120b", &[tags::MANTLE][..])]);
		let catalog: crate::model_catalog::Catalog = Some(&cat);
		assert_eq!(
			p.resolve_endpoint(RouteType::Completions, Some("openai.gpt-oss-120b"), catalog),
			BedrockEndpoint::Mantle
		);
		assert_eq!(
			p.resolve_endpoint(
				RouteType::Completions,
				Some("anthropic.claude-3-5-sonnet-20241022-v2:0"),
				catalog
			),
			BedrockEndpoint::Runtime
		);
		p.guardrail_identifier = Some(strng::new("test-guardrail"));
		p.guardrail_version = Some(strng::new("1"));
		assert_eq!(
			p.resolve_endpoint(RouteType::Completions, Some("openai.gpt-oss-120b"), catalog),
			BedrockEndpoint::Runtime
		);
	}

	#[test]
	fn resolve_endpoint_mantle_preferred_is_the_mirror_of_runtime_preferred() {
		use crate::model_catalog::{TestCatalog, tags};
		let p = provider(BedrockEndpointPreference::MantlePreferred);
		let cat = TestCatalog::new([
			("only.mantle", &[tags::MANTLE][..]),
			("only.runtime", &[tags::RUNTIME][..]),
			("both.endpoints", &[tags::MANTLE, tags::RUNTIME][..]),
		]);
		let catalog: crate::model_catalog::Catalog = Some(&cat);
		assert_eq!(
			p.resolve_endpoint(RouteType::Completions, Some("only.runtime"), catalog),
			BedrockEndpoint::Runtime
		);
		assert_eq!(
			p.resolve_endpoint(RouteType::Completions, Some("only.mantle"), catalog),
			BedrockEndpoint::Mantle
		);
		assert_eq!(
			p.resolve_endpoint(RouteType::Completions, Some("both.endpoints"), catalog),
			BedrockEndpoint::Mantle
		);
		assert_eq!(
			p.resolve_endpoint(RouteType::Completions, Some("untagged.model"), catalog),
			BedrockEndpoint::Mantle
		);
		assert_eq!(
			p.resolve_endpoint(RouteType::Completions, Some("untagged.model"), None),
			BedrockEndpoint::Mantle
		);
		assert_eq!(
			p.resolve_endpoint(RouteType::Embeddings, Some("only.mantle"), catalog),
			BedrockEndpoint::Runtime
		);
	}

	#[test]
	fn mantle_endpoint_uses_correct_host_path_and_signing() {
		let p = provider(BedrockEndpointPreference::MantleOnly);
		let ep = BedrockEndpoint::Mantle;
		assert_eq!(
			p.get_host(RouteType::Messages, ep).as_str(),
			"bedrock-mantle.us-east-1.api.aws"
		);
		assert_eq!(p.signing_service_name(ep), Some("bedrock-mantle"));
		assert_eq!(
			p.get_path_for_route(RouteType::Messages, false, "m", ep)
				.as_str(),
			"/anthropic/v1/messages"
		);
		assert_eq!(
			p.get_path_for_route(RouteType::Responses, false, "m", ep)
				.as_str(),
			"/v1/responses"
		);
		assert_eq!(
			p.get_path_for_route(RouteType::Completions, false, "m", ep)
				.as_str(),
			"/v1/chat/completions"
		);
	}

	#[test]
	fn runtime_endpoint_uses_correct_host_path_and_signing() {
		let p = provider(BedrockEndpointPreference::RuntimeOnly);
		let ep = BedrockEndpoint::Runtime;
		assert_eq!(
			p.get_host(RouteType::Messages, ep).as_str(),
			"bedrock-runtime.us-east-1.amazonaws.com"
		);
		assert_eq!(p.signing_service_name(ep), None);
		assert_eq!(
			p.get_path_for_route(
				RouteType::Messages,
				false,
				"anthropic.claude-3-5-haiku-20241022-v1:0",
				ep
			)
			.as_str(),
			"/model/anthropic.claude-3-5-haiku-20241022-v1:0/converse"
		);
	}

	#[test]
	fn rerank_always_uses_agent_runtime_host() {
		for ep in [BedrockEndpoint::Mantle, BedrockEndpoint::Runtime] {
			assert_eq!(
				provider(BedrockEndpointPreference::RuntimePreferred)
					.get_host(RouteType::Rerank, ep)
					.as_str(),
				"bedrock-agent-runtime.us-east-1.amazonaws.com"
			);
		}
	}

	#[test]
	fn supported_chat_formats_without_catalog_falls_back_to_model_family() {
		let mantle = provider(BedrockEndpointPreference::MantleOnly);
		assert_eq!(
			mantle.supported_chat_formats("deepseek.v3.1", None),
			vec![ChatFormat::OpenAICompletions]
		);
		assert_eq!(
			mantle.supported_chat_formats("openai.gpt-oss-120b", None),
			vec![ChatFormat::OpenAICompletions, ChatFormat::OpenAIResponses]
		);
		assert_eq!(
			mantle.supported_chat_formats("xai.grok-4-6", None),
			vec![ChatFormat::OpenAICompletions, ChatFormat::OpenAIResponses]
		);
		assert_eq!(
			mantle.supported_chat_formats("anthropic.claude-3-sonnet", None),
			vec![ChatFormat::AnthropicMessages]
		);
		let runtime = provider(BedrockEndpointPreference::RuntimeOnly);
		assert_eq!(
			runtime.supported_chat_formats("any", None),
			vec![ChatFormat::BedrockConverse]
		);
	}

	#[test]
	fn supported_chat_formats_prefers_catalog_declared_formats() {
		use crate::model_catalog::{TestCatalog, tags};
		let mantle = provider(BedrockEndpointPreference::MantleOnly);
		let cat = TestCatalog::new([(
			"openai.gpt-oss-safeguard-120b",
			&[tags::MANTLE, tags::OPENAI_COMPLETIONS][..],
		)]);
		let catalog: crate::model_catalog::Catalog = Some(&cat);
		assert_eq!(
			mantle.supported_chat_formats("openai.gpt-oss-safeguard-120b", catalog),
			vec![ChatFormat::OpenAICompletions]
		);
		// A Mantle model with no format tags still falls back to the model-family guess
		let untagged = TestCatalog::new([("some.model", &[tags::MANTLE][..])]);
		let untagged: crate::model_catalog::Catalog = Some(&untagged);
		assert_eq!(
			mantle.supported_chat_formats("some.model", untagged),
			vec![ChatFormat::OpenAICompletions]
		);
	}

	#[test]
	fn supported_chat_formats_keeps_claude_on_messages_even_when_mistagged() {
		use crate::model_catalog::{TestCatalog, tags};
		let mantle = provider(BedrockEndpointPreference::MantleOnly);
		// Claude only speaks the Messages API on Mantle. Even if the catalog wrongly tags it with the
		// OpenAI formats, we must not advertise them — otherwise a Completions request would be sent to
		// Claude as OpenAI chat completions, which Mantle rejects.
		let cat = TestCatalog::new([(
			"anthropic.claude-sonnet-5",
			&[
				tags::MANTLE,
				tags::ANTHROPIC_MESSAGES,
				tags::OPENAI_COMPLETIONS,
				tags::OPENAI_RESPONSES,
			][..],
		)]);
		let catalog: crate::model_catalog::Catalog = Some(&cat);
		assert_eq!(
			mantle.supported_chat_formats("anthropic.claude-sonnet-5", catalog),
			vec![ChatFormat::AnthropicMessages]
		);
	}

	#[test]
	fn resolved_model_drives_endpoint_and_formats() {
		use crate::model_catalog::{TestCatalog, tags};
		let mut p = provider(BedrockEndpointPreference::RuntimePreferred);
		p.model_override = Some(strng::new("anthropic.claude-sonnet-4-5"));
		let cat = TestCatalog::new([(
			"openai.gpt-oss-120b",
			&[tags::MANTLE, tags::OPENAI_COMPLETIONS][..],
		)]);
		let catalog: crate::model_catalog::Catalog = Some(&cat);
		assert_eq!(
			p.resolve_endpoint(RouteType::Completions, Some("openai.gpt-oss-120b"), catalog),
			BedrockEndpoint::Mantle
		);
		assert_eq!(
			p.supported_chat_formats("openai.gpt-oss-120b", catalog),
			vec![ChatFormat::OpenAICompletions]
		);
	}

	#[test]
	fn resolve_endpoint_maps_every_route_type() {
		use RouteType::*;
		let mantle = provider(BedrockEndpointPreference::MantleOnly);
		let all = [
			Completions,
			Messages,
			Responses,
			AnthropicTokenCount,
			Models,
			Embeddings,
			Realtime,
			Rerank,
			GeminiCountTokens,
			GenerateContent,
			Detect,
			Passthrough,
		];
		for rt in all {
			let expected = match rt {
				Completions | Messages | Responses | AnthropicTokenCount | Models => {
					BedrockEndpoint::Mantle
				},
				Embeddings | Realtime | Rerank | GeminiCountTokens | GenerateContent | Detect
				| Passthrough => BedrockEndpoint::Runtime,
			};
			assert_eq!(
				mantle.resolve_endpoint(rt, Some("m"), None),
				expected,
				"{rt:?} resolved to the wrong endpoint"
			);
		}
	}

	#[test]
	fn non_chat_routes_stay_on_runtime_even_for_mantle_only() {
		// These routes exist only on Runtime, so MantleOnly must not force them to Mantle.
		let p = provider(BedrockEndpointPreference::MantleOnly);
		for rt in [
			RouteType::Embeddings,
			RouteType::Rerank,
			RouteType::Realtime,
			// Gemini-native routes are not served by Bedrock (Mantle has no generateContent /
			// count-tokens API), so they must not be forced onto Mantle either.
			RouteType::GenerateContent,
			RouteType::GeminiCountTokens,
		] {
			assert_eq!(
				p.resolve_endpoint(rt, Some("m"), None),
				BedrockEndpoint::Runtime,
				"{rt:?} must stay on Runtime under MantleOnly"
			);
		}
		// Host + path for embeddings must be the Runtime invoke path, not a Mantle path.
		let ep = p.resolve_endpoint(RouteType::Embeddings, Some("m"), None);
		assert_eq!(
			p.get_host(RouteType::Embeddings, ep).as_str(),
			"bedrock-runtime.us-east-1.amazonaws.com"
		);
		assert_eq!(
			p.get_path_for_route(RouteType::Embeddings, false, "m", ep)
				.as_str(),
			"/model/m/invoke"
		);
	}

	#[test]
	fn transport_defaults_to_converse_and_serializes_stable_names() {
		// Default must remain Converse: an unmodified backend config keeps the
		// proven production path (fail-closed rollout contract).
		let p = Provider {
			model_override: None,
			region: strng::new("eu-west-1"),
			guardrail_identifier: None,
			guardrail_version: None,
			endpoint_preference: BedrockEndpointPreference::RuntimePreferred,
			transport: BedrockTransport::default(),
		};
		assert_eq!(p.transport, BedrockTransport::Converse);
	}

	#[test]
	fn native_transport_forces_runtime_and_serves_messages_format() {
		// NativeMessages is Runtime-only: even MantlePreferred/MantleOnly
		// must resolve chat to Runtime (fail-closed, documented surface).
		for pref in [
			BedrockEndpointPreference::RuntimePreferred,
			BedrockEndpointPreference::MantlePreferred,
			BedrockEndpointPreference::MantleOnly,
		] {
			let p = native_provider(pref);
			assert_eq!(
				p.resolve_endpoint(
					RouteType::Messages,
					Some("global.anthropic.claude-opus-5-5"),
					None
				),
				BedrockEndpoint::Runtime,
				"native transport must stay on Runtime under {pref:?}"
			);
		}
		let p = native_provider(BedrockEndpointPreference::RuntimePreferred);
		assert_eq!(
			p.supported_chat_formats("global.anthropic.claude-opus-5-5", None),
			vec![ChatFormat::AnthropicMessages, ChatFormat::BedrockConverse]
		);
	}

	#[test]
	fn native_transport_uses_invoke_paths_for_messages_only() {
		let p = native_provider(BedrockEndpointPreference::RuntimePreferred);
		let ep = BedrockEndpoint::Runtime;
		assert_eq!(
			p.get_path_for_route(
				RouteType::Messages,
				false,
				"global.anthropic.claude-opus-5-5",
				ep
			)
			.as_str(),
			"/model/global.anthropic.claude-opus-5-5/invoke"
		);
		assert_eq!(
			p.get_path_for_route(
				RouteType::Messages,
				true,
				"global.anthropic.claude-opus-5-5",
				ep
			)
			.as_str(),
			"/model/global.anthropic.claude-opus-5-5/invoke-with-response-stream"
		);
		// Non-Messages routes keep the historical Converse paths: Responses
		// input falls back to Converse translation, and embeddings/token-count
		// keep their dedicated APIs.
		assert_eq!(
			p.get_path_for_route(RouteType::Responses, true, "m", ep)
				.as_str(),
			"/model/m/converse-stream"
		);
		assert_eq!(
			p.get_path_for_route(RouteType::Completions, false, "m", ep)
				.as_str(),
			"/model/m/converse"
		);
		assert_eq!(
			p.get_path_for_route(RouteType::Embeddings, false, "m", ep)
				.as_str(),
			"/model/m/invoke"
		);
		assert_eq!(
			p.get_path_for_route(RouteType::AnthropicTokenCount, false, "m", ep)
				.as_str(),
			"/model/m/count-tokens"
		);
	}

	#[test]
	fn converse_transport_keeps_historical_paths_and_formats() {
		// Regression guard: the Converse default must be byte-identical to the
		// pre-native-transport behavior.
		let p = provider(BedrockEndpointPreference::RuntimePreferred);
		let ep = BedrockEndpoint::Runtime;
		assert_eq!(
			p.get_path_for_route(RouteType::Messages, false, "m", ep)
				.as_str(),
			"/model/m/converse"
		);
		assert_eq!(
			p.get_path_for_route(RouteType::Messages, true, "m", ep)
				.as_str(),
			"/model/m/converse-stream"
		);
		assert_eq!(
			p.supported_chat_formats("any", None),
			vec![ChatFormat::BedrockConverse]
		);
		// Mantle resolution is untouched by the transport field when Converse.
		let mantle = provider(BedrockEndpointPreference::MantleOnly);
		assert_eq!(
			mantle.resolve_endpoint(RouteType::Messages, Some("m"), None),
			BedrockEndpoint::Mantle
		);
	}

	#[test]
	fn native_transport_serializes_from_config_value() {
		// serde round-trip of the backend config value. The schema_enum!
		// convention camel-cases variant names ("nativeMessages"), matching
		// how BedrockEndpointPreference serializes; the K8s CRD surface
		// exposes the PascalCase "NativeMessages" via the controller types
		// and the telemetry attribute uses the kebab "native-messages".
		let native: Provider =
			serde_json::from_str(r#"{"region":"eu-west-1","transport":"nativeMessages"}"#)
				.expect("nativeMessages config must deserialize");
		assert_eq!(native.transport, BedrockTransport::NativeMessages);
		let implicit: Provider =
			serde_json::from_str(r#"{"region":"eu-west-1"}"#).expect("missing transport must default");
		assert_eq!(implicit.transport, BedrockTransport::Converse);
		// Wire-name assertions (telemetry attribute contract).
		assert_eq!(BedrockTransport::Converse.as_str(), "converse");
		assert_eq!(BedrockTransport::NativeMessages.as_str(), "native-messages");
	}

	#[test]
	fn anthropic_count_tokens_follows_the_endpoint() {
		// Runtime uses the Bedrock CountTokens API; Mantle uses Anthropic's native count_tokens.
		let runtime = provider(BedrockEndpointPreference::RuntimeOnly);
		let runtime_ep = runtime.resolve_endpoint(RouteType::AnthropicTokenCount, Some("m"), None);
		assert_eq!(runtime_ep, BedrockEndpoint::Runtime);
		assert_eq!(
			runtime
				.get_path_for_route(RouteType::AnthropicTokenCount, false, "m", runtime_ep)
				.as_str(),
			"/model/m/count-tokens"
		);

		let mantle = provider(BedrockEndpointPreference::MantleOnly);
		let mantle_ep = mantle.resolve_endpoint(RouteType::AnthropicTokenCount, Some("m"), None);
		assert_eq!(mantle_ep, BedrockEndpoint::Mantle);
		assert_eq!(
			mantle
				.get_host(RouteType::AnthropicTokenCount, mantle_ep)
				.as_str(),
			"bedrock-mantle.us-east-1.api.aws"
		);
		assert_eq!(
			mantle.signing_service_name(mantle_ep),
			Some("bedrock-mantle")
		);
		assert_eq!(
			mantle
				.get_path_for_route(RouteType::AnthropicTokenCount, false, "m", mantle_ep)
				.as_str(),
			"/anthropic/v1/messages/count_tokens"
		);
	}
}
