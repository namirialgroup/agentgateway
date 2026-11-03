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

/// Environment override for the bedrock-mantle endpoint Region (host AND SigV4
/// signing region). bedrock-mantle serves Claude models In-Region in a small
/// set of Regions that differs per model (model cards; e.g. Opus 5.5:
/// us-east-1 / ap-southeast-4, Sonnet 5.5: none commercial at card time), so a
/// gateway running elsewhere (our eu-west-1) must call a supported Region.
/// Read per call (not cached) so tests and rolling config stay independent.
const MANTLE_REGION_ENV: &str = "AGENTGATEWAY_BEDROCK_MANTLE_REGION";

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
			// Claude models that launch CRIS-only on bedrock-runtime (geo-prefixed
			// inference profiles such as global.anthropic.claude-*-5-5) have no
			// Region-specific CountTokens surface there — the Runtime CountTokens
			// API denies them (AWS model cards). bedrock-mantle serves Anthropic's
			// native count_tokens for every Claude model, so Anthropic count-tokens
			// routes there regardless of the CHAT transport (NativeMessages forces
			// Runtime for chat only; count-tokens is transport-independent). An
			// explicit RuntimeOnly pin stays honored.
			RT::AnthropicTokenCount
				if model_id.is_some_and(|m| self.is_anthropic_model(m))
					&& !matches!(
						self.endpoint_preference,
						BedrockEndpointPreference::RuntimeOnly
					) =>
			{
				BedrockEndpoint::Mantle
			},
			// Chat (and Anthropic count-tokens under an explicit RuntimeOnly pin)
			// follows the model's endpoint: Runtime's Converse / CountTokens APIs,
			// or Mantle's native OpenAI/Anthropic APIs.
			RT::Completions | RT::Messages | RT::Responses | RT::AnthropicTokenCount => {
				self.chat_endpoint(model_id, catalog)
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
	pub fn mantle_region(&self) -> String {
		std::env::var(MANTLE_REGION_ENV)
			.ok()
			.filter(|r| !r.is_empty())
			.unwrap_or_else(|| self.region.as_str().to_owned())
	}

	pub fn get_host(&self, route_type: super::RouteType, endpoint: BedrockEndpoint) -> Strng {
		// Rerank always uses the agent-runtime host, independent of endpoint choice.
		if matches!(route_type, super::RouteType::Rerank) {
			return strng::format!("bedrock-agent-runtime.{}.amazonaws.com", self.region);
		}
		match endpoint {
			BedrockEndpoint::Mantle => {
				strng::format!("bedrock-mantle.{}.api.aws", self.mantle_region())
			},
			BedrockEndpoint::Runtime => {
				strng::format!("bedrock-runtime.{}.amazonaws.com", self.region)
			},
		}
	}
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

	/// Scoped test guard for `AGENTGATEWAY_BEDROCK_MANTLE_REGION`: sets the
	/// override and removes it on drop, so parallel tests observe the default
	/// again as soon as the guarded test ends.
	struct MantleRegionOverrideGuard;

	impl MantleRegionOverrideGuard {
		fn set(region: &str) -> Self {
			// Same single-test scoped pattern as the tool-search env guard:
			// the override lives only for the duration of this test.
			unsafe { std::env::set_var(MANTLE_REGION_ENV, region) };
			Self
		}
	}

	impl Drop for MantleRegionOverrideGuard {
		fn drop(&mut self) {
			unsafe { std::env::remove_var(MANTLE_REGION_ENV) };
		}
	}

	#[test]
	fn resolve_endpoint_explicit_preferences_ignore_model_table() {
		let mantle = provider(BedrockEndpointPreference::MantleOnly);
		let runtime = provider(BedrockEndpointPreference::RuntimeOnly);
		assert_eq!(
			mantle.resolve_endpoint(RouteType::Messages, Some("any-model"), None),
			BedrockEndpoint::Mantle
		);
		assert_eq!(
			runtime.resolve_endpoint(RouteType::Messages, Some("any-model"), None),
			BedrockEndpoint::Runtime
		);
	}

	#[test]
	fn resolve_endpoint_runtime_preferred_routes_non_chat_by_route_type() {
		// These routes resolve by route type alone (no catalog lookup), so this is deterministic.
		let p = provider(BedrockEndpointPreference::RuntimePreferred);
		assert_eq!(
			p.resolve_endpoint(RouteType::Embeddings, None, None),
			BedrockEndpoint::Runtime
		);
		assert_eq!(
			p.resolve_endpoint(RouteType::Rerank, None, None),
			BedrockEndpoint::Runtime
		);
		assert_eq!(
			p.resolve_endpoint(RouteType::Models, None, None),
			BedrockEndpoint::Mantle
		);
	}

	#[test]
	fn resolve_endpoint_routes_anthropic_count_tokens_to_mantle() {
		// CRIS-only Claude models (geo inference profiles) have no CountTokens
		// surface on bedrock-runtime; the Anthropic-native count_tokens API on
		// bedrock-mantle is the documented surface for them.
		let p = provider(BedrockEndpointPreference::RuntimePreferred);
		for model in [
			"global.anthropic.claude-opus-5-5",
			"global.anthropic.claude-sonnet-5-5",
			"eu.anthropic.claude-sonnet-4-6",
			"us.anthropic.claude-haiku-4-5-20251001-v1:0",
			"anthropic.claude-3-5-sonnet-20241022-v2:0",
		] {
			assert_eq!(
				p.resolve_endpoint(RouteType::AnthropicTokenCount, Some(model), None),
				BedrockEndpoint::Mantle,
				"model={model}"
			);
		}
		// Count-tokens rides Mantle even when the chat transport is native:
		// the transport selector forces Runtime for chat only.
		let native = native_provider(BedrockEndpointPreference::RuntimePreferred);
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
		// Chat and non-Anthropic count-tokens keep their current resolution.
		assert_eq!(
			p.resolve_endpoint(
				RouteType::Messages,
				Some("global.anthropic.claude-opus-5-5"),
				None
			),
			BedrockEndpoint::Runtime
		);
		assert_eq!(
			p.resolve_endpoint(
				RouteType::AnthropicTokenCount,
				Some("openai.gpt-6-luna"),
				None
			),
			BedrockEndpoint::Runtime
		);
	}

	#[test]
	fn mantle_host_and_signing_region_follow_override() {
		let p = provider(BedrockEndpointPreference::RuntimePreferred);
		// Default: the mantle host lives in the provider's own region…
		assert_eq!(
			p.get_host(RouteType::Messages, BedrockEndpoint::Mantle),
			"bedrock-mantle.us-east-1.api.aws"
		);
		// …unless the mantle region override points at a region that
		// actually serves the model In-Region (per-model card). The
		// override drives the host AND the SigV4 signing region (see
		// setup_request); the Runtime host never follows it.
		let _guard = MantleRegionOverrideGuard::set("eu-west-1");
		assert_eq!(
			p.get_host(RouteType::Messages, BedrockEndpoint::Mantle),
			"bedrock-mantle.eu-west-1.api.aws"
		);
		assert_eq!(
			p.get_host(RouteType::Messages, BedrockEndpoint::Runtime),
			"bedrock-runtime.us-east-1.amazonaws.com"
		);
	}

	#[test]
	fn resolve_endpoint_runtime_only_pin_keeps_runtime_count_tokens() {
		let p = provider(BedrockEndpointPreference::RuntimeOnly);
		assert_eq!(
			p.resolve_endpoint(
				RouteType::AnthropicTokenCount,
				Some("global.anthropic.claude-opus-5-5"),
				None
			),
			BedrockEndpoint::Runtime
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
