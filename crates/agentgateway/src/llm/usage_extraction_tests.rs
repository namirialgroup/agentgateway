//! Buffered cross-format usage extraction regression tests.
//!
//! The buffered path logs usage from the TRANSLATED client-format response.
//! Those translations materialize protocol-mandated usage placeholders
//! (Anthropic `usage.input_tokens` is non-optional; a missing OpenAI usage
//! becomes `unwrap_or(0)`), which fabricate complete zero-token provider
//! attempts. The proxy therefore overlays usage parsed from the
//! upstream-native wire format: real values override, a missing upstream
//! usage clears the placeholder back to unknown (never zero).

use bytes::Bytes;

use super::ChatFormat;
use crate::llm::AIProvider;

fn completions_body(usage_json: &str) -> Bytes {
	Bytes::from(format!(
		r#"{{"id":"cmpl-1","object":"chat.completion","created":1,"model":"accounts/fireworks/models/kimi-k3","choices":[{{"index":0,"message":{{"role":"assistant","content":"hi"}},"finish_reason":"stop"}}],{usage_json}}}"#
	))
}

/// A successful upstream response WITH usage: the overlay carries the real
/// token counts (prompt_tokens semantics — total prompt including cache —
/// consistent with the streaming path).
#[test]
fn native_upstream_usage_present() {
	let bytes = completions_body(
		r#""usage":{"prompt_tokens":1000,"completion_tokens":50,"total_tokens":1050,"prompt_tokens_details":{"cached_tokens":800}}"#,
	);
	let native = AIProvider::native_upstream_llm_response(ChatFormat::OpenAICompletions, &bytes)
		.expect("native parse");
	assert_eq!(native.input_tokens, Some(1000));
	assert_eq!(native.output_tokens, Some(50));
	assert_eq!(native.total_tokens, Some(1050));
	assert_eq!(native.cached_input_tokens, Some(800));
}

/// A successful upstream response WITHOUT usage: every dimension stays
/// unknown. This is the evidence that must CLEAR the translated response's
/// fabricated `usage: {input_tokens: 0, output_tokens: 0}`.
#[test]
fn native_upstream_usage_missing_is_unknown_never_zero() {
	let bytes = completions_body(r#""usage":null"#);
	let native = AIProvider::native_upstream_llm_response(ChatFormat::OpenAICompletions, &bytes)
		.expect("native parse");
	assert_eq!(native.input_tokens, None);
	assert_eq!(native.output_tokens, None);
	assert_eq!(native.total_tokens, None);
	assert_eq!(native.cached_input_tokens, None);
	assert_eq!(native.cache_creation_input_tokens, None);
	// No usage evidence: completeness must stay untracked — a successful
	// response without counters is "unknown", not "complete".
	assert_eq!(
		native.usage_complete, None,
		"no counts means no completeness claim"
	);
}

/// A garbage body must not break request processing: the overlay is
/// best-effort and falls back to the translated response evidence.
#[test]
fn native_upstream_parse_failure_is_best_effort() {
	let bytes = Bytes::from_static(b"<not json>");
	let native = AIProvider::native_upstream_llm_response(ChatFormat::OpenAICompletions, &bytes);
	assert!(native.is_none());
}

/// The overlay moves ONLY usage dimensions; provider model (and content,
/// held by the translated response) stay untouched.
#[test]
fn overlay_moves_usage_only() {
	let mut translated = super::LLMResponse {
		input_tokens: Some(0),
		output_tokens: Some(0),
		..Default::default()
	};
	translated.provider_model = Some("client-visible-model".into());
	let native = super::LLMResponse {
		input_tokens: Some(91),
		output_tokens: Some(16),
		total_tokens: Some(107),
		usage_complete: Some(true),
		..Default::default()
	};
	AIProvider::overlay_upstream_usage(&mut translated, native);
	assert_eq!(translated.input_tokens, Some(91));
	assert_eq!(translated.output_tokens, Some(16));
	assert_eq!(translated.total_tokens, Some(107));
	assert_eq!(
		translated.provider_model.as_deref(),
		Some("client-visible-model")
	);
	assert_eq!(
		translated.usage_complete,
		Some(true),
		"completeness follows the native evidence, not the translated shape"
	);
}

/// Cross-format translation where the upstream reported NO usage: the
/// overlay clears the placeholder counts AND drops the translated
/// response's own `usage_complete` — no native evidence, no claim.
#[test]
fn overlay_without_native_usage_drops_completeness() {
	let mut translated = super::LLMResponse {
		input_tokens: Some(0),
		output_tokens: Some(0),
		usage_complete: Some(true),
		..Default::default()
	};
	let native = super::LLMResponse::default();
	AIProvider::overlay_upstream_usage(&mut translated, native);
	assert_eq!(translated.input_tokens, None);
	assert_eq!(translated.output_tokens, None);
	assert_eq!(translated.usage_complete, None);
}

/// Only true cross-format pairs need the overlay; native passthrough pairs
/// already parse the upstream wire format as the client format.
#[test]
fn crosses_formats_matrix() {
	use super::InputFormat as IF;
	assert!(AIProvider::crosses_chat_formats(
		IF::Messages,
		ChatFormat::OpenAICompletions
	));
	assert!(AIProvider::crosses_chat_formats(
		IF::Completions,
		ChatFormat::AnthropicMessages
	));
	assert!(!AIProvider::crosses_chat_formats(
		IF::Messages,
		ChatFormat::AnthropicMessages
	));
	assert!(!AIProvider::crosses_chat_formats(
		IF::Completions,
		ChatFormat::OpenAICompletions
	));
	assert!(!AIProvider::crosses_chat_formats(
		IF::Responses,
		ChatFormat::OpenAIResponses
	));
	assert!(!AIProvider::crosses_chat_formats(
		IF::Gemini,
		ChatFormat::VertexGemini
	));
}

/// Anthropic `input_tokens` EXCLUDES cached tokens: a fully cached request
/// legitimately reports `input_tokens: 0` with positive cache dimensions.
/// The native extraction must keep every dimension — cache-only usage is
/// real billing evidence, not a placeholder.
#[test]
fn native_upstream_anthropic_cache_only_usage_is_evidence() {
	let bytes = Bytes::from(
		r#"{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-4-5","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":5,"cache_creation_input_tokens":512,"cache_read_input_tokens":2048}}"#,
	);
	let native = AIProvider::native_upstream_llm_response(ChatFormat::AnthropicMessages, &bytes)
		.expect("native parse");
	assert_eq!(
		native.input_tokens,
		Some(0),
		"fully cached prompt bills 0 uncached input tokens"
	);
	assert_eq!(native.output_tokens, Some(5));
	assert_eq!(native.total_tokens, Some(5));
	assert_eq!(native.cached_input_tokens, Some(2048));
	assert_eq!(native.cache_creation_input_tokens, Some(512));
	// Buffered bodies are complete by construction: usage from them is final.
	assert_eq!(native.usage_complete, Some(true));
}

/// The native recovery exists only to overlay usage dimensions; it must not
/// extract response content even when the body carries choices/text.
#[test]
fn native_upstream_extraction_is_usage_only() {
	let bytes =
		completions_body(r#""usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}"#);
	let native = AIProvider::native_upstream_llm_response(ChatFormat::OpenAICompletions, &bytes)
		.expect("native parse");
	assert_eq!(native.input_tokens, Some(10));
	assert!(
		native.completion.is_none(),
		"usage-only conversion must not extract completion text"
	);
	assert!(
		native.output_messages.is_none(),
		"usage-only conversion must not extract tool-call output"
	);
}

/// Anthropic-native responses carry the per-TTL cache-write split in
/// `usage.cache_creation.ephemeral_5m/1h_input_tokens`. The extraction must
/// surface it for accounting (exact 5m/1h metering) while the client-facing
/// wire stays untouched.
#[test]
fn native_upstream_anthropic_cache_ttl_split_is_extracted() {
	let bytes = Bytes::from(
		r#"{"id":"msg_2","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":100,"output_tokens":5,"cache_creation_input_tokens":300,"cache_creation":{"ephemeral_5m_input_tokens":200,"ephemeral_1h_input_tokens":100},"cache_read_input_tokens":64}}"#,
	);
	let native = AIProvider::native_upstream_llm_response(ChatFormat::AnthropicMessages, &bytes)
		.expect("native parse");
	assert_eq!(native.cache_creation_input_tokens, Some(300));
	assert_eq!(native.cache_creation_5m_input_tokens, Some(200));
	assert_eq!(native.cache_creation_1h_input_tokens, Some(100));

	// Aggregate-only (older provider response shape): split stays unknown —
	// never inferred from the aggregate, never zero.
	let aggregate_only = Bytes::from(
		r#"{"id":"msg_3","type":"message","role":"assistant","model":"claude-sonnet-5-5","content":[{"type":"text","text":"hi"}],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":100,"output_tokens":5,"cache_creation_input_tokens":300}}"#,
	);
	let native =
		AIProvider::native_upstream_llm_response(ChatFormat::AnthropicMessages, &aggregate_only)
			.expect("native parse");
	assert_eq!(native.cache_creation_input_tokens, Some(300));
	assert_eq!(native.cache_creation_5m_input_tokens, None);
	assert_eq!(native.cache_creation_1h_input_tokens, None);
}
