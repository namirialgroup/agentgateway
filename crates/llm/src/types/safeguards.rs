//! Claude Code "safeguards" (server-side classifier review) wire protocol for
//! the Anthropic Messages API: request detection and stripping, result shapes,
//! bounded review context, and completed tool-use collection for buffered and
//! streamed responses.
//!
//! Pure data and JSON helpers — no I/O and no policy. The policy engine and the
//! classifier adapter live in the gateway crate; this module guarantees protocol
//! conformance (field names, enum values, placement of `safeguard_results`) and
//! the "never classify an incomplete tool input" rule.
//!
//! Wire contract (Claude Code >= 2.1.278, beta `dangerous-tool-use-2026-09-03`):
//!
//! ```text
//! request : "safeguards": [{"type":"dangerous_tool_use","classifier_context":{...}}]
//! response: "safeguard_results": [{"type":"dangerous_tool_use","status":
//!             {"type":"available","tool_uses":{"<tool_use_id>":
//!               {"type":"evaluated","outcome":"not_flagged"} |
//!               {"type":"evaluated","outcome":"flagged","explanation":"..."} |
//!               {"type":"unavailable","reason":"timeout|error|refused|input_too_long|truncated"}}}
//!             | {"type":"unsupported"}}]
//! stream  : the same array inside `delta` of the final `message_delta` event.
//! ```

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const DANGEROUS_TOOL_USE: &str = "dangerous_tool_use";
pub const DANGEROUS_TOOL_USE_BETA: &str = "dangerous-tool-use-2026-09-03";
pub const SAFEGUARDS_FIELD: &str = "safeguards";
pub const SAFEGUARD_RESULTS_FIELD: &str = "safeguard_results";

/// The classifier never sees the raw conversation: the review context is a
/// bounded excerpt. Budgets are characters of rendered JSON lines.
pub const TRANSCRIPT_CHAR_BUDGET: usize = 24_000;
pub const TRANSCRIPT_LINE_CHAR_LIMIT: usize = 2_000;
pub const TOOL_RESULT_CHAR_LIMIT: usize = 500;
const TRUNCATION_MARKER: &str = " [truncated]";
const OMISSION_MARKER: &str = r#"{"meta":"earlier turns omitted"}"#;

/// `classifier_context` keys forwarded to the classifier. Anything else the
/// client sends (or will send in later protocol versions) is dropped.
const CONTEXT_KEYS: [&str; 7] = [
	"permission_mode",
	"platform",
	"live_cwd",
	"home_dir",
	"trusted_directories",
	"rules",
	"auto_mode",
];
const GIT_STATE_KEYS: [&str; 5] = ["cwd", "root", "branch", "default_branch", "visibility"];

// ---------------------------------------------------------------------------
// Request side
// ---------------------------------------------------------------------------

/// The `dangerous_tool_use` entry of a request's `safeguards` array, reduced to
/// the sanitized classifier context (an object, or `Null` when absent).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SafeguardsRequest {
	pub classifier_context: Value,
}

/// Find the `dangerous_tool_use` safeguards entry in a request's unknown
/// fields (`rest`). Malformed entries are ignored, never rejected: a request
/// that does not conform simply carries no classifier review.
pub fn requested_dangerous_tool_use(rest: &Value) -> Option<SafeguardsRequest> {
	let entries = rest.get(SAFEGUARDS_FIELD)?.as_array()?;
	entries
		.iter()
		.filter_map(Value::as_object)
		.find(|entry| entry.get("type").and_then(Value::as_str) == Some(DANGEROUS_TOOL_USE))
		.map(|entry| SafeguardsRequest {
			classifier_context: entry
				.get("classifier_context")
				.map(sanitized_classifier_context)
				.unwrap_or(Value::Null),
		})
}

/// Remove the `safeguards` field before the request reaches an upstream that
/// does not implement the protocol. Returns whether anything was removed.
pub fn remove_safeguards(rest: &mut Value) -> bool {
	match rest.as_object_mut() {
		Some(map) => map.remove(SAFEGUARDS_FIELD).is_some(),
		None => false,
	}
}

/// Whether any `anthropic-beta` header value names the safeguards beta.
pub fn has_dangerous_tool_use_beta(headers: &::http::HeaderMap) -> bool {
	headers
		.get_all("anthropic-beta")
		.iter()
		.filter_map(|value| value.to_str().ok())
		.flat_map(|value| value.split(','))
		.any(|beta| beta.trim() == DANGEROUS_TOOL_USE_BETA)
}

/// Drop the safeguards beta from the `anthropic-beta` header values, keeping
/// every other beta in place. Returns whether anything was removed.
pub fn strip_dangerous_tool_use_beta(headers: &mut ::http::HeaderMap) -> bool {
	if !has_dangerous_tool_use_beta(headers) {
		return false;
	}
	let values: Vec<::http::HeaderValue> =
		headers.get_all("anthropic-beta").iter().cloned().collect();
	headers.remove("anthropic-beta");
	for value in values {
		let Ok(text) = value.to_str() else {
			headers.append("anthropic-beta", value);
			continue;
		};
		let kept: Vec<&str> = text
			.split(',')
			.map(str::trim)
			.filter(|beta| !beta.is_empty() && *beta != DANGEROUS_TOOL_USE_BETA)
			.collect();
		if kept.is_empty() {
			continue;
		}
		if let Ok(joined) = ::http::HeaderValue::from_str(&kept.join(",")) {
			headers.append("anthropic-beta", joined);
		}
	}
	true
}

/// Reduce the client's `classifier_context` to the documented keys.
pub fn sanitized_classifier_context(raw: &Value) -> Value {
	let Some(context) = raw.as_object() else {
		return Value::Null;
	};
	let mut out = serde_json::Map::new();
	for key in CONTEXT_KEYS {
		if let Some(value) = context.get(key) {
			out.insert(key.to_string(), value.clone());
		}
	}
	if let Some(git) = context.get("git_state").and_then(Value::as_object) {
		let mut reduced = serde_json::Map::new();
		for key in GIT_STATE_KEYS {
			if let Some(value) = git.get(key) {
				reduced.insert(key.to_string(), value.clone());
			}
		}
		out.insert("git_state".to_string(), Value::Object(reduced));
	}
	Value::Object(out)
}

// ---------------------------------------------------------------------------
// Result side
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
	Flagged,
	NotFlagged,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UnavailableReason {
	Timeout,
	Error,
	Refused,
	InputTooLong,
	Truncated,
}

impl UnavailableReason {
	pub fn as_str(self) -> &'static str {
		match self {
			Self::Timeout => "timeout",
			Self::Error => "error",
			Self::Refused => "refused",
			Self::InputTooLong => "input_too_long",
			Self::Truncated => "truncated",
		}
	}
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ToolUseVerdict {
	Evaluated {
		outcome: Outcome,
		#[serde(default, skip_serializing_if = "Option::is_none")]
		explanation: Option<String>,
	},
	Unavailable {
		reason: UnavailableReason,
	},
}

impl ToolUseVerdict {
	pub fn not_flagged() -> Self {
		Self::Evaluated {
			outcome: Outcome::NotFlagged,
			explanation: None,
		}
	}

	pub fn flagged(explanation: impl Into<String>) -> Self {
		Self::Evaluated {
			outcome: Outcome::Flagged,
			explanation: Some(explanation.into()),
		}
	}

	pub fn unavailable(reason: UnavailableReason) -> Self {
		Self::Unavailable { reason }
	}

	/// Static label for telemetry (never carries content).
	pub fn label(&self) -> &'static str {
		match self {
			Self::Evaluated {
				outcome: Outcome::Flagged,
				..
			} => "flagged",
			Self::Evaluated {
				outcome: Outcome::NotFlagged,
				..
			} => "not_flagged",
			Self::Unavailable { .. } => "unavailable",
		}
	}
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SafeguardStatus {
	Available {
		tool_uses: BTreeMap<String, ToolUseVerdict>,
	},
	Unsupported,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SafeguardResult {
	#[serde(rename = "type")]
	pub kind: String,
	pub status: SafeguardStatus,
}

pub type SafeguardResults = Vec<SafeguardResult>;

impl SafeguardResult {
	pub fn available(tool_uses: BTreeMap<String, ToolUseVerdict>) -> Self {
		Self {
			kind: DANGEROUS_TOOL_USE.to_string(),
			status: SafeguardStatus::Available { tool_uses },
		}
	}

	pub fn unsupported() -> Self {
		Self {
			kind: DANGEROUS_TOOL_USE.to_string(),
			status: SafeguardStatus::Unsupported,
		}
	}
}

/// `available` results — including the empty map for a response without tool
/// calls, which is what tells the client that server review is active.
pub fn results_available(tool_uses: BTreeMap<String, ToolUseVerdict>) -> SafeguardResults {
	vec![SafeguardResult::available(tool_uses)]
}

/// The whole feature is not served for this request; the client falls back to
/// its own classifier for the session.
pub fn results_unsupported() -> SafeguardResults {
	vec![SafeguardResult::unsupported()]
}

/// Every pending tool use gets the same `unavailable` reason (classifier
/// failure that is a verdict, never an error on the agent's turn).
pub fn results_all_unavailable(
	pending: &[PendingToolUse],
	reason: UnavailableReason,
) -> SafeguardResults {
	results_available(
		pending
			.iter()
			.map(|use_| {
				(
					use_.id().to_string(),
					match use_ {
						PendingToolUse::Truncated { .. } => {
							ToolUseVerdict::unavailable(UnavailableReason::Truncated)
						},
						PendingToolUse::Complete { .. } => ToolUseVerdict::unavailable(reason),
					},
				)
			})
			.collect(),
	)
}

// ---------------------------------------------------------------------------
// Tool-use collection (buffered + streamed)
// ---------------------------------------------------------------------------

/// A tool use the model emitted in this response. `Truncated` inputs (cut by
/// `max_tokens`, or whose streamed JSON does not parse) are never sent to a
/// classifier; they are reported `unavailable/truncated`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PendingToolUse {
	Complete {
		id: String,
		name: String,
		input: Value,
	},
	Truncated {
		id: String,
	},
}

impl PendingToolUse {
	pub fn id(&self) -> &str {
		match self {
			Self::Complete { id, .. } | Self::Truncated { id } => id,
		}
	}
}

/// Collect tool uses from a buffered (non-streaming) Anthropic Messages response.
pub fn pending_tool_uses_from_response(response: &Value) -> Vec<PendingToolUse> {
	let Some(blocks) = response.get("content").and_then(Value::as_array) else {
		return Vec::new();
	};
	let hit_max_tokens = response.get("stop_reason").and_then(Value::as_str) == Some("max_tokens");
	let cut_off_index = if hit_max_tokens {
		blocks.len().checked_sub(1)
	} else {
		None
	};
	blocks
		.iter()
		.enumerate()
		.filter_map(|(index, block)| pending_from_block(block, Some(index) == cut_off_index))
		.collect()
}

fn pending_from_block(block: &Value, cut_off: bool) -> Option<PendingToolUse> {
	let object = block.as_object()?;
	if object.get("type").and_then(Value::as_str) != Some("tool_use") {
		return None;
	}
	let id = object.get("id").and_then(Value::as_str)?.to_string();
	if cut_off {
		return Some(PendingToolUse::Truncated { id });
	}
	Some(PendingToolUse::Complete {
		id,
		name: object
			.get("name")
			.and_then(Value::as_str)
			.unwrap_or_default()
			.to_string(),
		input: object.get("input").cloned().unwrap_or(Value::Null),
	})
}

/// Whether an upstream already answered (native pass-through): results are then
/// preserved and local classification must not run.
pub fn response_has_safeguard_results(response: &Value) -> bool {
	response
		.get(SAFEGUARD_RESULTS_FIELD)
		.is_some_and(|results| !results.is_null())
}

/// Attach results to a buffered response (top level).
pub fn stamp_safeguard_results(response: &mut Value, results: &SafeguardResults) -> bool {
	let Some(object) = response.as_object_mut() else {
		return false;
	};
	match serde_json::to_value(results) {
		Ok(value) => {
			object.insert(SAFEGUARD_RESULTS_FIELD.to_string(), value);
			true
		},
		Err(_) => false,
	}
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct StreamedToolUse {
	id: String,
	name: String,
	start_input: Value,
	partial_json: String,
}

/// Accumulates the `tool_use` content blocks of one streamed message so the
/// final `message_delta` can carry a verdict per completed tool use.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StreamedToolUses {
	by_index: BTreeMap<usize, StreamedToolUse>,
	last_block_index: Option<usize>,
}

impl StreamedToolUses {
	/// Feed every parsed SSE event; only `content_block_start` and
	/// `input_json_delta` deltas change the state.
	pub fn observe(&mut self, event: &Value) {
		match event.get("type").and_then(Value::as_str) {
			Some("content_block_start") => self.start(event),
			Some("content_block_delta") => self.delta(event),
			_ => {},
		}
	}

	fn start(&mut self, event: &Value) {
		let Some(index) = event.get("index").and_then(Value::as_u64) else {
			return;
		};
		let index = index as usize;
		self.last_block_index = Some(index);
		let Some(block) = event.get("content_block").and_then(Value::as_object) else {
			return;
		};
		if block.get("type").and_then(Value::as_str) != Some("tool_use") {
			return;
		}
		let Some(id) = block.get("id").and_then(Value::as_str) else {
			return;
		};
		self.by_index.insert(
			index,
			StreamedToolUse {
				id: id.to_string(),
				name: block
					.get("name")
					.and_then(Value::as_str)
					.unwrap_or_default()
					.to_string(),
				start_input: block.get("input").cloned().unwrap_or(Value::Null),
				partial_json: String::new(),
			},
		);
	}

	fn delta(&mut self, event: &Value) {
		let Some(index) = event.get("index").and_then(Value::as_u64) else {
			return;
		};
		let Some(current) = self.by_index.get_mut(&(index as usize)) else {
			return;
		};
		let Some(delta) = event.get("delta").and_then(Value::as_object) else {
			return;
		};
		if delta.get("type").and_then(Value::as_str) != Some("input_json_delta") {
			return;
		}
		if let Some(fragment) = delta.get("partial_json").and_then(Value::as_str) {
			current.partial_json.push_str(fragment);
		}
	}

	pub fn is_empty(&self) -> bool {
		self.by_index.is_empty()
	}

	pub fn len(&self) -> usize {
		self.by_index.len()
	}

	/// Resolve the collected blocks. The last block is `Truncated` when the
	/// message stopped on `max_tokens`; streamed input that is not valid JSON
	/// is `Truncated` as well.
	pub fn finish(&self, hit_max_tokens: bool) -> Vec<PendingToolUse> {
		let cut_off_index = if hit_max_tokens {
			self.last_block_index
		} else {
			None
		};
		self
			.by_index
			.iter()
			.map(|(index, use_)| {
				if Some(*index) == cut_off_index {
					return PendingToolUse::Truncated {
						id: use_.id.clone(),
					};
				}
				if use_.partial_json.is_empty() {
					return PendingToolUse::Complete {
						id: use_.id.clone(),
						name: use_.name.clone(),
						input: use_.start_input.clone(),
					};
				}
				match serde_json::from_str::<Value>(&use_.partial_json) {
					Ok(input) => PendingToolUse::Complete {
						id: use_.id.clone(),
						name: use_.name.clone(),
						input,
					},
					Err(_) => PendingToolUse::Truncated {
						id: use_.id.clone(),
					},
				}
			})
			.collect()
	}
}

/// What the stamping stage must do with a parsed SSE event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MessageDeltaCheck {
	/// Not a `message_delta`, or one without `stop_reason`: pass through.
	PassThrough,
	/// The upstream already stamped `safeguard_results`: pass through, and the
	/// local classifier must not run for this message.
	AlreadyAnswered,
	/// The final delta: hold it until results exist, then stamp.
	NeedsResults { hit_max_tokens: bool },
}

pub fn inspect_message_delta(event: &Value) -> MessageDeltaCheck {
	if event.get("type").and_then(Value::as_str) != Some("message_delta") {
		return MessageDeltaCheck::PassThrough;
	}
	let Some(delta) = event.get("delta").and_then(Value::as_object) else {
		return MessageDeltaCheck::PassThrough;
	};
	let Some(stop_reason) = delta.get("stop_reason").filter(|v| !v.is_null()) else {
		return MessageDeltaCheck::PassThrough;
	};
	if delta
		.get(SAFEGUARD_RESULTS_FIELD)
		.is_some_and(|results| !results.is_null())
	{
		return MessageDeltaCheck::AlreadyAnswered;
	}
	MessageDeltaCheck::NeedsResults {
		hit_max_tokens: stop_reason.as_str() == Some("max_tokens"),
	}
}

/// Attach results inside `delta` of a `message_delta` event.
pub fn stamp_message_delta(event: &mut Value, results: &SafeguardResults) -> bool {
	let Some(delta) = event.get_mut("delta").and_then(Value::as_object_mut) else {
		return false;
	};
	match serde_json::to_value(results) {
		Ok(value) => {
			delta.insert(SAFEGUARD_RESULTS_FIELD.to_string(), value);
			true
		},
		Err(_) => false,
	}
}

// ---------------------------------------------------------------------------
// Review plan (request-time context carried to the response side)
// ---------------------------------------------------------------------------

/// Everything the response side needs to classify this request's tool uses,
/// captured when the request is parsed. Lives for the request only and is
/// never serialized or logged: `Debug` prints sizes, not content.
#[derive(Clone, PartialEq, Eq)]
pub struct SafeguardsReviewPlan {
	pub request_model: String,
	/// Authenticated principal (opaque subject id) for correlation; never content.
	pub identity: Option<String>,
	pub classifier_context: Value,
	/// Bounded transcript, one JSON object per line (newest last).
	pub transcript: Vec<String>,
	pub transcript_truncated: bool,
}

impl std::fmt::Debug for SafeguardsReviewPlan {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("SafeguardsReviewPlan")
			.field("request_model", &self.request_model)
			.field("identity", &self.identity.as_ref().map(|_| "[set]"))
			.field(
				"classifier_context_keys",
				&context_key_count(&self.classifier_context),
			)
			.field("transcript_lines", &self.transcript.len())
			.field("transcript_truncated", &self.transcript_truncated)
			.finish()
	}
}

fn context_key_count(context: &Value) -> usize {
	context.as_object().map(|o| o.len()).unwrap_or(0)
}

impl SafeguardsReviewPlan {
	/// Build the plan from the request's `messages` (as JSON) and the sanitized
	/// safeguards entry. The system prompt and thinking blocks are deliberately
	/// excluded from the transcript.
	pub fn from_request(
		request_model: &str,
		messages: &Value,
		safeguards: &SafeguardsRequest,
	) -> Self {
		let (transcript, transcript_truncated) = render_transcript(messages);
		Self {
			request_model: request_model.to_string(),
			identity: None,
			classifier_context: safeguards.classifier_context.clone(),
			transcript,
			transcript_truncated,
		}
	}

	pub fn transcript_text(&self) -> String {
		self.transcript.join("\n")
	}

	/// Trusted directories from the classifier context (`live_cwd` is included
	/// as a trusted root when present), for deterministic path rules. Two wire
	/// shapes are accepted: a plain array of paths, and the object form the
	/// real Claude Code 2.1.304 client sends —
	/// `{"primary":{"path":…,"resolved":[…]},"additional":[…],"network":[…]}` —
	/// where the `primary`/`additional` entries are either paths or
	/// `{path, resolved}` objects.
	pub fn trusted_directories(&self) -> Vec<String> {
		let mut dirs: Vec<String> = Vec::new();
		match self.classifier_context.get("trusted_directories") {
			Some(Value::Array(items)) => {
				dirs.extend(items.iter().filter_map(Value::as_str).map(str::to_string));
			},
			Some(Value::Object(map)) => {
				for group in ["primary", "additional"] {
					match map.get(group) {
						Some(Value::Array(items)) => {
							for entry in items {
								collect_trusted_entry(entry, &mut dirs);
							}
						},
						Some(entry) => collect_trusted_entry(entry, &mut dirs),
						None => {},
					}
				}
			},
			_ => {},
		}
		if let Some(cwd) = self
			.classifier_context
			.get("live_cwd")
			.and_then(Value::as_str)
			&& !dirs.iter().any(|d| d == cwd)
		{
			dirs.push(cwd.to_string());
		}
		let mut seen = std::collections::HashSet::new();
		dirs.retain(|dir| seen.insert(dir.clone()));
		dirs
	}

	pub fn permission_mode(&self) -> Option<&str> {
		self
			.classifier_context
			.get("permission_mode")
			.and_then(Value::as_str)
	}
}

/// One `primary`/`additional` entry of the object-form `trusted_directories`:
/// either a path or a `{path, resolved}` object.
fn collect_trusted_entry(entry: &Value, out: &mut Vec<String>) {
	match entry {
		Value::String(path) => out.push(path.clone()),
		Value::Object(map) => {
			if let Some(path) = map.get("path").and_then(Value::as_str) {
				out.push(path.to_string());
			}
			if let Some(resolved) = map.get("resolved").and_then(Value::as_array) {
				out.extend(
					resolved
						.iter()
						.filter_map(Value::as_str)
						.map(str::to_string),
				);
			}
		},
		_ => {},
	}
}

/// Render the conversation to budgeted JSON lines. The opening message keeps
/// its newest lines within half the budget (Claude Code puts the user's request
/// after its reminders, so it is cut from the start); later turns keep the
/// newest lines that fit, with a marker where lines were dropped.
pub fn render_transcript(messages: &Value) -> (Vec<String>, bool) {
	let Some(messages) = messages.as_array() else {
		return (Vec::new(), false);
	};
	let opening: Vec<String> = messages.first().map(transcript_lines).unwrap_or_default();
	let later: Vec<String> = messages.iter().skip(1).flat_map(transcript_lines).collect();
	if rendered_size(&opening) + rendered_size(&later) <= TRANSCRIPT_CHAR_BUDGET {
		let mut all = opening;
		all.extend(later);
		return (all, false);
	}
	let kept_opening = newest_lines_within(&opening, TRANSCRIPT_CHAR_BUDGET / 2);
	let budget_for_later = TRANSCRIPT_CHAR_BUDGET
		.saturating_sub(rendered_size(&kept_opening))
		.saturating_sub(OMISSION_MARKER.len() + 1);
	let mut out = kept_opening;
	out.push(OMISSION_MARKER.to_string());
	out.extend(newest_lines_within(&later, budget_for_later));
	(out, true)
}

fn rendered_size(lines: &[String]) -> usize {
	lines.iter().map(|line| line.len() + 1).sum()
}

fn newest_lines_within(lines: &[String], budget: usize) -> Vec<String> {
	let mut size = 0usize;
	let mut kept = 0usize;
	for line in lines.iter().rev() {
		size += line.len() + 1;
		if size > budget {
			break;
		}
		kept += 1;
	}
	lines[lines.len() - kept..].to_vec()
}

fn transcript_lines(message: &Value) -> Vec<String> {
	let Some(role) = message.get("role").and_then(Value::as_str) else {
		return Vec::new();
	};
	if role != "user" && role != "assistant" {
		return Vec::new();
	}
	match message.get("content") {
		Some(Value::String(text)) => vec![prompt_json(&serde_json::json!({
			role: clipped(text, TRANSCRIPT_LINE_CHAR_LIMIT)
		}))],
		Some(Value::Array(blocks)) => blocks
			.iter()
			.filter_map(|block| block_line(role, block))
			.collect(),
		_ => Vec::new(),
	}
}

fn block_line(role: &str, block: &Value) -> Option<String> {
	let fields = block.as_object()?;
	match fields.get("type").and_then(Value::as_str)? {
		"text" => {
			let text = fields
				.get("text")
				.and_then(Value::as_str)
				.unwrap_or_default();
			Some(prompt_json(&serde_json::json!({
				role: clipped(text, TRANSCRIPT_LINE_CHAR_LIMIT)
			})))
		},
		"tool_use" => Some(prompt_json(&serde_json::json!({
			"assistant_tool_call": {
				"name": fields.get("name").cloned().unwrap_or(Value::Null),
				"input": clipped_input(fields.get("input").unwrap_or(&Value::Null)),
			}
		}))),
		"tool_result" => Some(prompt_json(&serde_json::json!({
			"tool_result": clipped(
				&content_text(fields.get("content").unwrap_or(&Value::Null)),
				TOOL_RESULT_CHAR_LIMIT
			)
		}))),
		"compaction" => {
			let summary = fields
				.get("content")
				.and_then(Value::as_str)
				.unwrap_or_default();
			Some(prompt_json(&serde_json::json!({
				"conversation_summary": clipped(summary, TRANSCRIPT_LINE_CHAR_LIMIT)
			})))
		},
		// thinking, redacted_thinking, images, documents, server tool blocks: not
		// part of what the user said or the agent did.
		_ => None,
	}
}

fn clipped_input(input: &Value) -> Value {
	let rendered = prompt_json(input);
	if rendered.len() <= TRANSCRIPT_LINE_CHAR_LIMIT {
		return input.clone();
	}
	Value::String(clipped(&rendered, TRANSCRIPT_LINE_CHAR_LIMIT))
}

/// Clip on a char boundary; the marker is appended only when clipping happened.
pub fn clipped(text: &str, limit: usize) -> String {
	if text.len() <= limit {
		return text.to_string();
	}
	let mut end = limit;
	while end > 0 && !text.is_char_boundary(end) {
		end -= 1;
	}
	format!("{}{}", &text[..end], TRUNCATION_MARKER)
}

fn content_text(content: &Value) -> String {
	match content {
		Value::String(text) => text.clone(),
		Value::Array(parts) => parts
			.iter()
			.filter_map(|part| part.get("text").and_then(Value::as_str))
			.collect::<Vec<_>>()
			.join(""),
		_ => String::new(),
	}
}

fn prompt_json(value: &Value) -> String {
	serde_json::to_string(value).unwrap_or_default()
}

#[cfg(test)]
#[path = "safeguards_tests.rs"]
mod tests;
