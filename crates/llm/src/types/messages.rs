use agent_core::prelude::Strng;
use agent_core::strng;
use itertools::Itertools;
use serde::{Deserialize, Serialize};

use crate::types::{
	ContentScope, NormalizedMessage, NormalizedMessagePart, OutputMessage, OutputMessagePart,
	RequestType, ResponseType, SimpleChatCompletionMessage, visit_json_at,
};
use crate::webhook::{Message, ResponseChoice};
use crate::{AIError, InputFormat, LLMRequest, LLMRequestParams, LLMResponse};

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct Request {
	pub messages: Vec<RequestMessage>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub model: Option<String>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub system: Option<TextBlock>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub top_p: Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub temperature: Option<f32>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub stream: Option<bool>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub max_tokens: Option<u64>,
	#[serde(flatten, default)]
	pub rest: serde_json::Value,
}

impl Request {
	/// Whether this request carries native Anthropic Tool Search (deferred tool
	/// loading) semantics: a `tool_search*` server tool offered in `tools`, or
	/// deferred-tool protocol blocks in the conversation — a `tool_reference`
	/// content block (an assistant turn referencing a deferred tool, or a
	/// `tool_result` embedding the schema reply), a `server_tool_use` block for
	/// the tool_search server tool, or a `tool_search_tool_result` block from a
	/// completed search round. These semantics cannot survive a chat format
	/// conversion (the Bedrock Converse/OpenAI/Gemini conversions drop the
	/// server tool and flatten `tool_reference` blocks into plain text), so
	/// callers must keep such requests on an Anthropic Messages-compatible
	/// upstream rather than silently degrading them. Detection runs on the
	/// passthrough shape, which preserves unknown blocks verbatim.
	pub fn uses_native_tool_search(&self) -> bool {
		tools_offer_native_tool_search(self.rest.get("tools"))
			|| self.messages.iter().any(|m| {
				m.content
					.as_ref()
					.is_some_and(content_references_deferred_tool)
			})
	}

	/// Downgrade this request's Tool Search (deferred tool loading) shapes to
	/// ordinary upfront-tool-loading inference:
	///
	/// * `tools`: drop every `tool_search*` server tool and strip
	///   `defer_loading` flags from the remaining (real) tool definitions,
	///   which stay fully declared — only their context-deferral hint is
	///   removed. A `tool_reference` that carries a full `input_schema` (some
	///   clients embed definitions in search results) is hoisted into `tools`
	///   so the eager model keeps access to that tool.
	/// * history: rewrite the deferred-tool protocol plumbing into an eager
	///   equivalent — remove `tool_reference` parts, remove the
	///   `server_tool_use`/`tool_search_tool_result` search-round blocks, and
	///   drop `tool_reference` parts nested inside `tool_result` content.
	///   These blocks are protocol plumbing between the client and a native
	///   Tool Search model: the tools themselves stay declared in `tools`
	///   (eager), ordinary `tool_use`/`tool_result` history is untouched, and
	///   no user content is affected. (2026-10-20 incident: Claude Desktop
	///   inserts `tool_reference` blocks on turns after tool use, so
	///   preserving them kept every non-Anthropic model failing with 400
	///   `unverified-provider` after an otherwise successful first turn.)
	/// * `tool_choice` pinned to a removed server tool — or any `tool_choice`
	///   once `tools` ends up empty — is dropped so the upstream never sees a
	///   choice referencing an undeclared tool.
	///
	/// A message whose content array becomes empty after the rewrite (e.g. a
	/// pure search-round assistant turn) is replaced with a fixed text marker
	/// so message count and user/assistant alternation survive intact.
	///
	/// After this call `uses_native_tool_search()` must return `false`; the
	/// caller re-checks and keeps the strict Anthropic-compatible-only guard
	/// for anything the rewrite could not translate (defense in depth).
	/// Callers must also drop the tool-search `anthropic-beta` header values
	/// from the forwarded request.
	pub fn downgrade_tool_search(&mut self) -> ToolSearchDowngrade {
		let mut report = ToolSearchDowngrade::default();

		if let Some(tools) = self.rest.get_mut("tools").and_then(|t| t.as_array_mut()) {
			for tool in tools.iter_mut() {
				if let Some(object) = tool.as_object_mut()
					&& object.remove("defer_loading").is_some()
				{
					report.stripped_defer_loading += 1;
				}
			}
			let server_tools = tools.len();
			tools.retain(|tool| {
				let tool_type = tool
					.get("type")
					.and_then(serde_json::Value::as_str)
					.unwrap_or_default();
				if is_builtin_server_tool(tool_type) {
					report.removed_builtin_server_tools += 1;
					false
				} else {
					!tool_type.starts_with("tool_search")
				}
			});
			report.removed_server_tools =
				server_tools - tools.len() - report.removed_builtin_server_tools;
		}

		let mut hoisted: Vec<serde_json::Value> = Vec::new();
		for message in self.messages.iter_mut() {
			let Some(ContentBlock::Array(parts)) = message.content.as_mut() else {
				continue;
			};
			sanitize_deferred_tool_parts(parts, &mut hoisted, &mut report);
			if parts.is_empty() {
				// Keep the message (count and alternation) with a neutral
				// marker instead of sending an empty content array upstream.
				*parts = vec![ContentPart::Text {
					r#type: "text".to_string(),
					text: "[tool search]".to_string(),
					rest: serde_json::Value::Null,
				}];
			}
		}

		if !hoisted.is_empty() {
			// Deduplicate against tools already declared AND against copies
			// hoisted earlier in this loop: the same deferred tool can embed
			// its definition in more than one history block, and a duplicate
			// tool name would make the upstream reject the whole request.
			//
			// A name collision with a DIFFERENT schema is never resolved
			// silently: identical definitions dedupe, but a divergent one is
			// resolved by origin — a declared tool in THIS request is the
			// client's current contract (it is what the client executes
			// against), while between two hoisted history references the
			// NEWEST reference wins (a mid-session tool upgrade). Either way
			// `hoisted_schema_conflicts` counts the divergence so operators
			// can see the downgrade had to choose between schemas.
			let tools = self
				.rest
				.as_object_mut()
				.map(|o| {
					o.entry("tools")
						.or_insert_with(|| serde_json::Value::Array(Vec::new()))
				})
				.and_then(|t| t.as_array_mut());
			if let Some(tools) = tools {
				#[derive(Clone, Copy, PartialEq)]
				enum Origin {
					Declared,
					Hoisted,
				}
				let mut known: Vec<(String, usize, Origin)> = tools
					.iter()
					.enumerate()
					.filter_map(|(index, tool)| {
						tool
							.get("name")
							.and_then(|n| n.as_str())
							.map(|n| (n.to_string(), index, Origin::Declared))
					})
					.collect();
				for definition in hoisted {
					let name = definition
						.get("name")
						.and_then(|n| n.as_str())
						.map(str::to_string);
					let Some(name) = name else {
						tools.push(definition);
						report.hoisted_schemas += 1;
						continue;
					};
					if let Some(&(_, index, origin)) =
						known.iter().find(|(known_name, _, _)| known_name == &name)
					{
						if tools[index] != definition {
							report.hoisted_schema_conflicts += 1;
							if origin == Origin::Hoisted {
								tools[index] = definition;
							}
						}
						continue;
					}
					known.push((name, tools.len(), Origin::Hoisted));
					tools.push(definition);
					report.hoisted_schemas += 1;
				}
			}
		}

		// A tool_choice pinned to a removed server tool would reference an
		// undeclared tool upstream; an empty tools array cannot back any
		// tool_choice at all.
		let remaining_names: Vec<String> = self
			.rest
			.get("tools")
			.and_then(|t| t.as_array())
			.map(|tools| {
				tools
					.iter()
					.filter_map(|t| t.get("name").and_then(|n| n.as_str()))
					.map(str::to_string)
					.collect()
			})
			.unwrap_or_default();
		let tools_empty = remaining_names.is_empty();
		if let Some(choice) = self.rest.get("tool_choice") {
			let pinned = choice.get("name").and_then(|n| n.as_str());
			let unresolvable =
				tools_empty || pinned.is_some_and(|name| !remaining_names.iter().any(|n| n == name));
			if unresolvable && let Some(object) = self.rest.as_object_mut() {
				object.remove("tool_choice");
				report.neutralized_tool_choice = true;
			}
		}
		if tools_empty && let Some(object) = self.rest.as_object_mut() {
			object.remove("tools");
		}

		report
	}
}

/// Sanitized outcome of a Tool Search downgrade — counts and flags only, for
/// telemetry and tests. Never carries tool names, schemas, or content.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ToolSearchDowngrade {
	pub removed_server_tools: usize,
	pub removed_builtin_server_tools: usize,
	pub stripped_defer_loading: usize,
	pub removed_tool_references: usize,
	pub removed_search_rounds: usize,
	pub hoisted_schemas: usize,
	/// Name collisions between hoisted history definitions that carried a
	/// DIFFERENT schema (or between a hoisted definition and the request's
	/// declared tool). Never resolved silently: counted here and resolved by
	/// documented origin rules (declared wins over hoisted, newest hoisted
	/// wins over older hoisted).
	pub hoisted_schema_conflicts: usize,
	/// Built-in server-tool rounds (web_search/web_fetch/code_execution use +
	/// result blocks from an earlier native session) rewritten into neutral
	/// text context so the upstream neither sees an executable server tool it
	/// cannot run nor loses the conversation meaning.
	pub translated_builtin_rounds: usize,
	pub neutralized_tool_choice: bool,
}

/// Anthropic built-in server tools — executed provider-side (web search,
/// web fetch, code execution), not client function calling. Third-party
/// Anthropic-compatible providers reject these tool types outright
/// (2026-10-20: Fireworks answered `web_search_20250329` with HTTP 400 in
/// 13 ms and the client recovered by silently retrying without it), so for
/// models without verified native support the downgrade removes them instead
/// of gambling on every client implementing the same recovery.
fn is_builtin_server_tool(tool_type: &str) -> bool {
	tool_type.starts_with("web_search")
		|| tool_type.starts_with("web_fetch")
		|| tool_type.starts_with("code_execution")
}

/// Same families as [`is_builtin_server_tool`], matched against the `name`
/// field of a history `server_tool_use` block (the block names the tool
/// without its version suffix, e.g. `web_search`).
fn is_builtin_server_tool_name(name: &str) -> bool {
	name.starts_with("web_search")
		|| name.starts_with("web_fetch")
		|| name.starts_with("code_execution")
}

/// History result blocks of the built-in server tools (`web_search_tool_result`,
/// `web_fetch_tool_result`, `code_execution_tool_result`).
fn is_builtin_result_type(block_type: &str) -> bool {
	block_type.starts_with("web_search_tool_result")
		|| block_type.starts_with("web_fetch_tool_result")
		|| block_type.starts_with("code_execution_tool_result")
}

/// Bounded UTF-8-safe truncation for translated history context.
fn truncate_chars(value: &str, max_chars: usize) -> &str {
	match value.char_indices().nth(max_chars) {
		Some((index, _)) => &value[..index],
		None => value,
	}
}

/// Text replacement for a history built-in `server_tool_use` block: states
/// plainly that the built-in ran in an earlier native session and is not
/// available now, keeping a bounded copy of the invocation input as context.
fn builtin_use_summary(name: &str, input: Option<&serde_json::Value>) -> String {
	let mut text = format!(
		"[built-in {name} invocation from an earlier native session — this model cannot run built-in server tools]"
	);
	if let Some(input) = input
		&& let Ok(compact) = serde_json::to_string(input)
	{
		text.push_str(" input: ");
		text.push_str(truncate_chars(&compact, 512));
	}
	text
}

/// Text replacement for a history built-in tool-result block: keeps titles,
/// URLs and short text fields of the results as bounded context instead of an
/// executable-looking server-tool payload (whose `encrypted_content` is
/// undecryptable for third-party providers anyway).
fn builtin_result_summary(block_type: &str, value: &serde_json::Value) -> String {
	let mut text = format!("[{block_type} from an earlier native session]");
	let Some(content) = value.get("content").and_then(|c| c.as_array()) else {
		return text;
	};
	for part in content {
		let part_type = part.get("type").and_then(|t| t.as_str()).unwrap_or("");
		if part_type.ends_with("_result") || part_type == "text" {
			let title = part.get("title").and_then(|t| t.as_str());
			let url = part.get("url").and_then(|u| u.as_str());
			let body = part.get("text").and_then(|t| t.as_str());
			match (title, url) {
				(Some(title), Some(url)) => {
					text.push_str("\n- ");
					text.push_str(truncate_chars(title, 160));
					text.push_str(" — ");
					text.push_str(truncate_chars(url, 200));
				},
				(_, Some(url)) => {
					text.push_str("\n- ");
					text.push_str(truncate_chars(url, 200));
				},
				_ => {
					if let Some(body) = body {
						text.push_str("\n- ");
						text.push_str(truncate_chars(body, 240));
					}
				},
			}
		}
	}
	truncate_chars(&text, 2048).to_string()
}

impl Request {
	/// Whether the `tools` array offers any Anthropic built-in server tool
	/// (`web_search*` / `web_fetch*` / `code_execution*`). Used as an
	/// additional entry-time trigger for the eager downgrade on models
	/// without verified native support — these requests carry no Tool Search
	/// shapes at all, so TS detection alone would forward the unsupported
	/// tool and leave the failure to the upstream.
	pub fn offers_builtin_server_tools(&self) -> bool {
		self
			.rest
			.get("tools")
			.and_then(|t| t.as_array())
			.is_some_and(|items| {
				items.iter().any(|tool| {
					tool
						.get("type")
						.and_then(serde_json::Value::as_str)
						.is_some_and(is_builtin_server_tool)
				})
			})
	}
}

/// Extract a full eager tool definition from a `tool_reference` block that
/// carries one (`input_schema` present). Most `tool_reference` blocks are bare
/// pointers; only search-result payloads embed definitions.
fn hoistable_definition(value: &serde_json::Value) -> Option<serde_json::Value> {
	let object = value.as_object()?;
	let schema = object.get("input_schema")?;
	let mut definition = serde_json::Map::new();
	definition.insert(
		"name".to_string(),
		object
			.get("name")
			.or_else(|| object.get("tool_name"))?
			.clone(),
	);
	if let Some(description) = object.get("description") {
		definition.insert("description".to_string(), description.clone());
	}
	definition.insert("input_schema".to_string(), schema.clone());
	Some(serde_json::Value::Object(definition))
}

/// Rewrite the deferred-tool protocol parts inside one message content array
/// into their eager equivalent, collecting any embedded full tool definitions
/// for hoisting into `tools`.
fn sanitize_deferred_tool_parts(
	parts: &mut Vec<ContentPart>,
	hoisted: &mut Vec<serde_json::Value>,
	report: &mut ToolSearchDowngrade,
) {
	let mut index = 0;
	while index < parts.len() {
		let ContentPart::Unknown(value) = &parts[index] else {
			index += 1;
			continue;
		};
		match value.get("type").and_then(serde_json::Value::as_str) {
			Some("tool_reference") => {
				if let Some(definition) = hoistable_definition(value) {
					hoisted.push(definition);
				}
				report.removed_tool_references += 1;
				parts.remove(index);
			},
			Some("server_tool_use") => {
				let name = value
					.get("name")
					.and_then(serde_json::Value::as_str)
					.unwrap_or_default();
				if name.starts_with("tool_search") {
					report.removed_search_rounds += 1;
					parts.remove(index);
				} else if is_builtin_server_tool_name(name) {
					// A built-in round from an earlier NATIVE session (e.g. the
					// user switched models mid-session). The upstream cannot
					// execute this server tool — forwarding the block verbatim
					// would invite the model to emit another server_tool_use it
					// cannot honor (a "successful" looking search that never
					// runs). Keep the invocation as bounded text context.
					let text = builtin_use_summary(name, value.get("input"));
					parts[index] = ContentPart::Text {
						r#type: "text".to_string(),
						text,
						rest: serde_json::Value::Null,
					};
					report.translated_builtin_rounds += 1;
					index += 1;
				} else {
					index += 1;
				}
			},
			Some(tool_result_type) if is_builtin_result_type(tool_result_type) => {
				let text = builtin_result_summary(tool_result_type, value);
				parts[index] = ContentPart::Text {
					r#type: "text".to_string(),
					text,
					rest: serde_json::Value::Null,
				};
				report.translated_builtin_rounds += 1;
				index += 1;
			},
			Some("tool_search_tool_result") => {
				if let Some(content) = value.get("content").and_then(|c| c.as_array()) {
					for part in content {
						if part.get("type").and_then(|t| t.as_str()) == Some("tool_reference")
							&& let Some(definition) = hoistable_definition(part)
						{
							hoisted.push(definition);
						}
					}
				}
				report.removed_search_rounds += 1;
				parts.remove(index);
			},
			Some("tool_result") => {
				if let ContentPart::Unknown(value) = &mut parts[index]
					&& let Some(nested) = value.get_mut("content").and_then(|c| c.as_array_mut())
				{
					let nested_before = nested.len();
					nested.retain(|part| part.get("type").and_then(|t| t.as_str()) != Some("tool_reference"));
					report.removed_tool_references += nested_before - nested.len();
				}
				index += 1;
			},
			_ => index += 1,
		}
	}
}

/// A `tools` array (raw JSON from the passthrough `rest` field) offering any
/// member of the `tool_search*` server-tool family — e.g.
/// `tool_search_tool_regex_20250922` or `tool_search_tool_bm25_20251119`.
/// The tool's `name` is optional upstream and irrelevant to detection.
fn tools_offer_native_tool_search(tools: Option<&serde_json::Value>) -> bool {
	tools.and_then(|t| t.as_array()).is_some_and(|items| {
		items.iter().any(|tool| {
			tool
				.get("type")
				.and_then(serde_json::Value::as_str)
				.is_some_and(|tool_type| tool_type.starts_with("tool_search"))
		})
	})
}

/// A message content block referencing a deferred tool: a top-level
/// `tool_reference` part (assistant turn), a `tool_result` whose content
/// array carries the deferred tool's schema reply as a `tool_reference` part,
/// a `server_tool_use` block invoking the tool_search server tool, or a
/// `tool_search_tool_result` block from a completed search round.
fn content_references_deferred_tool(block: &ContentBlock) -> bool {
	let ContentBlock::Array(parts) = block else {
		return false;
	};
	parts.iter().any(part_references_deferred_tool)
}

fn part_references_deferred_tool(part: &ContentPart) -> bool {
	let ContentPart::Unknown(value) = part else {
		return false;
	};
	match value.get("type").and_then(serde_json::Value::as_str) {
		Some("tool_reference") => true,
		Some("tool_search_tool_result") => true,
		Some("server_tool_use") => value
			.get("name")
			.and_then(serde_json::Value::as_str)
			.is_some_and(|name| name.starts_with("tool_search")),
		Some("tool_result") => value
			.get("content")
			.and_then(serde_json::Value::as_array)
			.is_some_and(|content| {
				content.iter().any(|part| {
					part.get("type").and_then(serde_json::Value::as_str) == Some("tool_reference")
				})
			}),
		_ => false,
	}
}

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct RequestMessage {
	pub role: String,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub content: Option<ContentBlock>,
	#[serde(flatten, default)]
	pub rest: serde_json::Value,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(untagged)]
pub enum ContentBlock {
	Text(String),
	Array(Vec<ContentPart>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ContentPart {
	Text {
		r#type: String,
		text: String,
		#[serde(flatten, default)]
		rest: serde_json::Value,
	},
	Unknown(serde_json::Value),
}

#[derive(Debug, Deserialize, Clone, Serialize)]
#[serde(untagged)]
pub enum TextBlock {
	Text(String),
	Array(Vec<TextPart>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(untagged)]
pub enum TextPart {
	Text {
		r#type: String,
		text: String,
		#[serde(flatten, default)]
		rest: serde_json::Value,
	},
	Unknown(serde_json::Value),
}

impl TextPart {
	fn text(&self) -> Option<&str> {
		match self {
			TextPart::Text { text, .. } => Some(text),
			TextPart::Unknown(_) => None,
		}
	}

	fn text_mut(&mut self) -> Option<&mut String> {
		match self {
			TextPart::Text { text, .. } => Some(text),
			TextPart::Unknown(_) => None,
		}
	}

	fn rest_mut(&mut self) -> Option<&mut serde_json::Value> {
		match self {
			TextPart::Text { rest, .. } => Some(rest),
			TextPart::Unknown(_) => None,
		}
	}
}

impl ContentPart {
	fn text(&self) -> Option<&str> {
		match self {
			ContentPart::Text { text, .. } => Some(text),
			ContentPart::Unknown(_) => None,
		}
	}

	fn text_mut(&mut self) -> Option<&mut String> {
		match self {
			ContentPart::Text { text, .. } => Some(text),
			ContentPart::Unknown(_) => None,
		}
	}

	fn rest_mut(&mut self) -> Option<&mut serde_json::Value> {
		match self {
			ContentPart::Text { rest, .. } => Some(rest),
			ContentPart::Unknown(_) => None,
		}
	}
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Response {
	pub id: String,
	pub r#type: String,
	pub role: String,
	pub model: String,
	pub stop_reason: Option<String>,
	pub stop_sequence: Option<String>,
	pub usage: Usage,
	pub content: Vec<Content>,
	#[serde(skip)]
	pub input_audio_tokens: Option<u64>,
	#[serde(skip)]
	pub output_audio_tokens: Option<u64>,
	#[serde(flatten, default)]
	pub rest: serde_json::Value,
}

#[derive(Debug, Deserialize, Clone, Serialize, Default)]
pub struct Content {
	#[serde(skip_serializing_if = "Option::is_none")]
	pub text: Option<String>,
	#[serde(flatten, default)]
	pub rest: serde_json::Value,
}

#[derive(Debug, Deserialize, Clone, Serialize)]
pub struct Usage {
	pub input_tokens: u64,
	pub output_tokens: u64,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub cache_creation_input_tokens: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub cache_read_input_tokens: Option<u64>,
	#[serde(skip_serializing_if = "Option::is_none")]
	pub service_tier: Option<String>,
	#[serde(flatten, default)]
	pub rest: serde_json::Value,
}

impl Usage {
	/// Per-TTL cache-write split from the pass-through `usage.cache_creation`
	/// object (Anthropic-native `ephemeral_5m_input_tokens` /
	/// `ephemeral_1h_input_tokens`). Read from `rest` so the client-facing
	/// wire format stays byte-identical — no re-parsed typed field, no key
	/// reordering. `None` = the provider reported no split; consumers must
	/// not guess the class from the aggregate.
	///
	/// A reported all-zero split next to a small aggregate is the
	/// provider's authoritative sub-minimum no-op write evidence (observed
	/// on captured production usage): the classes are authoritative and the
	/// residual aggregate meters at zero — not an inconsistency to hide.
	pub fn cache_ttl_split(&self) -> (Option<u64>, Option<u64>) {
		let cache_creation = self.rest.get("cache_creation");
		let five_m = cache_creation
			.and_then(|c| c.get("ephemeral_5m_input_tokens"))
			.and_then(|v| v.as_u64());
		let one_h = cache_creation
			.and_then(|c| c.get("ephemeral_1h_input_tokens"))
			.and_then(|v| v.as_u64());
		(five_m, one_h)
	}
}

pub fn get_messages_helper(
	messages: &[RequestMessage],
	system: &Option<TextBlock>,
) -> Vec<SimpleChatCompletionMessage> {
	let mut out = Vec::new();
	if let Some(system) = system {
		let content = match system {
			TextBlock::Text(t) => strng::new(t),
			TextBlock::Array(parts) => {
				crate::types::join_text(parts.iter().filter_map(TextPart::text), '\n')
			},
		};
		if !content.is_empty() {
			out.push(SimpleChatCompletionMessage {
				role: strng::literal!("system"),
				content,
			});
		}
	}

	out.extend(messages.iter().map(|m| {
		let content = m
			.content
			.as_ref()
			.map(|c| match c {
				ContentBlock::Text(t) => strng::new(t),
				ContentBlock::Array(parts) => {
					crate::types::join_text(parts.iter().filter_map(ContentPart::text), ' ')
				},
			})
			.unwrap_or_default();
		SimpleChatCompletionMessage {
			role: strng::new(&m.role),
			content,
		}
	}));
	out
}

/// `rest` keys preserved when a masked text run collapses; see `scan_text_runs`.
const PRESERVED_REST_KEYS: &[&str] = &[
	// Anthropic prompt-cache breakpoint
	"cache_control",
];

impl RequestType for Request {
	fn input_format() -> crate::InputFormat {
		crate::InputFormat::Messages
	}
	fn body_is_json(&self) -> bool {
		true
	}
	fn model(&mut self) -> &mut Option<String> {
		&mut self.model
	}

	fn to_value(&self) -> serde_json::Result<serde_json::Value> {
		serde_json::to_value(self)
	}

	fn prepend_prompts(&mut self, prompts: Vec<SimpleChatCompletionMessage>) {
		prepend_prompts_helper(&mut self.messages, &mut self.system, prompts);
	}

	fn append_prompts(&mut self, prompts: Vec<SimpleChatCompletionMessage>) {
		append_prompts_helper(&mut self.messages, &mut self.system, prompts);
	}

	fn to_llm_request(&self, provider: Strng, tokenize: bool) -> Result<LLMRequest, AIError> {
		let model = strng::new(self.model.as_deref().unwrap_or_default());
		let input_tokens = if tokenize {
			let messages = self.get_messages();
			let tokens = crate::tokenizer::num_tokens_from_messages(&model, &messages)?;
			Some(tokens)
		} else {
			None
		};
		// Pass the original body through
		let llm = LLMRequest {
			input_tokens,
			input_format: InputFormat::Messages,
			cache_convention: crate::CacheTokenConvention::pending(),
			request_model: model,
			provider,
			streaming: self.stream.unwrap_or_default(),
			params: LLMRequestParams {
				temperature: self.temperature.map(Into::into),
				top_p: self.top_p.map(Into::into),
				frequency_penalty: None,
				presence_penalty: None,
				seed: None,
				max_tokens: self.max_tokens,
				encoding_format: None,
				dimensions: None,
			},
			prompt: Default::default(),
			provider_state: None,
			safeguards: None,
			transport_hints: Default::default(),
		};
		Ok(llm)
	}

	fn get_messages(&self) -> Vec<SimpleChatCompletionMessage> {
		get_messages_helper(&self.messages, &self.system)
	}

	fn get_messages_v2(&self) -> Vec<NormalizedMessage> {
		let mut messages = self
			.system
			.as_ref()
			.map(|system| NormalizedMessage {
				role: strng::literal!("system"),
				parts: match system {
					TextBlock::Text(text) => {
						vec![NormalizedMessagePart::text(strng::new(text))]
					},
					TextBlock::Array(parts) => parts
						.iter()
						.filter_map(TextPart::text)
						.map(|text| NormalizedMessagePart::text(strng::new(text)))
						.collect(),
				},
			})
			.into_iter()
			.collect::<Vec<_>>();
		messages.extend(self.messages.iter().map(|message| NormalizedMessage {
			role: strng::new(&message.role),
			parts: match &message.content {
				Some(ContentBlock::Text(text)) => {
					vec![NormalizedMessagePart::text(strng::new(text))]
				},
				Some(ContentBlock::Array(parts)) => {
					parts.iter().filter_map(normalized_anthropic_part).collect()
				},
				None => Vec::new(),
			},
		}));
		crate::types::attach_tool_result_names(&mut messages);
		messages
	}

	fn set_messages(&mut self, messages: Vec<SimpleChatCompletionMessage>) {
		let (system_prompts, message_prompts): (Vec<_>, Vec<_>) = messages
			.into_iter()
			.partition(|m| m.role.as_str() == "system");

		self.system = if system_prompts.is_empty() {
			None
		} else {
			Some(TextBlock::Array(
				system_prompts
					.into_iter()
					.map(|p| TextPart::Text {
						r#type: "text".to_string(),
						text: p.content.to_string(),
						rest: Default::default(),
					})
					.collect(),
			))
		};
		self.messages = message_prompts.into_iter().map(Into::into).collect();
	}

	fn visit_text_mut(&mut self, f: &mut dyn FnMut(ContentScope, &mut String)) {
		match &mut self.system {
			Some(TextBlock::Text(text)) => f(ContentScope::SystemPrompt, text),
			Some(TextBlock::Array(parts)) => {
				crate::types::scan_text_runs(
					parts,
					"\n",
					TextPart::text_mut,
					TextPart::rest_mut,
					PRESERVED_REST_KEYS,
					&mut |text| f(ContentScope::SystemPrompt, text),
				);
			},
			None => {},
		}
		for msg in &mut self.messages {
			match &mut msg.content {
				Some(ContentBlock::Text(text)) => f(ContentScope::Messages, text),
				Some(ContentBlock::Array(parts)) => {
					for part in parts.iter_mut() {
						if let ContentPart::Unknown(value) = part {
							visit_tool_part_text(value, f);
						}
					}
					crate::types::scan_text_runs(
						parts,
						" ",
						ContentPart::text_mut,
						ContentPart::rest_mut,
						PRESERVED_REST_KEYS,
						&mut |text| f(ContentScope::Messages, text),
					);
				},
				None => {},
			}
		}
	}
}

fn normalized_anthropic_part(part: &ContentPart) -> Option<NormalizedMessagePart> {
	match part {
		ContentPart::Text { text, .. } => Some(NormalizedMessagePart::text(strng::new(text))),
		ContentPart::Unknown(value) => match value.get("type").and_then(serde_json::Value::as_str) {
			Some("tool_use" | "server_tool_use" | "mcp_tool_use") => {
				crate::types::normalized_tool_call(value)
			},
			Some(item_type) if item_type == "tool_result" || item_type.ends_with("_tool_result") => {
				Some(NormalizedMessagePart::tool_result(
					value
						.get("tool_use_id")
						.or_else(|| value.get("id"))
						.and_then(serde_json::Value::as_str)
						.map(strng::new),
					value
						.get("name")
						.and_then(serde_json::Value::as_str)
						.map(strng::new),
					value
						.get("content")
						.cloned()
						.unwrap_or_else(|| serde_json::Value::Null),
					value.get("is_error").and_then(serde_json::Value::as_bool),
				))
			},
			Some("thinking" | "redacted_thinking") => {
				Some(NormalizedMessagePart::reasoning(value.clone()))
			},
			_ => None,
		},
	}
}

// visit every documented part type
// known-ignored items should be listed
// unknown items should be logged for future review
// https://github.com/anthropics/anthropic-sdk-typescript/blob/main/.stats.yml
// may give us a way to keep an eye on changes
fn visit_tool_part_text(
	value: &mut serde_json::Value,
	f: &mut dyn FnMut(ContentScope, &mut String),
) {
	match value.get("type").and_then(|t| t.as_str()) {
		Some(
			"tool_result"
			| "mcp_tool_result"
			| "code_execution_tool_result"
			| "bash_code_execution_tool_result"
			| "text_editor_code_execution_tool_result"
			| "tool_search_tool_result"
			| "web_fetch_tool_result"
			| "advisor_tool_result",
		) => {
			visit_json_at(value, &["content"], ContentScope::ToolOutput, f);
		},
		Some("tool_use" | "server_tool_use" | "mcp_tool_use") => {
			visit_json_at(value, &["input"], ContentScope::ToolInput, f);
		},
		// User-provided context blocks: message content, not tool traffic.
		Some("document") => {
			visit_json_at(value, &["source"], ContentScope::Messages, f);
			visit_json_at(value, &["title"], ContentScope::Messages, f);
			visit_json_at(value, &["context"], ContentScope::Messages, f);
		},
		Some("search_result") => {
			visit_json_at(value, &["title"], ContentScope::Messages, f);
			visit_json_at(value, &["content"], ContentScope::Messages, f);
		},
		// Replayed conversation summary.
		Some("compaction") => {
			visit_json_at(value, &["content"], ContentScope::Messages, f);
		},
		// Mid-conversation system instructions: text blocks under `content`.
		Some("mid_conv_system") => {
			visit_json_at(value, &["content"], ContentScope::SystemPrompt, f);
		},
		// No readable text: base64 payloads and file/tool references.
		Some("image" | "container_upload" | "tool_addition" | "tool_removal" | "fallback") => {},
		// Signature/encrypted content the API integrity-checks on replay; a mask would 400.
		Some("thinking" | "redacted_thinking" | "web_search_tool_result") => {},
		other => {
			tracing::debug!(
				block_type = other.unwrap_or("<none>"),
				"unrecognized content block; not scanned by prompt guards"
			);
		},
	}
}

pub fn prepend_prompts_helper(
	messages: &mut Vec<RequestMessage>,
	system: &mut Option<TextBlock>,
	prompts: Vec<SimpleChatCompletionMessage>,
) {
	let (system_prompts, message_prompts): (Vec<_>, Vec<_>) = prompts
		.into_iter()
		.partition(|p| p.role.as_str() == "system");

	if !system_prompts.is_empty() {
		let mut items: Vec<TextPart> = match std::mem::take(system) {
			Some(TextBlock::Array(existing)) => existing,
			Some(TextBlock::Text(text)) => vec![TextPart::Text {
				r#type: "text".to_string(),
				text,
				rest: Default::default(),
			}],
			None => Vec::new(),
		};

		items.splice(
			0..0,
			system_prompts.into_iter().map(|p| TextPart::Text {
				r#type: "text".to_string(),
				text: p.content.to_string(),
				rest: Default::default(),
			}),
		);

		*system = Some(TextBlock::Array(items));
	}

	if !message_prompts.is_empty() {
		messages.splice(..0, message_prompts.into_iter().map(Into::into));
	}
}

pub fn append_prompts_helper(
	messages: &mut Vec<RequestMessage>,
	system: &mut Option<TextBlock>,
	prompts: Vec<SimpleChatCompletionMessage>,
) {
	let (system_prompts, message_prompts): (Vec<_>, Vec<_>) = prompts
		.into_iter()
		.partition(|p| p.role.as_str() == "system");

	if !system_prompts.is_empty() {
		let mut items: Vec<TextPart> = match std::mem::take(system) {
			Some(TextBlock::Text(text)) => vec![TextPart::Text {
				r#type: "text".to_string(),
				text,
				rest: Default::default(),
			}],
			Some(TextBlock::Array(existing)) => existing,
			None => Vec::new(),
		};

		items.extend(system_prompts.into_iter().map(|p| TextPart::Text {
			r#type: "text".to_string(),
			text: p.content.to_string(),
			rest: Default::default(),
		}));

		*system = Some(TextBlock::Array(items));
	}

	if !message_prompts.is_empty() {
		messages.extend(message_prompts.into_iter().map(Into::into));
	}
}

impl From<SimpleChatCompletionMessage> for RequestMessage {
	fn from(r: SimpleChatCompletionMessage) -> Self {
		RequestMessage {
			role: r.role.to_string(),
			content: Some(ContentBlock::Text(r.content.to_string())),
			rest: Default::default(),
		}
	}
}

fn extract_output_message(resp: &Response) -> Option<Vec<OutputMessage>> {
	let mut content = Vec::new();

	for c in &resp.content {
		let typ = c.rest.get("type").and_then(|v| v.as_str());
		if matches!(typ, Some("tool_use" | "server_tool_use")) {
			let id = c.rest.get("id").and_then(|v| v.as_str()).unwrap_or("");
			let name = c.rest.get("name").and_then(|v| v.as_str()).unwrap_or("");
			let arguments = c
				.rest
				.get("input")
				.cloned()
				.unwrap_or(serde_json::Value::Object(Default::default()));
			content.push(OutputMessagePart::ToolCall {
				id: strng::new(id),
				name: strng::new(name),
				arguments,
			});
		}
	}

	if content.is_empty() {
		return None;
	}

	Some(vec![OutputMessage {
		role: strng::new(&resp.role),
		content,
		finish_reason: resp.stop_reason.as_deref().map(strng::new),
	}])
}

impl ResponseType for Response {
	fn to_llm_response(&self, log_content: crate::LogContentFields) -> LLMResponse {
		let output_messages = if log_content.tool_calls {
			extract_output_message(self)
		} else {
			None
		};

		LLMResponse {
			usage_complete: Some(true),
			input_tokens: Some(self.usage.input_tokens),
			input_image_tokens: None,
			input_text_tokens: None,
			input_audio_tokens: self.input_audio_tokens,
			output_tokens: Some(self.usage.output_tokens),
			output_image_tokens: None,
			output_text_tokens: None,
			output_audio_tokens: self.output_audio_tokens,
			total_tokens: Some(self.usage.output_tokens + self.usage.input_tokens),
			pages: None,
			provider_model: Some(strng::new(&self.model)),
			count_tokens: None,
			reasoning_tokens: None,
			cache_creation_input_tokens: self.usage.cache_creation_input_tokens,
			cache_creation_5m_input_tokens: self.usage.cache_ttl_split().0,
			cache_creation_1h_input_tokens: self.usage.cache_ttl_split().1,
			cached_input_tokens: self.usage.cache_read_input_tokens,
			service_tier: self.usage.service_tier.as_deref().map(Into::into),
			completion: if log_content.completion {
				Some(
					self
						.content
						.iter()
						.flat_map(|c| c.text.clone())
						.collect_vec(),
				)
			} else {
				None
			},
			output_messages,
			first_token: Default::default(),
			last_token_at: Default::default(),
			inter_chunk_latencies: Default::default(),
		}
	}

	fn set_webhook_choices(&mut self, choices: Vec<ResponseChoice>) -> anyhow::Result<()> {
		if self.content.len() != choices.len() {
			anyhow::bail!("webhook response message count mismatch");
		}
		for (m, wh) in self.content.iter_mut().zip(choices) {
			m.text = Some(wh.message.content.to_string());
		}
		Ok(())
	}

	fn to_webhook_choices(&self) -> Vec<ResponseChoice> {
		self
			.content
			.iter()
			.map(|c| {
				let content = c.text.clone().unwrap_or_default();
				ResponseChoice {
					message: Message {
						role: "assistant".into(),
						content: content.into(),
					},
				}
			})
			.collect()
	}

	fn serialize(&self) -> serde_json::Result<Vec<u8>> {
		serde_json::to_vec(&self)
	}

	fn visit_text_mut(&mut self, f: &mut dyn FnMut(&mut String)) {
		for c in &mut self.content {
			if let Some(text) = &mut c.text {
				f(text);
			}
		}
	}
}

// 'typed' provides a typed accessor
pub mod typed {
	use async_openai::types::chat::PromptCacheBreakpointParam;
	use serde::{Deserialize, Deserializer, Serialize};
	use serde_json::Value;

	use crate::serdes::is_default;

	#[derive(Copy, Clone, Deserialize, Serialize, Debug, PartialEq, Eq, Default)]
	#[serde(rename_all = "snake_case")]
	pub enum Role {
		#[default]
		User,
		Assistant,
		System,
	}

	#[derive(Clone, Deserialize, Serialize, Debug)]
	#[serde(rename_all = "snake_case")]
	pub struct ContentTextBlock {
		pub text: String,

		#[serde(skip_serializing_if = "Option::is_none")]
		pub citations: Option<Value>,

		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_control: Option<CacheControlEphemeral>,
	}

	#[derive(Clone, Deserialize, Serialize, Debug)]
	#[serde(rename_all = "snake_case")]
	pub struct ContentImageBlock {
		pub source: Value,
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_control: Option<CacheControlEphemeral>,
	}

	#[derive(Clone, Deserialize, Serialize, Debug)]
	#[serde(rename_all = "snake_case")]
	pub struct ContentSearchResultBlock {
		pub content: Vec<Value>,
		pub source: String,
		pub title: String,
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_control: Option<CacheControlEphemeral>,
	}

	#[derive(Clone, Deserialize, Serialize, Debug)]
	#[serde(rename_all = "snake_case")]
	pub struct ContentDocumentBlock {
		pub source: Value,
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_control: Option<CacheControlEphemeral>,
		#[serde(skip_serializing_if = "Option::is_none")]
		pub citations: Option<Value>,
		#[serde(skip_serializing_if = "Option::is_none")]
		pub context: Option<String>,
		#[serde(skip_serializing_if = "Option::is_none")]
		pub title: Option<String>,
	}

	#[derive(Clone, Deserialize, Serialize, Debug)]
	#[serde(rename_all = "snake_case", tag = "type")]
	pub enum ContentBlock {
		Text(ContentTextBlock),
		Image(ContentImageBlock),
		Document(ContentDocumentBlock),
		SearchResult(ContentSearchResultBlock),
		Thinking {
			thinking: String,
			signature: String,
		},
		RedactedThinking {
			data: String,
		},
		/// Tool use content
		ToolUse {
			id: String,
			name: String,
			input: serde_json::Value,
			#[serde(skip_serializing_if = "Option::is_none")]
			cache_control: Option<CacheControlEphemeral>,
		},
		/// Tool result content
		ToolResult {
			tool_use_id: String,
			content: ToolResultContent,
			#[serde(skip_serializing_if = "Option::is_none")]
			cache_control: Option<CacheControlEphemeral>,
			#[serde(skip_serializing_if = "Option::is_none")]
			is_error: Option<bool>,
		},
		ServerToolUse {
			id: String,
			name: String,
			input: serde_json::Value,
			#[serde(skip_serializing_if = "Option::is_none")]
			cache_control: Option<CacheControlEphemeral>,
		},
		/// Web search tool result content
		WebSearchToolResult {
			tool_use_id: String,
			#[serde(skip_serializing_if = "Option::is_none")]
			content: Option<serde_json::Value>,
			#[serde(skip_serializing_if = "Option::is_none")]
			cache_control: Option<CacheControlEphemeral>,
		},
		// There are LOTs of possible values; since we don't support them all, just allow them without failing
		#[serde(other)]
		Unknown,
	}

	#[derive(Debug, Serialize, Deserialize, Clone)]
	#[serde(untagged)]
	pub enum ToolResultContent {
		/// The text contents of the tool message.
		Text(String),
		/// An array of content parts with a defined type. For tool messages, only type `text` is supported.
		Array(Vec<ToolResultContentPart>),
	}

	#[derive(Debug, Serialize, Deserialize, Clone)]
	#[serde(tag = "type", rename_all = "snake_case")]
	pub enum ToolResultContentPart {
		Text {
			text: String,
			#[serde(skip_serializing_if = "Option::is_none")]
			citations: Option<Value>,
			#[serde(skip_serializing_if = "Option::is_none")]
			cache_control: Option<CacheControlEphemeral>,
		},
		Image {
			source: Value,
			#[serde(skip_serializing_if = "Option::is_none")]
			cache_control: Option<CacheControlEphemeral>,
		},
		Document {
			source: Value,
			#[serde(skip_serializing_if = "Option::is_none")]
			cache_control: Option<CacheControlEphemeral>,
			#[serde(skip_serializing_if = "Option::is_none")]
			citations: Option<Value>,
			#[serde(skip_serializing_if = "Option::is_none")]
			context: Option<String>,
			#[serde(skip_serializing_if = "Option::is_none")]
			title: Option<String>,
		},
		SearchResult {
			content: Vec<Value>,
			source: String,
			title: String,
			#[serde(skip_serializing_if = "Option::is_none")]
			cache_control: Option<CacheControlEphemeral>,
		},
		ToolReference {
			tool_name: String,
			#[serde(skip_serializing_if = "Option::is_none")]
			cache_control: Option<CacheControlEphemeral>,
		},
		// Same tolerance as ContentBlock: an unrecognized part must not reject the whole request
		#[serde(other)]
		Unknown,
	}

	#[derive(Clone, Deserialize, Serialize, Debug, PartialEq, Eq)]
	#[serde(rename_all = "snake_case", tag = "type")]
	pub enum CacheControlEphemeral {
		Ephemeral {
			#[serde(default)]
			#[serde(skip_serializing_if = "Option::is_none")]
			ttl: Option<String>,
		},
	}

	#[derive(Clone, Deserialize, Serialize, Debug)]
	#[serde(rename_all = "snake_case")]
	pub struct Message {
		pub role: Role,
		#[serde(deserialize_with = "deserialize_content")]
		pub content: Vec<ContentBlock>,
	}

	// Custom deserializer that handles both string and array formats
	fn deserialize_content<'de, D>(deserializer: D) -> Result<Vec<ContentBlock>, D::Error>
	where
		D: Deserializer<'de>,
	{
		use serde::de::Error;
		use serde_json::Value;

		let value = Value::deserialize(deserializer)?;

		match value {
			// If it's a string, wrap it in a Text content block
			Value::String(text) => Ok(vec![ContentBlock::Text(ContentTextBlock {
				text,
				citations: None,
				cache_control: None,
			})]),
			// If it's an array, deserialize normally
			Value::Array(_) => Vec::<ContentBlock>::deserialize(value).map_err(D::Error::custom),
			// Reject other types
			_ => Err(D::Error::custom(
				"content must be either a string or an array",
			)),
		}
	}

	/// System prompt format - can be either a simple string or an array of content blocks
	#[derive(Clone, Debug, Serialize, Deserialize)]
	#[serde(untagged)]
	pub enum SystemPrompt {
		Text(String),
		Blocks(Vec<SystemContentBlock>),
	}

	/// System content block for structured system prompts
	#[derive(Clone, Debug, Serialize, Deserialize)]
	#[serde(tag = "type", rename_all = "snake_case")]
	pub enum SystemContentBlock {
		Text {
			text: String,
			#[serde(skip_serializing_if = "Option::is_none")]
			cache_control: Option<CacheControlEphemeral>,
		},
	}

	impl From<CacheControlEphemeral> for super::super::completions::typed::PromptCacheBreakpointParam {
		fn from(_: CacheControlEphemeral) -> Self {
			PromptCacheBreakpointParam {
				mode: super::super::completions::typed::PromptCacheBreakpointParamMode::Explicit,
			}
		}
	}

	#[derive(Deserialize, Serialize, Default, Debug)]
	pub struct Request {
		/// The User/Assistent prompts.
		pub messages: Vec<Message>,
		/// The System prompt - can be a string or array of content blocks
		#[serde(skip_serializing_if = "Option::is_none")]
		pub system: Option<SystemPrompt>,
		/// The model to use.
		pub model: String,
		/// The maximum number of tokens to generate before stopping.
		pub max_tokens: usize,
		/// The stop sequences to use.
		#[serde(default, skip_serializing_if = "Vec::is_empty")]
		pub stop_sequences: Vec<String>,
		/// Whether to incrementally stream the response.
		#[serde(default, skip_serializing_if = "is_default")]
		pub stream: bool,
		/// Amount of randomness injected into the response.
		///
		/// Defaults to 1.0. Ranges from 0.0 to 1.0. Use temperature closer to 0.0 for analytical /
		/// multiple choice, and closer to 1.0 for creative and generative tasks. Note that even
		/// with temperature of 0.0, the results will not be fully deterministic.
		#[serde(skip_serializing_if = "Option::is_none")]
		pub temperature: Option<f32>,
		/// Use nucleus sampling.
		///
		/// In nucleus sampling, we compute the cumulative distribution over all the options for each
		/// subsequent token in decreasing probability order and cut it off once it reaches a particular
		/// probability specified by top_p. You should either alter temperature or top_p, but not both.
		/// Recommended for advanced use cases only. You usually only need to use temperature.
		#[serde(skip_serializing_if = "Option::is_none")]
		pub top_p: Option<f32>,
		/// Only sample from the top K options for each subsequent token.
		/// Used to remove "long tail" low probability responses. Learn more technical details here.
		/// Recommended for advanced use cases only. You usually only need to use temperature.
		#[serde(skip_serializing_if = "Option::is_none")]
		pub top_k: Option<usize>,
		/// Tools that the model may use
		#[serde(skip_serializing_if = "Option::is_none")]
		pub tools: Option<Vec<Tool>>,
		/// How the model should use tools
		#[serde(skip_serializing_if = "Option::is_none")]
		pub tool_choice: Option<ToolChoice>,
		/// Request metadata
		#[serde(skip_serializing_if = "Option::is_none")]
		pub metadata: Option<Metadata>,

		#[serde(skip_serializing_if = "Option::is_none")]
		pub thinking: Option<ThinkingInput>,

		#[serde(skip_serializing_if = "Option::is_none")]
		pub output_config: Option<OutputConfig>,
	}

	#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq, Default)]
	pub struct OutputConfig {
		#[serde(skip_serializing_if = "Option::is_none")]
		pub effort: Option<ThinkingEffort>,
		#[serde(skip_serializing_if = "Option::is_none")]
		pub format: Option<OutputFormat>,
	}

	#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
	#[serde(rename_all = "snake_case", tag = "type")]
	pub enum OutputFormat {
		JsonSchema { schema: serde_json::Value },
	}

	#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
	#[serde(rename_all = "snake_case", tag = "type")]
	pub enum ThinkingInput {
		Enabled { budget_tokens: u64 },
		Disabled {},
		Adaptive {},
	}

	#[derive(Clone, Copy, Serialize, Deserialize, Debug, Eq, PartialEq)]
	#[serde(rename_all = "snake_case")]
	pub enum ThinkingEffort {
		Low,
		Medium,
		High,
		Xhigh,
		Max,
	}

	/// Response body for the Messages API.
	#[derive(Debug, Serialize, Deserialize, Clone)]
	pub struct MessagesResponse {
		/// Unique object identifier.
		/// The format and length of IDs may change over time.
		pub id: String,
		/// Object type.
		/// For Messages, this is always "message".
		pub r#type: String,
		/// Conversational role of the generated message.
		/// This will always be "assistant".
		pub role: Role,
		/// Content generated by the model.
		/// This is an array of content blocks, each of which has a type that determines its shape.
		/// Currently, the only type in responses is "text".
		///
		/// Example:
		/// `[{"type": "text", "text": "Hi, I'm Claude."}]`
		///
		/// If the request input messages ended with an assistant turn, then the response content
		/// will continue directly from that last turn. You can use this to constrain the model's
		/// output.
		///
		/// For example, if the input messages were:
		/// `[ {"role": "user", "content": "What's the Greek name for Sun? (A) Sol (B) Helios (C) Sun"},
		///    {"role": "assistant", "content": "The best answer is ("} ]`
		///
		/// Then the response content might be:
		/// `[{"type": "text", "text": "B)"}]`
		pub content: Vec<ContentBlock>,
		/// The model that handled the request.
		pub model: String,
		/// The reason that we stopped.
		/// This may be one the following values:
		/// - "end_turn": the model reached a natural stopping point
		/// - "max_tokens": we exceeded the requested max_tokens or the model's maximum
		/// - "stop_sequence": one of your provided custom stop_sequences was generated
		///
		/// Note that these values are different than those in /v1/complete, where end_turn and
		/// stop_sequence were not differentiated.
		///
		/// In non-streaming mode this value is always non-null. In streaming mode, it is null
		/// in the message_start event and non-null otherwise.
		pub stop_reason: Option<StopReason>,
		/// Which custom stop sequence was generated, if any.
		/// This value will be a non-null string if one of your custom stop sequences was generated.
		pub stop_sequence: Option<String>,
		/// Billing and rate-limit usage.
		/// Anthropic's API bills and rate-limits by token counts, as tokens represent the underlying
		/// cost to our systems.
		///
		/// Under the hood, the API transforms requests into a format suitable for the model. The
		/// model's output then goes through a parsing stage before becoming an API response. As a
		/// result, the token counts in usage will not match one-to-one with the exact visible
		/// content of an API request or response.
		///
		/// For example, output_tokens will be non-zero, even for an empty string response from Claude.
		pub usage: Usage,

		// Internal fields not shown to user but used for our internal accounting.
		#[serde(skip)]
		pub input_audio_tokens: Option<usize>,
		#[serde(skip)]
		pub output_audio_tokens: Option<usize>,
	}

	#[derive(Clone, Serialize, Deserialize, Debug)]
	#[serde(rename_all = "snake_case", tag = "type")]
	pub enum MessagesStreamEvent {
		MessageStart {
			message: MessagesResponse,
		},
		ContentBlockStart {
			index: usize,
			content_block: ContentBlock,
		},
		ContentBlockDelta {
			index: usize,
			delta: ContentBlockDelta,
		},
		ContentBlockStop {
			index: usize,
		},
		MessageDelta {
			delta: MessageDelta,
			usage: MessageDeltaUsage,
		},
		MessageStop,
		Ping,
		Error {
			error: MessagesError,
		},
	}

	impl MessagesStreamEvent {
		/// Get the SSE event name for this event type
		#[allow(dead_code)] // Used by Bedrock streaming translation
		pub fn event_name(&self) -> &'static str {
			match self {
				Self::MessageStart { .. } => "message_start",
				Self::ContentBlockStart { .. } => "content_block_start",
				Self::ContentBlockDelta { .. } => "content_block_delta",
				Self::ContentBlockStop { .. } => "content_block_stop",
				Self::MessageDelta { .. } => "message_delta",
				Self::MessageStop => "message_stop",
				Self::Ping => "ping",
				Self::Error { .. } => "error",
			}
		}

		/// Convert to (event_name, self) tuple for transform_multi
		#[allow(dead_code)] // Used by Bedrock streaming translation
		pub fn into_sse_tuple(self) -> (&'static str, Self) {
			let name = self.event_name();
			(name, self)
		}
	}

	// Note: event_name() and into_sse_tuple() are used by Bedrock streaming translation

	#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
	#[serde(rename_all = "snake_case", tag = "type")]
	#[allow(clippy::enum_variant_names)]
	pub enum ContentBlockDelta {
		TextDelta {
			text: String,
		},
		InputJsonDelta {
			partial_json: String,
		},
		ThinkingDelta {
			thinking: String,
		},
		SignatureDelta {
			signature: String,
		},
		CitationsDelta {
			#[serde(default)]
			citations: Vec<serde_json::Value>,
		},
	}

	#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
	pub struct MessageDeltaUsage {
		/// Cumulative input tokens
		pub input_tokens: Option<usize>,
		/// Cumulative output tokens
		pub output_tokens: Option<usize>,
		/// Cumulative cache creation tokens
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_creation_input_tokens: Option<usize>,
		/// Cumulative cache read tokens
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_read_input_tokens: Option<usize>,
		/// Per-TTL cache-write split, when the provider reports the
		/// `usage.cache_creation` object on the cumulative delta (Anthropic
		/// prompt caching: 5m default bucket, 1h extended bucket). Absent
		/// means the split is unknown — never zero.
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_creation: Option<CacheCreationSplit>,
	}

	/// The Anthropic `usage.cache_creation` object: explicit per-TTL
	/// buckets. A present `Some(0)` bucket is real evidence (the provider
	/// reported the split); a missing key or object means unknown.
	#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
	pub struct CacheCreationSplit {
		#[serde(skip_serializing_if = "Option::is_none")]
		pub ephemeral_5m_input_tokens: Option<usize>,
		#[serde(skip_serializing_if = "Option::is_none")]
		pub ephemeral_1h_input_tokens: Option<usize>,
	}

	#[derive(Clone, Serialize, Deserialize, Debug, Eq, PartialEq)]
	pub struct MessageDelta {
		/// The reason that we stopped.
		/// This may be one the following values:
		/// - "end_turn": the model reached a natural stopping point
		/// - "max_tokens": we exceeded the requested max_tokens or the model's maximum
		/// - "stop_sequence": one of your provided custom stop_sequences was generated
		///
		/// Note that these values are different than those in /v1/complete, where end_turn and
		/// stop_sequence were not differentiated.
		///
		/// In non-streaming mode this value is always non-null. In streaming mode, it is null
		/// in the message_start event and non-null otherwise.
		pub stop_reason: Option<StopReason>,
		/// Which custom stop sequence was generated, if any.
		/// This value will be a non-null string if one of your custom stop sequences was generated.
		pub stop_sequence: Option<String>,
	}

	/// Response body for the Messages API.
	#[derive(Debug, Deserialize, Serialize, Clone)]
	pub struct MessagesErrorResponse {
		pub r#type: String,
		pub error: MessagesError,
	}

	#[derive(Debug, Deserialize, Serialize, Clone)]
	pub struct MessagesError {
		pub r#type: String,
		pub message: String,
	}

	/// Reason for stopping the response generation.
	#[derive(Copy, Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
	#[serde(rename_all = "snake_case")]
	pub enum StopReason {
		/// The model reached a natural stopping point.
		EndTurn,
		/// The requested max_tokens or the model's maximum was exceeded.
		MaxTokens,
		/// One of the provided custom stop_sequences was generated.
		StopSequence,
		/// The model invoked one or more tools.
		ToolUse,
		/// The model's response was refused.
		Refusal,
		/// The model paused generation (for long-running responses).
		PauseTurn,
		/// The model exceeded the context window.
		ModelContextWindowExceeded,
	}

	/// Billing and rate-limit usage.
	#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
	pub struct Usage {
		/// The number of input tokens which were used.
		pub input_tokens: usize,

		/// The number of output tokens which were used.
		pub output_tokens: usize,

		/// The number of input tokens used to create the cache entry.
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_creation_input_tokens: Option<usize>,

		/// The number of input tokens read from the cache.
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_read_input_tokens: Option<usize>,

		/// The service tier used to serve the request.
		#[serde(skip_serializing_if = "Option::is_none")]
		pub service_tier: Option<String>,

		/// Per-TTL cache-write split, when the provider reports the
		/// `usage.cache_creation` object (see [`CacheCreationSplit`]).
		/// Absent means the split is unknown — never zero.
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_creation: Option<CacheCreationSplit>,
	}

	/// Tool definition. A client-defined custom tool always carries `input_schema` and no `type`
	/// tag. An Anthropic server tool (`web_search_20250305`, `bash_20250124`, `computer_20250124`,
	/// `text_editor_20250728`, `code_execution_20250522`, etc.) is tagged with `type` and never
	/// carries `input_schema` since it runs server-side. `Custom` is tried first so existing custom
	/// tool payloads (no `type` field) keep matching without a discriminant lookup.
	#[derive(Debug, Serialize, Deserialize)]
	#[serde(untagged)]
	pub enum Tool {
		Custom(CustomTool),
		Server(ServerTool),
	}

	impl Tool {
		pub fn name(&self) -> &str {
			match self {
				Tool::Custom(tool) => &tool.name,
				Tool::Server(tool) => &tool.name,
			}
		}

		pub fn cache_control(&self) -> Option<&CacheControlEphemeral> {
			match self {
				Tool::Custom(tool) => tool.cache_control.as_ref(),
				Tool::Server(tool) => tool.cache_control.as_ref(),
			}
		}
	}

	#[derive(Debug, Serialize, Deserialize)]
	pub struct CustomTool {
		/// Name of the tool
		pub name: String,
		/// Description of the tool
		#[serde(skip_serializing_if = "Option::is_none")]
		pub description: Option<String>,
		/// JSON schema for tool input
		pub input_schema: serde_json::Value,
		/// Enforce the tool input schema strictly.
		#[serde(skip_serializing_if = "Option::is_none")]
		pub strict: Option<bool>,
		/// Create a cache control breakpoint at this content block
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_control: Option<CacheControlEphemeral>,
	}

	/// An Anthropic server-executed tool (runs upstream of the provider, e.g. `web_search_20250305`).
	/// We don't model every server tool's specific fields — just enough to round-trip the block
	/// without failing deserialization. Providers that can't execute a server tool (e.g. Bedrock)
	/// reject the request explicitly rather than silently changing its semantics; see `conversion::bedrock`.
	#[derive(Debug, Serialize, Deserialize)]
	pub struct ServerTool {
		/// Discriminant, e.g. "web_search_20250305"
		#[serde(rename = "type")]
		pub tool_type: String,
		/// Name of the tool
		pub name: String,
		/// Create a cache control breakpoint at this content block
		#[serde(skip_serializing_if = "Option::is_none")]
		pub cache_control: Option<CacheControlEphemeral>,
		/// Any other server-tool-specific fields (max_uses, allowed_domains, etc.)
		#[serde(flatten)]
		pub extra: std::collections::HashMap<String, serde_json::Value>,
	}

	/// Tool choice configuration
	#[derive(Debug, Serialize, Deserialize)]
	#[serde(tag = "type", rename_all = "snake_case")]
	pub enum ToolChoice {
		/// Let model choose whether to use tools
		Auto {
			#[serde(default, skip_serializing_if = "Option::is_none")]
			disable_parallel_tool_use: Option<bool>,
		},
		/// Model must use one of the provided tools
		Any {
			#[serde(default, skip_serializing_if = "Option::is_none")]
			disable_parallel_tool_use: Option<bool>,
		},
		/// Model must use a specific tool
		Tool {
			name: String,
			#[serde(default, skip_serializing_if = "Option::is_none")]
			disable_parallel_tool_use: Option<bool>,
		},
		/// Model must not use any tools
		None {},
	}

	/// Message metadata
	#[derive(Debug, Serialize, Deserialize, Default)]
	pub struct Metadata {
		/// Custom metadata fields
		#[serde(flatten)]
		pub fields: std::collections::HashMap<String, String>,
	}

	fn extract_output_message_from_typed(
		resp: &MessagesResponse,
	) -> Option<Vec<super::OutputMessage>> {
		let mut content = Vec::new();

		for block in &resp.content {
			match block {
				ContentBlock::ToolUse {
					id, name, input, ..
				}
				| ContentBlock::ServerToolUse {
					id, name, input, ..
				} => {
					content.push(super::OutputMessagePart::ToolCall {
						id: agent_core::strng::new(id),
						name: agent_core::strng::new(name),
						arguments: input.clone(),
					});
				},
				_ => {},
			}
		}

		if content.is_empty() {
			return None;
		}

		let finish_reason = resp.stop_reason.as_ref().map(|sr| {
			let s = serde_json::to_value(sr)
				.ok()
				.and_then(|v| v.as_str().map(String::from))
				.unwrap_or_default();
			agent_core::strng::new(&s)
		});

		let role_str = match resp.role {
			Role::User => "user",
			Role::Assistant => "assistant",
			Role::System => "system",
		};

		Some(vec![super::OutputMessage {
			role: agent_core::strng::new(role_str),
			content,
			finish_reason,
		}])
	}

	impl super::ResponseType for MessagesResponse {
		fn to_llm_response(&self, log_content: crate::LogContentFields) -> crate::LLMResponse {
			let output_messages = if log_content.tool_calls {
				extract_output_message_from_typed(self)
			} else {
				None
			};

			crate::LLMResponse {
				usage_complete: Some(true),
				input_tokens: Some(self.usage.input_tokens as u64),
				input_image_tokens: None,
				input_text_tokens: None,
				input_audio_tokens: self.input_audio_tokens.map(|i| i as u64),
				output_tokens: Some(self.usage.output_tokens as u64),
				output_image_tokens: None,
				output_text_tokens: None,
				output_audio_tokens: self.output_audio_tokens.map(|i| i as u64),
				total_tokens: Some((self.usage.input_tokens + self.usage.output_tokens) as u64),
				pages: None,
				reasoning_tokens: None,
				cache_creation_input_tokens: self.usage.cache_creation_input_tokens.map(|i| i as u64),
				// Aggregate-only in this accumulated shape — no split reported.
				cache_creation_5m_input_tokens: None,
				cache_creation_1h_input_tokens: None,
				cached_input_tokens: self.usage.cache_read_input_tokens.map(|i| i as u64),
				service_tier: self.usage.service_tier.as_deref().map(Into::into),
				provider_model: Some(agent_core::strng::new(&self.model)),
				count_tokens: None,
				completion: if log_content.completion {
					Some(
						self
							.content
							.iter()
							.filter_map(|c| match c {
								ContentBlock::Text(t) => Some(t.text.clone()),
								_ => None,
							})
							.collect(),
					)
				} else {
					None
				},
				output_messages,
				first_token: Default::default(),
				last_token_at: Default::default(),
				inter_chunk_latencies: Default::default(),
			}
		}

		fn set_webhook_choices(
			&mut self,
			choices: Vec<crate::webhook::ResponseChoice>,
		) -> anyhow::Result<()> {
			if self.content.len() != choices.len() {
				anyhow::bail!("webhook response message count mismatch");
			}
			for (block, wh) in self.content.iter_mut().zip(choices) {
				if let ContentBlock::Text(t) = block {
					t.text = wh.message.content.to_string();
				}
			}
			Ok(())
		}

		fn to_webhook_choices(&self) -> Vec<crate::webhook::ResponseChoice> {
			self
				.content
				.iter()
				.map(|c| {
					let content = match c {
						ContentBlock::Text(t) => t.text.clone(),
						_ => String::new(),
					};
					crate::webhook::ResponseChoice {
						message: crate::webhook::Message {
							role: "assistant".into(),
							content: content.into(),
						},
					}
				})
				.collect()
		}

		fn serialize(&self) -> serde_json::Result<Vec<u8>> {
			serde_json::to_vec(&self)
		}

		fn visit_text_mut(&mut self, f: &mut dyn FnMut(&mut String)) {
			for block in &mut self.content {
				if let ContentBlock::Text(t) = block {
					f(&mut t.text);
				}
			}
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::types::ResponseType;

	#[test]
	fn tool_result_parts_accept_tool_reference_and_unknown_types() {
		let parsed: typed::ToolResultContent = serde_json::from_value(serde_json::json!([
			{"type": "tool_reference", "tool_name": "mcp__example__list_widgets"},
			{"type": "future_block", "foo": 1}
		]))
		.expect("unrecognized tool_result parts must not fail parsing");
		let typed::ToolResultContent::Array(parts) = parsed else {
			panic!("expected array content");
		};
		assert!(matches!(
			&parts[0],
			typed::ToolResultContentPart::ToolReference { tool_name, .. } if tool_name == "mcp__example__list_widgets"
		));
		assert!(matches!(parts[1], typed::ToolResultContentPart::Unknown));
	}

	fn make_typed_response_with_tool_use() -> typed::MessagesResponse {
		typed::MessagesResponse {
			id: "msg_test".to_string(),
			r#type: "message".to_string(),
			role: typed::Role::Assistant,
			content: vec![
				typed::ContentBlock::Text(typed::ContentTextBlock {
					text: "I'll call the tool.".to_string(),
					citations: None,
					cache_control: None,
				}),
				typed::ContentBlock::ToolUse {
					id: "toolu_01A".to_string(),
					name: "get_weather".to_string(),
					input: serde_json::json!({"location": "San Francisco"}),
					cache_control: None,
				},
				typed::ContentBlock::ServerToolUse {
					id: "srvtoolu_01B".to_string(),
					name: "web_search".to_string(),
					input: serde_json::json!({"query": "rust programming"}),
					cache_control: None,
				},
			],
			model: "claude-sonnet-4-20250514".to_string(),
			stop_reason: Some(typed::StopReason::ToolUse),
			stop_sequence: None,
			usage: typed::Usage {
				input_tokens: 100,
				output_tokens: 50,
				cache_creation_input_tokens: None,
				cache_read_input_tokens: None,
				service_tier: None,
				cache_creation: None,
			},
			input_audio_tokens: None,
			output_audio_tokens: None,
		}
	}

	#[test]
	fn test_typed_response_output_messages_populated_when_flag_true() {
		let response = make_typed_response_with_tool_use();
		let llm_response = response.to_llm_response(crate::LogContentFields {
			completion: true,
			tool_calls: true,
		});

		let messages = llm_response
			.output_messages
			.expect("output_messages should be Some");
		assert_eq!(messages.len(), 1);
		assert_eq!(messages[0].role.as_str(), "assistant");
		assert_eq!(messages[0].finish_reason.as_deref(), Some("tool_use"));

		let tool_calls = messages[0].tool_calls();
		assert_eq!(tool_calls.len(), 2);

		assert_eq!(tool_calls[0].id.as_str(), "toolu_01A");
		assert_eq!(tool_calls[0].name.as_str(), "get_weather");
		assert_eq!(
			tool_calls[0].arguments,
			serde_json::json!({"location":"San Francisco"})
		);

		assert_eq!(tool_calls[1].id.as_str(), "srvtoolu_01B");
		assert_eq!(tool_calls[1].name.as_str(), "web_search");
		assert_eq!(
			tool_calls[1].arguments,
			serde_json::json!({"query":"rust programming"})
		);
	}

	#[test]
	fn test_typed_response_text_only_omits_output_messages() {
		let response = typed::MessagesResponse {
			id: "msg_test2".to_string(),
			r#type: "message".to_string(),
			role: typed::Role::Assistant,
			content: vec![typed::ContentBlock::Text(typed::ContentTextBlock {
				text: "Hello!".to_string(),
				citations: None,
				cache_control: None,
			})],
			model: "claude-sonnet-4-20250514".to_string(),
			stop_reason: Some(typed::StopReason::EndTurn),
			stop_sequence: None,
			usage: typed::Usage {
				input_tokens: 50,
				output_tokens: 20,
				cache_creation_input_tokens: None,
				cache_read_input_tokens: None,
				service_tier: None,
				cache_creation: None,
			},
			input_audio_tokens: None,
			output_audio_tokens: None,
		};

		let llm_response = response.to_llm_response(crate::LogContentFields {
			completion: true,
			tool_calls: true,
		});
		assert!(llm_response.output_messages.is_none());
	}

	#[test]
	fn test_weakly_typed_response_output_messages_from_rest() {
		let response = Response {
			id: "msg_test3".to_string(),
			r#type: "message".to_string(),
			role: "assistant".to_string(),
			model: "claude-sonnet-4-20250514".to_string(),
			stop_reason: Some("tool_use".to_string()),
			stop_sequence: None,
			usage: Usage {
				input_tokens: 100,
				output_tokens: 50,
				cache_creation_input_tokens: None,
				cache_read_input_tokens: None,
				service_tier: None,
				rest: Default::default(),
			},
			content: vec![
				Content {
					text: Some("I'll call the tool.".to_string()),
					rest: serde_json::json!({"type": "text"}),
				},
				Content {
					text: None,
					rest: serde_json::json!({
						"type": "tool_use",
						"id": "toolu_01A",
						"name": "get_weather",
						"input": {"location": "San Francisco"}
					}),
				},
			],
			input_audio_tokens: None,
			output_audio_tokens: None,
			rest: Default::default(),
		};

		let llm_response = response.to_llm_response(crate::LogContentFields {
			completion: true,
			tool_calls: true,
		});

		let messages = llm_response
			.output_messages
			.expect("output_messages should be Some");
		assert_eq!(messages[0].role.as_str(), "assistant");
		assert_eq!(messages[0].finish_reason.as_deref(), Some("tool_use"));

		assert_eq!(messages[0].content.len(), 1);

		let tool_calls = messages[0].tool_calls();
		assert_eq!(tool_calls.len(), 1);
		assert_eq!(tool_calls[0].id.as_str(), "toolu_01A");
		assert_eq!(tool_calls[0].name.as_str(), "get_weather");
		assert_eq!(
			tool_calls[0].arguments,
			serde_json::json!({"location":"San Francisco"})
		);
	}

	fn parse_request(body: serde_json::Value) -> Request {
		serde_json::from_value(body).expect("request must parse")
	}

	fn toolsearch_body_base() -> serde_json::Value {
		serde_json::json!({
			"model": "claude-sonnet-4-5",
			"max_tokens": 64,
			"messages": [{"role": "user", "content": "find a tool for that"}],
		})
	}

	#[test]
	fn native_tool_search_detected_from_regex_search_tool() {
		let mut body = toolsearch_body_base();
		body["tools"] = serde_json::json!([
			{"type": "tool_search_tool_regex_20250922", "name": "tool_search"}
		]);
		assert!(parse_request(body).uses_native_tool_search());
	}

	#[test]
	fn native_tool_search_detected_from_unnamed_search_tool() {
		// The tool_search server tool's `name` is optional upstream.
		let mut body = toolsearch_body_base();
		body["tools"] = serde_json::json!([{"type": "tool_search_tool_regex_20250922"}]);
		assert!(parse_request(body).uses_native_tool_search());
	}

	#[test]
	fn native_tool_search_detected_from_bm25_variant() {
		let mut body = toolsearch_body_base();
		body["tools"] = serde_json::json!([
			{"type": "tool_search_tool_bm25_20251119", "name": "tool_search"}
		]);
		assert!(parse_request(body).uses_native_tool_search());
	}

	#[test]
	fn native_tool_search_detected_from_tool_reference_in_assistant_turn() {
		let body = serde_json::json!({
			"model": "claude-sonnet-4-5",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "list the widgets"},
				{"role": "assistant", "content": [
					{"type": "tool_reference", "tool_name": "mcp__example__list_widgets"}
				]}
			]
		});
		assert!(parse_request(body).uses_native_tool_search());
	}

	#[test]
	fn native_tool_search_detected_from_schema_reply_in_tool_result() {
		let body = serde_json::json!({
			"model": "claude-sonnet-4-5",
			"max_tokens": 64,
			"messages": [
				{"role": "assistant", "content": [
					{"type": "tool_reference", "tool_name": "mcp__example__list_widgets"}
				]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "toolu_1", "content": [
						{"type": "tool_reference", "tool_name": "mcp__example__list_widgets"}
					]}
				]}
			]
		});
		assert!(parse_request(body).uses_native_tool_search());
	}

	#[test]
	fn ordinary_tools_do_not_trigger_native_tool_search() {
		let mut body = toolsearch_body_base();
		body["tools"] = serde_json::json!([
			{
				"name": "get_weather",
				"description": "Get the weather",
				"input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
			},
			{"type": "web_search_20250305", "name": "web_search"}
		]);
		assert!(!parse_request(body).uses_native_tool_search());
	}

	#[test]
	fn text_only_conversation_has_no_native_tool_search() {
		assert!(!parse_request(toolsearch_body_base()).uses_native_tool_search());
	}

	#[test]
	fn downgrade_strips_tool_search_tool_and_defer_flags() {
		let mut body = toolsearch_body_base();
		body["tools"] = serde_json::json!([
			{"type": "tool_search_tool_regex_20250922", "name": "tool_search"},
			{
				"name": "get_weather",
				"description": "Get the weather",
				"input_schema": {"type": "object"},
				"defer_loading": true
			},
			{
				"name": "convert_currency",
				"description": "Convert",
				"input_schema": {"type": "object"},
				"defer_loading": true
			}
		]);
		let mut req = parse_request(body);
		assert!(req.uses_native_tool_search());
		req.downgrade_tool_search();
		assert!(
			!req.uses_native_tool_search(),
			"fully downgraded request must no longer count as native Tool Search"
		);
		let tools = req.rest["tools"].as_array().unwrap();
		assert_eq!(tools.len(), 2, "only the real tools remain: {tools:?}");
		assert!(
			tools.iter().all(|t| !t.get("defer_loading").is_some()),
			"defer_loading flags must be stripped so tools load upfront: {tools:?}"
		);
		assert!(tools[0]["name"].as_str() == Some("get_weather"));
	}

	#[test]
	fn downgrade_without_tools_array_is_a_no_op() {
		let mut req = parse_request(toolsearch_body_base());
		req.downgrade_tool_search();
		assert!(!req.uses_native_tool_search());
	}

	#[test]
	fn downgrade_translates_tool_reference_history_to_eager() {
		// 2026-10-20 production incident (Claude Desktop + non-Anthropic
		// models): Desktop's engine inserts `tool_reference` blocks on turns
		// after tool use. The downgrade must rewrite that history into its
		// eager equivalent — tool_reference plumbing removed, ordinary
		// tool_use/tool_result untouched — instead of keeping the request
		// native and failing the strict guard with 400.
		let body = serde_json::json!({
			"model": "glm-5p3",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "list the widgets"},
				{"role": "assistant", "content": [
					{"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"city": "Rome"}},
					{"type": "tool_reference", "tool_name": "mcp__example__list_widgets"}
				]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "toolu_1", "content": [
						{"type": "text", "text": "18C"},
						{"type": "tool_reference", "tool_name": "mcp__example__list_widgets"}
					]}
				]}
			],
			"tools": [
				{"type": "tool_search_tool_regex_20250922", "name": "tool_search"},
				{
					"name": "get_weather",
					"description": "Get the weather",
					"input_schema": {"type": "object"},
					"defer_loading": true
				}
			]
		});
		let mut req = parse_request(body);
		assert!(req.uses_native_tool_search());
		let report = req.downgrade_tool_search();
		assert!(
			!req.uses_native_tool_search(),
			"history rewrite must complete the downgrade"
		);
		assert_eq!(report.removed_server_tools, 1);
		assert_eq!(report.stripped_defer_loading, 1);
		assert_eq!(report.removed_tool_references, 2);
		assert_eq!(report.removed_search_rounds, 0);
		let serialized = serde_json::to_value(&req).unwrap();
		assert!(
			!serialized.to_string().contains("tool_reference"),
			"no tool_reference plumbing may survive: {serialized}"
		);
		assert!(
			serialized.to_string().contains("toolu_1"),
			"ordinary tool_use/tool_result history must survive: {serialized}"
		);
	}

	#[test]
	fn native_search_round_history_is_detected_and_downgraded() {
		// Mid-conversation model switch (native Tool Search turn on an
		// Anthropic model, then a non-Anthropic model): the replayed history
		// carries the search-round blocks. Detection must catch them and the
		// downgrade must remove them — before the 2026-10-20 fix these blocks
		// were neither detected nor removed and the upstream rejected the
		// unknown content types with 400.
		let body = serde_json::json!({
			"model": "glm-5p3",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "find a tool for that"},
				{"role": "assistant", "content": [
					{"type": "server_tool_use", "id": "srvtoolu_1", "name": "tool_search", "input": {"query": "widgets"}},
					{"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
						{
							"type": "tool_reference",
							"tool_name": "mcp__example__list_widgets",
							"description": "List widgets",
							"input_schema": {"type": "object"}
						}
					]},
					{"type": "text", "text": "found it"}
				]}
			],
			"tools": [
				{
					"name": "get_weather",
					"description": "Get the weather",
					"input_schema": {"type": "object"},
					"defer_loading": true
				}
			]
		});
		let mut req = parse_request(body);
		assert!(
			req.uses_native_tool_search(),
			"search-round blocks must count as native Tool Search"
		);
		let report = req.downgrade_tool_search();
		assert!(!req.uses_native_tool_search());
		assert_eq!(report.removed_search_rounds, 2);
		assert_eq!(
			report.hoisted_schemas, 1,
			"embedded definition must be hoisted"
		);
		let serialized = serde_json::to_value(&req).unwrap();
		assert!(
			!serialized.to_string().contains("server_tool_use")
				&& !serialized.to_string().contains("tool_search_tool_result"),
			"search-round blocks must be gone: {serialized}"
		);
		let tools = serialized["tools"].as_array().unwrap();
		assert!(
			tools
				.iter()
				.any(|t| t["name"].as_str() == Some("mcp__example__list_widgets")),
			"hoisted definition must be declared eager: {tools:?}"
		);
	}

	#[test]
	fn empty_search_round_turn_becomes_marker_text() {
		// A pure search-round assistant turn must keep its slot (message count
		// and user/assistant alternation) instead of forwarding an empty
		// content array upstream.
		let body = serde_json::json!({
			"model": "glm-5p3",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "find a tool"},
				{"role": "assistant", "content": [
					{"type": "server_tool_use", "id": "srvtoolu_2", "name": "tool_search", "input": {"query": "x"}},
					{"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_2", "content": []}
				]},
				{"role": "user", "content": "thanks"}
			]
		});
		let mut req = parse_request(body);
		assert!(req.uses_native_tool_search());
		req.downgrade_tool_search();
		assert_eq!(req.messages.len(), 3, "message count must be preserved");
		let serialized = serde_json::to_value(&req).unwrap();
		let middle = &serialized["messages"][1]["content"];
		assert_eq!(
			middle[0]["type"].as_str(),
			Some("text"),
			"emptied turn becomes a text marker: {middle}"
		);
	}

	#[test]
	fn eager_downgrade_preserves_every_tool_definition() {
		// Availability proof for the eager compatibility path: a realistic
		// Desktop/CLI shape (30 deferred MCP tools + 5 eager core tools + the
		// tool_search server tool, with a tool round and a Desktop-style
		// pre-load in history) must reach the upstream with all 35 ordinary
		// definitions byte-identical, so every deferred tool stays invocable.
		let deferred: Vec<serde_json::Value> = (0..30)
			.map(|i| {
				serde_json::json!({
					"name": format!("mcp__srv__tool_{i}"),
					"description": format!("Deferred tool {i}"),
					"input_schema": {
						"type": "object",
						"properties": {"arg_{i}": {"type": "string", "description": format!("argument {i}")}},
						"required": ["arg_{i}"]
					},
					"defer_loading": true
				})
			})
			.collect();
		let eager: Vec<serde_json::Value> = (0..5)
			.map(|i| {
				serde_json::json!({
					"name": format!("core_tool_{i}"),
					"description": format!("Core tool {i}"),
					"input_schema": {"type": "object", "properties": {"x": {"type": "number"}}}
				})
			})
			.collect();
		let mut tools = deferred.clone();
		tools.extend(eager.clone());
		tools
			.push(serde_json::json!({"type": "tool_search_tool_regex_20250922", "name": "tool_search"}));

		let body = serde_json::json!({
			"model": "glm-5p3",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "use core tool 0"},
				{"role": "assistant", "content": [
					{"type": "tool_use", "id": "toolu_1", "name": "core_tool_0", "input": {"x": 1}}
				]},
				{"role": "user", "content": [
					{"type": "tool_result", "tool_use_id": "toolu_1", "content": "ok"},
					{"type": "tool_reference", "tool_name": "mcp__srv__tool_7"}
				]},
				{"role": "user", "content": "now use deferred tool 29"}
			],
			"tools": tools,
			"tool_choice": {"type": "auto"}
		});
		let mut req = parse_request(body);
		assert!(req.uses_native_tool_search());
		let report = req.downgrade_tool_search();
		assert_eq!(report.stripped_defer_loading, 30);
		assert_eq!(report.removed_server_tools, 1);
		assert_eq!(report.removed_tool_references, 1);

		let serialized = serde_json::to_value(&req).unwrap();
		let out_tools = serialized["tools"].as_array().unwrap();
		assert_eq!(
			out_tools.len(),
			35,
			"all ordinary definitions must survive: {report:?}"
		);
		let mut expected: Vec<serde_json::Value> = deferred
			.iter()
			.map(|t| {
				let mut t = t.clone();
				t.as_object_mut().unwrap().remove("defer_loading");
				t
			})
			.collect();
		expected.extend(eager);
		let by_name =
			|v: &serde_json::Value| v.get("name").and_then(|n| n.as_str()).unwrap().to_string();
		let mut out_sorted = out_tools.clone();
		out_sorted.sort_by_key(by_name);
		let mut expected_sorted = expected;
		expected_sorted.sort_by_key(by_name);
		assert_eq!(
			out_sorted, expected_sorted,
			"surviving definitions must be byte-identical to the client's (minus defer_loading)"
		);
		assert_eq!(
			serialized["tool_choice"],
			serde_json::json!({"type": "auto"}),
			"ordinary tool_choice must be preserved"
		);
		assert!(
			serialized["messages"][2]["content"]
				.as_array()
				.unwrap()
				.iter()
				.any(|p| p["type"].as_str() == Some("tool_result")),
			"the ordinary tool round must stay in history"
		);
	}

	#[test]
	fn builtin_web_search_is_removed_by_downgrade() {
		// 2026-10-20 incident class C: Claude Desktop sends its integrated
		// `web_search` server tool alongside ordinary tools; Fireworks
		// rejects the unknown tool type with HTTP 400 in 13 ms and Desktop
		// recovered only by silently retrying without it. The downgrade must
		// remove the built-in server tool deterministically (telemetry
		// carries the count) while keeping every ordinary tool.
		let body = serde_json::json!({
			"model": "glm-5p3-flash",
			"max_tokens": 64,
			"messages": [{"role": "user", "content": "prezzo dell'oro"}],
			"tools": [
				{"type": "web_search_20250329", "name": "web_search"},
				{"name": "get_weather", "description": "Get the weather", "input_schema": {"type": "object"}},
				{"type": "code_execution_20250522", "name": "code"}
			]
		});
		let mut req = parse_request(body);
		assert!(
			req.offers_builtin_server_tools(),
			"web_search must be detected without any TS shapes"
		);
		assert!(!req.uses_native_tool_search());
		let report = req.downgrade_tool_search();
		assert_eq!(report.removed_builtin_server_tools, 2);
		assert_eq!(
			report.removed_server_tools, 0,
			"tool_search family is separate"
		);
		let serialized = serde_json::to_value(&req).unwrap();
		let tools = serialized["tools"].as_array().unwrap();
		assert_eq!(tools.len(), 1, "ordinary tool stays: {serialized}");
		assert_eq!(tools[0]["name"].as_str(), Some("get_weather"));
		assert!(
			!serialized.to_string().contains("web_search")
				&& !serialized.to_string().contains("code_execution"),
			"no built-in server tool may survive: {serialized}"
		);
	}

	#[test]
	fn hoisted_definitions_deduplicate_by_tool_name() {
		// The same deferred tool can surface in more than one search round or
		// be pre-loaded across several Desktop turns. Each occurrence embeds
		// its own copy of the definition; hoisting every copy would declare
		// the tool twice and upstreams reject duplicate tool names with 400.
		let definition = serde_json::json!({
			"type": "tool_reference",
			"tool_name": "mcp__example__list_widgets",
			"description": "List widgets",
			"input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}
		});
		let body = serde_json::json!({
			"model": "glm-5p3",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "find a tool"},
				{"role": "assistant", "content": [
					{"type": "server_tool_use", "id": "srvtoolu_1", "name": "tool_search", "input": {"query": "widgets"}},
					{"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [definition.clone()]},
					{"type": "text", "text": "found it"}
				]},
				{"role": "user", "content": "again"},
				{"role": "assistant", "content": [
					{"type": "server_tool_use", "id": "srvtoolu_2", "name": "tool_search", "input": {"query": "widgets"}},
					{"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_2", "content": [definition]},
					{"type": "text", "text": "still found"}
				]}
			],
			"tools": []
		});
		let mut req = parse_request(body);
		assert!(req.uses_native_tool_search());
		let report = req.downgrade_tool_search();
		assert_eq!(
			report.hoisted_schemas, 1,
			"the second embedded copy must not be hoisted again: {report:?}"
		);
		let serialized = serde_json::to_value(&req).unwrap();
		let names: Vec<&str> = serialized["tools"]
			.as_array()
			.unwrap()
			.iter()
			.filter_map(|t| t["name"].as_str())
			.collect();
		assert_eq!(names, vec!["mcp__example__list_widgets"], "{serialized}");
	}

	#[test]
	fn hoisted_schema_conflict_keeps_newest_reference_and_counts() {
		// A mid-session tool upgrade embeds a NEWER schema in a later history
		// reference. The newest reference wins (it is the contract the client
		// will execute against going forward), and the divergence is COUNTED —
		// never resolved silently.
		let older = serde_json::json!({
			"type": "tool_reference",
			"tool_name": "mcp__example__list_widgets",
			"description": "List widgets",
			"input_schema": {"type": "object", "properties": {"a": {"type": "string"}}}
		});
		let newer = serde_json::json!({
			"type": "tool_reference",
			"tool_name": "mcp__example__list_widgets",
			"description": "List widgets (v2, paginated)",
			"input_schema": {"type": "object", "properties": {"a": {"type": "string"}, "cursor": {"type": "string"}}}
		});
		let body = serde_json::json!({
			"model": "glm-5p3",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "find a tool"},
				{"role": "assistant", "content": [
					{"type": "server_tool_use", "id": "srvtoolu_1", "name": "tool_search", "input": {"query": "widgets"}},
					{"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [older]},
					{"type": "text", "text": "found it"}
				]},
				{"role": "user", "content": "again after the server restarted"},
				{"role": "assistant", "content": [
					{"type": "server_tool_use", "id": "srvtoolu_2", "name": "tool_search", "input": {"query": "widgets"}},
					{"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_2", "content": [newer]},
					{"type": "text", "text": "newer schema"}
				]}
			],
			"tools": []
		});
		let mut req = parse_request(body);
		let report = req.downgrade_tool_search();
		assert_eq!(report.hoisted_schemas, 1, "{report:?}");
		assert_eq!(report.hoisted_schema_conflicts, 1, "{report:?}");
		let serialized = serde_json::to_value(&req).unwrap();
		let tools = serialized["tools"].as_array().unwrap();
		assert_eq!(tools.len(), 1, "{serialized}");
		assert!(
			tools[0]["description"]
				.as_str()
				.is_some_and(|d| d.contains("v2")),
			"the NEWEST reference must win: {serialized}"
		);
	}

	#[test]
	fn hoisted_schema_conflict_against_declared_tool_keeps_declared() {
		// The request's OWN tools array is the client's current contract; a
		// stale history reference embedding an older schema must not override
		// it, and the divergence is counted.
		let stale = serde_json::json!({
			"type": "tool_reference",
			"tool_name": "get_weather",
			"description": "Get the weather (old)",
			"input_schema": {"type": "object"}
		});
		let body = serde_json::json!({
			"model": "glm-5p3",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "find a tool"},
				{"role": "assistant", "content": [
					{"type": "server_tool_use", "id": "srvtoolu_1", "name": "tool_search", "input": {"query": "weather"}},
					{"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [stale]},
					{"type": "text", "text": "found it"}
				]}
			],
			"tools": [
				{"name": "get_weather", "description": "Get the weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}
			]
		});
		let mut req = parse_request(body);
		let report = req.downgrade_tool_search();
		assert_eq!(report.hoisted_schemas, 0, "{report:?}");
		assert_eq!(report.hoisted_schema_conflicts, 1, "{report:?}");
		let serialized = serde_json::to_value(&req).unwrap();
		let tools = serialized["tools"].as_array().unwrap();
		assert_eq!(tools.len(), 1, "{serialized}");
		assert!(
			tools[0]["input_schema"]["properties"]["city"].is_object(),
			"the DECLARED definition must win: {serialized}"
		);
	}

	#[test]
	fn builtin_history_rounds_are_translated_to_neutral_text() {
		// Mixed native→third-party session: the native leg executed the real
		// web_search built-in, so history carries server_tool_use +
		// web_search_tool_result blocks. Forwarding them verbatim invites the
		// model to emit another server_tool_use it cannot honor (a search that
		// looks successful but never runs). The downgrade rewrites both into
		// bounded text context, keeping the search's meaning without the
		// executable-looking blocks.
		let body = serde_json::json!({
			"model": "glm-5p3",
			"max_tokens": 64,
			"messages": [
				{"role": "user", "content": "prezzo dell'oro"},
				{"role": "assistant", "content": [
					{"type": "server_tool_use", "id": "srvtoolu_ws", "name": "web_search", "input": {"query": "gold price"}},
					{"type": "web_search_tool_result", "tool_use_id": "srvtoolu_ws", "content": [
						{"type": "web_search_result", "title": "Gold spot price", "url": "https://example.com/gold"},
						{"type": "web_search_result", "title": "Metals dashboard", "url": "https://example.com/metals"}
					]},
					{"type": "text", "text": "Gold is trading at $2,700/oz."}
				]},
				{"role": "user", "content": "e ora?"}
			],
			"tools": []
		});
		let mut req = parse_request(body);
		let report = req.downgrade_tool_search();
		assert_eq!(report.translated_builtin_rounds, 2, "{report:?}");
		let serialized = serde_json::to_value(&req).unwrap();
		let history = serialized.to_string();
		assert!(
			!history.contains("\"server_tool_use\"") && !history.contains("\"web_search_tool_result\""),
			"no executable-looking server tool blocks may survive (the neutral text marker may name what it replaced): {serialized}"
		);
		assert!(
			history.contains("gold price") && history.contains("example.com/gold"),
			"the invocation and its results must stay as text context: {serialized}"
		);
		assert!(
			history.contains("cannot run built-in server tools"),
			"the marker must state plainly the tool is unavailable: {serialized}"
		);
		// The assistant text and the user turn stay intact.
		assert!(history.contains("$2,700/oz."), "{serialized}");
	}

	#[test]
	fn downgrade_neutralizes_tool_choice_pinned_to_server_tool() {
		let mut body = toolsearch_body_base();
		body["tools"] = serde_json::json!([
			{"type": "tool_search_tool_regex_20250922", "name": "tool_search"},
			{"name": "get_weather", "description": "Get the weather", "input_schema": {"type": "object"}}
		]);
		body["tool_choice"] = serde_json::json!({"type": "tool", "name": "tool_search"});
		let mut req = parse_request(body);
		let report = req.downgrade_tool_search();
		assert!(report.neutralized_tool_choice);
		let serialized = serde_json::to_value(&req).unwrap();
		assert!(serialized.get("tool_choice").is_none(), "{serialized}");
		assert!(
			serialized["tools"].as_array().unwrap().len() == 1,
			"the real tool stays: {serialized}"
		);
	}

	#[test]
	fn downgrade_drops_tools_when_only_the_server_tool_remained() {
		let mut body = toolsearch_body_base();
		body["tools"] =
			serde_json::json!([{"type": "tool_search_tool_regex_20250922", "name": "tool_search"}]);
		body["tool_choice"] = serde_json::json!({"type": "auto"});
		let mut req = parse_request(body);
		assert!(req.uses_native_tool_search());
		req.downgrade_tool_search();
		let serialized = serde_json::to_value(&req).unwrap();
		assert!(
			serialized.get("tools").is_none(),
			"empty tools array is dropped entirely: {serialized}"
		);
		assert!(serialized.get("tool_choice").is_none(), "{serialized}");
	}

	#[test]
	fn search_round_blocks_are_detected_but_ordinary_server_tools_are_not() {
		let mut body = toolsearch_body_base();
		body["messages"] = serde_json::json!([
			{"role": "user", "content": "search"},
			{"role": "assistant", "content": [
				{"type": "server_tool_use", "id": "srvtoolu_3", "name": "web_search", "input": {"query": "x"}},
				{"type": "web_search_tool_result", "tool_use_id": "srvtoolu_3", "content": []}
			]}
		]);
		assert!(
			!parse_request(body.clone()).uses_native_tool_search(),
			"non-tool-search server tools are not deferred-tool semantics"
		);
		body["messages"][1]["content"][0]["name"] = serde_json::json!("tool_search");
		assert!(
			parse_request(body).uses_native_tool_search(),
			"a tool_search server_tool_use must be detected"
		);
	}
}
