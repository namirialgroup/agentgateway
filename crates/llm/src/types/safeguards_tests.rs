use serde_json::{Value, json};

use super::*;

fn request_rest(context: Value) -> Value {
	json!({
		"safeguards": [{"type": "dangerous_tool_use", "classifier_context": context}],
		"tools": [{"name": "Bash", "input_schema": {"type": "object"}}]
	})
}

#[test]
fn detects_dangerous_tool_use_entry_and_sanitizes_context() {
	let rest = request_rest(json!({
		"v": 1,
		"permission_mode": "auto",
		"platform": "darwin",
		"live_cwd": "/work/repo",
		"home_dir": "/Users/dev",
		"trusted_directories": ["/work/repo"],
		"rules": {"allow": ["Bash(git:*)"], "deny": []},
		"auto_mode": true,
		"git_state": {"cwd": "/work/repo", "root": "/work/repo", "branch": "main",
									"default_branch": "main", "visibility": "private", "remote_url": "git@x"},
		"future_field": "dropped"
	}));
	let found = requested_dangerous_tool_use(&rest).expect("entry detected");
	let context = found.classifier_context.as_object().expect("object");
	assert_eq!(context.get("permission_mode"), Some(&json!("auto")));
	assert_eq!(
		context.get("trusted_directories"),
		Some(&json!(["/work/repo"]))
	);
	assert!(context.get("v").is_none(), "undocumented keys are dropped");
	assert!(context.get("future_field").is_none());
	let git = context
		.get("git_state")
		.and_then(Value::as_object)
		.expect("git_state");
	assert_eq!(git.get("branch"), Some(&json!("main")));
	assert!(
		git.get("remote_url").is_none(),
		"git_state is reduced to documented keys"
	);
}

#[test]
fn ignores_requests_without_or_with_malformed_safeguards() {
	assert!(requested_dangerous_tool_use(&json!({})).is_none());
	assert!(requested_dangerous_tool_use(&Value::Null).is_none());
	assert!(requested_dangerous_tool_use(&json!({"safeguards": "nope"})).is_none());
	assert!(requested_dangerous_tool_use(&json!({"safeguards": [{"type": "other"}]})).is_none());
	// Entry without a context is still a request for review.
	let found =
		requested_dangerous_tool_use(&json!({"safeguards": [{"type": "dangerous_tool_use"}]}))
			.expect("detected");
	assert_eq!(found.classifier_context, Value::Null);
}

#[test]
fn remove_safeguards_strips_only_that_field() {
	let mut rest = request_rest(json!({"permission_mode": "auto"}));
	assert!(remove_safeguards(&mut rest));
	assert!(rest.get("safeguards").is_none());
	assert!(rest.get("tools").is_some(), "other unknown fields survive");
	assert!(!remove_safeguards(&mut rest));
	let mut null = Value::Null;
	assert!(!remove_safeguards(&mut null));
}

#[test]
fn beta_header_strip_keeps_other_betas_and_header_order() {
	let mut headers = ::http::HeaderMap::new();
	headers.append(
		"anthropic-beta",
		"interleaved-thinking-2025-05-14,dangerous-tool-use-2026-09-03"
			.parse()
			.unwrap(),
	);
	headers.append(
		"anthropic-beta",
		"dangerous-tool-use-2026-09-03".parse().unwrap(),
	);
	headers.append("anthropic-beta", "context-1m-2025-08-07".parse().unwrap());
	assert!(has_dangerous_tool_use_beta(&headers));
	assert!(strip_dangerous_tool_use_beta(&mut headers));
	let values: Vec<&str> = headers
		.get_all("anthropic-beta")
		.iter()
		.map(|v| v.to_str().unwrap())
		.collect();
	assert_eq!(
		values,
		vec!["interleaved-thinking-2025-05-14", "context-1m-2025-08-07"]
	);
	assert!(!has_dangerous_tool_use_beta(&headers));
	assert!(!strip_dangerous_tool_use_beta(&mut headers), "idempotent");
}

#[test]
fn result_json_shapes_match_protocol() {
	let mut tool_uses = BTreeMap::new();
	tool_uses.insert("toolu_01".to_string(), ToolUseVerdict::not_flagged());
	tool_uses.insert(
		"toolu_02".to_string(),
		ToolUseVerdict::flagged("[Destructive operation] deletes the repository"),
	);
	tool_uses.insert(
		"toolu_03".to_string(),
		ToolUseVerdict::unavailable(UnavailableReason::Truncated),
	);
	let available = serde_json::to_value(results_available(tool_uses)).unwrap();
	assert_eq!(
		available,
		json!([{
			"type": "dangerous_tool_use",
			"status": {
				"type": "available",
				"tool_uses": {
					"toolu_01": {"type": "evaluated", "outcome": "not_flagged"},
					"toolu_02": {"type": "evaluated", "outcome": "flagged",
											 "explanation": "[Destructive operation] deletes the repository"},
					"toolu_03": {"type": "unavailable", "reason": "truncated"}
				}
			}
		}])
	);
	let unsupported = serde_json::to_value(results_unsupported()).unwrap();
	assert_eq!(
		unsupported,
		json!([{"type": "dangerous_tool_use", "status": {"type": "unsupported"}}])
	);
	let empty = serde_json::to_value(results_available(BTreeMap::new())).unwrap();
	assert_eq!(
		empty,
		json!([{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": {}}}])
	);
	for reason in [
		UnavailableReason::Timeout,
		UnavailableReason::Error,
		UnavailableReason::Refused,
		UnavailableReason::InputTooLong,
		UnavailableReason::Truncated,
	] {
		assert_eq!(
			serde_json::to_value(reason).unwrap(),
			json!(reason.as_str())
		);
	}
}

#[test]
fn results_round_trip_through_serde() {
	let mut tool_uses = BTreeMap::new();
	tool_uses.insert("a".to_string(), ToolUseVerdict::flagged("why"));
	let results = results_available(tool_uses);
	let text = serde_json::to_string(&results).unwrap();
	let back: SafeguardResults = serde_json::from_str(&text).unwrap();
	assert_eq!(back, results);
}

#[test]
fn buffered_collection_marks_last_block_truncated_on_max_tokens() {
	let response = json!({
		"content": [
			{"type": "text", "text": "Running"},
			{"type": "tool_use", "id": "toolu_a", "name": "Bash", "input": {"command": "ls"}},
			{"type": "tool_use", "id": "toolu_b", "name": "Write", "input": {"file_path": "/x"}}
		],
		"stop_reason": "max_tokens"
	});
	let pending = pending_tool_uses_from_response(&response);
	assert_eq!(
		pending,
		vec![
			PendingToolUse::Complete {
				id: "toolu_a".into(),
				name: "Bash".into(),
				input: json!({"command": "ls"})
			},
			PendingToolUse::Truncated {
				id: "toolu_b".into()
			},
		]
	);
	let complete = json!({"content": [
		{"type": "tool_use", "id": "toolu_a", "name": "Bash", "input": {"command": "ls"}}
	], "stop_reason": "tool_use"});
	assert!(matches!(
		pending_tool_uses_from_response(&complete)[..],
		[PendingToolUse::Complete { .. }]
	));
	assert!(
		pending_tool_uses_from_response(&json!({"content": [{"type": "text", "text": "hi"}]}))
			.is_empty()
	);
	assert!(pending_tool_uses_from_response(&json!({"id": "msg"})).is_empty());
}

#[test]
fn stamping_places_results_at_top_level_and_in_message_delta() {
	let results = results_available(BTreeMap::new());
	let mut response = json!({"id": "msg_1", "content": []});
	assert!(stamp_safeguard_results(&mut response, &results));
	assert_eq!(
		response["safeguard_results"][0]["status"]["type"],
		json!("available")
	);
	assert!(response_has_safeguard_results(&response));
	assert!(!response_has_safeguard_results(
		&json!({"safeguard_results": null})
	));

	let mut event = json!({"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": null},
	                       "usage": {"output_tokens": 12}});
	assert!(stamp_message_delta(&mut event, &results));
	assert_eq!(event["delta"]["stop_reason"], json!("tool_use"));
	assert_eq!(
		event["delta"]["safeguard_results"][0]["type"],
		json!("dangerous_tool_use")
	);
	assert_eq!(event["usage"]["output_tokens"], json!(12));
}

#[test]
fn message_delta_inspection() {
	assert_eq!(
		inspect_message_delta(&json!({"type": "content_block_stop", "index": 0})),
		MessageDeltaCheck::PassThrough
	);
	assert_eq!(
		inspect_message_delta(&json!({"type": "message_delta", "delta": {"stop_reason": null}})),
		MessageDeltaCheck::PassThrough
	);
	assert_eq!(
		inspect_message_delta(&json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}})),
		MessageDeltaCheck::NeedsResults {
			hit_max_tokens: false
		}
	);
	assert_eq!(
		inspect_message_delta(
			&json!({"type": "message_delta", "delta": {"stop_reason": "max_tokens"}})
		),
		MessageDeltaCheck::NeedsResults {
			hit_max_tokens: true
		}
	);
	assert_eq!(
		inspect_message_delta(&json!({"type": "message_delta", "delta": {
			"stop_reason": "end_turn",
			"safeguard_results": [{"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": {}}}]
		}})),
		MessageDeltaCheck::AlreadyAnswered
	);
}

#[test]
fn streamed_tool_uses_reassemble_inputs_and_preserve_ids() {
	let mut streamed = StreamedToolUses::default();
	for event in [
		json!({"type": "message_start", "message": {"id": "msg_1"}}),
		json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
		json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "I will"}}),
		json!({"type": "content_block_stop", "index": 0}),
		json!({"type": "content_block_start", "index": 1,
		       "content_block": {"type": "tool_use", "id": "call_501f1ce0", "name": "Bash", "input": {}}}),
		json!({"type": "content_block_delta", "index": 1,
		       "delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"rm -rf"}}),
		json!({"type": "content_block_delta", "index": 1,
		       "delta": {"type": "input_json_delta", "partial_json": " /tmp/build\"}"}}),
		json!({"type": "content_block_stop", "index": 1}),
		json!({"type": "content_block_start", "index": 2,
		       "content_block": {"type": "tool_use", "id": "toolu_prefilled", "name": "Read",
		                         "input": {"file_path": "/work/README.md"}}}),
		json!({"type": "content_block_stop", "index": 2}),
	] {
		streamed.observe(&event);
	}
	assert_eq!(streamed.len(), 2);
	let pending = streamed.finish(false);
	assert_eq!(
		pending,
		vec![
			PendingToolUse::Complete {
				id: "call_501f1ce0".into(),
				name: "Bash".into(),
				input: json!({"command": "rm -rf /tmp/build"})
			},
			PendingToolUse::Complete {
				id: "toolu_prefilled".into(),
				name: "Read".into(),
				input: json!({"file_path": "/work/README.md"})
			},
		]
	);
	// max_tokens cuts the LAST content block, whatever its type.
	let cut = streamed.finish(true);
	assert_eq!(
		cut[1],
		PendingToolUse::Truncated {
			id: "toolu_prefilled".into()
		}
	);
	assert!(matches!(cut[0], PendingToolUse::Complete { .. }));
}

#[test]
fn streamed_tool_use_with_unparseable_json_is_truncated_not_classified() {
	let mut streamed = StreamedToolUses::default();
	streamed.observe(&json!({"type": "content_block_start", "index": 0,
		"content_block": {"type": "tool_use", "id": "toolu_x", "name": "Bash", "input": {}}}));
	streamed.observe(&json!({"type": "content_block_delta", "index": 0,
		"delta": {"type": "input_json_delta", "partial_json": "{\"command\": \"ec"}}));
	assert_eq!(
		streamed.finish(false),
		vec![PendingToolUse::Truncated {
			id: "toolu_x".into()
		}]
	);
	// Deltas for unknown indexes or non-input types are ignored.
	streamed.observe(&json!({"type": "content_block_delta", "index": 7,
		"delta": {"type": "input_json_delta", "partial_json": "{}"}}));
	streamed.observe(&json!({"type": "content_block_delta", "index": 0,
		"delta": {"type": "text_delta", "text": "ho\"}"}}));
	assert_eq!(streamed.len(), 1);
}

#[test]
fn all_unavailable_results_keep_truncated_reason() {
	let pending = vec![
		PendingToolUse::Complete {
			id: "a".into(),
			name: "Bash".into(),
			input: json!({}),
		},
		PendingToolUse::Truncated { id: "b".into() },
	];
	let results = results_all_unavailable(&pending, UnavailableReason::Timeout);
	let SafeguardStatus::Available { tool_uses } = &results[0].status else {
		panic!("available expected");
	};
	assert_eq!(
		tool_uses["a"],
		ToolUseVerdict::unavailable(UnavailableReason::Timeout)
	);
	assert_eq!(
		tool_uses["b"],
		ToolUseVerdict::unavailable(UnavailableReason::Truncated)
	);
}

#[test]
fn transcript_renders_roles_tool_calls_and_excerpts_but_not_thinking() {
	let messages = json!([
		{"role": "user", "content": "please clean the build dir"},
		{"role": "assistant", "content": [
			{"type": "thinking", "thinking": "secret reasoning"},
			{"type": "text", "text": "Sure."},
			{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls dist"}}
		]},
		{"role": "user", "content": [
			{"type": "tool_result", "tool_use_id": "t1", "content": [{"type": "text", "text": "x".repeat(900)}]}
		]},
		{"role": "system", "content": "ignored"}
	]);
	let (lines, truncated) = render_transcript(&messages);
	assert!(!truncated);
	assert_eq!(lines.len(), 4);
	assert_eq!(lines[0], r#"{"user":"please clean the build dir"}"#);
	assert_eq!(lines[1], r#"{"assistant":"Sure."}"#);
	assert_eq!(
		serde_json::from_str::<Value>(&lines[2]).unwrap(),
		json!({"assistant_tool_call": {"name": "Bash", "input": {"command": "ls dist"}}})
	);
	let result_line: Value = serde_json::from_str(&lines[3]).unwrap();
	let excerpt = result_line["tool_result"].as_str().unwrap();
	assert!(excerpt.ends_with(" [truncated]"));
	assert_eq!(excerpt.len(), TOOL_RESULT_CHAR_LIMIT + " [truncated]".len());
	assert!(!lines.iter().any(|l| l.contains("secret reasoning")));
}

#[test]
fn transcript_budget_keeps_newest_lines_and_marks_omission() {
	let long = "y".repeat(TRANSCRIPT_LINE_CHAR_LIMIT + 10);
	let mut messages = vec![json!({"role": "user", "content": "opening request"})];
	for _ in 0..40 {
		messages.push(json!({"role": "assistant", "content": long}));
	}
	messages.push(json!({"role": "assistant", "content": "newest line"}));
	let (lines, truncated) = render_transcript(&Value::Array(messages));
	assert!(truncated);
	assert_eq!(lines[0], r#"{"user":"opening request"}"#);
	assert_eq!(lines[1], OMISSION_MARKER);
	assert_eq!(lines.last().unwrap(), r#"{"assistant":"newest line"}"#);
	let total: usize = lines.iter().map(|l| l.len() + 1).sum();
	assert!(total <= TRANSCRIPT_CHAR_BUDGET, "total {total}");
	// Every long line is clipped to the per-line limit plus the marker.
	assert!(lines[2].len() <= TRANSCRIPT_LINE_CHAR_LIMIT + 32);
}

#[test]
fn review_plan_debug_is_redacted_and_exposes_trusted_dirs() {
	let rest = request_rest(json!({
		"permission_mode": "auto",
		"live_cwd": "/work/repo",
		"trusted_directories": ["/work/other"]
	}));
	let safeguards = requested_dangerous_tool_use(&rest).unwrap();
	let plan = SafeguardsReviewPlan::from_request(
		"zai.glm-5.3",
		&json!([{"role": "user", "content": "delete the SECRET_TOKEN=abc file"}]),
		&safeguards,
	);
	let debug = format!("{plan:?}");
	assert!(
		!debug.contains("SECRET_TOKEN"),
		"debug must not leak content: {debug}"
	);
	assert!(debug.contains("transcript_lines: 1"));
	assert_eq!(
		plan.trusted_directories(),
		vec!["/work/other", "/work/repo"]
	);
	assert_eq!(plan.permission_mode(), Some("auto"));
	assert!(
		plan.transcript_text().contains("SECRET_TOKEN"),
		"content stays available to the engine"
	);
}

/// The object-form `trusted_directories` the real Claude Code 2.1.304 client
/// sends (captured 2026-10-20 in the isolated container E2E; synthetic /work
/// paths). The parser must accept it next to the plain-array form: dropping
/// `additional` roots would wrongly treat trusted paths as outside-trusted.
#[test]
fn trusted_directories_object_form_from_real_client() {
	let rest = request_rest(json!({
		"permission_mode": "auto",
		"platform": "linux",
		"live_cwd": "/work",
		"home_dir": "/root",
		"trusted_directories": {
			"primary": {"path": "/work", "resolved": ["/work"]},
			"additional": [{"path": "/work/extra", "resolved": ["/work/extra"]}],
			"network": [],
			"block_reads_outside_working_directories": false
		}
	}));
	let safeguards = requested_dangerous_tool_use(&rest).unwrap();
	let plan = SafeguardsReviewPlan::from_request(
		"claude-sonnet-5-5",
		&json!([{"role": "user", "content": "list files"}]),
		&safeguards,
	);
	assert_eq!(
		plan.trusted_directories(),
		vec!["/work", "/work/extra"],
		"primary plus additional roots, deduplicated, live_cwd not duplicated"
	);

	// Legacy array form still works, and a string-form additional entry too.
	let rest = request_rest(json!({
		"live_cwd": "/a",
		"trusted_directories": {"primary": "/b", "additional": ["/c"]}
	}));
	let safeguards = requested_dangerous_tool_use(&rest).unwrap();
	let plan = SafeguardsReviewPlan::from_request(
		"claude-sonnet-5-5",
		&json!([{"role": "user", "content": "x"}]),
		&safeguards,
	);
	assert_eq!(plan.trusted_directories(), vec!["/b", "/c", "/a"]);
}

#[test]
fn clipping_respects_char_boundaries() {
	let text = "héllo wörld";
	let out = clipped(text, 2);
	assert!(out.starts_with('h'));
	assert!(out.ends_with(" [truncated]"));
	assert_eq!(clipped("short", 10), "short");
}
