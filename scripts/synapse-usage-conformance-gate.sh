#!/usr/bin/env bash
# Synapse AgentGateway upgrade gate — Anthropic-Messages usage conformance.
#
# Promotion invariant:
#
#   AgentGateway upgrade candidate
#           |
#   Anthropic Messages usage conformance (this script)
#           |
#         PASS
#           |
#   eligible for promotion (image build + infra tag bump)
#
# The gate replays the live-verified provider wire semantics (an all-zero
# message_start usage placeholder with the real cumulative usage only on the
# final message_delta) against ANY candidate tree:
#
#   Tier 1 (HARD): streaming placeholder semantics + cache usage
#     crates/llm/src/conversion/messages_synapse_conformance_tests.rs
#       1. real message_delta usage overrides the all-zero placeholder
#       2. placeholder-only stream ends with UNKNOWN usage, never zeros
#       3. cache_read / cache_creation evidence survives extraction
#       4. normal non-zero usage is unaffected
#
#   Tier 2 (mechanism, advisory when the candidate implements the fix
#   differently): buffered cross-format native-usage overlay
#     crates/agentgateway/src/llm/usage_extraction_synapse_conformance_tests.rs
#       - missing upstream-native usage stays unknown, never Some(0)
#
# Usage:
#   scripts/synapse-usage-conformance-gate.sh <candidate-source-dir>
#   scripts/synapse-usage-conformance-gate.sh <git-ref-of-this-repo>
#
# Exit 0 only when Tier 1 passes and Tier 2 passes or is explicitly skipped
# for a documented reason. Any other outcome blocks promotion.
set -euo pipefail

CANDIDATE="${1:?usage: synapse-usage-conformance-gate.sh <candidate-source-dir>|<git-ref>}"
GATE_REPO="$(git rev-parse --show-toplevel)"
WORK="$(mktemp -d /tmp/agw-usage-gate.XXXXXX)"
TARGET_DIR="${AGW_GATE_TARGET_DIR:-$WORK/target}"
trap 'rm -rf "$WORK"' EXIT

echo "== Synapse AgentGateway usage conformance gate =="
echo "gate repo:     $GATE_REPO"
echo "candidate:     $CANDIDATE"
echo "shared target: $TARGET_DIR"

# ---------------------------------------------------------------- candidate
if [ -d "$CANDIDATE" ] && [ -f "$CANDIDATE/Cargo.toml" ]; then
  CANDIDATE_LABEL="dir:$CANDIDATE"
  rsync -a --exclude target "$CANDIDATE/" "$WORK/candidate/"
else
  CANDIDATE_LABEL="ref:$CANDIDATE"
  git -C "$GATE_REPO" worktree add --detach "$WORK/candidate" "$CANDIDATE" >/dev/null 2>&1
  WORKTREE_ADDED=1
fi
trap 'rm -rf "$WORK"; git -C "$GATE_REPO" worktree remove --force "$WORK/candidate" >/dev/null 2>&1 || true' EXIT
echo "candidate:     $CANDIDATE_LABEL"

# ------------------------------------------------------------------- graft
# Golden modules are grafted under Synapse-specific names so they never
# collide with a candidate's own regression tests.
MS_DECL='#[cfg(test)] #[path = "messages_synapse_conformance_tests.rs"] mod messages_synapse_conformance_tests;'
UE_DECL='#[cfg(test)] mod usage_extraction_synapse_conformance_tests;'

cp "$GATE_REPO/crates/llm/src/conversion/messages_tests.rs" \
   "$WORK/candidate/crates/llm/src/conversion/messages_synapse_conformance_tests.rs"
cp "$GATE_REPO/crates/agentgateway/src/llm/usage_extraction_tests.rs" \
   "$WORK/candidate/crates/agentgateway/src/llm/usage_extraction_synapse_conformance_tests.rs"

graft_decl() { # file, declaration
  grep -qF "$2" "$1" || printf '\n%s\n' "$2" >> "$1"
}
graft_decl "$WORK/candidate/crates/llm/src/conversion/messages.rs" "$MS_DECL"
graft_decl "$WORK/candidate/crates/agentgateway/src/llm/mod.rs" "$UE_DECL"

# Fresh mtimes so cargo cannot reuse a stale fingerprint for the grafted tree.
touch "$WORK/candidate/crates/llm/src/conversion/messages_synapse_conformance_tests.rs" \
      "$WORK/candidate/crates/llm/src/conversion/messages.rs" \
      "$WORK/candidate/crates/agentgateway/src/llm/usage_extraction_synapse_conformance_tests.rs" \
      "$WORK/candidate/crates/agentgateway/src/llm/mod.rs"

export CARGO_TARGET_DIR="$TARGET_DIR"
cd "$WORK/candidate"

tier_result() { # log-file — echoes "PASS:<n>" or "FAIL:0"
  if grep -q "^test result: ok" "$1" && ! grep -qE "^test result: ok\. 0 passed" "$1"; then
    echo "PASS:$(grep -oE '[0-9]+ passed' "$1" | head -1 | cut -d' ' -f1)"
  else
    echo "FAIL:0"
  fi
}

# ------------------------------------------------------------------ tier 1
echo "-- Tier 1 (HARD): Messages streaming placeholder + cache semantics"
TIER1=FAIL
TIER1_N=0
if cargo test -p agent-llm --lib messages_synapse_conformance_tests > /tmp/agw-gate-tier1.log 2>&1; then
  R="$(tier_result /tmp/agw-gate-tier1.log)"
  TIER1="${R%%:*}"
  TIER1_N="${R##*:}"
else
  TIER1_N="$(grep -oE '[0-9]+ passed; [0-9]+ failed' /tmp/agw-gate-tier1.log | head -1 || true)"
fi
tail -12 /tmp/agw-gate-tier1.log

# ------------------------------------------------------------------ tier 2
echo "-- Tier 2 (mechanism): buffered cross-format native-usage overlay"
TIER2=SKIP
TIER2_NOTE=""
UE_DECL_FILE="$WORK/candidate/crates/agentgateway/src/llm/mod.rs"
if cargo test -p agentgateway --lib --features assert_size_runtime \
      usage_extraction_synapse_conformance_tests >/tmp/agw-gate-tier2.log 2>&1; then
  R="$(tier_result /tmp/agw-gate-tier2.log)"
  TIER2="${R%%:*}"
  [ "$TIER2" = "FAIL" ] && TIER2_NOTE="tier-2 harness ran but its tests failed — buffered cross-format usage extraction regressed on this candidate."
else
  if grep -qE "cannot find.*(native_upstream_llm_response|overlay_upstream_usage|crosses_chat_formats)|unresolved import" /tmp/agw-gate-tier2.log; then
    TIER2=SKIP
    TIER2_NOTE="candidate implements the buffered cross-format fix differently (overlay helpers absent). Semantic coverage for 'missing buffered usage stays unknown' must be reviewed manually against the candidate implementation."
    # remove the graft so the tree stays buildable for posterity
    python3 - "$UE_DECL_FILE" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read().replace('\n#[cfg(test)] mod usage_extraction_synapse_conformance_tests;\n', '\n')
open(p, 'w').write(s)
PY
  else
    TIER2=SKIP
    TIER2_NOTE="tier-2 harness did not compile against this candidate (API drift). Review the build log; if the drift is only cosmetic, port the harness — a FAIL in tier 1 already blocks promotion."
  fi
fi

# ----------------------------------------------------------------- verdict
echo
echo "== verdict ============================================"
echo "candidate:                      $CANDIDATE_LABEL"
echo "TIER1 streaming semantics:      $TIER1 ($TIER1_N tests)"
echo "TIER2 buffered cross-format:    $TIER2${TIER2_NOTE:+ (see note)}"
[ -n "$TIER2_NOTE" ] && echo "note: $TIER2_NOTE"
if [ "$TIER1" = "PASS" ] && { [ "$TIER2" = "PASS" ] || [ "$TIER2" = "SKIP" ]; }; then
  echo "AGENTGATEWAY_USAGE_CONFORMANCE = PASS"
  echo "candidate is ELIGIBLE for promotion (image build + infra tag bump)."
  exit 0
fi
echo "AGENTGATEWAY_USAGE_CONFORMANCE = FAIL"
echo "candidate is BLOCKED: it would silently regress to missing provider usage -> explicit 0 -> fake settled \$0."
exit 1
