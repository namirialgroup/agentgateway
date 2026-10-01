# Synapse AgentGateway usage-conformance gate

## Why this exists

Production defect (proven 2026-10-01, live wire evidence): Anthropic-compatible
upstreams (verified against the Fireworks `/inference/v1/messages` endpoint)
send an **all-zero usage placeholder** in `message_start` and the real
cumulative usage only on the final `message_delta`. Unpatched AgentGateway
materialized the placeholder as final usage evidence, producing
succeeded-but-zero-token provider attempts that downstream accounting settled
as $0.

The fix (streaming placeholder guard + buffered cross-format native-usage
overlay) lives in this fork and is proposed upstream in
agentgateway/agentgateway#3740 (issue #3739). Until a release containing an
equivalent fix is adopted, **every future AgentGateway upgrade candidate must
pass this gate before its image is built and pinned in the Synapse infra
repo** (`agentgateway-parameters.yaml` `image.tag`).

## Promotion invariant

```
AgentGateway upgrade candidate
        |
Anthropic Messages usage conformance
  scripts/synapse-usage-conformance-gate.sh <candidate>
        |
       PASS
        |
eligible for promotion (image build + infra tag bump)
```

No upgrade may silently regress to:
`missing provider usage -> explicit 0 -> fake settled $0`.

## What the gate tests

| # | Invariant | Where |
|---|-----------|-------|
| 1 | Real `message_delta` usage overrides the all-zero `message_start` placeholder | Tier 1 (hard) |
| 2 | Placeholder-only stream ends with **unknown** usage — never explicit zeros | Tier 1 (hard) |
| 3 | `cache_read` / `cache_creation` evidence survives extraction | Tier 1 (hard) |
| 4 | Normal non-zero usage unaffected | Tier 1 (hard) |
| 5 | Buffered cross-format: missing upstream-native usage stays unknown, never `Some(0)` | Tier 2 (mechanism) |

Tier 2 is advisory when a candidate implements the fix with different
internals (the harness references this fork's overlay helpers); Tier 1 is the
non-negotiable semantic floor.

## Usage

```bash
# candidate = upstream checkout, release tag ref of this repo, or a patched build
scripts/synapse-usage-conformance-gate.sh /path/to/agentgateway-checkout
scripts/synapse-usage-conformance-gate.sh v1.6.0-alpha.2
scripts/synapse-usage-conformance-gate.sh "$(git rev-parse HEAD)"
```

The gate grafts the golden test modules into a disposable candidate tree
(never mutates the candidate checkout), reuses one cargo target dir
(`AGW_GATE_TARGET_DIR` to pin it), and prints
`AGENTGATEWAY_USAGE_CONFORMANCE = PASS|FAIL` with a promotion verdict.

Known upstream quirk (pre-existing on agentgateway main as of 2026-10):
`cargo test -p agentgateway --lib` fails a compile-time `SizeAtMost` future
-size assertion unrelated to usage handling; the gate runs Tier 2 with
`--features assert_size_runtime` to execute the suite regardless. Tier 1 is
unaffected.

## Retirement path

When upstream ships a release containing a semantically equivalent fix
(candidate for that: PR #3740), the gate still runs — it is the acceptance
test for that upgrade. The local patch can be dropped only after:
1. the upstream release passes this gate, and
2. the patched image has been replaced by the upstream image in
   `agentgateway-parameters.yaml`, and
3. a post-rollout survey shows zero `succeeded zero-token usage_complete=true`
   rows on the new image.
