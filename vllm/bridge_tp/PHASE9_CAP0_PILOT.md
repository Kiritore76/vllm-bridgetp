# BridgeTP Phase 9 CAP-0 Engineering Pilot

## Status and scope

CAP-0 is a **single-anchor engineering reachability pilot** based on commit
`faf16b5adf9321b7df139c3eaf0f537f190fdf6e`.  It answers three implementation
questions before a formal capacity experiment is attempted:

1. Can one named TP1 anchor publish/migrate while other requests share its
   scheduler batch?
2. Can current TP1 KV headroom trigger Shadow only after the TP4 target becomes
   admissible?
3. If pressure clears before cutover, can staging be drained while the user
   request continues on TP1?

It does **not** implement or validate multi-candidate scheduling, calibrated
OOM probability, target KV reservation, exact copy-backlog tracking, or a
formal E-series result.  Do not cite CAP-0 latency/goodput as paper evidence.

## What this patch changes

- `BRIDGETP_STREAM_SOURCE_REQUEST_ID_PREFIX` selects one explicit anchor from a
  multi-request source batch.  When unset, the original Phase 6-8 rule remains:
  a publisher runs only for a one-request batch.
- `/bridge_tp/v1/cleanup` accepts `abort_source=false`.  The default remains
  `true` for the Phase 8 cleanup experiment; Phase 9 policy abandonment uses
  `false` so ownership stays on TP1.
- The source delta mirror becomes terminal before its workers are drained, so
  continued TP1 decoding cannot enqueue deltas into stopped queues.
- `CapacityHeadroomTracker` records free KV tokens, an EWMA decline rate, and
  time-to-guard with enter/clear hysteresis.  This is a deterministic signal,
  not `p_oom` or `p_cap`.
- Every armed migration records one trigger path.  CAP-0 uses
  `CAPACITY_PILOT`; the old policy paths remain distinguishable.
- The background-load schedule runs in a separate process.  The controller
  receives only current metrics and never receives future arrivals.

## Preflight: identify the real server baseline

Do not start from the older `d3-phase9-greedy-contract` branch.  First run:

```bash
git -C /root/autodl-tmp/bridgetp/vllm_bridge status --short --branch
git -C /root/autodl-tmp/bridgetp/vllm_bridge branch --show-current
git -C /root/autodl-tmp/bridgetp/vllm_bridge rev-parse HEAD
git -C /root/autodl-tmp/bridgetp/vllm_bridge log -5 --oneline
```

Preserve any server-only changes before applying this package.  The expected
public base is `faf16b5`; if the server is newer, inspect the patch rather than
forcing the branch backward.

Run the pure regression set after applying the patch:

```bash
python -m unittest \
  tests.bridge_tp.test_phase9_capacity_pilot \
  tests.bridge_tp.test_phase9_predictor_policy \
  tests.bridge_tp.test_phase9_state_and_proxy \
  tests.bridge_tp.test_phase9_online_integration \
  tests.bridge_tp.test_phase9_d3_batch
```

## Calibrate a guard before enabling migration

The checked-in controller template deliberately contains zero for three
machine-specific fields and must fail closed until they are replaced:

- `tp1_total_kv_blocks`
- `tp4_total_kv_blocks`
- `capacity_pilot.guard_free_kv_tokens`

Obtain total block counts from the exact TP1/TP4 startup configuration used by
the pilot.  Do not substitute `max_num_batched_tokens`; it is not a KV reserve.

Use at least three **no-migration calibration repetitions** of the same load
shape.  From the controller/metrics time series, locate the free-KV level at
the first preemption (or the closest approach if no preemption occurs), choose
a conservative guard before that point, document the rule, then freeze it for
the held-out reachability runs.  Do not retune the guard on each reported run.

The example background manifest is only a schema-valid starting point.  Adjust
request counts and lengths during calibration so B1 creates measurable TP1
pressure without turning every run into an immediate unrecoverable overload.

## Server launch contract

Follow `PHASE9_SERVER_TEST.md` for the five-GPU TP1/TP4/stager layout.  In
addition to its existing Phase 8/9 variables, set an anchor prefix before the
TP1 server starts.  The controller constructs its external request ID from the
run-directory basename:

```bash
export CAP0_ID=cap0-run001
export CAP0_DIR=/root/autodl-tmp/bridgetp/results/phase9_cap0/$CAP0_ID
export BRIDGETP_STREAM_SOURCE_REQUEST_ID_PREFIX=bridgetp-phase9-$CAP0_ID
```

Use the same `$CAP0_DIR` and migration ID for the TP1 server, takeover API,
stager, and controller exactly as required by the existing Phase 9 runbook.
An empty prefix restores the old single-request behavior.

Prepare safe, run-specific copies of both templates:

```bash
cp experiments/phase9/configs/cap0_controller.template.json \
  /root/autodl-tmp/bridgetp/cap0_controller.json
cp experiments/phase9/configs/cap0_background.template.json \
  /root/autodl-tmp/bridgetp/cap0_background.json
```

Fill the three measured controller fields, the model alias, prompts, and the
frozen workload parameters.  Validate the background manifest without sending
requests:

```bash
python tools/bridge_tp/run_phase9_capacity_background.py \
  --manifest /root/autodl-tmp/bridgetp/cap0_background.json \
  --out-dir "$CAP0_DIR/background" \
  --validate-only
```

## One reachability run

Start TP4 background work first, then the controller anchor, while the manifest
introduces the TP1 burst at its declared relative time.  The two commands are
separate so the schedule remains workload-generator state rather than policy
input:

```bash
python tools/bridge_tp/run_phase9_capacity_background.py \
  --manifest /root/autodl-tmp/bridgetp/cap0_background.json \
  --source-url http://127.0.0.1:8001 \
  --target-url http://127.0.0.1:8200 \
  --out-dir "$CAP0_DIR/background"
```

```bash
python tools/bridge_tp/run_phase9_controller.py \
  --config /root/autodl-tmp/bridgetp/cap0_controller.json \
  --run-dir "$CAP0_DIR" \
  --source-request experiments/phase9/configs/request_long.json \
  --migration-id "$CAP0_ID"
```

For a clean no-migration control, run the same command with `--dry-run`.  That
keeps telemetry and policy audit active while suppressing every actuator;
merely setting `capacity_pilot.enabled=false` would re-enable the legacy
performance trigger and is therefore not a valid B1 control.  Preserve the
same workload manifest and seed.

## Required CAP-0 outcomes

Run these as engineering assertions, not a large factorial evaluation:

1. **No-op:** pressure enters while TP4 fails its current admission guard;
   audit records `STAY`, and no Shadow starts.
2. **Rescue reachability:** TP4 becomes admissible while the headroom signal is
   active; audit records `trigger_path=CAPACITY_PILOT`, four target ranks become
   ready, and takeover commits.
3. **Safe abandon:** the signal reaches `CLEAR` before cutover; cleanup records
   `source_abort_dispatched=false` and the anchor completes on TP1.

Every run must preserve:

- `phase9_audit.jsonl`
- `source_progress.json`
- `session_manifest.json`
- `staging_manifest.json` and rank receipts when Shadow starts
- `takeover_state.json`
- `response_proxy_stats.json`
- source/target responses and the background manifest/events/summary
- exact git revision, dirty diff, server command lines, dtype, model path,
  block counts, and GPU inventory

## Stop/go gate for formal capacity E

Proceed only if all three CAP-0 outcomes repeat without source loss, deadlock,
or cross-request contamination.  The next code phase must then add, in order:

1. request registry and multiple migration candidates;
2. joint source growth/risk using only causal observations;
3. explicit TP4 reservation for current KV plus guarded future growth;
4. actual copy-backlog receipts and a deadline-feasibility check;
5. candidate ranking and concurrent-migration accounting;
6. preemption/rejection/goodput measurement on the exact vLLM commit.

Until those gates are implemented, CAP-0 is evidence that the mechanism is
reachable under real contention—not evidence that the final controller is
capacity-safe or optimal.
