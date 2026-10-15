# Pressure feedback policy v1 (experimental integration)

This implementation follows `control/CHAT_HANDOFF_PRESSURE_BALANCED_MIGRATION_20261015_CN.md`.
Experimental multi-stream ingress and a rolling-session driver now exist.
They have CPU contract checks, **not GPU end-to-end acceptance**.

## Implemented boundary

- `controller/bounded_length.py`: condition the original categorical PMF on
  `R >= generated - predicted_at`, subtract lag, and clip to
  `max(0, min(8192, frozen_total_output_cap) - generated)`. Finite integer bins
  are integrated under a uniform approximation. Open-tail mass is retained at
  the stopping cap; an unresolved tail below that cap is rejected. Point
  estimates and distribution-free probability bounds are separate APIs.
- `controller/pressure_policy.py`: separate decode, prefill, and physical
  capacity readings; persistence, hysteresis, cooldown, capacity fast path;
  inventory-wide filtering and a longest-remaining tier before physical release,
  transfer bytes, and stable request ID tie breaks. U is audit data, not a gate.
- Admission fixes the output stop cap to the chosen instance's context limit.
  Both admission and migration reserve through one ledger. Physical free blocks
  are never decremented in telemetry by pending prompt demand. Partial physical
  allocation acknowledgements subtract only that allocated part from future
  promises. Commit does not discard still-unallocated future output capacity.
- `controller/pressure_runtime.py`: serialized, in-process coordination with a
  bound callback for the selected request/session. Audit precedes START.
  Ambiguous START failures retain promises until confirmed cleanup. Commit
  requires ownership and source-release evidence; repeated settlement is
  idempotent and conflicting outcomes fail. Every candidate is queried through
  M5 independently, with explicit capture/publication timestamps, age, and lag.
- `tools/bridge_tp/replay_pressure_policy.py`: mutually isolated CPU replay mode;
  no old M1 probability, benefit, NOW/WAIT, or generated-token gates are imported.

The existing single-anchor `run_phase9_controller.py` is unchanged. M2, M3,
rolling transfer, four-rank receipts, the NCCL channel, and CUDA paths are
unchanged. `pressure_session.py` binds a selected existing ingress stream to
those rolling/ready helpers without relaunching its source. The separate
`run_pressure_balanced_pilot.py` imports only mechanism geometry/rates from old
configuration and bypasses the old benefit-model validation gate. One migration
per fresh process is an explicit function-pilot limit, not a scheduling optimum.

## Draft configuration and evidence

`experiments/phase9/controller/pressure_balanced_v1_draft.json` is for CPU contract
scenarios. Its prompt boundary 2048 and pending-prefill budget 2048 refer to
previously measured 1024/2048 prompt workloads (~373/~750 ms GPU prefill spans).
They are not validated routing boundaries or SLO limits. In particular a single
2048 prompt can already occupy a large service interval. Neither span is client
ITL. Waiting=4 is a provisional workload budget, not evidence of optimal count.

Decode ratios must use the SAME SLO semantics on both pools (for example the
maximum of recent native ITL p95 / reference and slow-interval fraction / 0.01).
Sample count and newest-sample timestamp are mandatory. Completion histograms,
GPU execution medians, and zero samples do not establish this ratio. Fresh idle
means an actually current empty scheduler report, with no intervening arrivals.
The present pure API relies on its caller to establish those facts.

Capacity guard=125 blocks assumes block size 16, i.e. the 2000-token warning.
Reconfigure this for other block geometry. Physical exhaustion is independent
of the warning. The capacity-path release-time check uses observed growth as a
short-horizon estimate; it is not an OOM guarantee under future prefill bursts.
Source/target service deltas must come from supported local batch lookups or be
missing; they must not be inferred merely from running request count. The
512-token length resolution is a draft uncertainty setting. The new controller
uses a common cost horizon: observed maximum START-to-free budget times the
fastest currently observed source request rate, plus the configured 256 recovery
tokens, clipped to8192. Missing source rate evidence disables actual START.
This is an engineering budget, not a confidence bound or a fixed generated-token
trigger. New contexts outside calibration still need pilot confirmation.

## CPU commands

Run from the active checkout using the existing environment:

```powershell
& D:/work/bridgetp/vllm_bridge/.venv/Scripts/python.exe -B -X utf8 -m unittest tests.bridge_tp.test_pressure_policy tests.bridge_tp.test_manager_m5 -v
& D:/work/bridgetp/vllm_bridge/.venv/Scripts/python.exe -B -X utf8 tools/bridge_tp/replay_pressure_policy.py --help
```

No root pytest/CUDA dependency setup is needed for these unittest contracts.
`pressure_cpu.py` imports the pure modules under a private namespace rather than
loading the GPU package. Replays are advisory: subsequent real pool states are
not counterfactual states following the proposed action. Saved-PMF lag offsets
test the algebra and do not assert the request reached those positions online.

## GPU pilot preparation and remaining integration

Experimental integration now implemented, pending actual server acceptance:

1. `pressure_inventory.py` publishes actual blocks/refcounts, cap, progress and
   recent scheduled batch. Set distinct `BRIDGETP_PRESSURE_INVENTORY_PATH` and
   `BRIDGETP_PRESSURE_POOL=tp1/tp4` on respective engines. Default is off. Shared
   and null blocks do not count as unique release; multiple KV groups are unknown.
   Request membership changes force immediate publication inside the sample period.
2. `pressure_observation.py` reads complete journal lines and computes live ITL,
   slow fraction and open gaps. An older empty report additionally requires fresh
   endpoint liveness and matching free blocks, with no intervening arrival.
3. `pressure_ingress.py` reserves before SSE submission and binds actual server
   IDs using prompt/cap/EOS checks. Allocation, visible decode service and pending
   prefill commitments reconcile separately. Preemption restores future promises.
   An admitted active request can migrate while retaining its source promise.
4. The session adapter reuses M2, rolling reservation and four-rank ready/commit
   helpers, then requires a matching source free-return receipt. Existing pressure
   or session artifacts refuse restart; the ledger has no automatic recovery.

The local service lookup is an engine-p95 proxy, not a causal native-ITL model.
Exact unsupported cells remain unknown; the bounded empirical decode fallback
described below is separately identified. Long prompts can join a nonidle pool
within shared physical commitments; service gaps do not hard-gate admission.
The owned-process launcher and four arrival trajectories are prepared. It verifies
committed HEAD/input/model/checkpoint identities and uses the real Linux serving/
stager parsers before any server starts. The full original rolling regression is blocked
on this Windows environment by missing `packaging`; offline uv found no cache.

Four function scenarios are specified, not executed:

| Scenario | Required evidence | Expected action |
| --- | --- | --- |
| Both pools light | Fresh native and capacity readings | STAY even if TP4 is emptier |
| Sustained source decode pressure | Two genuinely migratable requests with different KV/PMFs and supported post-move service estimates | Choose exactly one, then validate safe takeover and reobserve |
| Source pressure, target prefill | Actual prompt allocation and native-service visibility through the prefill phase | Restrict ordinary start; re-evaluate after prefill; missing decode is not idle |
| Source capacity warning, target arrival in flight | Target ingress uses the same commitments; no physical OOM | Reserve one migration; arrival consumes remaining budget; no duplicate START |

After function success, compare identical initial routing and arrival traces
against a no-migration baseline. Keep H300 soft quality, full-request SLO,
TTFT/ITL, handoff, bytes, recovery, reservation errors and invalid migrations.
Retain STAY, negative benefit, no-candidate and long-handoff outcomes. Preserve
the first two technical failures and stop on the third cumulative technical
failure. Do not count expected STAY or valid negative benefit as technical errors.

Pilot launcher must use the actual new committed HEAD, parser-verified full
argv and regenerated file/input/model/checkpoint SHA manifests. Keep SSH22
server pull, original server checkout/environment/model paths, five-GPU hardware
checks, real return codes and function/return shell style from the handoff.
The controller parser/preflight and transport argv construction are CPU-checked.
Linux dependency-backed serving validation and the full rolling regression remain
mandatory pre-GPU gates. Normal EOS before binding retains uncertain migration
commitments until all owned process groups end; it is a coverage gap, not a
technical failure. This first pilot owns fresh groups with at most one migration
per scene. First-run results and required fixes are described below.

First function pilot (68fe581) exposed an idle bootstrap defect: the allocator's
usage denominator excludes its null block, but the controller used total pool
blocks. Three scenes dispatched zero requests and failed; the fourth never ran.
Inventory now explicitly publishes usage_denominator_blocks, and the pressure
metrics adapter uses it. Matching fresh idle evidence remains mandatory.
Short metrics failures omit the heartbeat; stale inventory is never refreshed.
Controller status is saved every five seconds and shown by the launcher every
ten seconds. Due work without token/admission/completion/session progress for
60 seconds fails with queue diagnostics; future arrivals alone do not time out.
Use --smoke-only for a single both-light admission check before rerunning the
four-scene pilot. Code/identity/parser gates still cover the full frozen spec.

The v2 workload is also available as a Git-tracked bundle under
experiments/phase9/pressure_balanced_v2. After updating the existing branch, run
`bash tools/bridge_tp/run_pressure_pilot.sh` for smoke or append `--full` for all
four scenes. No input archive upload is needed. SHA checks are retained; the
actual clean checkout HEAD and source spec digest are recorded at runtime.

The four-scene run at 3a9902b completed 13 requests but produced zero migrations:
only both-light was covered. Full prompt length had incorrectly been constrained
by a one-step prefill budget; sparse three-request lookup cells also delayed
admission. Revised pressure records actual prefill step tokens separately from
pending full prompts. A draft 8192-token backlog reference is for pressure only;
there is no idle-only long-prompt admission exception or service-pressure gate.
Physical blocks and original output stop caps remain protected.

For decode-only gaps, a separately audited empirical P95 envelope can use
supported heavier request-count/mean-context shapes. It assumes monotone cost,
is not a confidence bound, records the historical maximum as well, and never
extends beyond a dominating observed shape. Fallback admission charges the whole
post-admission step; fallback migration predicts no positive source relief.
Unknown mixed/recompute costs stay unknown but do not queue feasible arrivals.
Their temporary service reservation protects migration feasibility until actual
stream service is observed. Decode thresholds, persistence,
cooldown, capacity guard, PMF and transport mechanism are unchanged.

Normal ingress prefers TP1. Prompts >=2048 prefer TP4 when TP1 is CRITICAL or
OVERLOAD; known prompts >=4096 consider TP4 directly. If TP1 cannot fit the
unchanged output cap/context or physical commitments, consider TP4 rather than
shrinking that cap. These are draft prompt thresholds, not predicted output.
Fixed scene routes remain explicit harness constraints, not this normal routing.
