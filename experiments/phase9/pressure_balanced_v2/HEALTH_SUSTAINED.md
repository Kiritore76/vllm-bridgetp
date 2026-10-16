# Health policy repeated migration matrix

Evaluation definitions and SLO attainment are frozen in
[EVALUATION_STANDARD_CN.md](EVALUATION_STANDARD_CN.md). Report raw throughput,
latency, soft goodoutput, and both all-request and multi-token SLO attainment.
Current calibration gaps and the deferred TP4 prefill experiment are recorded
in [PARAMETER_STATUS_CN.md](PARAMETER_STATUS_CN.md).

Run the existing tracked wrapper:
`bash tools/bridge_tp/run_pressure_pilot.sh --health-sustained`.
This replaces the previous four-run health sustained plan. Historical KV-only
`--sustained` and single-move `--health-screen` entries remain separate.

Use efficiency0.90, target load0.80 and **3s persistence**, retaining the closed
10s causal health window,125-block guard and measured16.86410117149353s
START-to-source-release lookahead. These remain engineering choices rather
than fitted optimal thresholds. No future-output ingress reservation gate.
Keep the existing bucket remaining-length predictor.

Ten fresh runs (five STAY/migrate pairs):

| Scenario | Runs | Expected observation |
| --- | --- | --- |
| Sustained service pressure, repetitions1/2 | 4 | At least two service migrations if measured eligibility persists |
| Both lightly loaded | 2 | No migration |
| Source pressure, busy target | 2 | Busy-target gate; no service migration |
| Two source KV-pressure waves | 2 | At least two KV migrations if physical/cost gates permit |

STAY precedes migrate in repetition1; repetition2 reverses order. Each pair
shares exact prompt token lists, greedy sampling, seed20261015 and arrivals.
Sustained cases retain151 requests/53388 output tokens per run: target
2048/6144 at0/90/180/270s; source3 requests2048/4096 at5/5.02/5.04s, four more
at110/165/220/275s, plus source2048/1 every2s from20 through298s. Busy retains
24 target requests,3 source requests and30 prefill competitors. Light has one
target2048/6144 and one source2048/4096. KV has one target2048/6144 plus two
waves of five source4096/3072 at5s and170s (0.02s within each wave).

Maximum migrations0 means no per-run count limit. Transfers are serialized:
require source KV release and stream termination, native target decode,
finalized boundary, clean stager completion, source channel IDLE and all four target
channels IDLE before switching the active session. The old target response
may keep decoding during the next transfer. Each transfer has an immutable
directory and unique identity. Old target futures, capacity commitments,
ownership and completion remain independently tracked. A new service transfer
rechecks current health and requires a fresh3s hold. KV relief bypasses hold
and cooldown, never physical, target service or channel safety. It does not
interrupt an in-flight transfer. Uncertain cleanup blocks reuse. Confirmed
cancellation can rotate while the source continues on TP1; this requires matching
source/stager cleanup, target abort and fresh absence, all five channel IDLE
receipts and clean stager exit. See `PARAMETER_STATUS_CN.md` for the new opt-in
pre-freeze cancellation rules and remaining GPU acceptance requirements.

Keep one reusable receive buffer per target rank across sessions instead of
releasing it when an older request completes: an old completion must not touch
a newer buffer or receipt directory. Buffer size follows the largest received
request, not the sum of migrated requests, and owned worker shutdown frees it.
Inspect GPU memory across sessions. Target prefill/scheduling budget is unchanged.

Score the first300s with full-request ITL100ms/slow1%/maxgap1s. Drain gaps affect
quality; drain tokens stay out of the numerator. Retain TTFT, raw throughput,
preemption, release, handoff and host/timing evidence. Sustained comparisons
require arrival lag<=0.5s, active coverage>=95% and output in all30 ten-second
bins in both arms. Other cases primarily test function/safety. Missing repeated
START is COVERAGE_GAP; never force migration, cherry-pick seeds or hide invalid
arms. No automatic ingress routing or universal optimum claim.

Progress shows scene number, submitted/completed requests, phase and
released/started migrations. Allow roughly70–100 minutes including ten server
startups and full drain; each run still has a900s deadline. Existing wrapper
preserves dirty work and stops, cleans only owned groups, removes only the
runtime active symlink after groups end, and packs all real session directories
once in `pressure-balanced-git-v2-health-sustained-*.tar.gz`. CPU checks do not
establish GPU acceptance; return that archive for analysis.


## Follow-up: two-wave KV regression, then cancellation mechanism smoke

Use the existing wrapper; no separate input package or cloud helper script:

- `bash tools/bridge_tp/run_pressure_pilot.sh --health-kv-only`: validate the full
  frozen matrix, then select only its original `health-kv-r2` recipe. Fresh
  migrate and STAY arms use unchanged prompts, stops, timing and policy. Two
  scenes, approximately15–25 minutes including startup and drain. Inspect both
  KV waves, actual source release, preemption, cancellation decisions, full
  visible output, native empty and final ledger. This is a finite functional
  pair, not a saturated capacity measurement. Do not splice an old STAY arm.
- After reviewing that archive, `bash tools/bridge_tp/run_pressure_pilot.sh
  --cancel-smoke` runs just one diagnostic migration arm using the same two-wave
  workload (approximately8–15 minutes). Only the first naturally STARTed shadow
  is eligible for the test cancellation: target must be admitted with zero
  output, its identity visible in fresh native inventory, all four initial
  history stage receipts present, and no freeze selected. It does not force
  START, slow the transport, disable pressure gates or change the online policy.

The diagnostic records `pressure_test_cancel_injected`, `test_only=true` and
`natural_cancellation_validated=false`. It requires confirmed cleanup, a matching
native target abort receipt, channel reuse from CANCELLED and a later successful
migration/release. Existing completion/ledger/native-empty checks still apply.
Missing a safe injection window or later natural START is COVERAGE_GAP, not
success. The diagnostic exits nonzero on missing coverage, and its results have
no paired benefit comparison. Do not interpret test cancellation under KV risk
as permission for the production policy to cancel then.

Both modes run CPU regressions first, show the usual concise progress and create
one `pressure-balanced-git-v2-health-kv-only-*.tar.gz` or
`pressure-balanced-git-v2-cancel-smoke-*.tar.gz` under the existing results folder.
Offline tests separately cover healthy recovery, persisting service/KV pressure,
UNKNOWN/stale observations, repeated reads of one closed window and freeze gates.
Natural pressure recovery before freeze remains a later coverage study; do not
relax cancellation rules or cherry-pick seeds to force it.
