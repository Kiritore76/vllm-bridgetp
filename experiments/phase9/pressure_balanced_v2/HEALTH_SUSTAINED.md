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
require arrival lag<=1s, active coverage>=95% and output in all30 ten-second
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

## Pool comparison with automatic routing and natural EOS (2026-10-16)

`bash tools/bridge_tp/run_pressure_pilot.sh --pool-compare` runs12 scenes:
weighted round-robin, normalized least-inflight, pressure routing without
migration (`stay`), and pressure routing with migration (`migrate`), three
repetitions each. Each repetition uses one shared arrivals file and rotates arm
order. Only `migrate` may start a shadow. All arms use the same instrumented
vLLM, default FCFS, continuous batching, disabled prefix cache, native KV
observations, predictor instrumentation, service settings and capacity fallback.
This measures policy effects; it is not an unmodified upstream vLLM overhead
comparison. GPU allocation remains one TP1 plus one TP4 on five GPUs.

The weights are frozen before execution: for each instance, take the median of
the measured peak healthy goodoutput in the shared2k and4k context bins of
health_reference_v2: TP1=231.0625, TP4=597.4625. Source cells and reference SHA
are saved. These are empirical capacity proxies, not maximum sustainable
capacities. Weighted round-robin minimizes (historical admissions+1)/weight;
least-inflight minimizes (currently tracked unfinished admissions+1)/weight.
Both operate on the same serialized ingress ledger; both retain prompt-fit and
context fallback. Neither consults future arrivals or true output lengths.

No new pressure probes or calibration traffic: reuse the tracked natural-long,
natural-small-kv, short-source, short-target and target-long-prefill prompts,
enable natural EOS for all, and keep their original output caps. Each of three
predeclared seeds shuffles15 copies of these five templates, yielding75 normal
arrivals every4s from0 to296s. Requests have no fixed pool. No padding is added
and no synthetic single-token competition stream is injected. This is an
initial controlled workload at0.25 requests/s, not a representative production
trace, a calibrated saturation level or a sweep of low/high load and bursts.
Those follow after observing this comparison. Do not claim a capacity ceiling
or global superiority from this one rate and five templates.

Use the same300s cutoff from the controller clock origin in every arm, drain
fully, and retain full-request ITL/max-gap penalties including drain. Record
raw output, soft goodoutput, hard SLO attainment and request/token goodput,
TTFT/latency, queue observations, initial routing counts, migration and release
counts, native-empty/ledger/shutdown evidence. Reuse the existing archive and
progress output. A source stream must terminate normally, or be aborted only
as part of a committed migration; missing completion or partial length output
is an error. Actual natural outputs and per-request token hashes are retained;
output differences between arms must be examined before attributing throughput
changes to scheduling. There is no forced minimum number of migrations.

`pool_comparison_acceptance.json` reports every repetition, pair deltas and
output parity. Late dispatch over1s yields a nonzero final return code and
preserves the evidence. Low utilization is recorded, not removed or presented
as saturated capacity. Technical failures stop subsequent scenes and still
produce the wrapper's single evidence archive. Inspect all repetitions,
including unfavorable outcomes, before choosing follow-up arrival rates.

Optional first-stage ablation only:
`bash tools/bridge_tp/run_pressure_pilot.sh --pool-compare-pair` runs six scenes,
the same three pressure-routing STAY/migration pairs. Choose one command;
running both repeats those pairs. Full mode is roughly1.5–3h and pair mode
45–90min, depending on model startup, EOS lengths and drain; neither estimate
has been measured for this new workload. CPU and real parser preflight gates
run before GPU startup. Five GPUs must be idle; no unrelated processes are killed.


## Workload-pressure repair pilot (2026-10-17)

`bash tools/bridge_tp/run_pressure_pilot.sh --workload-pilot` selects only the
first fresh natural-EOS repetition of the pool comparison and runs STAY and
migration arms (two scenes,75 identical planned arrivals each). It uses the
new workload_pressure profile and retains all latency/output-parity, fixed
300s scoring, full drain, native-empty, ledger, trace and owned-process checks.
There is no forced START, workload padding or online pressure probe. The two
arms share the same file; both are new runs, never splice old STAY evidence.

Inspect passive_pressure_coverage, routing reasons, service/KV migration paths,
preemptions, actual output lengths, goodoutput and hard SLO. A scene without
START is valid functional data but cannot validate migration coverage or
benefit. The small pilot is a regression stage, not a full three-repeat result.
Expect roughly15–30min for two startups,300s arrival windows and drain, subject
to actual EOS lengths and platform performance; this estimate is unmeasured.

Progress now shows native workload pressure rather than treating a missing
eta as the only health verdict. The terminal remains concise and one archive
is created underresults withprefixpressure-balanced-git-v2-workload-pilot-.
If this passes, use--pool-compare for the12-scene four-arm comparison; select
new load levels only after reviewing the small pilot, not by chasing a win.


## Arrival-isolation repeat (2026-10-17)

Repeat the existing `--workload-pilot` after pulling the arrival-isolation fix.
It still runs two fresh STAY/MIGRATE scenes with the original 75 arrivals,
300s scoring window, routing/migration settings, 1s dispatch gate and full
drain/trace/ledger/owned-process checks. No extra cloud helper or probe is added.

Arrival submission now uses an independent monotonic scheduler. Inventory is
read in per-pool workers and routing uses age-preserving cached readings. The
control loop, migration START and synchronous critical fsync no longer hold
the arrival routing/observation lock across slow I/O. Identity-only admissions
are async observations in health mode; migration critical records stay durable.

Inspect each request's `dispatch_timing` together with existing HTTP transport
timestamps. A shared host or GIL stall can still delay both threads/services;
this repair isolates control I/O and does not certify the previous 30s output
pause fixed. Keep both arrival validity and visible output gaps in acceptance.


## Isolated inventory-process repeat (2026-10-17)

Use the same `--workload-pilot` after pulling the inventory-process fix. The
two fresh STAY/MIGRATE scenes retain all 75 arrivals, routing/migration rules,
300s scoring and full drain. The user-requested dispatch limit is now 1s;
ITL 100ms/1%, maximum output gap 1s and penalty definitions are unchanged.

The internal observe_pressure_inventory.py worker is a required component,
not a separate experiment launcher. Each pool reads/validates its complete raw
journal in an exec'd CPU-only process and sends bounded compact snapshots.
The parent retains a latest cache with original evidence timestamps. Block
lists remain in the raw evidence; no request is removed from policy inputs.

Inspect inventory_tp1/tp4_process_receipt.json, worker.log and timing.jsonl
beside the existing controller evidence. Parent timing also includes IPC decode
and GC evidence; workers record read/parse/GC independently. Abnormal observer
exit, corrupt evidence or missing shutdown receipt fails the run. Children
inherit the Linux controller process group and also stop on parent stdin EOF.
The small CPU-held-GIL regression proves process separation, not GPU benefit.


## Compact retained history repeat (2026-10-17)

Repeat the same --workload-pilot. Pressure source/target transports no longer
retain duplicate nested SSE chunks in Futures. The response proxy retains all
token IDs, original timestamps and origins in primitive arrays and exports the
same full response records. The token journal and failure/timing contracts stay
in place; chunks_retained explicitly marks transport retention. Other proxy
modes/default callers retain their previous storage behavior.

Inspect controller generation-2 GC, arrival timing, visible gaps, handoff and
G300/SLO. No GC disabling, timestamp rewriting, policy threshold changes or new
GPU probes are included. Exact historical response replay and a synthetic heap
benchmark validate representation/retention, not GPU performance improvement.
