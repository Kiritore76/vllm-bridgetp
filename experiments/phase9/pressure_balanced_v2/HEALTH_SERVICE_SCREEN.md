# Health-reference service-path screen

Cloud entry: `bash tools/bridge_tp/run_pressure_pilot.sh --health-screen`.
This is an opt-in experiment. Existing function, KV-release and sustained modes
retain their policy settings. No production switch, automatic cloud GPU run,
new predictor, transport change, or ordinary-ingress KV admission gate is added.

## Health quantities

`health_reference_v2.json` re-scores the five original calibration archives.
Only past-window intervals enter the ten-second score: per request take the
minimum of the 100ms/1% slow-interval weight and 1s/maximum-gap weight, including
right-censored unfinished output gaps. Do not use eventual full-request quality
in online decisions. Whole-run G300 remains a separate, full-drain outcome.

There are 22 repeated operating points, 20 usable and 2 TP4 Nd32 diagnostic
points kept isolated. Of 504 re-scored windows, 20 changed. Two reference
medians changed: target Nd24/context2k..4k 613.60 to 613.40, and target
Nd8/context4k..6k 364.20 to 362.425 tokens/s. The other 20 medians are unchanged.
Migration arms, interrupted prefixes and later sustained outcomes do not enter
this reference. Reference lookup requires exact Nd and 2048-token context bin.

The Nd2 retest adds five repeated points from the standalone no-migration
supplement `pressure-balanced-git-v2-health-nd2-20261016T033757Z-36267.tar.gz`
(SHA256 `1e82634108326cb9515e34c2993db347165f64f74a778e41d693e7332afaa0bb`).
Both runs and all 31 qualified windows are retained. TP1 Nd2 reference values
are 75.75 / 74.20 tokens/s in context bins 2k..4k / 4k..6k; TP4 values are
125.70 / 124.05 / 124.90 in bins 2k..4k / 4k..6k / 6k..8k. Original 22 cells
are unchanged: 27 total, 25 usable, with Nd32 still isolated. These are measured
working points; no new threshold fit or capacity-ceiling claim is made.

Use the same `--health-screen` entry and the same four recipes and policy
thresholds for the retest. Check that source/target Nd2 health recovers after
the migration and a qualifying ten-second window closes. A transitional
UNKNOWN window remains expected; missing source Nd2 contexts above 6143 tokens
remain UNKNOWN. Inspect both pools' `pressure_decision` readings and native
inventory timestamps, then check matched arrivals, finalization and full-drain
goodoutput. This rerun still permits only one migration per scene and does
not establish a long-duration or repeated migration benefit.

For a supported current shape:

    eta = observed past-window goodoutput / same-shape pure-decode reference
    reference_load = same-shape reference / best measured reference in context bin

The second quantity is relative to measured working points, not a fraction of
true hardware capacity. A high goodoutput source is not marked unhealthy just
because its ITL is higher. Prefill interference may reduce eta below the pure
decode baseline; mixed windows are marked explicitly rather than silently
claimed to be calibrated mixed-prefill capacity.

Actual SSE callbacks feed an additional causal observer. Native inventory
generations establish window shape: at least 80% same Nd/context bin, with
no observation gap over 2s and no stale current shape. Startup, unsupported
shapes and missing data return UNKNOWN. A fresh native-confirmed idle pool is
handled explicitly. Diagnostic Nd32 never becomes a spare capacity reference.

## Experimental trigger

`health_service_policy.json` enables both the service path and independent KV
lookahead. Defaults in existing modes remain unchanged.

- Source eta below 0.90, target eta at least 0.90 and target reference_load at
  most 0.80 must persist for 10s, normally requiring two closed windows.
- These 0.90/0.80/10s settings are explicit engineering choices, not fitted
  optimal thresholds. Unknown health never establishes target spare status.
- Closed-window persistence tolerates polling jitter up to the existing2s
  freshness limit; a longer gap resets it. Legacy modes retain their0.5s limit.
- Target current service and queue evidence must also be acceptable; an old
  healthy window cannot override an ongoing slow output gap or fresh ingress.
- For service START, additionally require a repeated heavier target shape in
  the same post-move mean-context bin. This monotonic support assumption is
  recorded explicitly; it does not interpolate unseen goodoutput or predict
  migration benefit. The original engine-step budget and physical migration
  promises still apply, together with ownership, context, remaining-life,
  unique release and channel checks.
- KV guard remains 2000 tokens and release budget 16.86410117149353 s. A KV
  warning may evaluate candidates even with UNKNOWN service health. Neither
  path forces START. Source-to-target only, at most one session per scene.
- Optional new-ingress routing prefers TP1; long prompts consider TP4 when
  source health/KV needs relief, and very long prompts consider TP4 directly,
  provided the target has known spare evidence. Context fallback remains.
  Pressure never rejects ordinary ingress. Fixed pilot routes override this
  preference; therefore this GPU screen does not validate automatic routing.

## Four fresh runs

Two target conditions, one STAY/migrate pair each, not two repeats of one
workload. Counterbalanced order: spare STAY then migrate; busy migrate then
STAY. Each pair uses identical frozen arrivals. There is no forced overload or
artificial injected ITL in the GPU workload.

- Source: 3 background requests at 5,5.02,5.04s, prompt2048/output4096.
- Source-only prefill competition: prompt2048/output1 at 20,22,...,78s (30).
- Spare target: one prompt2048/output6144 at 0s.
- Busy target: 24 prompt2048/output4096 arrivals at 0,0.02,...,0.46s.

These make 34/57 requests per arm. Fresh post-warmup capacity must cover all
source background stop caps plus one injection and guard, and target caps
plus one migrated request and guard. This verifies intended KV headroom;
it does not reserve ingress KV. The longer prefill train is chosen to cover
two complete health windows. Historical audit found only two instantaneous
service opportunities and zero sustained opportunities, so the new coverage
must be measured, not assumed.

Spare-arm acceptance requires one health-path move (not KV-path), commit,
source release and at least1024 remaining tokens at selection. Busy arm must
show simultaneous source efficiency loss and known target busy/inefficient
evidence without START. Any busy-arm START fails the screen. Missing coverage
is reported as COVERAGE_GAP. Both arms fully drain and finalize native/GPU
evidence and owned cleanup. Paired arrivals must remain within0.5s.

G300/raw rate and old ITL-only score are still saved, but this short function
screen may underfill300s. Do not call function coverage a sustained throughput
or net-benefit validation. Progress additionally prints each pool's health
status, eta and reference load. There is no general multishot controller yet.

## Local audit

The past-window support audit is not an exact native-observer replay or an
independent holdout: 216 source windows, 186 supported/30 unknown; target
183 supported/105 unknown. Mixed windows are compared to supported pure-decode
reference shapes and marked mixed online. There are29 source eta<0.9 windows,
2 contemporaneous opportunities, zero satisfying two-window persistence.
No historical migration is inferred or counted from those opportunities.

Local reproduction scripts and detailed outputs live under
`D:/work/bridgetp/control/pressure_health_service_stage_20261016` and its report.
Actual serving-parser validation, full rolling CPU gate and GPU acceptance
must be run on the cloud host before interpreting this screen.

## Pause diagnosis in the same four-run screen

`--health-screen` now also owns a nice(10) host monitor. Load/CPU/memory/PSI,
disk/cgroup counters and registered owned process trees are sampled every2s;
GPU queries every10s have a3s timeout and report their own duration. Per-process
stat/status/wchan/io/schedstat helps distinguish CPU scheduling and storage
waits from absent application work. Process enumeration follows only registered
server/stager/controller roots and their current children, not unrelated jobs.
Sampling gaps remain data; no seed search or slow-run deletion is performed.

Only this screen enables controller `pressure_timing.jsonl`. It uses bounded
asynchronous evidence and records operation summaries about once per second,
plus operations taking at least100ms (tick intervals at least500ms). Timed
paths include inventory, metrics, predictor reads, policy/migration ticks,
progress snapshots, atomic JSON writes/replaces, and audit serialization,
write/flush/fsync and enclosing lock time. Thread CPU time accompanies operation
wall time; tick intervals include intentional tick sleep. Nested durations
overlap and must not be added. Both controller helper module namespaces use the
same observer so migration helper writes remain visible.

The initial diagnostic revision kept audit fsync frequency unchanged. The
observation-I/O optimization below now reduces syncs in health mode. Atomic
JSON write semantics remain unchanged. Timings observe the remaining causes.
The four recipes, seeds, health reference, thresholds, goodoutput scoring and
single-migration limit are unchanged. Other pilot modes do not enable timing.

Each scene must save `host_monitor_receipt.json` and
`pressure_timing_receipt.json`, with finalized journals and no timing drops or
writer errors. Monitor failure is a technical failure after owned cleanup.
The monitor stops through an owned stop file before process-group cleanup;
this adds no unrelated process termination. All records remain in the existing
single `pressure-balanced-git-v2-health-screen-*.tar.gz` archive.

## Observation I/O optimization

Health-service mode now uses one bounded background metrics sampler per pool:
one request at a time, about once per second after the previous request ends,
HTTP socket timeout0.75s. The controller reads memory without waiting for HTTP.
Cache timestamps conservatively use the earlier of response timestamp and
request start; reading the cache never refreshes age. Failed, future or older
than max_age_s samples are unavailable. Native inventory validation, KV growth,
in-memory ingress reconciliation, health-window collection and migration ticks
retain their original cadence and freshness requirements.

Unchanged ingress reconciliation logs are limited to once per second per
request, with suppressed counts. Binding/ownership/reservation-state changes
are recorded immediately. Metrics sample logs are limited to once per second
per pool. Memory reconciliation itself still executes every controller tick.

`pressure_audit.jsonl` remains the full, analysis-compatible timeline, now
written through the existing bounded AsyncJournal with a1s flush interval.
It includes the existing integrity footer; loss/write errors fail collection.
Pure metrics/progress/unchanged reconciliation use only this async path.
Every decision, binding/state change, admission, dispatch, ownership event,
migration start/commit/cancel/release and all unknown event kinds additionally
use synchronous AuditLog in `pressure_audit_durable.jsonl` before returning.
Critical persistence is independent of the async observation queue. An abrupt
crash may lose recent async observations; retain both files for diagnosis.
The controller still refuses uncertain restart and does not recover from a
partial observation timeline.

Normal shutdown drains observations and writes `controller/pressure_io_receipt.json`
with total records, critical fsync count and dropped count. The metrics workers
must stop before successful shutdown. Legacy non-health modes retain their
original audit and polling behavior. Seeds, four recipes, threshold values,
health references, token timestamp evidence and scoring are unchanged. Compare
fsync counts, tick gaps, dispatch lag and full-pool goodoutput in the same
four-run `--health-screen`; neither lower logging overhead nor removal of all
GPU stalls is asserted before that GPU rerun.
