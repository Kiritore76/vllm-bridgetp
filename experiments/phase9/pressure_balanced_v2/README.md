# Pressure-balanced function pilot: repository bundle

The four arrival traces, policy, empirical service summary, controller template
and launcher are versioned together. No separate input archive or file upload is
needed. The workload remains the same as the v2 pilot. Decode/migration trigger
thresholds are unchanged; revised prefill admission and service support are below.
These reused development-family inputs are for function checks, not independent
benefit evaluation. Model weights, predictor checkpoint and results stay outside Git.

On the existing five-A100 server, update the clean checkout on
`bridgetp/predictor-nonuniform84-20261010` using SSH22, then run from
`/root/autodl-tmp/bridgetp/vllm_bridge`:

```bash
bash tools/bridge_tp/run_pressure_pilot.sh
```

This defaults to the both-light admission smoke test. Require two completed
formal requests and FUNCTION_OBSERVED before running the four function scenes:

```bash
bash tools/bridge_tp/run_pressure_pilot.sh --full
```

The launcher preserves the original venv/model/checkpoint paths and five-GPU
checks. It validates file/input/model SHA, actual serving/stager argv and the
rolling/session/bootstrap CPU regression before starting servers. Dirty work is
backed up and the run stops without reset/stash/clean. All result paths are fresh.

Progress shows server startup, warmup, submission/completion counts, queued jobs,
pool states, migration, cleanup and packaging. During formal execution it prints
every ten seconds. Due work without actual progress for sixty seconds fails with
diagnostics. Future arrivals do not count as stalled due work.

`pilot_spec.json` contains repository-relative paths and SHA256 of runtime code
and input files. Git-tracked bundle mode requires a tracked spec, the expected
branch and a clean checkout, then records the actual HEAD and spec digest in
the frozen runtime spec. The manifest does not embed its own containing commit
ID, avoiding self-reference. The original external, exact-HEAD spec mode remains
supported for earlier runs. JSON and shell files have explicit LF attributes.

One evidence archive is written to the external results directory under
`pressure-balanced-git-v2-smoke-<UTC>-<pid>` (or `function` for --full), including
the runtime spec, HEAD, original input bundle, native logs and return codes.
The service summary includes historical calibration provenance; original source
paths in that provenance are not dependencies of this run. No old final benefit
test or training features are read by this launcher.

## Routing and pressure feedback

Ordinary ingress has no controller KV-capacity admission gate and no prompt,
output, or decode-service reservation. The controller chooses TP1/TP4 and submits
every due legal request once. vLLM owns its waiting queue, prefill scheduling,
actual KV allocation, and preemption. Its existing full-input-fit scheduler check
is retained; this controls entry into execution, not receipt of HTTP requests.

Normal routing prefers TP1. Prompts >=2048 prefer TP4 when TP1 is CRITICAL or
OVERLOAD; known prompts >=4096 consider TP4 directly. These are draft prompt
thresholds. The original output stop cap must fit the selected instance's
configured context limit; try the other pool if it does not, without shrinking
that cap. Invalid context contracts fail instead of entering a capacity queue.

Zero free blocks, migration commitments, overload, service lookup gaps and stale
or missing pressure readings never prevent a new request from being submitted.
With missing telemetry, routing uses the configured serving context limits and
retains the TP1-first/very-long-prompt rules; new migration decisions wait for
fresh physical and service evidence. The launcher passes the actual source and
target serving context limits into the controller and checks published matches.

Ingress records retain identity and unpublished prompt demand only, with zero
KV blocks and zero decode-service cost. Until the request appears in scheduler
inventory, its prompt contributes to routing pressure. Once visible, including
while waiting with zero allocated blocks, native waiting/pending-prompt counters
own that demand. Completion or migration retires the identity record. Preemption
does not create an ingress capacity promise. These records cannot lock blocks.

Physical free blocks, their growth rate, live ITL, waiting requests, pending
prompt tokens and actual prefill chunks drive pressure/migration decisions.
The scheduled step reference is 2048 tokens and the full backlog reference is
8192 tokens. Both are observations, not ingress admission limits. Terminal
progress distinguishes pending dispatch from engine_waiting on TP1 and TP4.

Only migration targets reserve capacity, retaining the full prompt plus frozen
output-cap commitment, four-rank readiness, ownership confirmation and source
free-return evidence. Unknown service shapes still restrict migration feasibility.
Exact local engine-P95 lookups are preferred; bounded heavier decode-shape
fallbacks assume monotone cost and claim zero positive source relief. They are
empirical proxies, not causal ITL predictions or hard bounds.

The fixed routes in these four function scenes intentionally override normal
initial routing to construct source/target conditions. Normal routing has CPU
regression coverage; the fixed-route pilot does not measure its effectiveness.
The cloud entrypoint and input folder stay fixed. CPU checks do not imply a new
GPU run or successful migration coverage.

## Tail batching and inventory/progress repair

Pressure thresholds, scene arrivals and serving settings stay unchanged for the
next four-scene rerun. The sender may wait for the final computed KV boundary
only when the unsent batch has at most 16 tokens and at most 4 tokens remain.
This wait is bounded at 200 ms. Normal/far/large batches send immediately; already
in-flight batches are immutable. After a bounded wait starts, its original small
batch can absorb the remaining tail (up to 20 total tokens) without resetting
the deadline. Contiguous slots use the last producer CUDA event. Timeout sends
the available prefix; cancellation and queue gaps preserve termination checks.
Sender receipts include tail_coalescing outcome/wait/limits when this wait runs.
Exact readback and four-rank readiness remain mandatory. These transport limits
are engineering defaults, not a claimed GPU handoff improvement.

Inventory appended after a poll's time snapshot is deferred unread to the next
poll. Future evidence is never used, and genuine identity/loss/writer corruption
remains fatal. Diagnostics retain the pool, row generation and timestamps.
No-progress detection includes constant-time proxy token counts, buffered target
tokens, completed streams and prefill computed-token progress. A migrated request
is not counted complete until its target stream completes. Inventory health and
request progress remain distinct; missing evidence does not permit migration.
Normal scene completion also waits for ACCOUNTED records to retire after fresh
inventory confirms request absence; process shutdown receipts remain mandatory.

Before GPU startup the launcher now also runs the inventory and pure tail queue
regressions. A new GPU result is still required to confirm normal final cleanup,
that a final tail actually coalesces, and its measured handoff latency.


## Pressure and output-length calibration (no migration)

Use the same fixed entry point, with tracked `calibration.json` inputs:

```bash
bash tools/bridge_tp/run_pressure_pilot.sh --calibrate
```

This supersedes the unexecuted local 8-scene/16-run draft. Only five missing
shapes run, each twice with the same token inputs and seed. Fresh server groups
are owned per repetition. The source control is unarmed; no controller or
stager is started, so no migration can be selected. Existing serving settings,
source predictor and target connector are retained. Runtime records the actual
HEAD, all canonical SHA, native inventory, complete client token timestamps,
sampled engine/four-rank GPU traces and explicit trace finalization receipts.
A test-shape capacity check includes the full output bounds and one injection;
this is a collection precheck, not an online ingress admission gate.

| Scene | Prompt tokens | TP1 / TP4 requests | Output stop cap |
|---|---:|---:|---:|
| output-short | 2048 | 1 / 1 | 128 |
| output-long | 2048 | 1 / 1 | 6144 |
| decode-high-2k | 2048 | 6 / 16 | 2048 |
| decode-high-6k | 6144 | 3 / 16 | 2048 |
| mixed-prefill | 2048 background | 3 / 8 background | 6144 background |

In mixed-prefill, inject 512/1024/2048/4096/6144 prompt requests twice each,
with output cap 16, one per pool at a time, separated by completion plus three
seconds. Backgrounds must still be alive before each injection. Analyze actual
scheduled chunks and actual concurrent decode, not nominal shape labels.
All outputs use ignore_eos=true only for controlled performance measurements;
this does not validate natural-EOS remaining-length prediction or migration
candidate selection. This batch is screening, not a final threshold benchmark.
A completed collection still needs actual Nd/context/chunk coverage analysis.

Reuse instead of recollecting:
- The 103928 both-light pilot provides the 1024-output reference, including its
  common stall; retain the anomaly rather than deleting the run.
- That pilot's natural requests provide current-version EOS/predictor records;
  its capacity-arrival provides long-output and KV-growth observations.
- The three service_evidence archives contain 59,419 descriptive scheduler
  batch observations. Historical six-STAY data supply decode/prefill cost
  references. Split STAY/START and code versions before service-SLO analysis;
  never treat these batches as independent runs or GPU span as visible ITL.

SLO remains visible ITL 100 ms with at most 1% slow intervals. TTFT and queue
wait are measured without inventing thresholds. No pressure/routing/PMF
threshold changes or predictor training are included. A technical failure stops
remaining scenes, preserves partial request tokens and packs one evidence
archive. Normal completion requires every forced stream to reach its cap,
advanced native inventory to be empty, performance trace integrity checks and
owned process-group shutdown. Console reports scene N/10, startup, warmup,
submitted/completed/token counts every ten seconds, finalization and cleanup.
Do not run --full to collect this batch; --full still runs the original four
function scenarios. Results stay outside Git; scripts and inputs replace the
fixed tracked files in place, without another uploaded input archive.


## Targeted pressure screening after offline calibration

The broad 10-run baseline is already complete. Do not repeat it to choose
persistence and target spare parameters. The held-repeat-out offline reference
covers only observed token/Nd/context shapes and returns UNKNOWN outside its
bounded support; it is not enabled in serving. The ordinary pressure gate
replay does not invent PMFs, migration feasibility or benefits.

A small follow-up separates brief source pressure from a longer train while
the target stays lightly loaded:

```bash
bash tools/bridge_tp/run_pressure_pilot.sh --pressure-screen
```

This runs two scenes, each twice (four fresh scenes total). Both have three
TP1 background requests (2k prompt / 2048 output) and one TP4 background request
(2k prompt / 4096 output). After all have started, wait five seconds to clear
the initial service window, then inject into TP1 only:

- source-prefill-pulse: one 2048-prompt / 16-output request;
- source-prefill-train: twelve successive 6144-prompt / 16-output requests,
  waiting for each completion with no added recovery delay.

All backgrounds must still be alive at every injection. This remains an
observational no-migration batch: it supplies pressure-positive cases for
offline persistence/spare screening, before selecting an actual migration
comparison. Nominal train duration is not an acceptance label; inspect its real
pressure and queue duration. It does not claim to cover emergency KV pressure.

The new mode additionally records native inventory generation/observation time
and real parsed /metrics heartbeats in controller/observation_events.jsonl.
An old empty inventory cannot become fresh merely because replay time advances.
Both pools must have valid observation records; scrape errors remain explicit.
The replay tool can consume these actual capture times instead of assuming
publication lag. Observer CPU/HTTP overhead is part of this instrumented run.

Offline tools (no model/server startup) are:

```bash
.venv/bin/python -B tools/bridge_tp/replay_pressure_calibration.py \
  --archive /path/to/returned-calibration.tar.gz --out-dir /path/to/analysis
```

Use the actual environment Python on the machine; on this GPU host it lives at
/root/autodl-tmp/bridgetp/.venv_bridge/bin/python. The offline tool writes a
prefill reference and held-repeat-out pressure replay, not a serving policy.
Results are pressure-balanced-git-v2-pressure-screen-*.tar.gz outside Git.
The normal --calibrate and original --full inputs/modes retain their meanings.
