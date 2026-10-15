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
