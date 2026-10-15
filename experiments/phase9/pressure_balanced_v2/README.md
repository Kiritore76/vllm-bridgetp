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

## Revised prefill and admission

The complete pending prompt backlog is observed against an 8192-token reference,
including unobserved ingress reservations. A scheduled prefill step is compared
with the existing 2048-token step reference. These are pressure observations,
not hard admission limits. Missing active step evidence stays unknown.

Normal ingress prefers TP1. A prompt of at least 2048 tokens prefers TP4 only
when TP1 is CRITICAL/OVERLOAD. A known prompt of at least 4096 tokens considers
TP4 directly. These prompt boundaries are explicit draft parameters. If the
preferred pool cannot hold the full unchanged stop cap/context or shared physical
commitments, try the other pool. Do not shrink the output cap to fit TP1.
max_tokens is a stop contract, not a prediction of expected output length.

Only fresh physical inventory, shared block commitments and context feasibility
gate ingress. Observed service pressure, waiting/backlog and lookup gaps do not
prevent a physically feasible submission. Unknown decode cost is flagged and
temporarily reserves a full service budget for MIGRATION feasibility until the
new stream is observed; this does not block other feasible admissions.
The fixed routes in these four function scenes intentionally override normal
initial routing to construct source/target conditions. Normal routing has CPU
regression coverage; the fixed-route pilot does not measure its effectiveness.

Exact local engine-P95 lookups remain the first choice. Sparse decode cells may
use the largest P95 among supported shapes with at least as many requests and
at least as much mean context as the query. This assumes monotone service cost
in those two features. It is an empirical working envelope, not a hard bound or
causal ITL prediction; contributing cell indices and observed tail maxima are
logged. Queries beyond supported heavier shapes and mixed/recompute gaps remain
unknown. The full post-admission step is reserved on fallback, and fallback
migration estimates claim zero source relief. No outlier records are removed.

Queue diagnostics name stale/missing physical evidence, context and pending
physical commitments. Service cost gaps and pressure remain separately audited.
The cloud entrypoint and input folder stay fixed. No new GPU run is implied by
the CPU counterfactual admission checks against the previous run's snapshots.
