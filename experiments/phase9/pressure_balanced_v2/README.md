# Pressure-balanced function pilot: repository bundle

The four arrival traces, policy, empirical service summary, controller template
and launcher are versioned together. No separate input archive or file upload is
needed. The workload and draft thresholds are unchanged from the v2 pilot.
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
