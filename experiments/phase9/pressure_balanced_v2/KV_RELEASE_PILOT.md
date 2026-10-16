# KV release pilot

Run `bash tools/bridge_tp/run_pressure_pilot.sh --kv-release` from the cloud
checkout after pulling this branch. Inputs are tracked in `kv_release.json`.
No separate input package is needed. The existing wrapper owns validation,
CPU regression, progress reporting, cleanup and one result archive.

Four fresh runs: repetition 1 STAY then migrate; repetition 2 migrate then STAY.
Each matched pair shares exactly the same arrival file and sampling contract.
TP4 receives one 2048-prompt/6144-output background at t=0. TP1 receives five
4096-prompt/3072-output requests at t=5, 5.02, 5.04, 5.06 and 5.08 seconds.
Outputs ignore EOS for this controlled performance experiment. Keep the existing
four-layer bucket predictor and all transport/chunk/cutover settings.

Only this mode enables KV lookahead and disables ordinary service-imbalance
triggers. Warning threshold in physical blocks:

    guard + not-yet-allocated prompt blocks
          + ceil(past decode growth * (inventory age + release budget))

Guard remains 2000 tokens / 125 blocks. Release budget is the frozen historical
maximum START-to-source-free-return, 16.86410117149353 s, not handoff latency or
a guaranteed bound. Growth uses unique native generations over the past two
seconds, with at least 0.5 s observed span. Already allocated prompt blocks are
not subtracted again. No ordinary ingress reservation/admission gate is added.

The migration arm may start at most one request. Existing source ownership,
remaining-work, target physical commitments, empirical service support, channel,
and release-before-exhaustion checks remain mandatory. A warning never forces
START. STAY may encounter native vLLM preemption; collecting that control is
intentional. After warmup, actual empty allocator capacity must establish that
initial prompts fit, full source stops exceed capacity, and one offload leaves
the guard. A changed machine/capacity fails the experiment rather than silently
changing concurrency. This check is not a production admission policy.

Each scene emits `kv_capacity_shape.json`, `comparison_metrics.json`, and
`kv_release_metrics.json`, along with native inventories, rolling transport,
performance traces, source-release receipt and owned shutdown receipt. Capture
START-to-actual-source-release separately from visible handoff; released block
count remains unknown unless independently proven. Record native preemptions
before START and over the run. Migration coverage requires one committed/released
KV-path move, at least 1024 remaining tokens at selection, no observed source
preemption before START, and valid fixed arrivals. Otherwise report COVERAGE_GAP.

This pilot measures feasibility and transport under growing KV pressure.
Compare whole-pool ITL100ms goodoutput curves, preemption, stalls and target
interference; successful completion alone does not establish net benefit. Keep
the legacy soft-SLO score separate. Do not retune the scoring window after seeing
which arm wins. Full live service-parser validation and GPU execution occur on
the cloud host, after the mandatory pre-GPU gates.
