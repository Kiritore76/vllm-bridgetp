# Nd2 health reference supplement

Run `bash tools/bridge_tp/run_pressure_pilot.sh --health-nd2` from the cloud
checkout after pulling this branch. Existing dirty-work preservation, CPU gates,
GPU ownership checks, per-scene shutdown and single-archive packaging apply.

Two fresh runs, no migration or stager. Each pool has two concurrent requests.
TP1 uses prompt 2048 / output 4096; TP4 uses prompt 2048 / output 6144.
Reuse tracked prompts, greedy decoding, ignore EOS and seed 20261015. Do not
search for a seed using observed pauses. A seed cannot prevent host contention.
No additional prefill arrivals are injected. Four SSE workers suffice.

Require at least two complete closed 10-second windows per run at actual Nd2
for each context bin: TP1 2048–4095 / 4096–6143; TP4 additionally 6144–8191.
Native batch identity and completed GPU traces must pass existing verification.
Missing coverage is reported explicitly, never replaced by nominal concurrency.
Retain both runs including pauses; annotate host anomalies before fitting.
Existing health references and online policy are unchanged by collection.

A separate nice(10) process records host CPU/memory/PSI/disk/cgroup every two
seconds and GPU utilization/memory/power/temperature every ten seconds. GPU
queries time out after three seconds. Journals flush without per-sample fsync.
Host sample gaps and GPU query errors remain in evidence; they do not silently
exclude a run. All monitor processes are owned and must finalize and terminate.
This reduces collection overhead but does not guarantee absence of pauses.

The final archive is `pressure-balanced-git-v2-health-nd2-*.tar.gz` under
`/root/autodl-tmp/bridgetp/results`. Inspect `health_nd2_coverage.json`,
`host_monitor_receipt.json`, native empty and owned-process cleanup receipts,
and request timestamps before extending the reference. Preserve the existing
ITL 100 ms / 1% and maximum-gap 1 s goodoutput definition when analyzing.
