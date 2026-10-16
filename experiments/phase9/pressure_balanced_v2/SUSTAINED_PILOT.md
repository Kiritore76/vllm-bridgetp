# Fixed-window continued-arrival KV pilot

Cloud entry: `bash tools/bridge_tp/run_pressure_pilot.sh --sustained`.
This extends the existing launcher, not a separate cloud script or input archive.
Tracked inputs: `sustained.json`. STAY/migrate, two repetitions with reversed
order in repetition 2, four fresh serving groups. Each pair shares one arrival
file. Model, bucket predictor, strict greedy sampling, ignore-EOS, transport,
2000-token guard and empirical release budget 16.86410117149353 s are unchanged.
Existing controller supports at most one migration per run; this experiment
tests one KV-relief event under continued arrivals, not a repeated-migration
controller. No forced START, external admission gate or new healthy-region
threshold is introduced.

After serving warmup, the workload clock starts. Initial source stratum is the
same five 4096-prompt/3072-output requests at 5..5.08 s as the KV-release pilot.
Source then receives one identical-shape request at 110,134,158,182,206,230,254,
278 s. Target receives one 2048-prompt/6144-output request at 0,80,160,240 s.
There are 13 source and 4 target requests, 64512 total output tokens per run.
No arrivals are generated from completion feedback; all times are frozen.

The later source rate is 128 output tokens/s (3072/24), near but below the
measured source Nd4/context4k..6k window goodoutput 138.2 tokens/s; target offered
rate is 76.8 tokens/s. These are engineering workload choices, not a fitted
queue stability guarantee. The initial source burst deliberately tests KV
pressure. Later arrivals need not reproduce another KV warning. Measure queue
trends and do not claim infinite-horizon stability from this finite experiment.

Primary scoring is predeclared as
`ITL100MS_SLOW1PCT_MAXGAP1S_MIN_WEIGHT_V1`:

    w_slow = min(1, 0.01 / fraction_of_visible_intervals_over_0.1s)
    w_gap  = min(1, 1s / maximum_visible_interval)
    w      = min(w_slow, w_gap)
    G300   = sum(window_output_tokens * request_w) / 300s

Zero denominators contribute weight 1. Window is [first actual HTTP arrival,
arrival+300s). Quality uses full completed visible streams including migration
seam and post-window drain intervals. TTFT is reported separately and is not
penalized. Drain output tokens are never added to the numerator. Preserve
raw window rate and old ITL-only G300 separately. Do not substitute makespan
for goodoutput or select another horizon after seeing results.

Final arrivals precede 300 s; after submissions end, fully drain all requests.
The original 900 s controller deadline and owned cleanup remain. Progress
prints elapsed seconds, submitted/total, completed, pool pressure and engine
waiting. Each scene writes `sustained_goodoutput.json` with request weights,
actual emitting pool contribution, drain count, TTFT, max gap, window coverage
and 30 s native queue summaries. Coverage needs >=95% whole-pool inflight time
and output in every 10 s bin; insufficient coverage is evidence, not proof of
benefit. Source/target final empty inventory and all 5 GPU journals still gate
technical acceptance. The wrapper packages one result archive automatically.

This preset is a first workload screening point. Compare paired whole-pool
G300, initial KV migration coverage, target existing-request harm, preemptions
and queue evolution before tuning rate or rolling transport parameters.
