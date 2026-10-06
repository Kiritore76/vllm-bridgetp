# Final collection SLO and remaining pilots

The user fixed the new collection slow-interval allowance at 2%. Select
`--slo-profile slow2pct` and the committed reference
`experiments/phase9/slo/slo_v6_slow2pct_a100_tp4_reference_20261006.json`.
Its policy ID is `v6_slow2pct_20261006`, SHA256
`0bea93bd3af4e6d44b3164c14b6e3504666c8adf3ad8679c411f775936f4f6cb`.
Slow means a visible interval greater than 100ms; each request may have up to
2% such intervals. Mean TPOT 100ms, max visible interval 1000ms, max handoff
1000ms, frozen TTFT curve/queue allowance and other criteria remain unchanged.
Preflight rejects a mismatched profile/reference hash, audit records the policy
ID and probability outcomes record the profile. Legacy defaults support old
replay only; new launchers must explicitly select `slow2pct` and `reduced2000`.

The user excludes prior data from formal fitting. Preserve those 1% archives
and labels; historical 2% sensitivity analysis does not turn them into formal
samples. Future formal batches use one frozen guard/SLO/window contract.

Already demonstrated: initial and delta four-rank exact restore, actual
START/STAY collection, guard-warning admission, guard2000 configuration,
continuous 300-second output coverage and retrospective full-request SLO
drain scoring. Do not rerun standalone versions of those checks. Validate the
new SLO in the next coverage batch instead.

Remaining minimum coverage is organized into two batches:

1. Eight arms: two fresh engineering request seeds (20261032, 20261033),
   thresholds 0.01 and 0.05, each with START/STAY. Keep B2 source3/target0,
   source peer prompts 4480, 24-second waves (later 8-second spacing), H=300.
   This tests earlier/intermediate candidates under the final 2% contract and
   estimates between-run variation. Actual online crossings govern timing;
   natural EOS before crossing is a valid policy outcome. It is not excluded.
2. Six arms: a higher source-pressure case at two thresholds, START/STAY
   (four arms), and a moderate target-background case at one threshold,
   START/STAY (two arms). These load recipes and threshold choices require
   capacity-budget and offline probability preparation before launch. Candidate
   source shape is 7168 anchor + three 6656 peers: occupied prompt budget 27136,
   free 4272, headroom to guard2000 2272 at historical capacity31408. Slower
   later arrivals should bound overlap. This is a proposed recipe, not a
   measured safe/accepted workload. Moderate target starts with two 1536-token
   peers and staggered later arrivals; avoid the historical target8 SLO floor.

The second batch verifies urgency coverage and destination interference. If
the first batch already covers the intended urgency range with valid SLO
headroom, remove the duplicate high-source test; do not require another pilot
solely because it has a name. Require actual measured load/feature coverage,
not configured counts. At least one high-urgency START must still have positive
physical headroom and feasible target/restore evidence. True OOM is never a
required event.

Inspect coverage, first crossing/candidate/execution/commit, source release,
predictor observation overhead, target cleanup, recipe identities and SLO
headroom. Preserve signed benefit and no-start results, and distinguish service
zeros from technical exclusions. If target/background all zeros or probability
thresholds always coincide, adjust the workload before bulk collection.

These pilots are engineering coverage/variance checks, not a fitted function
or a positive-benefit qualification. Once representative coverage is adequate,
freeze code/threshold family/request splits and enter formal expansion. Retain
theta=0 exploration and late thresholds as appropriate in the formal matrix,
with held-out seed/request-tree/load splits; do not treat ticks or requests
within an episode as independent migration samples. Freeze predictor training.
