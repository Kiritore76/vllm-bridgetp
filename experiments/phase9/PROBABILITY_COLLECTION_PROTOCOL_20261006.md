# F1/F2 probability collection protocol

This implements the 2026-10-06 migration function fitting plan. Predictor
training is frozen. The eventual label is whole-pool SLO v6 GoodOutput at H;
no benefit model or calibrated pool OOM classifier is published here.

## F1 observations

`risk_urgency_snapshot` records block-rounded H = free - guard - unallocated
pending prefill, the full frozen distribution/edges/checkpoint/model/input
SHAs, candidate decode speed, separated source decode growth, target state,
actual initial M2 preview, probability bounds, T_guard, T_release, S and U.

The bucket calculation is P(min(R - lag, runtime_cap) >= ceil(r) | R >= lag).
Observed emitted tokens establish R >= lag; they do not establish a future
visible token. At lag zero the zero-remaining category is retained. Bucket
mass can lie anywhere in its integer support, including the open tail.
There is no uniform-bin interpolation. Ignore-EOS and stale/missing evidence
are invalid for this natural-length risk. Runtime cap changes are applied at
each tick. Direct capacity probability and shared-growth probability are
separate; the gate uses the lower shared-growth bound.

T_guard = H / source decode growth; r_guard = candidate speed * T_guard.
Growth omits isolated prefill allocations and does not assume future arrivals
or peer EOS. Missing/zero/negative growth has no finite projected guard risk.
The EWMA candidate speed is estimated from successive actual output progress.

T_release includes history and ongoing candidate decode catch-up at the M2
initial rate, plus the configured five-second final-sync/handoff/source-release
tail. S and U also include a two-second margin. Time envelopes assume
0.5..1.5 times current growth and 0.5..1.0 times effective initial bandwidth.
These are explicit engineering envelopes, not calibrated confidence intervals
or guarantees. F3 must measure effective rates and actual source release to
check their adequacy. No new joint rate search is performed.

Ordinary experimental admission requires fresh pools, positive trustworthy
growth and candidate rate, valid natural prediction, a local request, free
channel, sufficient output budget, target capped-request KV plus its existing
15% KV reserve and unallocated target pending prefill, and positive conservative
source time slack. Target running/waiting and old M1 survival-table/length soft
rules remain observations rather than experimental admission switches.
The physical gate rejects starts that cannot catch up or reach source release
before the projected guard. Protection demand is logged separately from an
actual safety override; this collector does not claim to implement a new rescue
path. M2, four-rank readiness, earliest-safe M3 takeover and M4 cleanup remain
in force.

## F2 gate and action assignment

`--probability-pilot --probability-thresholds ...` enables the new collection
path in `run_randomized_goodoutput_pilot_a100.py`. Legacy experiment options
are mutually exclusive with it. The new path has no fixed output eligibility
or single-refresh wait action.

Each threshold has a START/STAY replay pair with the same anchor and arrival
manifest. Pair execution order is randomized by seed/load block; both assigned
actions run the same predictor and physical gate. Threshold candidate selection
uses current p_lower and common feasibility independently of assigned action.
The two complementary assignments have marginal probability 1/2 under the
preregistered randomized ordering; they are dependent paired arms, not IID draws.
The controller also supports a Bernoulli assignment at a candidate when no
explicit assignment is supplied.

The first valid crossing and first feasible candidate are separate records.
Simultaneous crossings retain their shared tick. STAY stays for that episode;
source EOS without a crossing is a legitimate policy result. The execution
record is written only after the action adapter arms Shadow. Merely proposing
START is not reported as an executed migration. Protection-demand and actual
safety-override fields are distinct, and affected episodes are excluded from
ordinary threshold effect samples while retaining their service outcomes.

Probability mode rotates distinct held-out background content on later arrival
waves. It checks tree/content hashes against predictor train/validation splits,
records effective token-prompt SHAs and source trees, and respects actual source
and target contexts. Input long-form augmentation provenance is retained.
These are engineering flows and previously inspected predictor test requests;
controller final testing needs a separate grouped holdout.

The historical p00 label is retained for compatibility, but probability mode
explicitly sets its target jobs to zero and records `target_idle_override`.
Read actual job counts and telemetry, not the old configuration label.

## Fixed-window settlement

A is the fixed arrival-schedule cutoff; H is the fixed scoring duration from
the earliest actual client request arrival, the existing pilot convention.
`horizon_goodoutput.score_horizon` uses the externally visible anchor stream
once, source peers and all target jobs. A request must complete by H and pass
the frozen v6 contract for its H-window tokens to count. Incomplete-by-H,
failed and SLO-failing requests contribute zero; drain after H cannot repair
the label. Output-cap completions are marked right-censored.

Recorded stream interruption/5xx/request failure is a service outcome.
4xx input errors, SHA/reference mismatch, malformed or missing logs, unsafe
migration receipts, startup/runtime-tool failures and unknown runner rejections
are technical exclusions with raw diagnostics. Partial failed source/target
and background responses retain their timestamps and visible tokens. An anchor
failure during a created migration session still needs valid mechanism/cleanup
evidence; missing such evidence is a technical exclusion rather than an inferred
successful migration. Unrecorded process-kill failures likewise remain technical.

`probability_outcomes.json` has one descriptive effect row per threshold's
intervention episode, plus no-start policy outcomes and predecision matching
differences. Reused content/seed/load groups remain grouped. Never replicate
one window delta over its ticks. No-start outcomes can measure policy value,
but do not become START-versus-STAY effect samples.

## Pilot preregistration and acceptance

The seven-archive F1 replay found natural p_lower up to 0.07079, with no natural
crossings of .2/.5/.8. Register [0, .001, .01, .05] for the natural collector
pilot. Artificially padded diagnostics are analyzed separately. This is a
coverage choice made before new outcomes, not a learned policy threshold.

F3: seed 20261021/p00 (idle target), 20261022/p02 (middle target),
20261023/p05 (busy target), eight paired arms per block, 24 episodes total.
A=60 seconds; wave period=10 seconds; H=300 seconds. Source context=16384;
target context=32768; natural EOS; BF16; frozen decoder:31 checkpoint and SLO v6.
Resume requires identical preflight, protocol, inputs and manifests. It skips
recorded successful episodes; a failed or unrecorded episode requires diagnosis.

Accept complete p/U/provenance and assignment/execution logs, timing diversity
or explicit simultaneous/no-crossing outcomes, fixed-H settlement, predictor
overhead/M2-rate/release-envelope measurements, and cancellation/target cleanup.
Inspect target running/waiting/TPOT/KV to confirm intended load coverage. Positive
benefit and guard/OOM events are not acceptance prerequisites. If a state cell
or actual timing contrast is missing, preregister only the specific supplemental
coverage after diagnosing it. Do not expand to fitting before collector review.
