# Controlled construction for migration benefit fitting

The user requested controlled artificial requests after the first F3 block
provided inadequate guard-risk coverage. This is an engineering sampling
change, not predictor retraining or a replacement of the frozen GoodOutput SLO.

`tools/bridge_tp/build_constructed_probability_workload.py` creates unique
inventory, workflow, and service-capacity records from explicit seeds. The
number of requested analysis sections controls the output task; the model
still chooses EOS. Output length is not guaranteed. It never uses ignore_eos
or an artificial minimum output length. Additional context contains distinct
generated records, rather than repeating a single token or a tiny prompt set.

The collector materializes the recipes with the actual model chat template
and tokenizer. It preserves chat/task boundaries and all primary records,
fits supplementary records to the requested prompt-token count, and records
the resulting token IDs and SHA. A recipe that cannot fit its required records
is rejected. Source and target context/output budgets remain checked.

Collection requires `--probability-pilot --constructed-workload` plus
`--expected-input-sha256`. The SHA override applies only to the constructed
request file. Model config, checkpoint, guard, base, survival, reference, full
code HEAD, GPU model/count and idle-device checks remain required. UUIDs are
only recorded in portable-hardware mode. Predictor checkpoint provenance
`capture_input_sha256` still identifies its frozen training capture dataset;
the newly constructed request-file SHA is recorded separately in the inputs
and preflight. Do not relabel predictor training provenance as live requests.

Use `--constructed-source-count 1..5` and `--constructed-target-count 0..24`
independently. No `--cases` is used in this mode; the case name records the
actual configured counts. Zero target backgrounds requires zero target
readiness jobs and remains subject to target capacity, channel and four-rank
mechanism checks. START/STAY use the same manifests at the same probability
gate. Prompt size and requested output task size can be crossed with both
source and target counts, rather than associating every high-risk task with
only a busy target.

Rows must explicitly declare `workload_origin=constructed` and one consistent
`engineering_train`, `engineering_validation` or `engineering_test` split.
The collector rejects mixed or implicit inputs. Construction recipes and
source-tree identifiers remain in manifests, including rotated arrival waves.
Use disjoint seeds/trees for controller splits, keep all variants of the same
tree together, and also test held-out task families and independent natural
requests. These rows are not predictor test500 requests or production traces.

Historical short prompts left 16K or more safe KV slots unused. Five 4096-token
source prompts previously approached the guard. The first revised profiles
therefore control source and target prompt occupation independently. The recipe
flag `--background-prompt-tokens` controls source backgrounds;
`--target-prompt-tokens` optionally supplies a separate target budget. Without
the optional flag, original v1 recipes and SHAs are unchanged.

At the historical 1963 x 16 = 31408 source capacity and guard 8448:

| Profile | Anchor prompt | Source peer prompt/count | Initial headroom | Target peers |
| --- | ---: | --- | ---: | --- |
| Mechanism | 4096 | 2048 / 1 | 16816 | 0 |
| Moderate coverage | 4096 | 4096 / 3 | 6576 | 0 |
| Pressure coverage | 8192 | 4096 / 3 | 2480 | 0 or 8 |

Cross the pressure profile with 8/24 requested anchor sections while keeping
source/target prompt budgets unchanged. Target prompts are 1536 tokens.
Start with a single burst; do not immediately replay overlapping long-context
waves. Initial headroom is a planning estimate, not a bound for the whole
episode. `--minimum-initial-source-headroom-tokens 1536` validates rounded
first-burst occupation against the newly measured source capacity. Decode,
later arrivals, readiness and time feasibility still use the runtime gates.

Constructed collectors also select candidates only after the planned source
concurrency, including the anchor, is active and pending prefill is zero. This
load-stratum condition is audited separately from physical feasibility and
does not rewrite probability or urgency. A theta=0 candidate no longer starts
before the planned peers arrive. STAY uses the identical eligibility rule.

These are state probes, not guaranteed p/U cells. Natural EOS, context truncation,
actual p bounds, U/slack and physical rejections determine observed coverage.
Right-censored output is labeled as such. Do not use a threshold crossing or
an OOM as a required outcome. Ordinary randomization stops at the physical
gate; protection episodes remain separately identified.

The failed raw F3 block started Shadow once but never cut over: history was
ready while delta lag remained above 16, and the future candidate expired.
Cancellation was routed to source-EOS acceptance. Routing now uses controller
termination: unexpected CANCELLED stays a diagnostic failure with its actual
abandon reason. It cannot become natural-EOS PASS or an eligible effect label.
Expected M4 cancellation remains handled by the separate M4 acceptance path.

Delta injection now scatters all token slots in one operation per layer and
retains complete exact readback. It directly indexes the strided cache view;
duplicate or out-of-range target blocks are rejected. The source candidate,
16-token lag/lead gate, M2 policy, four-rank commit and cleanup rules remain.
`benchmark_delta_restore.py` compares full-cache fidelity and local apply times
against the former blockwise path on the four target GPUs before engines start.
This microbenchmark does not establish end-to-end catch-up or release times.

`--pre-episode-warmup` completes ordinary two-token requests for distinct prompt
shapes on both pools before arrival clocks start. It strips migration metadata
and records excluded warmup receipts in provenance. Prefix caching remains
disabled. This warms encountered shapes but does not guarantee all online JIT
is eliminated, especially for subsequent batched prefill or predictor capture.

First validate one actual START/STAY pair at theta=0, with complete handoff
and cleanup evidence. Then use a small coverage batch to observe constructed
STAY p/U trajectories. Preregister the next probability family using coverage
without looking at GoodOutput differences; expand independent blocks after
that. Do not resume the failed raw
directory or present its excluded START label as a measured migration effect.
