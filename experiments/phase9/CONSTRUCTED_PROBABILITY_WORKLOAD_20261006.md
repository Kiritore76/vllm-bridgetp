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

The first candidates for input preparation are 4K/8K/10K anchor contexts,
1K/1.5K backgrounds and 8/24 requested response sections. These are intended
state probes, not guaranteed p/U cells. Natural EOS, context truncation,
actual p bounds, U/slack and physical rejections determine observed coverage.
Right-censored output is labeled as such. Do not use a threshold crossing or
an OOM as a required outcome. Ordinary randomization stops at the physical
gate; protection episodes remain separately identified.

The failed raw F3 block started Shadow once but never cut over: history was
ready while delta lag remained above 16, and the future candidate expired.
Cancellation was routed to source-EOS acceptance. Resolve mechanism and
acceptance handling and check runtime JIT/warmup before a new broad pilot.
First validate one actual START/STAY pair at theta=0, with complete handoff
and cleanup evidence; then preregister thresholds using constructed-input
p/U coverage and expand independent blocks. Do not resume the failed raw
directory or present its excluded START label as a measured migration effect.
