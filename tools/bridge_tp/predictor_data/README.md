# OASST1 predictor smoke input

## Qwen3 recipe development pilot (2026-10-27)

`qwen3_recipe_pilot48_20261027.jsonl` contains 48 raw user instructions:
24 OASST1 roots, 12 LongWriter-6k roots, 8 independent augmented OASST1
roots and 4 project-authored controlled tasks. English/Chinese are 24/24.
The adjacent manifest records fixed source SHA/revisions and excluded
source-family IDs. Both public datasets declare Apache-2.0. LongWriter source
revision: `0db15c0624f19d63e2efe1021595af933cc5b6cc`.
Reference assistant answers, old model hidden states and old EOS labels are
not included. Every record is recipe-development `pilot`, not a formal holdout.

`run_qwen3_recipe_pilot.py` partitions them into four disjoint 12-request
batches, each with 6 natural, 3 LongWriter, 2 augmented, 1 controlled request
and 6 English/6 Chinese requests. It renders and budgets inputs using the
installed Qwen3 tokenizer with thinking disabled before any GPU model load.
The shared output cap is at most 15872 and is lowered, with a receipt, if
required by that batch's longest prompt. Total context is 16384.
One batch captures real hidden states and output tokens on a single 4090D;
it does not train or qualify a predictor. Successful and failed runs are
archived with raw features, labels, decoded output, logs and provenance.

## Historical Qwen2 smoke

`oasst1_smoke16.jsonl` contains 16 public root user prompts from
[OpenAssistant/oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1),
licensed Apache-2.0. It is a small, pinned input for checking the feature
capture path on an A100 when the server cannot reach Hugging Face. The original
assistant replies are not included or used as length labels.

- Source file: `2023-04-12_oasst_prompts.messages.jsonl.gz`
- Source SHA256: `621ccd86a6ef320ca4e24c137121bd4b39bcc7a0df839f0897fcc965ef2076ed`
- Selection: `prepare_oasst1_predictor_inputs.py --pilot-en 8 --pilot-zh 8`
- Input SHA256: `d037cbc80e0eb46d1e48b676056184413fac32acadd32ff2743ec85a5c64b5bf`

The source archive and larger generated corpora remain outside this Git
repository. For the 160-request pilot, transfer the local generated file or
prepare it from the pinned source on a machine with dataset access.
