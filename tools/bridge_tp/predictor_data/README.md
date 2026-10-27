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
Offline capture explicitly pins `VLLM_USE_V2_MODEL_RUNNER=0`: these BridgeTP
hooks are implemented in `gpu_model_runner.py` (V1), while Qwen3 may otherwise
select V2 automatically. Capture checks writer initialization before generation
and requires each completed request's prefill/decode samples immediately.
Use `--probe-only` first to run two selected requests capped at 128 output
tokens, and require both prefill and decode evidence before repeating a full batch.

## Additional coding and diverse long-task pilot

`qwen3_diverse_long12_20261027.jsonl` is a separate project-authored pilot:
6 complete coding projects, 2 technical tutorials, 2 system-design/incident
analyses and 2 document-analysis tasks. English/Chinese are 6/6. Tasks ask
for substantive deliverables and tests rather than appending thousands of
words to short emails or scheduling questions. The manifest freezes the
input SHA and source-family IDs remain distinct from the original 48.

Select it with `run_qwen3_recipe_pilot.py --profile diverse12 --batch-index 0`.
The same Qwen3 tokenizer budgets, V1 runner, natural EOS and feature audits
apply. All tasks are recipe-development pilots, and “long” is an intent,
not an observed length. This 50% coding diagnostic allocation is not a
claim about production traffic proportions. Generated source code is stored
as response data and is not automatically executed.

### 24K capture and nonuniform remaining-length categories

The pilot accepts `--max-model-len 24576` as an explicit opt-in. It uses the
same fixed instructions but reserves an output cap of at most 24064, reduced
for the longest rendered prompt. The 16K default remains reproducible.
First validate model initialization and feature capture with the short
`recipe48 --probe-only --max-model-len 24576` probe; this checks the 24K engine
configuration, not actual 24K-length generation. Then collect diverse tasks.

Scratch training supports `--bucket-profile nonuniform24k100`: 99 finite
inclusive bounds through 24576 plus one open tail, with progressively wider bins:
512 tokens from 8192 to 16384 and 1024 tokens from 16384 to 24576. This is a proposed geometry,
not evidence that the model has observed sufficient natural long outputs.
All inherited features, normalization, calibration and head weights must be
regenerated for Qwen3. Existing valid 16K Qwen3 captures retain value within
their observed domain; censored outputs retain only lower-bound labels.

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
# Qwen3 formal 16K corpus (2026-10-27)

`qwen3_formal16k2000_20261027.jsonl` freezes 2000 public instructions with
1600/200/200 source-tree-disjoint train/validation/test requests. The matching
manifest pins source revisions, raw file digests, attribution and licenses.
Reference answers are omitted; character lengths are selection proxies only.
No local model output lengths or coverage claims are supplied by this corpus.

Attribution and source-specific terms remain applicable to selected instructions:
OpenAssistant ([Apache-2.0](https://huggingface.co/datasets/OpenAssistant/oasst1)),
Databricks ([CC-BY-SA-3.0](https://huggingface.co/datasets/databricks/databricks-dolly-15k)),
Google Research ([CC-BY-4.0](https://huggingface.co/datasets/google-research-datasets/mbpp)),
OpenAI ([MIT](https://huggingface.co/datasets/openai/gsm8k)), and
Z.ai ([Apache-2.0](https://huggingface.co/datasets/zai-org/LongWriter-6k)).
Modifications consist of selection, task formatting, metadata and predictor split
assignment; the data is not relicensed as vLLM source code. Dolly-derived records
retain CC-BY-SA-3.0. Keep this attribution and the provenance manifest with copies.

`prepare_qwen3_formal16k.py` renders with the installed Qwen3 tokenizer, freezes
context eligibility and generation caps before collection, and produces inputs
for `run_predictor_formal_collection.py`. Train new Qwen3 weights from scratch
with `--bucket-profile nonuniform16k91 --network-depth 4`: 91 finite categories,
maximum inclusive bound 16384, no overflow category. Whole capped requests remain
diagnostic evidence and are excluded from this finite-support training.
