# OASST1 predictor smoke input

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
