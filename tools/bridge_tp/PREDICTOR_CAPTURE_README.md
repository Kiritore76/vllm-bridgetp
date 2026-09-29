# Length predictor: first capture pilot

`run_predictor_capture.py` runs the existing base model on one GPU. With the
opt-in capture flag, the GPU runner saves the final-layer state of the token
used for logits once at request-level prefill completion and then every `k`
generated tokens. It does not load or train a prediction head and does not
change the migration controller.
The pilot explicitly disables asynchronous scheduling because the feature
index is taken from the synchronous request batch.

Input is local JSONL, one request per line, with a unique string `id` and
either a text `prompt` or chat-template-compatible `messages`. The bundled
`predictor_pilot_prompts.jsonl` is only a six-request plumbing test. vLLM's
default random serving benchmark normally ignores EOS and targets configured
lengths, so those outputs are not natural remaining-length training labels.
For training, use representative real requests or a locally staged public
conversation corpus, with the same base model and sampling policy intended
for evaluation. Keep entire requests in one train/validation/test split.

Outputs in a new directory:

- `preflight.json`: HEAD, original input path and SHA256, model path/config
  SHA256, GPU inventory, and generation settings.
- `features/*.npz`: float16 hidden vectors with engine request IDs, generated
  token counts, and `PREFILL_COMPLETE` / `DECODE` phases.
- `labels.jsonl`: one final response per request with finish reason.
- `sample_index.jsonl`: joins each hidden vector to its remaining-length
  target. `remaining_tokens` is null if the request hit the output cap;
  `observed_remaining_lower_bound` is recorded separately.
- `summary.json`: request counts, exact versus censored samples, hidden width,
  and capture phase counts.

Collection synchronizes selected GPU vectors and writes them to disk. Use
this first pilot to measure correctness and overhead before scaling up.
Do not treat capped runs as exact-length training examples. The script refuses
dirty Git checkouts, unexpected HEAD/input SHA/GPU inventories, or an existing
nonempty output directory.
