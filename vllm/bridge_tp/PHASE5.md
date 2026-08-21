# BridgeTP D3 Phase 5: TP4 file restore MVP

Status: passed on 2026-08-21. See `PHASE5_VALIDATION.md` for the recorded
restore receipts, exact continuation tokens, and evidence boundary.

Phase 5 uses vLLM's KV Connector lifecycle to let the target scheduler allocate
real TP4 KV blocks before each worker loads its authenticated Phase 4 shard.
Every worker immediately reads the target blocks back and requires exact tensor
equality before the pending token is computed.

This is a synchronous, shared-filesystem validation path. It proves allocation,
TP4 placement, restore, and readback. It is not background network transfer,
source-request cancellation, or atomic ownership takeover.

## Preconditions

- Four GPUs from the same A100 PCIe platform as the source evidence.
- vLLM v0.23.0 at the Phase 5 branch commit.
- Qwen2.5-14B-Instruct with the same weights and tokenizer.
- A regenerated Phase 4 directory containing the schema fields required by
  Phase 5.
- Prefix caching, speculative decoding, async scheduling, and the hybrid KV
  cache manager disabled for this first proof.

## Start the TP4 target

Replace the two paths below with the server's real paths:

```bash
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
cd /root/autodl-tmp/bridgetp/vllm_bridge

export PHASE4_DIR=/root/autodl-tmp/bridgetp/bridge_dumps/phase4_reshard_tp4
export RESTORE_RECEIPTS=/root/autodl-tmp/bridgetp/bridge_dumps/phase5_receipts

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m vllm.entrypoints.openai.api_server \
  --model /root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master \
  --served-model-name bridgetp-model \
  --tensor-parallel-size 4 \
  --port 8200 \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  --kv-transfer-config "$(python -c 'import json,os; print(json.dumps({
    "kv_connector":"BridgeTPFileRestoreConnector",
    "kv_connector_module_path":"vllm.bridge_tp.file_restore_connector",
    "kv_role":"kv_consumer",
    "kv_load_failure_policy":"fail",
    "kv_connector_extra_config":{
      "bridgetp_reshard_dir":os.environ["PHASE4_DIR"],
      "bridgetp_restore_receipt_dir":os.environ["RESTORE_RECEIPTS"]
    }}))')"
```

## Submit the exact continuation request

The request prompt is the complete known token history: 137 tokens already
represented by KV plus the one pending token that must be computed next.

```bash
python tools/bridge_tp/build_restore_request.py \
  "$PHASE4_DIR" \
  --model bridgetp-model \
  --max-tokens 32 \
  --output /tmp/bridgetp_restore_request.json

curl -sS http://127.0.0.1:8200/v1/completions \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/bridgetp_restore_request.json \
  | tee /tmp/bridgetp_tp4_continuation.json
```

The response includes `choices[0].token_ids` because the request sets
`return_token_ids=true`.

## Validate restore receipts

Find the target request directory written below `$RESTORE_RECEIPTS`, then run:

```bash
python tools/bridge_tp/inspect_restore_receipts.py \
  "$PHASE4_DIR" \
  "$RESTORE_RECEIPTS/<target-request-id>"
```

The required result is four receipts, the same scheduler-issued logical block
table on every TP rank, and `all_ranks_exact_readback: true`. Each rank applies
those IDs to its own GPU-local KV pool. They are target allocator results and
are not required to equal source block IDs `[1..9]`.

## Continuation control

Run the same 138-token prompt without the connector on a clean TP1 server with
the same model and greedy sampling. Compare `choices[0].token_ids` from TP1 and
restored TP4. Exact equality proves this narrow continuation case. It still
does not prove transfer of arbitrary sampling state or atomic live ownership.
