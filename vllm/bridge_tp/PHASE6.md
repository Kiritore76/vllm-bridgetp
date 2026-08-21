# BridgeTP D3 Phase 6: five-GPU live streaming transfer

## What this branch proves

Phase 6 is one unified five-GPU experiment on one A100 PCIe node:

```text
GPU 0: TP1 source request continues decoding
                 |
                 | iteration-boundary CPU snapshot
                 | four head-sharded, framed TCP streams
                 v
GPU 1,2,3,4: TP4 target scheduler allocation -> GPU KV injection/readback
                                                   |
                                                   v
                                      32-token shadow continuation
```

There is no separate Phase 6A or Phase 6B. The branch
`bridgetp/d3-phase6-streaming-transfer` contains the whole Phase 6 path.

The source is still authoritative and is not cancelled. This phase proves a
live TP1 snapshot, real byte transfer, TP4 restore, and greedy shadow
continuation. Target-ready commit, source cancellation, rollback, and atomic
ownership takeover remain Phase 7.

## Fixed correctness boundary

- one live TP1 request and one TP4 target request;
- synchronous scheduling, no prefix cache, no speculative decoding;
- snapshot after 128 source output tokens;
- exactly one known token is pending at the snapshot boundary;
- TP1 KV heads are split contiguously across four TP4 ranks;
- each TP4 worker receives only its own rank payload through TCP;
- every frame and full payload is SHA256 checked before deserialization;
- the TP4 scheduler allocates the destination block table;
- all ranks require exact GPU readback before returning `READY`;
- the next 32 greedy token IDs must equal the live source continuation.

The small `session_manifest.json` is shared control metadata. KV tensors never
pass through a shared shard file.

## One-time server preparation

Run after pulling this branch. Editable installation means Python source edits
are visible immediately; the existing native extension does not need rebuilding
because Phase 6 changes Python files only.

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

git fetch origin
git switch bridgetp/d3-phase6-streaming-transfer
git pull --ff-only origin bridgetp/d3-phase6-streaming-transfer

python -m py_compile \
  vllm/bridge_tp/stream_protocol.py \
  vllm/bridge_tp/kv_stream.py \
  vllm/bridge_tp/streaming_connector.py \
  tools/bridge_tp/run_live_migration.py \
  tools/bridge_tp/inspect_stream_run.py

python -m unittest \
  tests.bridge_tp.test_stream_protocol \
  tests.bridge_tp.test_kv_reshard \
  tests.bridge_tp.test_kv_restore
```

All tests must pass, including `test_corrupted_frame_is_rejected`.

## Create one explicit run

Use a new ID for every invocation. Run this once in a setup terminal:

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export PHASE6_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export PHASE6_ROOT=/root/autodl-tmp/bridgetp/results/phase6
export PHASE6_DIR="$PHASE6_ROOT/$PHASE6_ID"
export PHASE6_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
mkdir -p "$PHASE6_DIR"

printf '%s\n' \
  "export PHASE6_ID=$PHASE6_ID" \
  "export PHASE6_ROOT=$PHASE6_ROOT" \
  "export PHASE6_DIR=$PHASE6_DIR" \
  "export PHASE6_MODEL=$PHASE6_MODEL" \
  > /root/autodl-tmp/bridgetp/phase6_session.env

git rev-parse HEAD > "$PHASE6_DIR/git_revision.txt"
python -c 'import sys,torch,vllm; print("python",sys.version); print("torch",torch.__version__); print("torch_cuda",torch.version.cuda); print("vllm",vllm.__version__); print("vllm_path",vllm.__file__)' \
  > "$PHASE6_DIR/environment.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader > "$PHASE6_DIR/gpu_identity.txt"

echo "$PHASE6_ID"
```

Do not reuse a directory that already contains `session_manifest.json`.

## Terminal 1: start TP4 target on GPUs 1-4

The target may start before the manifest exists; it loads the selected session
when the migration request arrives.

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase6_session.env

export PHASE6_MANIFEST="$PHASE6_DIR/session_manifest.json"
export PHASE6_RECEIPTS="$PHASE6_DIR/receiver_receipts"

CUDA_VISIBLE_DEVICES=1,2,3,4 \
python -m vllm.entrypoints.openai.api_server \
  --model "$PHASE6_MODEL" \
  --served-model-name bridgetp-model \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --port 8200 \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  --kv-transfer-config "$(python -c 'import json,os; print(json.dumps({
    "kv_connector":"BridgeTPStreamingConnector",
    "kv_connector_module_path":"vllm.bridge_tp.streaming_connector",
    "kv_role":"kv_consumer",
    "kv_load_failure_policy":"fail",
    "kv_connector_extra_config":{
      "bridgetp_stream_manifest":os.environ["PHASE6_MANIFEST"],
      "bridgetp_stream_receipt_dir":os.environ["PHASE6_RECEIPTS"],
      "bridgetp_stream_socket_timeout_s":600
    }}))')" \
  2>&1 | tee "$PHASE6_DIR/target_tp4.log"
```

## Terminal 2: start live TP1 source on GPU 0

`BRIDGETP_STREAM_RATE_GIB_S=0` means unlimited. For later shaped observations,
use a fresh run ID and set `0.7` or `0.4`; those values are A100 PCIe debug
points from prior P2 evidence, not universal constants.

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase6_session.env

export BRIDGETP_DUMP_ENABLED=0
export BRIDGETP_STREAM_ENABLED=1
export BRIDGETP_STREAM_MIGRATION_ID="$PHASE6_ID"
export BRIDGETP_STREAM_RUN_DIR="$PHASE6_DIR"
export BRIDGETP_STREAM_HOST=127.0.0.1
export BRIDGETP_STREAM_BASE_PORT=29600
export BRIDGETP_STREAM_TARGET_TP=4
export BRIDGETP_STREAM_HEAD_AXIS=3
export BRIDGETP_STREAM_EXPECTED_KV_HEADS=8
export BRIDGETP_STREAM_AFTER_OUTPUT_TOKENS=128
export BRIDGETP_STREAM_CHUNK_BYTES=1048576
export BRIDGETP_STREAM_RATE_GIB_S=0
export BRIDGETP_STREAM_SOCKET_TIMEOUT_S=600
export BRIDGETP_STREAM_PIN_MEMORY=1
export BRIDGETP_STREAM_STRICT=1

CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model "$PHASE6_MODEL" \
  --served-model-name bridgetp-model \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --port 8001 \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  2>&1 | tee "$PHASE6_DIR/source_tp1.log"
```

## Terminal 3: health check and run

First confirm that the five GPUs are occupied by the intended servers:

```bash
source /root/autodl-tmp/bridgetp/phase6_session.env
curl -f http://127.0.0.1:8001/v1/models
curl -f http://127.0.0.1:8200/v1/models
nvidia-smi
```

Create a source request that is forced past the 128-token snapshot boundary:

```bash
python - "$PHASE6_ID" <<'PY'
import json
import sys
from pathlib import Path

run_id = sys.argv[1]
request = {
    "model": "bridgetp-model",
    "prompt": "Explain tensor parallel inference as a numbered technical tutorial. Continue until every major implementation stage is covered.",
    "max_tokens": 160,
    "temperature": 0,
    "ignore_eos": True,
    "stream": False,
    "return_token_ids": True,
}
path = Path(f"/tmp/phase6_source_request_{run_id}.json")
path.write_text(json.dumps(request, indent=2) + "\n")
print(path)
PY

cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

python tools/bridge_tp/run_live_migration.py \
  --source-url http://127.0.0.1:8001 \
  --target-url http://127.0.0.1:8200 \
  --source-request "/tmp/phase6_source_request_$PHASE6_ID.json" \
  --run-dir "$PHASE6_DIR" \
  --continuation-tokens 32 \
  | tee "$PHASE6_DIR/controller_output.txt"
```

The controller prints `"status": "PASS"` and
`"exact_token_continuity": true` only when the live source's tokens 129-160
exactly equal all 32 tokens generated by the restored TP4 request.

## Inspect and archive the exact run

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase6_session.env

python tools/bridge_tp/inspect_stream_run.py "$PHASE6_DIR" \
  | tee "$PHASE6_DIR/inspection.json"

(cd "$PHASE6_DIR" && sha256sum \
  session_manifest.json \
  continuity_result.json \
  source_request.json source_response.json \
  target_request.json target_response.json \
  sender_receipts/tp_rank_*.json \
  receiver_receipts/*/tp_rank_*.json) \
  > "$PHASE6_DIR/SHA256SUMS"

tar -C "$PHASE6_ROOT" -czf \
  "/root/autodl-tmp/bridgetp/phase6_${PHASE6_ID}.tar.gz" \
  "$PHASE6_ID"
sha256sum "/root/autodl-tmp/bridgetp/phase6_${PHASE6_ID}.tar.gz"
```

Pass requires exactly four `READY` sender receipts, exactly four `READY`
receiver receipts, exact readback on all ranks, matching sender/receiver hashes
and byte counts, and exact 32-token continuity. Report the receiver critical
path as the maximum rank time, never the sum.

## Stop

Stop both API server terminals with `Ctrl-C`, then verify that no model process
remains:

```bash
ps -ef | grep 'vllm.entrypoints.openai.api_server' | grep -v grep
nvidia-smi
```

Do not claim ownership takeover from a Phase 6 pass. That is the next phase.
