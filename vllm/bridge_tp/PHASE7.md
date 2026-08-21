# BridgeTP D3 Phase 7：五卡原子接管与回滚

## 目标

Phase 7建立在已经通过的Phase 6实时传输上，在同一个五卡A100 PCIe节点中加入
应用控制面的两阶段接管：

```text
GPU0 TP1 source：继续decode并保留ownership
        |
        | 实时KV快照与四路TCP传输
        v
GPU1-4 TP4 target：scheduler分配 -> 注入 -> 精确回读
        |
        | 四rank停在TARGET_READY，不执行forward
        v
source takeover API：验证四rank -> COMMITTING -> abort(source)
        |
        | 持久化COMMITTED
        v
TP4四rank同时越过barrier，接管剩余generation budget
```

失败路径只允许在commit前发生：控制面写入`ROLLED_BACK`，不abort源请求；TP4目标
退出，TP1继续成为唯一owner。Phase 7必须分别完成一次commit run和一次forced
rollback run，两轮使用不同migration ID并分别归档。

## 精确证据边界

Phase 7迁移的是可重建的请求执行状态：完整token history、computed/pending边界、
KV cache、greedy sampling配置和剩余generation budget。目标端使用新的vLLM内部
Request对象；本阶段不声称把同一个Python对象或任意sampling RNG状态跨进程搬移。

“原子”限定为本实验进程正常运行时的应用控制面顺序：

1. target在四rank exact readback后停在`TARGET_READY`；
2. commit API只有看到四个sender READY和四个receiver TARGET_READY才接受commit；
3. API先持久化`COMMITTING`，再调用vLLM `engine_client.abort()`；
4. abort调用返回后才原子写入`COMMITTED`；
5. TP4只在读取到`COMMITTED + source_abort_dispatched=true`后继续forward。

它不是跨节点共识协议。API进程在`COMMITTING`期间崩溃等故障仍需要后续恢复日志和
reconciler；不能从本实验宣称crash-consistent distributed transaction。

## 一次性准备

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

git fetch origin
git switch bridgetp/d3-phase7-atomic-takeover
git pull --ff-only origin bridgetp/d3-phase7-atomic-takeover

python -m py_compile \
  vllm/bridge_tp/takeover_api.py \
  vllm/bridge_tp/kv_stream.py \
  vllm/bridge_tp/streaming_connector.py \
  tools/bridge_tp/run_atomic_takeover.py \
  tools/bridge_tp/inspect_takeover_run.py

python -m unittest \
  tests.bridge_tp.test_stream_protocol \
  tests.bridge_tp.test_kv_reshard \
  tests.bridge_tp.test_kv_restore
```

## 为一轮实验创建唯一session

commit和rollback必须分别重新执行这一节，并重启两个server。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export PHASE7_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export PHASE7_ROOT=/root/autodl-tmp/bridgetp/results/phase7
export PHASE7_DIR="$PHASE7_ROOT/$PHASE7_ID"
export PHASE7_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
mkdir -p "$PHASE7_DIR"

printf '%s\n' \
  "export PHASE7_ID=$PHASE7_ID" \
  "export PHASE7_ROOT=$PHASE7_ROOT" \
  "export PHASE7_DIR=$PHASE7_DIR" \
  "export PHASE7_MODEL=$PHASE7_MODEL" \
  > /root/autodl-tmp/bridgetp/phase7_session.env

git rev-parse HEAD > "$PHASE7_DIR/git_revision.txt"
python -c 'import sys,torch,vllm; print("python",sys.version); print("torch",torch.__version__); print("torch_cuda",torch.version.cuda); print("vllm",vllm.__version__); print("vllm_path",vllm.__file__)' \
  > "$PHASE7_DIR/environment.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader > "$PHASE7_DIR/gpu_identity.txt"
nvidia-smi topo -m > "$PHASE7_DIR/gpu_topology.txt"
echo "$PHASE7_ID"
```

## 终端1：GPU1-4启动带commit barrier的TP4

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase7_session.env

export PHASE7_MANIFEST="$PHASE7_DIR/session_manifest.json"
export PHASE7_RECEIPTS="$PHASE7_DIR/receiver_receipts"
export PHASE7_CONTROL="$PHASE7_DIR/takeover_state.json"

CUDA_VISIBLE_DEVICES=1,2,3,4 \
python -m vllm.entrypoints.openai.api_server \
  --model "$PHASE7_MODEL" \
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
      "bridgetp_stream_manifest":os.environ["PHASE7_MANIFEST"],
      "bridgetp_stream_receipt_dir":os.environ["PHASE7_RECEIPTS"],
      "bridgetp_stream_socket_timeout_s":600,
      "bridgetp_takeover_control_path":os.environ["PHASE7_CONTROL"],
      "bridgetp_takeover_control_timeout_s":600
    }}))')" \
  2>&1 | tee "$PHASE7_DIR/target_tp4.log"
```

## 终端2：GPU0启动带takeover API的TP1

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase7_session.env

export BRIDGETP_DUMP_ENABLED=0
export BRIDGETP_STREAM_ENABLED=1
export BRIDGETP_STREAM_MIGRATION_ID="$PHASE7_ID"
export BRIDGETP_STREAM_RUN_DIR="$PHASE7_DIR"
export BRIDGETP_STREAM_HOST=127.0.0.1
export BRIDGETP_STREAM_BASE_PORT=29700
export BRIDGETP_STREAM_TARGET_TP=4
export BRIDGETP_STREAM_HEAD_AXIS=3
export BRIDGETP_STREAM_EXPECTED_KV_HEADS=8
export BRIDGETP_STREAM_AFTER_OUTPUT_TOKENS=128
export BRIDGETP_STREAM_CHUNK_BYTES=1048576
export BRIDGETP_STREAM_RATE_GIB_S=0
export BRIDGETP_STREAM_SOCKET_TIMEOUT_S=600
export BRIDGETP_STREAM_PIN_MEMORY=1
export BRIDGETP_STREAM_STRICT=1

export BRIDGETP_TAKEOVER_ENABLED=1
export BRIDGETP_TAKEOVER_MIGRATION_ID="$PHASE7_ID"
export BRIDGETP_TAKEOVER_RUN_DIR="$PHASE7_DIR"

CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model "$PHASE7_MODEL" \
  --served-model-name bridgetp-model \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --port 8001 \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  2>&1 | tee "$PHASE7_DIR/source_tp1.log"
```

## 终端3：创建请求

源请求总generation budget为192个token。快照发生在第128个输出token，commit后
TP4必须接管剩余64个token，而不是重新给目标一个无关的固定budget。

```bash
source /root/autodl-tmp/bridgetp/phase7_session.env

python - "$PHASE7_ID" <<'PY'
import json
import sys
from pathlib import Path

run_id = sys.argv[1]
request = {
    "model": "bridgetp-model",
    "prompt": "Explain tensor parallel inference as a numbered technical tutorial. Continue until every major implementation stage is covered.",
    "max_tokens": 192,
    "temperature": 0,
    "ignore_eos": True,
    "stream": False,
    "return_token_ids": True,
}
path = Path(f"/tmp/phase7_source_request_{run_id}.json")
path.write_text(json.dumps(request, indent=2) + "\n")
print(path)
PY

curl -f http://127.0.0.1:8001/v1/models
curl -f http://127.0.0.1:8200/v1/models
nvidia-smi
```

## Run A：正式commit

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase7_session.env

python tools/bridge_tp/run_atomic_takeover.py \
  --source-url http://127.0.0.1:8001 \
  --target-url http://127.0.0.1:8200 \
  --source-request "/tmp/phase7_source_request_$PHASE7_ID.json" \
  --run-dir "$PHASE7_DIR" \
  | tee "$PHASE7_DIR/controller_output.txt"

python tools/bridge_tp/inspect_takeover_run.py "$PHASE7_DIR" \
  | tee "$PHASE7_DIR/inspection.json"
```

commit run只有同时满足以下条件才PASS：

- 四rank都经过`TARGET_READY`并最终变成`OWNERSHIP_COMMITTED`；
- takeover state为`COMMITTED`；
- `source_abort_dispatched=true`；
- 原TP1请求的`finish_reason=abort`；
- TP4生成完整剩余64个token；
- “源端快照前128 token + TP4后64 token”与干净TP1 control的192 token逐个一致。

## Run B：强制rollback

先停止两个server，为rollback创建新的`PHASE7_ID`，重新启动终端1和终端2，再创建
新的source request。然后执行：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase7_session.env

python tools/bridge_tp/run_atomic_takeover.py \
  --source-url http://127.0.0.1:8001 \
  --target-url http://127.0.0.1:8200 \
  --source-request "/tmp/phase7_source_request_$PHASE7_ID.json" \
  --run-dir "$PHASE7_DIR" \
  --force-rollback-after-ready \
  | tee "$PHASE7_DIR/controller_output.txt"

python tools/bridge_tp/inspect_takeover_run.py "$PHASE7_DIR" \
  | tee "$PHASE7_DIR/inspection.json"
```

rollback run中TP4目标请求报错退出属于预期行为。PASS要求：

- 四rank先完成KV精确回读；
- takeover state为`ROLLED_BACK`；
- `source_abort_dispatched=false`；
- 原TP1请求不是`finish_reason=abort`，并完成全部192个token；
- TP1输出前缀与干净control逐个一致；
- 四个receiver receipt最终全部为`ROLLED_BACK`。

rollback可能使专用TP4验证server进入error状态，因此必须在下一轮前重启，不能复用。

## 每轮归档

```bash
source /root/autodl-tmp/bridgetp/phase7_session.env

(cd "$PHASE7_DIR" && sha256sum \
  session_manifest.json takeover_state.json takeover_result.json \
  source_request.json source_response.json \
  target_request.json target_response.json \
  control_request.json control_response.json \
  sender_receipts/tp_rank_*.json \
  receiver_receipts/*/tp_rank_*.json) \
  > "$PHASE7_DIR/SHA256SUMS"

tar -C "$PHASE7_ROOT" -czf \
  "/root/autodl-tmp/bridgetp/phase7_${PHASE7_ID}.tar.gz" \
  "$PHASE7_ID"
sha256sum "/root/autodl-tmp/bridgetp/phase7_${PHASE7_ID}.tar.gz"
```

rollback run的`target_response.json`可能是空对象，并额外包含`target_error.txt`，这是
预期证据，不能删除。

## Phase 7通过后仍不能宣称

- 任意temperature/top-p请求的RNG状态迁移；
- API或节点崩溃期间的分布式共识和自动恢复；
- new-KV双写与background old-KV并行；
- 多请求并发下的正式handoff stall分布；
- 跨节点或非A100平台的同等性能。

Phase 8再加入new-KV双写、background old-KV movement以及early EOS/cancellation
清理；Phase 9才接控制器并进行正式在线性能实验。
