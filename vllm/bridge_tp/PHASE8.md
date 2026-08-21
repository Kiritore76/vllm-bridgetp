# BridgeTP D3 Phase 8：后台 old-KV、新 KV 双写与取消清理

## 本阶段做什么

Phase 8 不在第160个输出 token 再复制一份完整 KV。它把工作拆成两条并行路径：

```text
第128个输出token
TP1 source ──完整历史KV──> CPU stager（后台四rank传输）
     │
     ├──继续decode 128→160
     └──每轮新增KV token──> CPU stager（四rank增量双写）

第160个输出token
CPU stager 检查增量无缺口 -> 组装最终12个逻辑block
     └──四路TCP──> TP4 exact readback -> TARGET_READY
                         └──复用Phase 7 commit -> abort TP1 -> TP4续写
```

`old-KV/new-KV overlap` 通过初始四rank最后完成时间与第一份增量完成时间比较；本轮
把旧KV聚合带宽限制为 `0.05 GiB/s`，目的是稳定制造可观测重叠，不是性能结论。

Phase 8 要跑两个独立 session：

1. `dualwrite_commit`：证明后台历史搬运、连续新KV增量、精确组装、原子接管和端到端
   token连续性；
2. `pre_cutover_controller_cancellation`：在已有历史KV和至少一份新KV增量后显式取消
   TP1，证明source mirror与CPU stager都释放资源，且不创建target请求、不commit。

## 一次性拉取和检查

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

git fetch origin
git switch bridgetp/d3-phase8-dualwrite-staging
git pull --ff-only origin bridgetp/d3-phase8-dualwrite-staging

python -m py_compile \
  vllm/bridge_tp/kv_stream.py \
  vllm/bridge_tp/phase8_source.py \
  vllm/bridge_tp/streaming_connector.py \
  vllm/bridge_tp/takeover_api.py \
  tools/bridge_tp/phase8_stager.py \
  tools/bridge_tp/run_phase8_bridge.py \
  tools/bridge_tp/run_phase8_cleanup.py \
  tools/bridge_tp/inspect_phase8_run.py

python -m unittest \
  tests.bridge_tp.test_stream_protocol \
  tests.bridge_tp.test_kv_reshard \
  tests.bridge_tp.test_kv_restore \
  tests.bridge_tp.test_phase8_staging
```

## 每一轮创建全新 session

commit与cancel分别执行本节，不能复用ID、目录或server进程。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export PHASE8_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export PHASE8_ROOT=/root/autodl-tmp/bridgetp/results/phase8
export PHASE8_DIR="$PHASE8_ROOT/$PHASE8_ID"
export PHASE8_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
mkdir -p "$PHASE8_DIR"

printf '%s\n' \
  "export PHASE8_ID=$PHASE8_ID" \
  "export PHASE8_ROOT=$PHASE8_ROOT" \
  "export PHASE8_DIR=$PHASE8_DIR" \
  "export PHASE8_MODEL=$PHASE8_MODEL" \
  > /root/autodl-tmp/bridgetp/phase8_session.env

git rev-parse HEAD > "$PHASE8_DIR/git_revision.txt"
python -c 'import sys,torch,vllm; print("python",sys.version); print("torch",torch.__version__); print("torch_cuda",torch.version.cuda); print("vllm",vllm.__version__); print("vllm_path",vllm.__file__)' \
  > "$PHASE8_DIR/environment.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader > "$PHASE8_DIR/gpu_identity.txt"
nvidia-smi topo -m > "$PHASE8_DIR/gpu_topology.txt"
echo "$PHASE8_ID"
```

## 终端1：GPU1-4 启动 TP4 target

取消清理轮不创建target请求，因此该轮可不启动本终端。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase8_session.env

export PHASE8_STAGING_MANIFEST="$PHASE8_DIR/staging_manifest.json"
export PHASE8_RECEIPTS="$PHASE8_DIR/receiver_receipts"
export PHASE8_CONTROL="$PHASE8_DIR/takeover_state.json"

CUDA_VISIBLE_DEVICES=1,2,3,4 \
python -m vllm.entrypoints.openai.api_server \
  --model "$PHASE8_MODEL" \
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
      "bridgetp_stream_manifest":os.environ["PHASE8_STAGING_MANIFEST"],
      "bridgetp_stream_receipt_dir":os.environ["PHASE8_RECEIPTS"],
      "bridgetp_stream_socket_timeout_s":600,
      "bridgetp_stream_expected_phase":"BridgeTP D3 Phase 8",
      "bridgetp_takeover_control_path":os.environ["PHASE8_CONTROL"],
      "bridgetp_takeover_control_timeout_s":600
    }}))')" \
  2>&1 | tee "$PHASE8_DIR/target_tp4.log"
```

## 终端2：GPU0 启动 TP1 source

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase8_session.env

export BRIDGETP_DUMP_ENABLED=0
export BRIDGETP_STREAM_ENABLED=1
export BRIDGETP_STREAM_MIGRATION_ID="$PHASE8_ID"
export BRIDGETP_STREAM_RUN_DIR="$PHASE8_DIR"
export BRIDGETP_STREAM_HOST=127.0.0.1
export BRIDGETP_STREAM_BASE_PORT=29800
export BRIDGETP_STREAM_TARGET_TP=4
export BRIDGETP_STREAM_HEAD_AXIS=3
export BRIDGETP_STREAM_EXPECTED_KV_HEADS=8
export BRIDGETP_STREAM_AFTER_OUTPUT_TOKENS=128
export BRIDGETP_STREAM_CHUNK_BYTES=1048576
export BRIDGETP_STREAM_RATE_GIB_S=0.05
export BRIDGETP_STREAM_SOCKET_TIMEOUT_S=600
export BRIDGETP_STREAM_PIN_MEMORY=1
export BRIDGETP_STREAM_STRICT=1

export BRIDGETP_PHASE8_ENABLED=1
export BRIDGETP_PHASE8_CUTOVER_OUTPUT_TOKENS=160
export BRIDGETP_PHASE8_DELTA_HOST=127.0.0.1
export BRIDGETP_PHASE8_DELTA_BASE_PORT=29900

export BRIDGETP_TAKEOVER_ENABLED=1
export BRIDGETP_TAKEOVER_MIGRATION_ID="$PHASE8_ID"
export BRIDGETP_TAKEOVER_RUN_DIR="$PHASE8_DIR"

CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model "$PHASE8_MODEL" \
  --served-model-name bridgetp-model \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --port 8001 \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  2>&1 | tee "$PHASE8_DIR/source_tp1.log"
```

## 终端3：启动 CPU stager

在发送请求前启动。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase8_session.env

python tools/bridge_tp/phase8_stager.py \
  --run-dir "$PHASE8_DIR" \
  --delta-host 127.0.0.1 \
  --delta-base-port 29900 \
  --delivery-host 127.0.0.1 \
  --delivery-base-port 30000 \
  --timeout-s 600 \
  2>&1 | tee "$PHASE8_DIR/stager.log"
```

## 终端4：创建公共请求文件

```bash
source /root/autodl-tmp/bridgetp/phase8_session.env

python - "$PHASE8_ID" <<'PY'
import json
import sys
from pathlib import Path

request = {
    "model": "bridgetp-model",
    "prompt": "Explain tensor parallel inference as a numbered technical tutorial. Continue until every major implementation stage is covered.",
    "max_tokens": 288,
    "temperature": 0,
    "ignore_eos": True,
    "stream": False,
    "return_token_ids": True,
}
path = Path(f"/tmp/phase8_source_request_{sys.argv[1]}.json")
path.write_text(json.dumps(request, indent=2) + "\n")
print(path)
PY
```

## Run A：dualwrite commit

确认三个server/stager都已启动：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase8_session.env

curl -f http://127.0.0.1:8001/v1/models
curl -f http://127.0.0.1:8200/v1/models

python tools/bridge_tp/run_phase8_bridge.py \
  --source-url http://127.0.0.1:8001 \
  --target-url http://127.0.0.1:8200 \
  --source-request "/tmp/phase8_source_request_$PHASE8_ID.json" \
  --run-dir "$PHASE8_DIR" \
  | tee "$PHASE8_DIR/controller_output.txt"

python tools/bridge_tp/inspect_phase8_run.py "$PHASE8_DIR" \
  | tee "$PHASE8_DIR/inspection.json"
```

PASS必须同时满足：四rank增量从初始computed边界连续覆盖到cutover、old/new传输时间
重叠、四rank exact readback、`COMMITTED + source abort`，以及组装后的288个输出token
与干净TP1 control逐个一致。

## Run B：pre-cutover cancellation cleanup

停止上一轮所有进程；重新创建ID和目录，重启终端2与终端3，重新创建请求文件。不需要
TP4 target。执行：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase8_session.env

python tools/bridge_tp/run_phase8_cleanup.py \
  --source-url http://127.0.0.1:8001 \
  --source-request "/tmp/phase8_source_request_$PHASE8_ID.json" \
  --run-dir "$PHASE8_DIR" \
  | tee "$PHASE8_DIR/controller_output.txt"

python tools/bridge_tp/inspect_phase8_run.py "$PHASE8_DIR" \
  | tee "$PHASE8_DIR/inspection.json"
```

PASS要求取消发生在初始old-KV和至少一份new-KV delta到达之后；TP1返回`abort`，source
mirror与CPU stager均写出`CLEANED` receipt，stager释放四rank buffer，并且不存在
`staging_manifest.json`和target request。

## 每轮只归档一次

```bash
source /root/autodl-tmp/bridgetp/phase8_session.env

(
  cd "$PHASE8_DIR" || exit 1
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum \
    > SHA256SUMS
  sha256sum -c SHA256SUMS
)

tar -C "$PHASE8_ROOT" -czf \
  "/root/autodl-tmp/bridgetp/phase8_${PHASE8_ID}.tar.gz" \
  "$PHASE8_ID"
sha256sum "/root/autodl-tmp/bridgetp/phase8_${PHASE8_ID}.tar.gz"
```

## 证据边界

Phase 8实现的是同节点CPU staging与token粒度new-KV mirror。它不等于GPU P2P/NVLink
直传，也不证明跨节点RDMA、任意sampling RNG、多请求并发、进程崩溃恢复或正式线上
性能。自然EOS可走同一cleanup信号，但本阶段的负向实验明确验证的是控制器显式取消；
不能把它写成“已验证所有EOS时序”。
