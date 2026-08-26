# BridgeTP Phase 9 numerical-fidelity 实验手册

本手册对应分支 `bridgetp/d3-phase9-numerical-fidelity`。目标是采集 D-0、D-1、
D-2 和配对 D-3 证据；它不重新设计 Phase 1-8，不恢复 agreement-gate，也不把
`migrated == TP1 control` 静默改写成已经通过的新标准。

当前所有正式 Phase 9 证据继续限定在同一台 `5 x NVIDIA A100-PCIE-40GB` 服务器。
其他 GPU 平台必须使用独立目录和独立结论。

## 0. 符号与四组实验

```text
K = 迁移时 KV cache 实际覆盖的精确输出 token 边界
A = native TP1 vs native TP1 rerun
B = native TP1 vs native TP4（纯 TP 拓扑对照）
C = native TP4 bs=1 vs native TP4 bs=8（batch/调度对照）
D = migrated vs native TP1 control（最终 migration 对象）
```

所有 B/C/D continuation 必须从相同固定 token IDs 开始：

```text
fixed prefix = original prompt token IDs + TP1 control 前 K 个输出 token IDs
```

禁止让 TP1 和 TP4 从同一文本 prompt 自由生成到 K。它们可能在 K 之前已经分叉。

## 1. 拉取代码与 CPU 检查

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

git fetch origin
git switch bridgetp/d3-phase9-numerical-fidelity
git pull --ff-only origin bridgetp/d3-phase9-numerical-fidelity

export BRIDGE_PY=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python

"$BRIDGE_PY" -m py_compile \
  vllm/bridge_tp/config.py \
  vllm/bridge_tp/kv_export.py \
  vllm/bridge_tp/logit_capture.py \
  vllm/bridge_tp/controller/numerics.py \
  vllm/v1/sample/sampler.py \
  vllm/v1/worker/gpu_model_runner.py \
  tools/bridge_tp/probe_logit_ulp.py \
  tools/bridge_tp/compare_kv_provenance.py \
  tools/bridge_tp/run_fixed_prefix_continuation.py \
  tools/bridge_tp/measure_agreement.py \
  tools/bridge_tp/summarize_agreement.py

"$BRIDGE_PY" -m unittest tests.bridge_tp.test_phase9_numerics
"$BRIDGE_PY" -m unittest discover -s tests -t . -p 'test_phase9_*.py'
```

所有 server、stager 和 controller 进程必须在切换分支后重启。

## 2. 每轮都要创建的新 session

以下变量示例用于一个请求。每轮必须使用全新 UUID 和目录。

```bash
export P9NF_ID="$("$BRIDGE_PY" -c 'import uuid; print(uuid.uuid4())')"
export P9NF_ROOT=/root/autodl-tmp/bridgetp/results/phase9_numerical_fidelity
export P9NF_DIR="$P9NF_ROOT/$P9NF_ID"
export P9NF_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
export P9NF_DTYPE=bfloat16
mkdir -p "$P9NF_DIR"

git rev-parse HEAD > "$P9NF_DIR/git_revision.txt"
"$BRIDGE_PY" -c 'import sys,torch,vllm; print("python",sys.version); print("torch",torch.__version__); print("cuda",torch.version.cuda); print("vllm",vllm.__version__); print("vllm_path",vllm.__file__)' \
  > "$P9NF_DIR/environment.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader > "$P9NF_DIR/gpu_identity.txt"
nvidia-smi topo -m > "$P9NF_DIR/gpu_topology.txt"

printf '%s\n' \
  "export BRIDGE_PY=$BRIDGE_PY" \
  "export P9NF_ID=$P9NF_ID" \
  "export P9NF_ROOT=$P9NF_ROOT" \
  "export P9NF_DIR=$P9NF_DIR" \
  "export P9NF_MODEL=$P9NF_MODEL" \
  "export P9NF_DTYPE=$P9NF_DTYPE" \
  > /root/autodl-tmp/bridgetp/phase9_nf_session.env
```

先按 `PHASE9_SERVER_TEST.md` 完成一次不带 D-0 capture 的 discovery run，得到：

```bash
export P9NF_K=53                         # 以本次 response_proxy_stats.json 为准
export P9NF_DIVERGENCE_INDEX=99          # 以本次 token_equivalence.json 为准
export P9NF_CONTROL_TOKEN_ID=8381        # 以本次真实结果为准
export P9NF_MIGRATED_TOKEN_ID=1372       # 以本次真实结果为准
```

这些数字只是现有单点示例。新请求不得照抄。

## 3. D-0/D-1 evidence rerun：启动完整迁移链

本节开启同步 D2H tensor capture，只用于 numerical-fidelity 正确性证据。该轮的
handoff stall、TPOT、E2E 和 throughput 不能进入正式性能统计。正式性能轮必须关闭
`BRIDGETP_LOGIT_CAPTURE_ENABLED` 和 `BRIDGETP_DUMP_ENABLED` 后另跑至少三轮。

### 3.1 终端 1：TP4 target

下面同时打开两项 opt-in 诊断：

- D-0：在真实 vLLM sampler 内采集指定全局 index 的 raw/processed logits；
- D-1：在 target 产生第一个本地输出时导出迁移后的 TP4 四 rank KV，此时新 token
  仍是 pending，dump 中 computed KV 正好覆盖 fixed prefix K。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_nf_session.env

export P9NF_K=替换为本次K
export P9NF_DIVERGENCE_INDEX=替换为本次首次分歧全局index
export P9NF_CONTROL_TOKEN_ID=替换为control token ID
export P9NF_MIGRATED_TOKEN_ID=替换为migrated token ID

export PHASE9_STAGING_MANIFEST="$P9NF_DIR/staging_manifest.json"
export PHASE9_RECEIPTS="$P9NF_DIR/receiver_receipts"
export PHASE9_CONTROL="$P9NF_DIR/takeover_state.json"

export BRIDGETP_LOGIT_CAPTURE_ENABLED=1
export BRIDGETP_LOGIT_CAPTURE_DIR="$P9NF_DIR/logit_captures/migrated"
export BRIDGETP_LOGIT_CAPTURE_INDICES="$P9NF_DIVERGENCE_INDEX"
export BRIDGETP_LOGIT_CAPTURE_GLOBAL_OFFSET="$P9NF_K"
export BRIDGETP_LOGIT_CAPTURE_CANDIDATE_TOKEN_IDS="$P9NF_CONTROL_TOKEN_ID,$P9NF_MIGRATED_TOKEN_ID"
export BRIDGETP_LOGIT_CAPTURE_TOPK=20
export BRIDGETP_LOGIT_CAPTURE_TENSORS=1
export BRIDGETP_LOGIT_CAPTURE_TP_RANK=0
export BRIDGETP_LOGIT_CAPTURE_STRICT=1

export BRIDGETP_DUMP_ENABLED=1
export BRIDGETP_DUMP_TENSORS=1
export BRIDGETP_DUMP_AFTER_OUTPUT_TOKENS=1
export BRIDGETP_DUMP_DIR="$P9NF_DIR/d1_migrated"
export BRIDGETP_DUMP_TP_WORLD_SIZES=1,4
export BRIDGETP_DUMP_STRICT=1

CUDA_VISIBLE_DEVICES=1,2,3,4 \
"$BRIDGE_PY" -m vllm.entrypoints.openai.api_server \
  --model "$P9NF_MODEL" \
  --served-model-name bridgetp-model \
  --tensor-parallel-size 4 \
  --dtype "$P9NF_DTYPE" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --port 8200 \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  --kv-transfer-config "$("$BRIDGE_PY" -c 'import json,os; print(json.dumps({
    "kv_connector":"BridgeTPStreamingConnector",
    "kv_connector_module_path":"vllm.bridge_tp.streaming_connector",
    "kv_role":"kv_consumer",
    "kv_load_failure_policy":"fail",
    "kv_connector_extra_config":{
      "bridgetp_stream_manifest":os.environ["PHASE9_STAGING_MANIFEST"],
      "bridgetp_stream_receipt_dir":os.environ["PHASE9_RECEIPTS"],
      "bridgetp_stream_socket_timeout_s":600,
      "bridgetp_stream_expected_phase":"BridgeTP D3 Phase 8",
      "bridgetp_takeover_control_path":os.environ["PHASE9_CONTROL"],
      "bridgetp_takeover_control_timeout_s":600}}))')" \
  2>&1 | tee "$P9NF_DIR/target_tp4.log"
```

### 3.2 终端 2：TP1 source 和后续 clean control capture

迁移 source 会在 K 被 abort，因此不会到达分歧 index。迁移完成后，同一 server 上的
clean control 会到达该 index，D-0 capture 目录中应当只有 control 的对应记录。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_nf_session.env

export BRIDGETP_LOGIT_CAPTURE_ENABLED=1
export BRIDGETP_LOGIT_CAPTURE_DIR="$P9NF_DIR/logit_captures/control"
export BRIDGETP_LOGIT_CAPTURE_INDICES="$P9NF_DIVERGENCE_INDEX"
export BRIDGETP_LOGIT_CAPTURE_GLOBAL_OFFSET=0
export BRIDGETP_LOGIT_CAPTURE_CANDIDATE_TOKEN_IDS="$P9NF_CONTROL_TOKEN_ID,$P9NF_MIGRATED_TOKEN_ID"
export BRIDGETP_LOGIT_CAPTURE_TOPK=20
export BRIDGETP_LOGIT_CAPTURE_TENSORS=1
export BRIDGETP_LOGIT_CAPTURE_TP_RANK=0
export BRIDGETP_LOGIT_CAPTURE_STRICT=1

export BRIDGETP_DUMP_ENABLED=0
export BRIDGETP_STREAM_ENABLED=1
export BRIDGETP_STREAM_MIGRATION_ID="$P9NF_ID"
export BRIDGETP_STREAM_RUN_DIR="$P9NF_DIR"
export BRIDGETP_STREAM_HOST=127.0.0.1
export BRIDGETP_STREAM_BASE_PORT=29800
export BRIDGETP_STREAM_TARGET_TP=4
export BRIDGETP_STREAM_HEAD_AXIS=3
export BRIDGETP_STREAM_EXPECTED_KV_HEADS=8
export BRIDGETP_STREAM_AFTER_OUTPUT_TOKENS=128
export BRIDGETP_STREAM_CHUNK_BYTES=1048576
export BRIDGETP_STREAM_RATE_GIB_S=0.50
export BRIDGETP_STREAM_SOCKET_TIMEOUT_S=600
export BRIDGETP_STREAM_PIN_MEMORY=1
export BRIDGETP_STREAM_STRICT=1
export BRIDGETP_PHASE8_ENABLED=1
export BRIDGETP_PHASE8_CUTOVER_OUTPUT_TOKENS=160
export BRIDGETP_PHASE8_DELTA_HOST=127.0.0.1
export BRIDGETP_PHASE8_DELTA_BASE_PORT=29900
export BRIDGETP_TAKEOVER_ENABLED=1
export BRIDGETP_TAKEOVER_MIGRATION_ID="$P9NF_ID"
export BRIDGETP_TAKEOVER_RUN_DIR="$P9NF_DIR"

CUDA_VISIBLE_DEVICES=0 \
"$BRIDGE_PY" -m vllm.entrypoints.openai.api_server \
  --model "$P9NF_MODEL" \
  --served-model-name bridgetp-model \
  --tensor-parallel-size 1 \
  --dtype "$P9NF_DTYPE" \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --port 8001 \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  2>&1 | tee "$P9NF_DIR/source_tp1.log"
```

### 3.3 终端 3：CPU stager

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_nf_session.env

"$BRIDGE_PY" tools/bridge_tp/phase8_stager.py \
  --run-dir "$P9NF_DIR" \
  --delta-host 127.0.0.1 \
  --delta-base-port 29900 \
  --delivery-host 127.0.0.1 \
  --delivery-base-port 30000 \
  --timeout-s 600 \
  2>&1 | tee "$P9NF_DIR/stager.log"
```

### 3.4 准备 controller config

从两个 server 日志读取每 rank KV block 数。TP4 数值不能乘 4。

```bash
grep -Ei 'GPU KV cache size|GPU blocks|num_gpu_blocks' \
  "$P9NF_DIR/source_tp1.log" "$P9NF_DIR/target_tp4.log" | tail -20

export TP1_BLOCKS=替换为TP1每rank实际值
export TP4_BLOCKS=替换为TP4每rank实际值
export P9NF_SURVIVAL=/root/autodl-tmp/bridgetp/calibration/survival_table.json

"$BRIDGE_PY" - "$P9NF_DIR" "$TP1_BLOCKS" "$TP4_BLOCKS" "$P9NF_SURVIVAL" <<'PY'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
cfg = json.loads(Path('experiments/phase9/configs/e1_correctness.json').read_text())
cfg['run_dir'] = str(run)
cfg['tp1_total_kv_blocks'] = int(sys.argv[2])
cfg['tp4_total_kv_blocks'] = int(sys.argv[3])
cfg['survival_table_path'] = sys.argv[4]
cfg['platform_note'] = '5x NVIDIA A100-PCIE-40GB; Phase 9 numerical fidelity'
(run / 'phase9_nf_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
PY
```

正式 D-3 不得把 smoke 中用于强制迁移的 `theta_0=0` 当作论文策略参数。

### 3.5 终端 4：运行 controller

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_nf_session.env

"$BRIDGE_PY" tools/bridge_tp/run_phase9_controller.py \
  --config "$P9NF_DIR/phase9_nf_config.json" \
  --run-dir "$P9NF_DIR" \
  --migration-id "$P9NF_ID" \
  --source-request experiments/phase9/configs/request_long.json \
  2>&1 | tee "$P9NF_DIR/controller_output.txt"
```

### 3.6 生成 clean TP1 control

```bash
"$BRIDGE_PY" - "$P9NF_DIR" <<'PY'
import json, sys, urllib.request
from pathlib import Path

run = Path(sys.argv[1])
payload = json.loads((run / 'source_request.json').read_text())
payload.pop('request_id', None)
payload['request_id'] = 'phase9-nf-control-' + run.name
payload['stream'] = False
payload['return_token_ids'] = True
(run / 'control_request.json').write_text(json.dumps(payload, indent=2) + '\n')
request = urllib.request.Request(
    'http://127.0.0.1:8001/v1/completions',
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'}, method='POST')
with urllib.request.urlopen(request, timeout=1800) as response:
    result = json.load(response)
tokens = result['choices'][0]['token_ids']
(run / 'control_response.json').write_text(json.dumps(result, indent=2) + '\n')
(run / 'control_tokens.json').write_text(json.dumps(tokens, indent=2) + '\n')
print('control tokens', len(tokens))
PY
```

### 3.7 原机械门与分层 inspector

```bash
"$BRIDGE_PY" tools/bridge_tp/probe_token_divergence.py \
  --run-dir "$P9NF_DIR" \
  --source-url http://127.0.0.1:8001 \
  --target-url http://127.0.0.1:8200 \
  --tokenizer "$P9NF_MODEL" \
  | tee "$P9NF_DIR/token_equivalence_output.txt"

"$BRIDGE_PY" tools/bridge_tp/inspect_phase9_run.py \
  --run-dir "$P9NF_DIR" \
  --expect commit \
  --control-tokens "$P9NF_DIR/control_tokens.json" \
  --json | tee "$P9NF_DIR/inspect.json"
```

`inspect.json` 的 tiered evidence 不会改写旧 PASS/FAIL；Tier 1 mechanical、Tier 2
numerical 和 Tier 3 semantic 分开记录。

## 4. D-0：真实 raw/processed logits + ULP

定位 control 和 migrated 各自唯一的 capture：

```bash
export P9NF_CONTROL_CAPTURE="$("$BRIDGE_PY" - "$P9NF_DIR" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'logit_captures/control'
matches = list(p.glob('*/global_*/capture.json'))
assert len(matches) == 1, matches
print(matches[0])
PY
)"

export P9NF_MIGRATED_CAPTURE="$("$BRIDGE_PY" - "$P9NF_DIR" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'logit_captures/migrated'
matches = list(p.glob('*/global_*/capture.json'))
assert len(matches) == 1, matches
print(matches[0])
PY
)"

"$BRIDGE_PY" tools/bridge_tp/probe_logit_ulp.py \
  --control-capture "$P9NF_CONTROL_CAPTURE" \
  --migrated-capture "$P9NF_MIGRATED_CAPTURE" \
  --candidate-token-ids "$P9NF_CONTROL_TOKEN_ID" "$P9NF_MIGRATED_TOKEN_ID" \
  --out "$P9NF_DIR/logit_ulp_analysis.json" \
  | tee "$P9NF_DIR/logit_ulp_output.txt"
```

只有实际 capture 中记录的 tensor dtype 和 raw values 才能用于 ULP 数。HF 单卡重算
只能作为辅助，不能替代这个文件，也不能由 `0.125` 单独推出“一个 BF16 ULP”。

## 5. D-1：构造相同固定 prefix

```bash
"$BRIDGE_PY" - "$P9NF_DIR" "$P9NF_K" "$P9NF_MODEL" <<'PY'
import json, sys
from pathlib import Path

run, k, model = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
staging = json.loads((run / 'staging_manifest.json').read_text())
control = json.loads((run / 'control_tokens.json').read_text())
num_prompt = int(staging['num_prompt_tokens'])
prompt = [int(x) for x in staging['all_known_token_ids'][:num_prompt]]
fixed = prompt + [int(x) for x in control[:k]]
assert len(control) >= k
payload = {
    'format_version': 1,
    'logical_request_id': run.name,
    'model': model,
    'boundary_k': k,
    'prompt_token_count': len(prompt),
    'fixed_token_ids': fixed,
}
(run / 'fixed_prefix.json').write_text(json.dumps(payload, indent=2) + '\n')
(run / 'fixed_prefix_tokens.json').write_text(json.dumps(fixed, indent=2) + '\n')
print('fixed prefix tokens', len(fixed), 'K', k)
PY
```

### 5.1 干净 TP1 dump（GPU0）

先停止迁移 session 的 TP1/TP4 server。然后启动：

```bash
export P9NF_D1_TP1_REQUEST="d1-$P9NF_ID-tp1"
export BRIDGETP_DUMP_ENABLED=1
export BRIDGETP_DUMP_TENSORS=1
export BRIDGETP_DUMP_AFTER_OUTPUT_TOKENS=1
export BRIDGETP_DUMP_REQUEST_ID="$P9NF_D1_TP1_REQUEST-primary"
export BRIDGETP_DUMP_DIR="$P9NF_DIR/d1_tp1"
export BRIDGETP_DUMP_TP_WORLD_SIZES=1,4
export BRIDGETP_DUMP_STRICT=1
export BRIDGETP_LOGIT_CAPTURE_ENABLED=0

CUDA_VISIBLE_DEVICES=0 \
"$BRIDGE_PY" -m vllm.entrypoints.openai.api_server \
  --model "$P9NF_MODEL" --served-model-name bridgetp-model \
  --tensor-parallel-size 1 --dtype "$P9NF_DTYPE" \
  --max-model-len 8192 --gpu-memory-utilization 0.88 --port 8001 \
  --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \
  --no-async-scheduling 2>&1 | tee "$P9NF_DIR/d1_tp1_server.log"
```

另一个终端发送精确 token IDs：

```bash
"$BRIDGE_PY" tools/bridge_tp/run_fixed_prefix_continuation.py \
  --base-url http://127.0.0.1:8001 \
  --model bridgetp-model \
  --fixed-prefix-token-ids "$P9NF_DIR/fixed_prefix.json" \
  --request-id "$P9NF_D1_TP1_REQUEST" \
  --max-tokens 1 --batch-size 1 \
  --out-dir "$P9NF_DIR/d1_tp1_request"
```

### 5.2 干净 TP4 dump（GPU1-4）

```bash
export P9NF_D1_TP4_REQUEST="d1-$P9NF_ID-native-tp4"
export BRIDGETP_DUMP_ENABLED=1
export BRIDGETP_DUMP_TENSORS=1
export BRIDGETP_DUMP_AFTER_OUTPUT_TOKENS=1
export BRIDGETP_DUMP_REQUEST_ID="$P9NF_D1_TP4_REQUEST-primary"
export BRIDGETP_DUMP_DIR="$P9NF_DIR/d1_native_tp4"
export BRIDGETP_DUMP_TP_WORLD_SIZES=1,4
export BRIDGETP_DUMP_STRICT=1
export BRIDGETP_LOGIT_CAPTURE_ENABLED=0

CUDA_VISIBLE_DEVICES=1,2,3,4 \
"$BRIDGE_PY" -m vllm.entrypoints.openai.api_server \
  --model "$P9NF_MODEL" --served-model-name bridgetp-model \
  --tensor-parallel-size 4 --dtype "$P9NF_DTYPE" \
  --max-model-len 8192 --gpu-memory-utilization 0.88 --port 8200 \
  --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \
  --no-async-scheduling 2>&1 | tee "$P9NF_DIR/d1_native_tp4_server.log"
```

另一个终端发送：

```bash
"$BRIDGE_PY" tools/bridge_tp/run_fixed_prefix_continuation.py \
  --base-url http://127.0.0.1:8200 \
  --model bridgetp-model \
  --fixed-prefix-token-ids "$P9NF_DIR/fixed_prefix.json" \
  --request-id "$P9NF_D1_TP4_REQUEST" \
  --max-tokens 1 --batch-size 1 \
  --out-dir "$P9NF_DIR/d1_native_tp4_request"
```

### 5.3 比较三种 provenance

先自动定位本次唯一 migrated request dump，拒绝 `find | head`：

```bash
export P9NF_MIGRATED_DUMP_ROOT="$("$BRIDGE_PY" - "$P9NF_DIR" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1]) / 'd1_migrated'
matches = [x for x in p.iterdir() if x.is_dir()]
assert len(matches) == 1, matches
print(matches[0])
PY
)"

"$BRIDGE_PY" tools/bridge_tp/compare_kv_provenance.py \
  --tp1-dump-dir \
    "$P9NF_DIR/d1_tp1/$P9NF_D1_TP1_REQUEST-primary/tp_rank_0" \
  --native-tp4-dump-dirs \
    "$P9NF_DIR/d1_native_tp4/$P9NF_D1_TP4_REQUEST-primary/tp_rank_0" \
    "$P9NF_DIR/d1_native_tp4/$P9NF_D1_TP4_REQUEST-primary/tp_rank_1" \
    "$P9NF_DIR/d1_native_tp4/$P9NF_D1_TP4_REQUEST-primary/tp_rank_2" \
    "$P9NF_DIR/d1_native_tp4/$P9NF_D1_TP4_REQUEST-primary/tp_rank_3" \
  --migrated-tp4-dump-dirs \
    "$P9NF_MIGRATED_DUMP_ROOT/tp_rank_0" \
    "$P9NF_MIGRATED_DUMP_ROOT/tp_rank_1" \
    "$P9NF_MIGRATED_DUMP_ROOT/tp_rank_2" \
    "$P9NF_MIGRATED_DUMP_ROOT/tp_rank_3" \
  --fixed-token-ids "$P9NF_DIR/fixed_prefix.json" \
  --boundary-k "$P9NF_K" \
  --logical-request-id "$P9NF_ID" \
  --out "$P9NF_DIR/kv_provenance.json" \
  | tee "$P9NF_DIR/kv_provenance_output.txt"
```

工具会先验证 model、token IDs、K 对应的 computed-token 数、layer、block size、dtype、
rank 和轴布局，再报告每层每 rank 的 `max_abs_diff`、RMS、绝对差分位数、稳健相对
误差和 ULP 距离直方图。它不会根据未经标定的阈值自动宣布 migration 无责。

## 6. D-2：精度敏感性（探索项）

至少完成 BF16 基线。若 A100 40GB 容量允许，再用全新 session 分别设置：

```bash
export P9NF_DTYPE=bfloat16
# 或
export P9NF_DTYPE=float16
```

然后完整重复第 2-5 节，不能只改变 TP4 一侧却把结果写成端到端精度实验。
`float32` 的 Qwen2.5-14B TP1 很可能放不下，不得强行要求；如果只提升局部 reduction 或
LM head，必须使用单独标签，例如 `bf16_weights_fp32_lm_head`。

每档用第 8 节的 group D record 记录 target-local agreement，再生成：

```json
[
  {"dtype":"bfloat16","request_id":"...","agreement_length":46},
  {"dtype":"float16","request_id":"...","agreement_length":118}
]
```

保存为 `$P9NF_DIR/precision_sweep.json`。单调变长只是探索性观察；不单调也必须保留。

## 7. D-3：准备 30-50 个完全配对的请求

正式结论至少 30 个、最好 50 个 request ID。冻结以下字段并写入每个请求的 metadata：

```bash
export P9NF_PLATFORM='5x NVIDIA A100-PCIE-40GB'
export P9NF_COMMIT="$(git rev-parse HEAD)"
export P9NF_CUTOVER_RULE='冻结后的正式cutover规则字符串'
export P9NF_TARGET_BUDGET=256
```

对每个请求先完成一次 D 组真实迁移，得到：

```text
requests/<RID>/control_tokens.json
requests/<RID>/migrated_tokens.json
requests/<RID>/fixed_prefix.json
requests/<RID>/K.txt
```

其中 `migrated_tokens.json` 是统一客户端完整输出，不能用 native TP4 替代。每个请求的
fixed prefix 都按第 5 节重建。

为该请求生成 metadata：

```bash
export RID=替换为逻辑request_id
export REQ_DIR="$P9NF_ROOT/requests/$RID"

"$BRIDGE_PY" - "$REQ_DIR" "$P9NF_MODEL" "$P9NF_PLATFORM" \
  "$P9NF_COMMIT" "$P9NF_CUTOVER_RULE" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
value = {
    'model': sys.argv[2],
    'gpu_platform': sys.argv[3],
    'vllm_commit': sys.argv[4],
    'cutover_rule': sys.argv[5],
}
(p / 'metadata.json').write_text(json.dumps(value, indent=2) + '\n')
PY
```

## 8. 在固定 prefix 后生成 A/B/C 对照

保持 clean TP1 和 clean TP4 server 常驻，均关闭 prefix caching、async scheduling 和
speculative decoding。对每个 RID 运行：

```bash
# A 的两个 TP1 rerun
"$BRIDGE_PY" tools/bridge_tp/run_fixed_prefix_continuation.py \
  --base-url http://127.0.0.1:8001 --model bridgetp-model \
  --fixed-prefix-token-ids "$REQ_DIR/fixed_prefix.json" \
  --request-id "$RID-tp1-a" --max-tokens "$P9NF_TARGET_BUDGET" \
  --batch-size 1 --out-dir "$REQ_DIR/tp1_a"

"$BRIDGE_PY" tools/bridge_tp/run_fixed_prefix_continuation.py \
  --base-url http://127.0.0.1:8001 --model bridgetp-model \
  --fixed-prefix-token-ids "$REQ_DIR/fixed_prefix.json" \
  --request-id "$RID-tp1-b" --max-tokens "$P9NF_TARGET_BUDGET" \
  --batch-size 1 --out-dir "$REQ_DIR/tp1_b"

# B 的 native TP4 bs=1
"$BRIDGE_PY" tools/bridge_tp/run_fixed_prefix_continuation.py \
  --base-url http://127.0.0.1:8200 --model bridgetp-model \
  --fixed-prefix-token-ids "$REQ_DIR/fixed_prefix.json" \
  --request-id "$RID-tp4-bs1" --max-tokens "$P9NF_TARGET_BUDGET" \
  --batch-size 1 --out-dir "$REQ_DIR/tp4_bs1"

# C 的 native TP4 bs=8；primary 是被测请求，另外 7 个是同时提交的 filler
"$BRIDGE_PY" tools/bridge_tp/run_fixed_prefix_continuation.py \
  --base-url http://127.0.0.1:8200 --model bridgetp-model \
  --fixed-prefix-token-ids "$REQ_DIR/fixed_prefix.json" \
  --request-id "$RID-tp4-bs8" --max-tokens "$P9NF_TARGET_BUDGET" \
  --batch-size 8 --out-dir "$REQ_DIR/tp4_bs8"
```

## 9. 为同一 RID 生成 A/B/C/D target-local record

```bash
export K="$(tr -d '[:space:]' < "$REQ_DIR/K.txt")"
mkdir -p "$P9NF_ROOT/agreement_records"/{A,B,C,D}

"$BRIDGE_PY" tools/bridge_tp/measure_agreement.py \
  --group A --request-id "$RID" \
  --left "$REQ_DIR/tp1_a/primary_tokens.json" \
  --right "$REQ_DIR/tp1_b/primary_tokens.json" \
  --boundary-k "$K" --budget "$P9NF_TARGET_BUDGET" \
  --fixed-prefix-token-ids "$REQ_DIR/fixed_prefix.json" \
  --metadata "$REQ_DIR/metadata.json" \
  --out "$P9NF_ROOT/agreement_records/A/$RID.json"

"$BRIDGE_PY" tools/bridge_tp/measure_agreement.py \
  --group B --request-id "$RID" \
  --left "$REQ_DIR/tp1_a/primary_tokens.json" \
  --right "$REQ_DIR/tp4_bs1/primary_tokens.json" \
  --boundary-k "$K" --budget "$P9NF_TARGET_BUDGET" \
  --fixed-prefix-token-ids "$REQ_DIR/fixed_prefix.json" \
  --metadata "$REQ_DIR/metadata.json" \
  --out "$P9NF_ROOT/agreement_records/B/$RID.json"

"$BRIDGE_PY" tools/bridge_tp/measure_agreement.py \
  --group C --request-id "$RID" \
  --left "$REQ_DIR/tp4_bs1/primary_tokens.json" \
  --right "$REQ_DIR/tp4_bs8/primary_tokens.json" \
  --boundary-k "$K" --budget "$P9NF_TARGET_BUDGET" \
  --fixed-prefix-token-ids "$REQ_DIR/fixed_prefix.json" \
  --metadata "$REQ_DIR/metadata.json" \
  --out "$P9NF_ROOT/agreement_records/C/$RID.json"

# D 才是 migration 对象；完整流双方都从全局 K 开始比较
"$BRIDGE_PY" tools/bridge_tp/measure_agreement.py \
  --group D --request-id "$RID" \
  --left "$REQ_DIR/migrated_tokens.json" \
  --right "$REQ_DIR/control_tokens.json" \
  --left-offset "$K" --right-offset "$K" \
  --boundary-k "$K" --budget "$P9NF_TARGET_BUDGET" \
  --fixed-prefix-token-ids "$REQ_DIR/fixed_prefix.json" \
  --metadata "$REQ_DIR/metadata.json" \
  --out "$P9NF_ROOT/agreement_records/D/$RID.json"
```

注意：B 组也必须从相同固定 prefix 后开始 continuation。不能拿“从 prompt 自由生成”的
B=29 与 D 的 target-local agreement 比较。

## 10. 配对 bootstrap 与非劣效性汇总

非劣效性 margin 必须在查看 D-3 结果之前冻结。下面不提供“从结果反推”的默认值：

```bash
export P9NF_NI_MARGIN=替换为预先注册的token margin

"$BRIDGE_PY" tools/bridge_tp/summarize_agreement.py \
  --group-a "$P9NF_ROOT/agreement_records/A" \
  --group-b "$P9NF_ROOT/agreement_records/B" \
  --group-c "$P9NF_ROOT/agreement_records/C" \
  --group-d "$P9NF_ROOT/agreement_records/D" \
  --confidence 0.95 \
  --bootstrap-resamples 10000 \
  --bootstrap-seed 17 \
  --noninferiority-margin-tokens "$P9NF_NI_MARGIN" \
  --out "$P9NF_ROOT/agreement_summary.json" \
  | tee "$P9NF_ROOT/agreement_summary.txt"
```

工具强制检查：相同 request IDs、固定 prefix hash、K、budget、模型、A100 平台、vLLM
commit 和 cutover rule。少于 30 对时不会产生正式结论；30-49 对会提示最好补到 50。

## 11. 打包与哈希

```bash
"$BRIDGE_PY" tools/bridge_tp/collect_results_bundle.py \
  --experiment D-NUMERICAL-FIDELITY \
  --runs "$P9NF_DIR" \
  --config "$P9NF_DIR/phase9_nf_config.json" \
  --summary "$P9NF_ROOT/agreement_summary.json" \
  --note "A100 PCIe only; fixed-prefix K; D is migration, B is topology control" \
  --out "$P9NF_ROOT/bundles"

sha256sum "$P9NF_ROOT"/bundles/*.tar.gz \
  > "$P9NF_ROOT/bundles/SHA256SUMS"
```

二进制 logits/KV tensor 默认不进入小型 evidence bundle；原始服务器目录必须另行保留，
并对完整归档计算 SHA256。

## 12. 允许与禁止的结论

数据完成前只能写：

- Phase 9 mechanical migration 已在 A100 PCIe 上取得四 rank exact readback、COMMITTED、
  source abort 和无 gap/duplicate 证据；
- D-0/D-1/D-3 是用于区分跨 TP 数值路径与 migration 增量误差的实验设计。

数据完成前不能写：

- `0.125` 已被证明是实际路径中的一个 BF16 ULP；
- 单点 `D=99 > B=29` 已证明 migration 统计上不劣；
- D-1 某个未经标定的阈值已经证明 migration 完全无责；
- 已实现跨 TP size bitwise deterministic inference；
- 其他 GPU 平台数据可与本 A100 PCIe 结果混合。
