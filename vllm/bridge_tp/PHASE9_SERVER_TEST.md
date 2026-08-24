# BridgeTP Phase 9：A100 服务器验证手册

本手册从代码已推送 GitHub 开始。所有正式结果继续使用同一台五卡
A100-PCIe-40GB 服务器；工程 smoke 与论文实验必须分开存放。

## 1. 拉取分支并做 CPU 侧检查

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

git fetch origin
git switch bridgetp/d3-phase9-online-controller
git pull --ff-only origin bridgetp/d3-phase9-online-controller

python -m py_compile \
  vllm/bridge_tp/runtime_control.py \
  vllm/bridge_tp/kv_stream.py \
  vllm/bridge_tp/phase8_source.py \
  vllm/bridge_tp/stream_protocol.py \
  vllm/bridge_tp/controller/*.py \
  tools/bridge_tp/run_phase9_controller.py \
  tools/bridge_tp/probe_token_divergence.py \
  tools/bridge_tp/inspect_phase9_run.py

python -m unittest discover -s tests -t . -p 'test_phase9_*.py'
```

预期：90 项测试全部通过。

本分支只修改 Python，不修改 C++/CUDA 扩展。editable 安装下无需重新编译 vLLM，
但所有 source、target、stager 进程都必须重启，才能加载新 Python 代码。

## 2. Phase 8 回归门

先按 `vllm/bridge_tp/PHASE8.md` 用全新 ID 重跑一次 `dualwrite_commit`。
不要创建 `runtime_control.json`，所有 Phase 8 环境变量保持原值。

必须重新确认：

- 四 rank exact readback；
- old-KV/new-KV overlap；
- `COMMITTED` 和 source abort；
- 组装输出与干净 TP1 control 逐 token 一致。

逐 token 对照若失败，不能只看总 `FAIL`：若增量覆盖、四 rank exact readback、
`COMMITTED` 与 source abort 均通过，应保留该轮并检查首次分叉。只有双方 topology
probe 都证明两个候选是并列最大 logprob，才归类为
`TIE_EQUIVALENT_DIVERGENCE`；不得通过反复重跑挑选碰巧全等的运行。非并列分叉仍然
停止 Phase 9。

对保留下来的 Phase 8 回归目录运行：

```bash
python tools/bridge_tp/probe_token_divergence.py \
  --run-dir "$PHASE8_DIR" \
  --source-url http://127.0.0.1:8001 \
  --target-url http://127.0.0.1:8200 \
  --tokenizer "$PHASE8_MODEL" \
  | tee "$PHASE8_DIR/token_equivalence_output.txt"
```

原 `inspection.json` 保持 `FAIL`，不覆盖；派生的 `token_equivalence.json` 单独记录
该失败能否由双方并列最大 logprob 解释。

## 3. 创建 Phase 9 工程 smoke session

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export PHASE9_ID="$(python -c 'import uuid; print(uuid.uuid4())')"
export PHASE9_ROOT=/root/autodl-tmp/bridgetp/results/phase9_smoke
export PHASE9_DIR="$PHASE9_ROOT/$PHASE9_ID"
export PHASE9_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
mkdir -p "$PHASE9_DIR"

printf '%s\n' \
  "export PHASE9_ID=$PHASE9_ID" \
  "export PHASE9_ROOT=$PHASE9_ROOT" \
  "export PHASE9_DIR=$PHASE9_DIR" \
  "export PHASE9_MODEL=$PHASE9_MODEL" \
  > /root/autodl-tmp/bridgetp/phase9_session.env

git rev-parse HEAD > "$PHASE9_DIR/git_revision.txt"
python -c 'import sys,torch,vllm; print("python",sys.version); print("torch",torch.__version__); print("torch_cuda",torch.version.cuda); print("vllm",vllm.__version__); print("vllm_path",vllm.__file__)' \
  > "$PHASE9_DIR/environment.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader > "$PHASE9_DIR/gpu_identity.txt"
nvidia-smi topo -m > "$PHASE9_DIR/gpu_topology.txt"
```

每一轮使用全新 ID、目录和 server 进程。

## 4. 终端 1：GPU1-4 启动 TP4 target

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_session.env

export PHASE9_STAGING_MANIFEST="$PHASE9_DIR/staging_manifest.json"
export PHASE9_RECEIPTS="$PHASE9_DIR/receiver_receipts"
export PHASE9_CONTROL="$PHASE9_DIR/takeover_state.json"

CUDA_VISIBLE_DEVICES=1,2,3,4 \
python -m vllm.entrypoints.openai.api_server \
  --model "$PHASE9_MODEL" \
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
      "bridgetp_stream_manifest":os.environ["PHASE9_STAGING_MANIFEST"],
      "bridgetp_stream_receipt_dir":os.environ["PHASE9_RECEIPTS"],
      "bridgetp_stream_socket_timeout_s":600,
      "bridgetp_stream_expected_phase":"BridgeTP D3 Phase 8",
      "bridgetp_takeover_control_path":os.environ["PHASE9_CONTROL"],
      "bridgetp_takeover_control_timeout_s":600
    }}))')" \
  2>&1 | tee "$PHASE9_DIR/target_tp4.log"
```

## 5. 终端 2：GPU0 启动 TP1 source

环境变量中的 128/160 是兼容基线。Controller 启动请求后会用
`runtime_control.json` 动态覆盖实际 trigger、cutover 和 rate。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_session.env

export BRIDGETP_DUMP_ENABLED=0
export BRIDGETP_STREAM_ENABLED=1
export BRIDGETP_STREAM_MIGRATION_ID="$PHASE9_ID"
export BRIDGETP_STREAM_RUN_DIR="$PHASE9_DIR"
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
export BRIDGETP_TAKEOVER_MIGRATION_ID="$PHASE9_ID"
export BRIDGETP_TAKEOVER_RUN_DIR="$PHASE9_DIR"

CUDA_VISIBLE_DEVICES=0 \
python -m vllm.entrypoints.openai.api_server \
  --model "$PHASE9_MODEL" \
  --served-model-name bridgetp-model \
  --tensor-parallel-size 1 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --port 8001 \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  2>&1 | tee "$PHASE9_DIR/source_tp1.log"
```

## 6. 终端 3：CPU stager

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_session.env

python tools/bridge_tp/phase8_stager.py \
  --run-dir "$PHASE9_DIR" \
  --delta-host 127.0.0.1 \
  --delta-base-port 29900 \
  --delivery-host 127.0.0.1 \
  --delivery-base-port 30000 \
  --timeout-s 600 \
  2>&1 | tee "$PHASE9_DIR/stager.log"
```

## 7. 读取真实 KV block 数并准备 smoke config

两个 server 就绪后：

```bash
source /root/autodl-tmp/bridgetp/phase9_session.env
curl -f http://127.0.0.1:8001/v1/models
curl -f http://127.0.0.1:8200/v1/models
grep -Ei 'GPU KV cache size|GPU blocks|num_gpu_blocks' \
  "$PHASE9_DIR/source_tp1.log" "$PHASE9_DIR/target_tp4.log" | tail -20
```

把日志中 TP1、TP4 各自报告的每 rank GPU KV block 数记为
`TP1_BLOCKS`、`TP4_BLOCKS`。不要把 TP4 的数再乘 4。

先用真实 M1 trace 建 survival 表：

```bash
export PHASE9_TRACE=/替换为/M1实际使用的trace.jsonl
mkdir -p calibration "$PHASE9_DIR/calibration"
python tools/bridge_tp/build_survival_table.py \
  --trace "$PHASE9_TRACE" \
  --output-field output_tokens \
  --train-frac 0.7 \
  --out "$PHASE9_DIR/calibration/survival_table.json"
```

生成仅用于工程验证的 config。下面三个模型值不是论文标定值，不能进入正式 E 系列：

```bash
export TP1_BLOCKS=替换为TP1实际值
export TP4_BLOCKS=替换为TP4实际值

python - <<'PY'
import json, os
from pathlib import Path

src = Path('experiments/phase9/configs/e1_correctness.json')
cfg = json.loads(src.read_text())
run = Path(os.environ['PHASE9_DIR'])
cfg['run_dir'] = str(run)
cfg['tp1_total_kv_blocks'] = int(os.environ['TP1_BLOCKS'])
cfg['tp4_total_kv_blocks'] = int(os.environ['TP4_BLOCKS'])
cfg['survival_table_path'] = str(run / 'calibration/survival_table.json')
cfg['platform_note'] = 'ENGINEERING SMOKE ONLY: 5x A100-PCIe-40GB'
cfg['tpot_tp1']['calibration_source'] = 'ENGINEERING SMOKE ONLY; anchor'
cfg['tpot_tp4']['calibration_source'] = 'ENGINEERING SMOKE ONLY; anchor'
cfg['interference']['calibration_source'] = 'ENGINEERING SMOKE ONLY; anchor'
cfg['policy']['theta_0'] = 0.0
cfg['policy']['theta_min'] = 0.0
cfg['policy']['min_output_tokens_before_eligible'] = 16
(run / 'phase9_smoke_config.json').write_text(json.dumps(cfg, indent=2) + '\n')
print(run / 'phase9_smoke_config.json')
PY
```

## 8. 终端 4：运行在线 controller

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_session.env

python tools/bridge_tp/run_phase9_controller.py \
  --config "$PHASE9_DIR/phase9_smoke_config.json" \
  --run-dir "$PHASE9_DIR" \
  --migration-id "$PHASE9_ID" \
  --source-request experiments/phase9/configs/request_long.json \
  2>&1 | tee "$PHASE9_DIR/controller_output.txt"
```

预期 controller 终态为 `TAKEOVER`，服务器落盘 `COMMITTED`。

## 9. 在同一 TP1 上生成干净 greedy control

第一条 source 已被 abort，但 TP1 server 仍然运行。它会拒绝为第二个请求复用迁移
session，因此可以生成同 prompt、同参数的 control：

```bash
source /root/autodl-tmp/bridgetp/phase9_session.env

python - "$PHASE9_DIR" <<'PY'
import json, sys, urllib.request
from pathlib import Path

run = Path(sys.argv[1])
payload = json.loads((run / 'source_request.json').read_text())
payload.pop('request_id', None)
payload['stream'] = False
request = urllib.request.Request(
    'http://127.0.0.1:8001/v1/completions',
    data=json.dumps(payload).encode(),
    headers={'Content-Type': 'application/json'},
    method='POST',
)
with urllib.request.urlopen(request, timeout=1800) as response:
    result = json.load(response)
tokens = result['choices'][0]['token_ids']
(run / 'control_response.json').write_text(json.dumps(result, indent=2) + '\n')
(run / 'control_tokens.json').write_text(json.dumps(tokens, indent=2) + '\n')
print('control tokens:', len(tokens))
PY
```

## 10. 生成 exact/tie token 等价证据

若 unified response 与 control 完全一致，本工具直接记录 `EXACT`。若不一致，工具在
首个共同前缀上分别向 TP1、TP4 发起一个无迁移、单 token、greedy、带 top-20
logprob 的 topology probe。只有 control token 和 target token 在两边都处于并列最大
logprob，才签发 `TIE_EQUIVALENT_DIVERGENCE`；缺失 logprob 或普通分叉直接失败。

```bash
python tools/bridge_tp/probe_token_divergence.py \
  --run-dir "$PHASE9_DIR" \
  --source-url http://127.0.0.1:8001 \
  --target-url http://127.0.0.1:8200 \
  --tokenizer "$PHASE9_MODEL" \
  | tee "$PHASE9_DIR/token_equivalence_output.txt"
```

正式 source/target 请求不启用 logprobs，避免改变 handoff/TPOT 测量开销。只有发生
分叉后才运行上述两个单 token probe。原始 probe request/response、tokenizer映射与
复算结果都会保存在运行目录，不能只保留最终分类。

## 11. 严格验收

```bash
source /root/autodl-tmp/bridgetp/phase9_session.env

python tools/bridge_tp/inspect_phase9_run.py \
  --run-dir "$PHASE9_DIR" \
  --expect commit \
  --control-tokens "$PHASE9_DIR/control_tokens.json" \
  --json | tee "$PHASE9_DIR/inspect.json"
```

关键证据缺失会直接 FAIL，不再以 SKIP 算通过。必须同时满足：

- controller 决策输入、benefit、cost、动作齐全；
- 四 rank exact readback；
- `COMMITTED` 且 source abort；
- source 和 target 都贡献客户端可见 token；
- `unified_response.jsonl` 是实时追加的客户端可见流，并与最终 proxy 状态一致；
- 全局 token index 无缺口、无重复；
- 与干净 TP1 control 逐 token完全相同，或者首分叉具有可复算的双方并列最大
  logprob证书；
- audit 终态为 `TAKEOVER`，无 invariant violation。

`EXACT` 和 `TIE_EQUIVALENT_DIVERGENCE` 必须分开统计。后者只解除“KV损坏”的误报，
不能写成与TP1 counterfactual逐token完全一致。

## 12. smoke 通过后再开始正式标定

工程 smoke 只回答在线闭环能否工作，不能写论文。正式顺序是：

1. C-1：同一 A100 平台重新提取或测量 TP1/TP4 TPOT；
2. C-2：同一 A100 平台标定迁移速率与 TP4 原生尾延迟；
3. C-3：使用 M1 同一 trace 和 0.7 时间序训练划分；
4. P9-2：离线确认策略边界；
5. 用真实标定替换 E-1 config，执行三个全新 session；
6. 每轮严格 inspect，三轮后再 summarize 和 collect bundle。

其他 GPU 平台的数据只能作为独立 R-3 结果，不能与本 A100 标定或正式 E 系列混合。
