# BridgeTP G3 Shadow 策略真实传输验证

更新时间：2026-09-09

## 1. 目标和边界

`run_shadow_strategy_transfer_validation.py`在5张GPU上比较两个固定的Shadow策略：

- `S_NEW`：Shadow每个decode step只把新KV发送到TP4；
- `S_NEW_OLD`：每个step先发送新KV并取得ACK，再从进入Shadow时的分界点向token 0发送历史
  KV block。

rank 0代表TP1，rank 1–4代表TP4。所有payload都通过NCCL真实传输，并在TP4验证后返回ACK。
payload使用合成值和可复用buffer，但字节数对应Qwen2.5-14B的48层、8个KV head、head dim
128、BF16几何：每个新token聚合192 KiB，每个16-token历史block聚合3 MiB。

进程启动时先执行无负载传输预热；此外，每个AB/BA策略臂正式计时前都执行相同次数的
NCCL＋该cell对应背景GEMM联合预热。联合预热不写入测量CSV，用于避免先运行的策略独自承担
cuBLAS初始化和GPU升频成本。

这是G3传输微基准，不是完整在线vLLM：它不释放真实paged-KV block、不生成真实token，也不在
Bridge内重复运行G2已经测量的远端Attention。G3-I在每个传输step前后插入相同负载的无传输
control，用两侧control均值抵消慢速漂移，分别估计Shadow复制和Bridge复制对TP4合成工作负载的
干扰。G4仍需把G2远端Attention成本纳入Bridge；G5再用真实TP4请求测量TPOT、P99和goodput。

## 2. 状态语义

- Shadow期间TP1始终保留全部KV，`source_released_tokens_in_shadow`必须为0；
- cancel episode不得进入Bridge，Shadow传输全部计为浪费；
- commit episode使用相同的Bridge规则：每步先把新KV直接放到TP4，再从当前边界向前清空历史
  backlog；
- `S_NEW`进入Bridge时保留完整历史backlog；
- `S_NEW_OLD`进入Bridge时只保留Shadow尚未回填的历史backlog；
- 每个payload必须由全部4个TP4 rank验证并ACK，才能记为成功。

## 3. 主要指标

- `shadow_new_ack_p50_ms/p95_ms`：高优先级新KV的ACK延迟；
- `shadow_history_ack_p50_ms/p95_ms`：机会型历史block的ACK延迟；
- `shadow_target_load_p50_ms/p95_ms`：复制期间TP4背景GEMM完成时间；
- `shadow_target_control_p50_ms/p95_ms`：同强度、无复制的配对TP4基线；
- `shadow_target_interference_harm_ms`：Shadow各step的
  `max(0, with_transfer - paired_control)`累加值；
- `shadow_transfer_driver_ms`：微基准串行驱动完全部Shadow复制步骤的时间，不等同于在线TP1
  TPOT或请求停顿；
- `history_backlog_after_shadow_tokens/bytes`：无论最终cancel还是commit，Shadow结束时尚未回填
  的历史债务；
- `bridge_entry_history_backlog_tokens/bytes`：进入不可逆Bridge时的历史债务；
- `bridge_catchup_ms`：微基准驱动器实际清空剩余历史backlog的时间，是G4输入而非在线vLLM
  wall-clock结论；
- `bridge_target_interference_harm_ms`：只包含Bridge复制的TP4干扰，不包含远端Attention；
- `transfer_only_target_interference_harm_ms`：Shadow与Bridge复制干扰之和；不能直接等同于真实
  TP4请求的SLO损失；
- `cancel_wasted_history_bytes/total_bytes`：最终反悔时的投机浪费；
- `takeover_ready`：commit episode是否完成全部传输和验证；
- `actual_transfer_bytes == expected_transfer_bytes`：传输量是否精确。

runner同时生成两级CSV：

- `measurements.csv`：每个策略臂的汇总；
- `step_measurements.csv`：每个Shadow/Bridge step的原始新KV ACK、历史ACK、TP4负载、实际字节
  和总耗时，用于跨AB/BA重复合并后重新计算尾延迟。验收器会用原始step重新核对汇总字节和
  step数量，缺失时fail closed。

## 4. 动态干扰轨迹

`--target-load-repeats`给出峰值强度，`--target-load-profiles`定义轨迹：

- `CONSTANT`：Shadow与Bridge始终维持峰值；
- `STEP_UP_BRIDGE`：Shadow空闲，进入Bridge时升到峰值；
- `STEP_DOWN_BRIDGE`：Shadow为峰值，进入Bridge时降为空闲；
- `PULSE`：每4个phase-local step出现一次峰值脉冲；
- `OSCILLATE`：phase-local step在峰值和空闲之间交替。

每个非空闲step按`control-before -> transfer -> control-after`执行，paired control为两侧均值。
空闲step保留在原始CSV中，但不进入slowdown比例的统计分母。轨迹只使用当前phase和step，
不读取未来的commit/cancel结果。

## 5. 静态检查

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export OMP_NUM_THREADS=1

"$VIRTUAL_ENV/bin/python" -m unittest \
  tests.bridge_tp.test_shadow_transfer_protocol \
  tests.bridge_tp.test_shadow_strategy_transfer_runner

"$VIRTUAL_ENV/bin/python" \
  tools/bridge_tp/run_shadow_strategy_transfer_validation.py \
  --validate-only \
  --out-dir /tmp/g3_shadow_validate_only \
  --expected-revision "$(git rev-parse HEAD)" \
  --history-tokens 1024 \
  --shadow-steps 8 32 \
  --target-load-repeats 4 \
  --target-load-profiles CONSTANT STEP_UP_BRIDGE STEP_DOWN_BRIDGE PULSE OSCILLATE \
  --outcomes CANCEL COMMIT
```

静态检查应报告20个paired cells、40行策略结果；不会创建`out-dir`。

## 6. GPU smoke

第一次只运行一个commit cell。AB与BA使用两个独立torchrun进程，以检验顺序效应：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export OMP_NUM_THREADS=1
export G3_HEAD="$(git rev-parse HEAD)"
export G3_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0 | sed -n '1p')"
export G3_ID="shadow-transfer-g3-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
export G3_PARENT="/root/autodl-tmp/bridgetp/results/shadow_transfer"
export G3_ROOT="$G3_PARENT/$G3_ID"
mkdir -p "$G3_ROOT"

CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
"$VIRTUAL_ENV/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node=5 \
  tools/bridge_tp/run_shadow_strategy_transfer_validation.py \
  --out-dir "$G3_ROOT/ab" \
  --expected-revision "$G3_HEAD" \
  --expected-gpu-name-substring "$G3_GPU_NAME" \
  --strategy-order S_NEW S_NEW_OLD \
  --history-tokens 1024 \
  --shadow-steps 8 \
  --target-load-repeats 4 \
  --outcomes COMMIT \
  --background-gemm-size 4096 \
  --transport-warmup-steps 3 \
  --strategy-warmup-steps 5
```

每个smoke进程应记录2行并通过验收。AB通过后再以新目录运行BA：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
"$VIRTUAL_ENV/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node=5 \
  tools/bridge_tp/run_shadow_strategy_transfer_validation.py \
  --out-dir "$G3_ROOT/ba" \
  --expected-revision "$G3_HEAD" \
  --expected-gpu-name-substring "$G3_GPU_NAME" \
  --strategy-order S_NEW_OLD S_NEW \
  --history-tokens 1024 \
  --shadow-steps 8 \
  --target-load-repeats 4 \
  --outcomes COMMIT \
  --background-gemm-size 4096 \
  --transport-warmup-steps 3 \
  --strategy-warmup-steps 5
```

smoke结果拿回检查后再冻结pilot/formal矩阵，不能跳过smoke直接运行大批次。

## 7. 验收含义

`acceptance.json`的`PASS`只证明调度不变量、真实传输、ACK和字节记账通过。最终策略结论还必须
基于AB/BA配对、cancel与commit分层、多个历史长度和Shadow窗口，以及G2远端Attention成本与
后续真实vLLM target-native请求伤害。当前干扰指标只回答“真实NCCL复制会让合成TP4 GEMM慢
多少”，不能单独得出最终策略胜负。
