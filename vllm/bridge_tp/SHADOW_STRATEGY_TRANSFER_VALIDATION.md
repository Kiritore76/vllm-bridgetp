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

这是G3传输微基准，不是完整在线vLLM：它不释放真实paged-KV block、不生成真实token，也不在
Bridge内重复运行G2已经测量的远端Attention。G4应把G2远端Attention表与本实验的传输/ACK表
组合；G5再实现真实Bridge ownership推进。

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
- `shadow_transfer_driver_ms`：微基准串行驱动完全部Shadow复制步骤的时间，不等同于在线TP1
  TPOT或请求停顿；
- `history_backlog_after_shadow_tokens/bytes`：无论最终cancel还是commit，Shadow结束时尚未回填
  的历史债务；
- `bridge_entry_history_backlog_tokens/bytes`：进入不可逆Bridge时的历史债务；
- `bridge_catchup_ms`：微基准驱动器实际清空剩余历史backlog的时间，是G4输入而非在线vLLM
  wall-clock结论；
- `cancel_wasted_history_bytes/total_bytes`：最终反悔时的投机浪费；
- `takeover_ready`：commit episode是否完成全部传输和验证；
- `actual_transfer_bytes == expected_transfer_bytes`：传输量是否精确。

## 4. 静态检查

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
  --target-load-repeats 0 4 8 \
  --outcomes CANCEL COMMIT
```

静态pilot应报告12个paired cells、24行策略结果；不会创建`out-dir`。

## 5. GPU smoke

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
  --transport-warmup-steps 3
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
  --transport-warmup-steps 3
```

smoke结果拿回检查后再冻结pilot/formal矩阵，不能跳过smoke直接运行大批次。

## 6. 验收含义

`acceptance.json`的`PASS`只证明调度不变量、真实传输、ACK和字节记账通过。最终策略结论还必须
基于AB/BA配对、cancel与commit分层、多个历史长度和Shadow窗口，以及G2远端Attention成本与
后续真实vLLM target-native请求伤害。
