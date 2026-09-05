# BridgeTP Shadow KV 策略对比 Demo

## 1. 比较对象

本 demo 比较两个明确策略：

1. `history_backfill`：Shadow 开始后复制全部历史 KV，同时镜像新 KV。新 KV 拥有块间优先级，历史块从 Shadow 分界点向 token 0 倒序传输。历史与增量全部 ACK 后，TP4 才能独立 takeover 并释放 TP1。
2. `new_kv_bridge`：只先传少量新 KV，然后让 TP4 成为计算端；历史 attention 继续由 TP1 提供。它启动快，但对不使用滑动窗口的 full-context 模型，TP1 必须一直参与到请求结束，因此不是独立 takeover。

该定义把两个方案真正的取舍显式化：前者支付一次历史回填成本以换取 source release；后者减少启动复制，但把成本改成逐 token remote-attention 延迟、通信和持续的 TP1 占用。

## 2. 模型和输出

事件模型同时推进 TP1 token 生成与限速 KV 复制。`history_backfill` 在一个历史 block 发完后优先发送所有已到达的新 KV，再继续倒序历史回填。`new_kv_bridge` 在指定数量的新 KV 获得 ACK 后进入 Bridge。

每个比较分别报告：

- 请求完成时间和相对纯 TP1 的延迟收益；
- TP1 释放时间；
- 历史、新 KV 和 remote attention 的通信字节；
- 是否在请求结束前达到独立 takeover；
- latency、source release、network bytes 三个互不混淆的 winner；
- new-KV Bridge 可以胜过历史回填的 remote-attention 每 token 延迟上限。

不提供任意加权总分。容量救援、用户延迟和网络占用是不同目标，应该分别看。

## 3. 默认几何

默认参数对应 Qwen2.5-14B-Instruct 的候选几何：48 layers、8 KV heads、head size 128、hidden size 5120、BF16/FP16 两字节。

- 聚合 KV：196,608 bytes/token。
- remote-attention 通信估计：983,040 bytes/token，即每层 query 与 attention output 的聚合大小，不含协议头。

remote-attention 的实际 RPC、kernel、同步和 PCIe/NVLink 开销尚未实现，因此 `--remote-penalty-ms` 必须做 sweep，不能只跑一个乐观常数。

## 4. 本地快速运行

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
export BRIDGE_PY=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python

"$BRIDGE_PY" tools/bridge_tp/run_shadow_copy_strategy_demo.py \
  --out-dir /root/autodl-tmp/bridgetp/results/shadow_copy_demo/quick \
  --history-tokens 512 1024 \
  --remaining-tokens 128 256 512 1024 \
  --copy-rate-gib-s 0.4 0.7 1.2 \
  --remote-penalty-ms 0.5 1 2 4 8 \
  --source-tpot-ms 30 \
  --target-tpot-ms 16
```

输出包括：

- `shadow_strategy_rows.csv`：每个条件两行策略数据；
- `shadow_strategy_decisions.csv`：每个条件一行的配对指标与 winner；
- `shadow_strategy_comparisons.json`：逐条件配对结果；
- `summary.json`：三个目标各自的胜负计数与 break-even 范围。

## 5. 决策方法

如果论文主目标是缓解 TP1 容量压力，首先检查 `source_release_winner`。对 full-context 模型，`new_kv_bridge` 在请求结束前不释放 TP1；只要倒序历史回填能在请求结束前追上，它通常更符合 capacity-first 目标。

如果主目标是缩短单请求完成时间，则检查 `latency_winner` 和 `remote_penalty_break_even_ms`。只有实测 remote-attention penalty 低于 break-even，并且持续占用 TP1 可以接受，new-KV Bridge 才有理由成为主方案。

如果 remote-attention wire bytes 或 TP1 kernel 占用过大，即使其启动时间短，也可能伤害两个 pool 的原生请求。应使用服务器实测值替换默认估计，再决定是否值得实现完整 remote attention。

### 5.1 默认 sweep 的代码自检结果

使用本页默认参数运行 120 个配对条件，得到：

- latency winner：`history_backfill=113`，`new_kv_bridge=7`；
- source-release winner：`history_backfill=120`，`new_kv_bridge=0`；
- network-bytes winner：`history_backfill=105`，`new_kv_bridge=15`；
- new-KV Bridge 的 remote-attention penalty break-even 为 0.032–1.741 ms/token。

这不是服务器实测结论，因为 remote penalty 仍是 sweep 输入。它说明下一步最关键的测量不是“新 KV 少传了多少”，而是 remote attention 的真实每 token 延迟、wire bytes 和 TP1 占用。若实测 penalty 普遍高于约 1.7 ms/token，默认范围内倒序历史回填在完成时间上也占优；即使 penalty 极低，new-only 仍不能提前释放 TP1，因此不适合作为 capacity rescue 的直接替代。

## 6. 证据边界

这是参数化决策 demo，不是端到端 serving 结果。它真实实现了倒序 block 调度、delta 优先和两种生命周期，但没有实现跨 TP remote-attention kernel/RPC。下一阶段只有在参数 sweep 显示 new-KV Bridge 存在稳定优势区间时，才值得实现 remote attention 并做配对 GPU 实验。
