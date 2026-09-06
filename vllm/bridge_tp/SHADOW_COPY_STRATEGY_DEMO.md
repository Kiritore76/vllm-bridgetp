# BridgeTP Shadow KV 传输策略模拟

## 1. 研究问题

本 demo 只比较 Shadow 阶段的数据迁移，不模拟 Bridge 或 remote attention：

1. `history_backfill`：从 Shadow 分界点向 token 0 回填历史 KV，同时传输新 KV。复制
   带宽先保证新 KV 不积压，剩余带宽消退历史 backlog。历史与新 KV 追平后达到
   `TAKEOVER_READY`。
2. `new_kv_only`：历史 KV 完全不动，按 decode step 镜像 Shadow 后产生的新 KV。它降低
   传输量和复制干扰，但历史 backlog 不会下降，因此不能单独达到 takeover-ready。

两者不使用一个加权总分。demo 分别报告传输开销、目标侧干扰和 takeover readiness。

## 2. 实测校准来源

模拟直接读取 Phase 9 C 系列冻结证据：

- C1 `tpot_model_load.json`：按 TP1 KV load 插值得到新 KV 产生速度；
- C2 `condition_inventory_48.csv`：使用 36 个非零复制单元，并与相同 load band、相同
  repetition 的 rate=0 单元配对，取得有效复制速率和目标侧 TPOT/ITL 干扰；
- C3 `survival_table.json`：在每个历史 prefix 使用条件剩余长度的 P50/P90 实际分位数。

默认 source load 为 0.22、0.48、0.62，全部位于 C1 的 TP1 支持区间。默认 prefix 为
128、256、512、768、1024、1536、2048 token。

## 3. 模型

历史回填的净 backlog 消退速率为：

```text
history_drain_bytes_s
= measured_copy_bytes_s - kv_bytes_per_token / source_tpot_s
```

如果净速率为正，历史追平时间为：

```text
history_ready_time_s = history_bytes / history_drain_bytes_s
```

若追平时间小于 C3 给出的剩余生成时间，则标记 `TAKEOVER_READY`；否则标记为请求先结束。

只传新 KV 的平均复制占空比为：

```text
new_only_duty_cycle
= new_kv_bytes / measured_copy_rate / remaining_generation_time
```

C2 测量的是连续复制干扰。new-only 是短 burst，因此 demo 用复制 active time 对 C2
配对 TPOT penalty 做 exposure scaling。这是有数据约束的估计，不是新的 GPU 测量。

## 4. 本地运行

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

EVIDENCE=/root/autodl-tmp/bridgetp/C_SERIES_LIGHTWEIGHT_EVIDENCE
OUT=/root/autodl-tmp/bridgetp/results/shadow_transfer_c_series

/root/autodl-tmp/bridgetp/.venv_bridge/bin/python \
  tools/bridge_tp/run_shadow_copy_strategy_demo.py \
  --out-dir "$OUT" \
  --interference-inventory \
    "$EVIDENCE/C2_INTERFERENCE/condition_inventory_48.csv" \
  --tpot-model "$EVIDENCE/C1_TPOT/tpot_model_load.json" \
  --survival-table "$EVIDENCE/C3_SURVIVAL/survival_table.json"
```

Windows 本地运行时只需替换 Python、证据目录和输出目录。

输出：

- `shadow_transfer_decisions.csv`：每个 C2 cell、source load、prefix 和剩余分位数一行；
- `shadow_transfer_comparisons.json`：两种策略的完整结果；
- `summary.json`：总体及按复制速率、剩余分位数拆分的 readiness 计数。

## 5. 证据边界

该结果适用于冻结的 A100 PCIe、Qwen2.5-14B、Phase 9 C 系列范围。历史回填采用连续流
近似，new-only 按现有 Phase 8 source 每次 decode progress 发布 delta 的语义建模。C2 能
约束 burst 发生时的干扰幅度，但没有实测低占空比 burst 的排队与尾延迟，因此不能替代
最终 GPU 配对实验。
