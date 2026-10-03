# TPOT 主指标与 TTFT 诊断口径（v4）

2026-10-03。此定义用于下一轮 STAY/MIGRATE 配对实验；旧 v1/v2/v3 结果继续保留用于历史对照。

## 为什么调整

同机五卡 A100 的 TP4 独立探针使用原始 A1D 四个 2048-token prompt、1024-token output 请求，预热后单请求 TTFT 为 1026/1027 ms；四请求在 60 ms 内到达时，TTFT 为 1030、3038、4031、4064 ms。然而四请求各自的 TPOT p99 只有约 16.7–16.9 ms。1 秒 TTFT 门槛连单请求都无法稳定通过，不适合该模型与负载的主要收益判定。第一秒左右包含真实 prefill 工作；突发请求的额外数秒主要与并发等待有关。突发前两个请求各有约 1 秒的单次输出暂停，mean TPOT 约 19.1/17.2 ms，故仍须单独监控最大间隔。

原始探针包：`control/results/target-solo-burst-20261003T132033Z-1267.tar.gz`，SHA-256 `4bd49879cbe63e8f26312541404a03f6a2990f7091280aab7b2dc117f51087bb`。探针独立启动 TP4，没有加载迁移 connector。

## 主指标

对完整、有逐 token 时间戳的请求，计算对外可见相邻输出 token 间隔的算术平均值，即请求级 mean TPOT。请求通过条件：

- mean TPOT ≤ 50 ms；单 token 输出没有间隔样本，不因此失败。
- E2E ≤ 60 s，作为防止长期饥饿的宽松保护；不单独限制 TTFT。
- 迁移锚请求的 handoff ≤ 1000 ms；源端和目标端同一锚请求只计一次。

`GoodOutput_v4 = 合格请求输出 token 总数 / 相同到达流从首个请求发出到最后请求结束的时间`。

TTFT 不参与请求合格判定，但其排队时间包含在共同实验时长中，因此仍可能降低 GoodOutput。离线审计的字段为 `goodoutput_v4_tokens_s`、`goodoutput_v4_success_requests` 和 `goodoutput_v4_success_rate`。`--mean-tpot-ms` 可以做阈值敏感性分析。

## 并列报告与限制

仍报告 TTFT 分布、E2E 分布、TPOT p95/p99、超过 50 ms 的间隔比例、最大可见间隔、handoff p95/p99、未完成/失败/OOM，以及原生 TP4 请求质量。mean TPOT 可能掩盖少数长暂停，因此不能只报平均值；策略上线门禁须比较这些尾部指标是否退化。固定到达流配对还需记录窗口末尾未完成工作，避免把排队代价移出观察窗口。

现有四个完整 M5 高压包按 v4 均是 13/13 请求达标，GoodOutput 分别为 310.59、309.23、307.55、267.99 tokens/s；慢间隔比例约 0.38%–0.65%。晚启动包按逐 token v1 为 93.82 good tokens/s、按 v2 请求级 p99 为 3/13。这个差异来自指标口径，不表示系统性能发生改变。样本均为合成的 1024-token 输出且没有 STAY 对照，不能据此宣称真实请求或迁移收益。
