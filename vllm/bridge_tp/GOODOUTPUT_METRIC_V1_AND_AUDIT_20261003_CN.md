# GoodOutput v1 指标与现有 M5 数据审计

日期：2026-10-03。用途：为下一步同机 STAY/MIGRATE 配对实验预先固定离线指标。此文件只审计**已完成的迁移样本**，不推断迁移收益。

后续请求级 SLO 定义曾在 [v2 修订](GOODOUTPUT_REQUEST_SLO_V2_20261003_CN.md)中更新为 p99 TPOT；当前主指标改为 [v4 TPOT 口径](GOODOUTPUT_TPOT_PRIMARY_V4_20261003_CN.md)。本文件保留 v1 逐 token GoodOutput 与旧严格口径的历史审计。

## 1. v1 计数规则

从每个在线实验包的 `online/contract.json` 读取 SLO 阈值。目前四个样本均为 ITL/TPOT 50 ms、TTFT 1000 ms、E2E 60000 ms、handoff 1000 ms。不从一次结果的好坏反向调整阈值。

对状态为 `COMPLETED`、逐 token 时间戳完整的请求，令 `N_i` 为对外可见的输出 token 数，`q_i = 1[TTFT_i <= 1000 ms 且 E2E_i <= 60000 ms]`：

```text
good_tokens_i = q_i × (1 + Σ_{j=2..N_i} 1[ITL_{i,j} <= 50 ms])
GoodOutput = Σ_i good_tokens_i / T
T = 从所有请求中最早 request_started 到最晚 request_ended 的时间
```

这是**逐 token 合格输出速率**。一次间隔超限只使其后那个 token 不合格；TTFT 或 E2E 超限使该请求所有 token 不合格。首 token 由 TTFT 判定。分母包括相同到达流的执行与 drain 时段；今后的配对须使用相同请求、到达时序与窗口定义。

另报严格请求级 SLO goodput：只有 TTFT、E2E、**全部**可见 ITL 均达标且已迁移请求 handoff 不超过 1000 ms，才把该请求的全部 token 计入。还单独报告超限间隔数/率、p95/p99、最大 gap、handoff、抢占/OOM。handoff 间隔若已反映在可见 ITL 中，不再从 GoodOutput 扣第二次；独立的 handoff 门禁仍有效。这两个 goodput 口径不能混称。

锚请求只从 `response_proxy_stats.json` 的**统一可见输出**计一次；source/target 的内部 response 仅用于核对开始与结束时间。背景 source/target 请求从 `background_summary.json` 计数。已明确记录失败且有起止时间的请求计 0 合格 token；遇到缺失、重复、非单调时间戳或数量不符，则报告 `computable=false`、列出错误，不把缺失值当零或合格。

实现：`vllm_bridge/tools/bridge_tp/audit_goodoutput.py`；测试：`vllm_bridge/tests/bridge_tp/test_audit_goodoutput.py`。计算器只读压缩包中的 JSON，不解压写入原目录。输出 `benefit_vs_stay=null`，直到有匹配的 STAY 对照。

## 2. 已有四包的样例审计

| 迁移样本 | 请求数 | 输出 tokens | 合格 tokens | GoodOutput tokens/s | 严格请求级 goodput |
| --- | ---: | ---: | ---: | ---: | ---: |
| 早启动 HIGH 091033 | 13 | 13,312 | 3,047 | 71.09 | 0 |
| 早启动 HIGH 091429 | 13 | 13,312 | 3,047 | 70.78 | 0 |
| 晚启动 HIGH 095734 | 13 | 13,312 | 4,061 | 93.82 | 0 |
| 先前 M5 影子包 040902 | 13 | 13,312 | 5,064 | 101.94 | 0 |

四包 `computable=true`、无缺失字段；每包含锚请求 1、TP1 source peers 8、TP4 target background 4。上述差异**不是早/晚启动或预测器的收益对比**：尚无匹配 STAY 包，而且运行条件与时间窗可能不同，不能从表中挑最高值当策略成绩。

另对失败包 `a100-m2-m5-shadow-20261003T045734Z-20739.tar.gz` 运行审计，结果为 `computable=false`：缺少背景汇总、可见锚请求输出和源请求响应。计算器没有从 runner 报错或零散日志推断合格输出。

晚启动包的逐池核对：锚请求 1024 token 中 1013 个合格、11 个可见 ITL 超过 50 ms；8 个 TP1 peers 合计 3048 个合格；4 个 TP4 背景请求 TTFT 全部超过 1000 ms，因此虽各完成 1024 token，按 v1 主指标计入 0。整池 59 个超限间隔、9 个背景请求 TTFT 超限；锚请求 handoff 347.53 ms，低于 1000 ms，但严格请求级 SLO 因 ITL 超限失败。严格请求级 goodput 为 0，说明当前高压 smoke 对这种全有或全无指标过于苛刻，不能只报它一个数字。

## 3. 本阶段结论与下一门禁

现有完整 M5 包**具备计算逐 token GoodOutput 的字段**；离线脚本和样例审计完成。当前四包均为迁移、`ignore_eos=true` 且锚请求输出上限 1024；只能验证计数与字段完整性，不能拟合长度分布收益、超过 1000 token 才启动的行为，或 STAY/MIGRATE 净增益。下一步进入计划阶段 A：同机复核 M1 的 TP1 KV release 时间与容量估计，并准备支持相同输入的 STAY 与 MIGRATE 两种运行方式。服务器实验前重新核对机器、HEAD、checkpoint/输入 SHA、原始数据路径。
