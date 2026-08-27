# BridgeTP Phase 9 当前状态与执行计划

更新时间：2026-08-27

适用源码：

```text
/root/autodl-tmp/bridgetp/vllm_bridge
branch: bridgetp/d3-phase9-numerical-fidelity
minimum commit: e8b3e5d
Section 7.2 stable-window runner: requires the commit containing this document
platform: 5 x NVIDIA A100-PCIE-40GB
```

本文是根据 2026-08-27 的实际代码、A100 PCIe 实验结果和当前证据边界整理的
执行计划。它更新旧版 `PHASE9_EXPERIMENT_PLAN.md` 中已经被后续发现修正的调度和
停止门，但不删除旧计划，也不重新解释 Phase 1-8 的既有结果。

## 1. 必须保持的边界

1. Phase 1-8 已完成并验证，不重新设计或整体重跑。
2. 当前 Phase 9 正式证据只属于 A100 PCIe，不与其他 GPU 平台混合。
3. 不恢复已经删除的 `bridgetp/phase9-agreement-gate` 分支。
4. `phase9new` 是数值保真研究方案的来源，不等同于已经验证的功能；只有实际合入
   `vllm_bridge` 的代码才算实现。
5. 迁移边界统一写作 `K`；A/B/C/D 中的 B 是纯 TP 拓扑对照组。
6. D 组真实 migration 是最终验证对象；B 组不能替代 D 组。
7. 已查看的 smoke RID 及其单点结果不得进入正式 D-3 统计。
8. 单点 `A=256/B=98/C=98/D=94`、单点 logit gap 和单次 handoff timing 只能作为
   工程诊断，不得写成总体统计结论。

## 2. 当前总体进度

| 阶段 | 当前状态 | 已有证据或产物 | 下一门槛 |
|---|---|---|---|
| Phase 1-8 | 完成 | 各阶段 validation；Phase 8 新鲜回归 continuity PASS | 不重跑 |
| P9-0 runtime control | 完成 | runtime control、honored marker、旧环境变量兼容 | 保持回归 |
| P9-1 online controller | 完成工程实现 | policy、state machine、rate controller、response proxy、inspector | 使用正式标定配置 |
| Phase 9 mechanical smoke | 单点通过 | 四 rank exact readback、COMMITTED、source abort、无 gap/duplicate | 不等于正式数值/性能结论 |
| D-0 logits/ULP | 单 RID 诊断完成 | global index 147 的真实 raw/processed tensor 分析 | 与 D-1、正式 D-3 联合解释 |
| D-1 KV provenance | 单 RID 测量完成 | 固定 prefix、K=53、TP1/native TP4/migrated TP4 逐层逐 rank 报告 | `formal_causal_conclusion` 仍为空 |
| D-2 precision sensitivity | 未开始，探索项 | 无正式 sweep | 不阻塞正式 D-3 停止门 |
| D-3 smoke | 工程链路通过 | A=256、B=98、C=98、D=94 | 必须排除该 RID |
| 更新版 C-1 / Section 7.1 | 进行中 | 自动 18 条件 runner；TPOT metric 修复至 `e8b3e5d` | 完整 fit、hash、支持范围审计 |
| 更新版 C-2 / Section 7.2 | 首轮 pilot 未通过，稳态重跑待开始 | 首轮全部 selected=null；stable-window runner 已实现 | 新 pilot READY；formal 36 条件 COMPLETE |
| C-3 survival table | 已使用但待冻结 | 服务器已有 discovery survival table | 审计 trace/split/source 并哈希 |
| P9-2 / Section 6 replay | 工程 replay 通过 | engineering anchor 下同时有 migrate/stay | 新模型实现后必须重跑 |
| 正式 policy-driven D-3 | 阻塞 | 尚无 preregistered 50-RID 结果 | 满足第 7 节全部停止门 |
| E-1 至 E-8 | 未开始正式运行 | 部分机制/工具已有 | D-3 后冻结 E-1 口径 |
| R-1 至 R-4 | 延后 | 无 | E 主实验和参数冻结后再跑 |

## 3. 已完成但不得扩大解释的 Phase 9 证据

### 3.1 机械迁移

当前 A100 PCIe smoke 已证明特定请求和配置下：

```text
TP1 KV export/reshard
四个 TP4 rank exact readback
ownership COMMITTED
source abort
source 与 target 都贡献客户端可见 token
unified stream 无 gap/duplicate
```

这支持 Tier 1 mechanical correctness，不支持“任意长 continuation 与 TP1 bitwise
一致”或“已完成正式在线性能评测”。

### 3.2 D-0 单点

已测 global output index 147：

```text
control raw gap:   0
migrated raw gap:  0.125（在记录的 BF16 dtype 下为 1 ULP）
```

processed tensor 中的 float32 ULP 数只是存储 dtype 下的描述，不能据此定位某个 CUDA
kernel、证明单次舍入因果或宣布 migration 无责。

### 3.3 D-1 单点

已有 `kv_provenance.json` 使用相同固定 token prefix，在 K=53、fixed token count=90
处比较 native TP1、native TP4 和 migrated TP4。报告包含逐层逐 rank 的绝对差、RMS、
分位数、稳健相对误差和 ULP；其 `formal_causal_conclusion` 仍为 `null`。正式解释必须
等待配对 D-3。

### 3.4 D-3 smoke

单 RID target-local smoke：

```text
A: 256/256
B: 98/256
C: 98/256
D: 94/256
```

它只证明 A/B/C/D 采集和汇总链路可运行。它已经被查看，且使用 diagnostic fixed
boundary，必须从正式统计中排除。

## 4. C 系列的当前映射

旧版 C 系列仍是逻辑上的必需标定，但执行方式已经更新。

| 旧编号 | 当前实现 | 当前决定 |
|---|---|---|
| C-1 TP1/TP4 TPOT-load | Section 7.1 tick-level sweep | 7.1 通过后不重复旧 C-1 |
| C-2 rate-load interference | Section 7.2 pilot + formal 36-cell | 7.2 通过后不重复旧 C-2 |
| C-3 survival table | `build_survival_table.py` 或审计现有表 | 仍必须冻结 |
| P9-2 policy boundary | 更新后的正式 replay | 新模型实现后重跑 |

### 4.1 Section 7.1 的完成条件

- 18 个条件全部完成：TP1/TP4 × QPS 1/2/4 × 3 reps；
- 使用实际 `num_requests_running`，不是 run-level max-concurrency proxy；
- 使用 request-level `vllm:request_time_per_output_token_seconds`；
- `tick_tpot_candidate.json` 和顶层 `SHA256SUMS` 存在；
- 拟合有足够非空 interval、合理的 weighted R²/RMSE；
- 记录 `num_running_min/max`，后续不在未测范围内无声明外推。

### 4.2 Section 7.2 的完成条件

1. TP1 停止并释放 GPU 0，TP4 在 GPU 1-4、端口 8200 常驻；
2. pilot 的预注册候选 QPS 全部完成；
3. `load_pilot_summary.json` 为 `READY`，low/medium/high 均有冻结 QPS；
4. formal 36 条件完整：3 bands × 4 rates × 3 reps；
5. `bridge_calibration_summary.json` 为 `COMPLETE`；
6. `missing/unexpected/duplicates/rejected` 全为空；
7. 保留非单调格点，不删除、不重选结果。

2026-08-27 首轮 pilot 使用固定 60 秒延迟后直接开始窗口，结果为
`MORE_QPS_CANDIDATES_REQUIRED`。其 tail 显示 QPS 0.7 可进入 low、QPS 1.5 穿过
medium，但原窗口主要记录负载爬升，且没有候选覆盖 high。该 pilot 必须保留为失败
诊断，不进入正式拟合。后续 runner 先验证 rolling stable-load window，再开始 300 秒
rate-zero/copy window；pilot timeout 会保留证据并继续下一候选，formal timeout 则
fail closed。

### 4.3 C-3 的完成条件

- 与论文 M1 使用同一 trace；
- 时间序 train split 固定为 0.7；
- 训练记录不少于 100；
- 表内有 `max_observed_length` 和来源；
- 超出支持范围不外推；
- survival table 和来源 manifest 已哈希。

## 5. Section 7 完成后的开发任务

Section 7 的完成本身不授权正式 D-3。必须先完成以下代码和分析工作：

1. 从 7.1 读取 TP1/TP4 的 `base_s`、`per_running_s` 和支持范围；
2. 从 7.2 拟合或选择可审计的 rate-aware interference model：

   ```text
   T_interference = f(kv_usage_frac, copy_rate_gib_s)
   ```

3. 不把 `inter_token_latency_seconds` 偷换为 request-level TPOT；
4. 更新 `InterferenceModel`、`ControllerConfig` 和 config serialization；
5. 更新 `replay_policy.py`，使其接收 TPOT slope 和 rate-aware model，而不是常数 TPOT
   加单一 `s_per_gib_at_ref`；
6. 为拟合结果、输入 summary、正式 config 和代码 commit 计算 SHA256；
7. 运行逻辑测试和离线 replay，确认边界方向与支持范围。

现有 P2-D 45 runs 仍可作为敏感性证据，但不能与新的 7.2 格点混合拟合后伪装成同一
实验协议。

## 6. 新模型的正式 replay 门

正式 replay 至少覆盖：

```text
decode progress: 64, 128, 256, 512, 768, 1024
target load:     0.1, 0.3, 0.5, 0.7, 0.85
copy rate:       正式模型支持范围内的低/中/高档
```

通过要求：

- 同时存在 migrate 与 do-not-migrate 区域；
- target load 增加时 N* 不反常降低；
- interference/copy rate 增加时策略不变得更积极；
- 决策边界位于 survival table 和标定模型支持范围内；
- 不从 replay 结果反向挑选只让论文结论更好的参数；
- 每一行能还原输入、benefit、cost、action 和 reason。

旧 engineering replay 的 `START_SHADOW=243/STAY=27` 只能证明工具和基本边界存在，
不能替代此门。

## 7. 正式 50-RID D-3 停止门

以下条件全部满足前，不运行正式 policy-driven D-3：

- [ ] Section 7.1 candidate fit 已生成、检查并哈希；
- [ ] Section 7.2 pilot 为 READY；
- [ ] Section 7.2 formal 为 COMPLETE；
- [ ] rate-aware interference model 已实现并测试；
- [ ] 正式 replay 同时存在 migrate/stay，且单调方向合理；
- [ ] survival table 来源已冻结并哈希；
- [ ] 50 个不同 prompt/RID 的 manifest 已冻结；
- [ ] target-local budget 已冻结；
- [ ] non-inferiority margin 在查看正式结果前预注册；
- [ ] cutover rule、controller config、manifest、commit 和 preregistration 已哈希；
- [ ] smoke RID `78fc430a-c0c1-47ff-8a54-27b8b0ab357f` 已明确排除；
- [ ] 正式运行不使用 `theta_0=theta_min=0`；
- [ ] 正式运行不使用 diagnostic trigger/cutover flags。

允许继续做固定 K 的 engineering D-3，但必须明确标成 numerical-fidelity diagnostic，
不能与正式 serving-policy D-3 合并。

## 8. 正式 D-3 计划

正式 D-3 使用至少 30、目标 50 个完全配对 RID。每个 RID 必须产生：

```text
真实 D 组 migration
相同固定 prefix 后的 A：TP1 vs TP1 rerun
相同固定 prefix 后的 B：TP1 vs native TP4 bs=1
相同固定 prefix 后的 C：native TP4 bs=1 vs bs=8
D：migrated unified stream vs clean TP1 control，从各自 K 后比较
```

每个 RID 固定并校验：model、A100 PCIe platform、commit、cutover rule、K、budget、
fixed-prefix hash 和 request ID。D 是最终迁移对象，B 只是拓扑基线。

完成后运行 paired bootstrap/non-inferiority 汇总。至少要求：

```text
formal_ready=true
baseline_reproducible=true
```

若 A 组不可重复，不允许通过修改 margin 挽救 D-vs-B 结论。

## 9. E 系列：D-3 后的正式在线实验

E 不是 A/B/C/D 之后的“实验组 E”，而是 Phase 9 在线系统评测编号。

| 编号 | 目标 | 当前状态 | 仍需开发或冻结 |
|---|---|---|---|
| E-1 | 单请求 commit 与统一响应流 | 未正式运行 | 按 D-3 冻结 Tier 1/2/3 验收；正式 config；3 fresh runs |
| E-2 | rollback/cancel/EOS race | 未开始 | fault injection、EOS/cancel 编排、cleanup 断言 |
| E-3 | HOLD_BACK vs GREEDY_FASTPATH stall | 未开始 | 配对 runner、stall 汇总；fastpath 分歧语义服从 D-3 |
| E-4 | fixed-low/fixed-high/AIMD | 未开始 | 稳态负载注入、时间序列收集、三组汇总与绘图 |
| E-5 | policy boundary 或候选排序 | 未开始 | 由正式 replay 决定形态；多候选 baseline harness |
| E-6 | 多请求并发 | 未开始 | multi-request orchestration、ID 隔离、队列/终态统计 |
| E-7 | fast/slow controller | 未开始 | transient/persistent harness；物理重配或明确 MVP |
| E-8 | 端到端 8 baselines | 未开始 | 统一 workload、baseline runner、SLO/goodput 汇总 |

旧 E-1 的“任意长 migrated stream 必须与 TP1 control 全程逐 token 相同”不能静默保留，
也不能静默放宽。正式 D-3 后必须一次性冻结：

1. Tier 1 mechanical correctness：严格二元门；
2. Tier 2 numerical fidelity：预注册的配对统计；
3. Tier 3 semantic/task equivalence：按任务类型补充。

如果仍要求跨 TP size bitwise deterministic inference，则这是额外的 deterministic
kernel/collective 开发任务，不能归功于 KV migration。

## 10. R 系列：稳健性复跑

R 不阻塞当前 C、正式 D-3 或 E-1。它应在 E 主实验和参数冻结后运行。

| 编号 | 内容 | 优先级与决定 |
|---|---|---|
| R-1 | 不同长请求比例/第二 trace 的 M1 稳健性 | 推荐；主结果后运行 |
| R-2 | transient/persistent 的额外随机种子 | 推荐；E-7 后运行 |
| R-3 | NVLink/NVSwitch 第二平台重跑 E-1/E-3/E-4 | 有机器则高价值；必须独立报告 |
| R-4 | 学习式 predictor 替换经验 CCDF | 可选消融 |

若不做 R-3，论文只能声称 A100 PCIe 标定和结果，不能外推速率参数。不得将 R-3 数据
并入当前 A100 PCIe 的 C/D/E 统计。

## 11. 推荐执行顺序

```text
现在
  └─ 完成 Section 7.1
       └─ 停 TP1，完成 Section 7.2 pilot
            ├─ pilot 非 READY → 扩展预注册 QPS，保留原 pilot
            └─ pilot READY → 完成 36-cell formal
                 └─ 带回并审计 7.1/7.2/C-3 产物
                      └─ 开发并拟合 rate-aware model
                           └─ 新模型正式 replay
                                └─ 冻结 D-3 preregistration
                                     └─ 正式 30-50 RID D-3
                                          └─ 冻结 E-1 Tier 1/2/3
                                               └─ E-1 → E-2/E-3/E-4/E-5
                                                    └─ E-6/E-7
                                                         └─ E-8
                                                              └─ R-1/R-2
                                                                   └─ 可用时 R-3
                                                                        └─ 可选 R-4
```

## 12. 最近一次交付检查点

Section 7 完成后，至少带回：

```text
7.1/tick_tpot_candidate.json
7.1/SHA256SUMS
7.1/tick_tpot_fit.log
7.2/load_pilot_summary.json
7.2/bridge_calibration_summary.json
7.2/SHA256SUMS
7.2/bridge_calibration_summary.log
C-3/survival_table.json
三个 driver log
```

正式归档还必须保留每个 condition 的 manifest、telemetry、benchmark、copy window、
analysis、result 和 SHA256；不能只保留 summary。

## 13. 当前允许与禁止的论文表述

当前允许：

- A100 PCIe 单点上机械迁移链路已得到四 rank exact readback、COMMITTED、source abort
  和无 gap/duplicate 证据；
- native TP1/TP4 和 migrated 路径可能产生不同 greedy 数值轨迹；
- D-0/D-1/D-3 的分层设计用于区分纯拓扑差异和 migration 增量差异；
- Section 7 正在标定 controller 实际使用的运行时坐标。

当前禁止：

- 把单点 D/B agreement 写成总体统计优势；
- 宣称 migration 已被证明没有引入任何额外数值误差；
- 宣称已实现跨 TP bitwise deterministic inference；
- 把 engineering replay 写成正式 calibrated policy；
- 把不同 GPU 平台数据混入当前 A100 PCIe 结论；
- 在正式 D-3 前根据结果反推 non-inferiority margin。
