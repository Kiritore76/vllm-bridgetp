# Shadow-only Takeover 原型与成对实验

## 要验证的命题

对同一条 TP1 长请求和同一份 TP4 背景负载，比较：

- `BRIDGE`：Shadow 只传新 KV；到冻结边界后才传历史 KV，控制状态为
  `LOCAL -> SHADOW -> HANDOFF -> TAKEOVER`。
- `SHADOW_ONLY`：Shadow 从请求第 0 个 token 开始，按 token 递增顺序传历史
  KV，同时传持续产生的
  新 KV；历史必须在冻结边界前完成 staging，四个 TP4 rank 完成精确恢复后，
  控制状态直接为 `LOCAL -> SHADOW -> TAKEOVER`。

核心假设是：若迁移有效吞吐长期大于 KV 生成速率，历史积压可在 TP1 KV
耗尽前追平。提前完成历史传输可以缩短最终同步/切换停顿；代价是 Shadow
期间对繁忙 TP4 的额外干扰。

## 当前原型真正实现了什么

该分支增加了显式 `--handoff-mode shadow-only`。它不进入 Bridge/Handoff
控制状态，并且状态机仍强制要求四个 rank 的 `TARGET_READY` 与 exact-readback
证据，之后才允许原子提交。默认 `bridge` 路径不变。

实验使用真实 Qwen、真实 TP1/TP4 vLLM、真实 KV 导出、TCP 传输、重分片、
TP4 restore、统一响应和 TP4 背景请求 TPOT。每轮交替运行顺序以降低热机和
时间漂移偏差。

当前实现的边界必须明确：Shadow 期间历史 KV 先到 CPU stager；最终冻结后
才把拼装后的完整 KV restore 到 TP4 GPU。因此这是“去掉 Bridge 状态”的可运行
原型和保守性能下界，不是最终的 TP4 GPU 原位增量写入实现，也不执行远端
attention。正式论文若要宣称 TP4 在 TP1 继续解码时已经完全热接收，还需增加
TP4 dormant-request KV 预分配与原位 patch 接口。

## 接受条件

每个成功的 `SHADOW_ONLY` run 必须同时满足：

1. 状态迁移严格为 `SHADOW -> TAKEOVER`，不能出现 `HANDOFF`。
2. 历史 KV 的四个 staging receipt 全部早于 source freeze/cutover。
3. 四个 TP4 rank 全部 exact-readback，takeover 为 `COMMITTED`。
4. 统一响应 token 索引连续，且同时包含 TP1 与 TP4 产生的 token。
5. 背景任务全部完成，各观测窗口达到最小 TPOT 样本数。

## 主要指标

- `bridge_to_commit_ms`：在新原型中解释为 final-freeze 到 commit 的时间。
- `handoff_stall_ms`：用户可见的输出缝隙。
- `history_ready_before_freeze_ms`：历史传输领先冻结边界的裕量；必须非负。
- `shadow_tpot_p99_ms`：提前复制对 TP4 原生流量的 Shadow 期干扰。
- `paired_comparisons.csv`：逐 repetition 的 Shadow-only 相对 Bridge 差值。
- `process_lifetimes.json`：TP1、TP4、stager、controller 和负载进程的占用时段。
- `slo`：可配置 TPOT、TTFT、E2E 和切换停顿阈值及对应违约数/违约率。

两个系统不必在同一分支执行。新分支使用 `--shadow-only-only` 只运行新方案；
随后回到 Bridge 分支，在相同冻结输入与参数下只运行旧方案，最后按 repetition
和运行顺序配对比较。

只有多轮配对结果稳定、置信区间不跨零时，才能判断哪条路径性能更好；单次
smoke 只验证机制与数据链路，不用于论文结论。
