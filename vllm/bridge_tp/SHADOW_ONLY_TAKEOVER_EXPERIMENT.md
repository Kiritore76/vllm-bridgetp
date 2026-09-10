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

## 两级实现边界

普通 `--handoff-mode shadow-only` 保留早期的 CPU staging 性能下界；显式增加
`--gpu-resident-shadow` 后才启用最终结构。最终结构在 Shadow 开始时向 TP4
提交 dormant target request，scheduler 为计划切换长度分配最终 paged-KV
blocks，并将请求保持在 `WAITING_FOR_REMOTE_KVS`，所以它不会在提交前执行
forward。

实验使用真实 Qwen、真实 TP1/TP4 vLLM、真实 KV 导出、TCP 传输、重分片、
TP4 restore、统一响应和 TP4 背景请求 TPOT。每轮交替运行顺序以降低热机和
时间漂移偏差。

GPU-resident 模式复用现有 TCP stager 作为 CPU 中继，但不等 source freeze 才
restore：历史 KV 按逻辑 block 从 0 开始写入最终 TP4 blocks，每块生成四 rank
readback receipt；新 KV delta 随 TP1 解码持续原位 patch，并为各 rank 推进连续
watermark。cutover 时用真实 token IDs 替换只用于预留长度的占位 token，四个
rank 都达到最终 watermark 后才能提交。提交前 TP1 是唯一 owner；提交后目标
请求才首次执行 forward，因此 TP4 从 source 的最后 pending token 继续生成。

这里仍不执行远端 attention，也没有 Bridge 状态。传输链路是
`TP1 GPU -> source CPU -> TCP stager -> relay file/page cache -> TP4 GPU`；因此
本实验能验证真实 GPU 常驻、增量覆盖、干扰和切换正确性，但中继实现尚不是
RDMA/NCCL 优化后的最终数据面。

## 接受条件

每个成功的 `SHADOW_ONLY` run 必须同时满足：

1. 状态迁移严格为 `SHADOW -> TAKEOVER`，不能出现 `HANDOFF`。
2. 历史 KV 的四个 staging receipt 全部早于 source freeze/cutover。
3. 四个 TP4 rank 全部 exact-readback，takeover 为 `COMMITTED`。
4. 统一响应 token 索引连续，且同时包含 TP1 与 TP4 产生的 token。
5. 背景任务全部完成，各观测窗口达到最小 TPOT 样本数。
6. GPU-resident 模式还必须有完整的逐 block、逐 delta receipt，四个最终
   watermark 等于 `cutover_manifest.num_computed_tokens`，且初始历史在 freeze
   前已经位于 TP4 GPU。

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
