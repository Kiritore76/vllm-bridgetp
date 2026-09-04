# BridgeTP Phase 9：正式 E 之前的 CAP-0 执行计划

更新日期：2026-09-04
代码基线：`faf16b5adf9321b7df139c3eaf0f537f190fdf6e`
集成分支：`bridgetp/phase9-cap0-pilot`

## 1. 文档目标和范围

本文只规划正式 E 之前的工程与实验门，包括：

1. CAP-0 本地集成与回归；
2. 服务器 E0 事实盘点；
3. `d3-formal-052` pending-tail-block 修复回归；
4. 至少三次 no-migration source guard 标定；
5. CAP-0 No-op、Rescue reachability 和 Safe abandon 三个场景；
6. 跨运行工件审计和进入 formal controller 开发的 go/no-go。

本文不规划正式 E 的 baseline、cell、seed、矩阵或论文统计。CAP-0 是单 anchor 工程可达性
验证，不是正式 capacity-aware controller，也不是论文性能结果。

## 2. 这一阶段最终要回答的问题

正式 E 之前需要依次回答四个问题：

1. CAP-0 补丁能否在 `faf16b5` 上干净集成，并保持既有 Phase 9 行为？
2. 052 的合法 pending tail block 能否在真实服务器 checkout 上完成 restore 和 takeover？
3. 当前可观测的 TP1 KV headroom 信号能否在真实 contention 下安全触达迁移数据面？
4. controller 能否正确执行三种互补行为：不迁移、完成迁移、安全放弃迁移？

只有四个问题都有可审计的肯定答案，才进入 formal capacity controller 开发。即使 CAP-0
全部通过，也不能直接声称正式 E 已具备运行条件；formal controller 仍需实现 request
registry、target reservation、真实 backlog/deadline、候选排序和 migration slot accounting。

## 3. 全程必须保持的不变量

- 不覆盖、清理或误提交原工作树中的用户改动。
- 所有服务器运行使用全新目录，不复用或覆盖旧结果。
- controller 只能读取当前及过去的因果 telemetry，不能读取 future workload schedule。
- background manifest 只属于独立 workload generator，不得复制到 controller 输入或
  controller run directory 作为决策依据。
- TP1/TP4 总 KV blocks 必须来自实际启动日志；不得使用 `max_num_batched_tokens` 代替。
- `tp1_total_kv_blocks`、`tp4_total_kv_blocks` 或
  `capacity_pilot.guard_free_kv_tokens` 为零时必须 fail closed。
- capacity path 必须记录 `trigger_path=CAPACITY_PILOT`，不能称为校准的 OOM 概率。
- target current guard 不是 reservation proof；审计中必须保持
  `target_reservation_proven=false`。
- cutover 前 TP1 始终是 owner；只有 commit 后才允许 target 成为唯一 owner。
- policy abandon 必须使用 `abort_source=false`，不能因取消迁移而终止用户请求。
- 052 post-fix 结果必须独立归档，不能并入旧 49 条 D 数值队列。
- CAP-0 latency、throughput 和 goodput 不进入论文 headline 表。

## 4. 执行总顺序

```text
L0 本地补丁集成与回归
  -> S0 服务器 E0 事实盘点
  -> S1 052 post-fix 工程回归
  -> C0 no-migration bring-up
  -> C1 至少三次 no-migration guard 标定并冻结
  -> P0 No-op bring-up + 至少三次重复
  -> P1 Rescue reachability bring-up + 至少三次重复
  -> P2 Safe abandon bring-up + 至少三次重复
  -> G0 跨运行工件审计与 CAP-0 go/no-go
  -> formal capacity controller 开发
```

后一步不得用来补救前一步缺失的事实。例如，Rescue 成功不能替代 052 回归，Safe abandon
成功也不能替代 guard 的事前冻结。

## 5. L0：本地 CAP-0 集成与回归

### 5.1 要证明什么

证明 ZIP 补丁与 `faf16b5` 的完整本地 checkout 兼容，新增路径可导入，关键安全路径有单元
测试覆盖，且没有意外修改补丁清单以外的既有文件。

### 5.2 操作

1. 从 `faf16b5` 创建独立分支/worktree；
2. 应用 `BridgeTP_CAP0_faf16b5.patch`；
3. 核对实际改动与 ZIP 中 `CHANGED_FILES.txt`；
4. 人工审查以下路径：
   - anchor prefix 精确选择、无配置兼容和多匹配 fail closed；
   - capacity exclusive trigger；
   - target current guard；
   - `abort_source=false` 的 cleanup API、adapter 和调用点；
   - source mirror terminal/drain 生命周期锁；
   - current/legacy preemption metric 兼容；
5. 使用仓库 venv 运行：
   - CAP-0 新增测试；
   - 补丁 runbook 指定的 Phase 9 测试；
   - 全部 `test_phase9_*.py` discovery；
   - 修改/新增 Python 文件的 `py_compile`；
   - `git diff --check`。

### 5.3 必须保存的证据

- branch、HEAD 和补丁 SHA-256；
- `git status --short`、`git diff --stat`、`git diff --check`；
- 每条测试命令、Python 路径、退出码和测试计数；
- 人工安全审查结论；
- 最终 commit ID；推送后保存 remote branch 名称。

### 5.4 通过标准

- 补丁清单与实际修改一致，额外计划文档单独列明；
- 所有可运行测试通过；若存在环境级 import blocker，必须明确区分于断言失败，并在服务器
  完整 checkout 复跑；
- `py_compile` 和 `git diff --check` 通过；
- anchor ambiguity、零 guard、非法 cleanup boolean 均 fail closed；
- 原工作树用户改动保持不变。

### 5.5 不能证明什么

本地通过不能证明 GPU 数据面、真实 scheduler contention、四 rank restore、owner 切换或
容量收益已经成立。

## 6. S0：服务器 E0 事实盘点

### 6.1 要证明什么

建立所有后续运行的可信 provenance，确认服务器实际代码、模型、dtype、KV geometry 和
硬件拓扑，而不是根据目录名或旧文档推测。

### 6.2 操作

只读记录：

- repository path、branch、HEAD、最近提交、remote；
- `git status`、dirty diff 和服务器独有改动；
- TP1/TP4/stager 启动命令和环境变量；
- GPU 型号、数量和 topology；
- Python、PyTorch、vLLM 版本；
- model path、model revision、dtype、block size；
- TP1/TP4 实际 `total_kv_blocks`；
- 旧 C/D 工件记录的 commit 和配置；
- 052 是否已经存在 post-fix 运行。

### 6.3 必须保存的证据

建立只追加的 E0 audit 目录，至少包含：

- `git_status.txt`、`git_head.txt`、`git_log.txt`、`git_remote.txt`；
- `server_dirty.patch`；
- `environment.txt`、`gpu_inventory.txt`、`gpu_topology.txt`；
- source/target 启动命令与日志；
- model、dtype、block geometry 和 KV block counts 的结构化摘要；
- 盘点目录的 `SHA256SUMS`。

### 6.4 通过标准

- 服务器 HEAD 是经审核的 CAP-0 commit，或能证明与该 commit 等价；
- 所有 dirty change 均已保存且不与 CAP-0 未审阅地混合；
- TP1/TP4 block counts 来自实际 server geometry；
- dtype、model revision 和启动参数均有日志证据。

### 6.5 不能证明什么

E0 只证明环境和 provenance 可追溯，不证明任何迁移正确性或容量效果。

## 7. S1：052 pending-tail-block post-fix 回归

### 7.1 要证明什么

证明 `faf16b5` 修复能在完整服务器 checkout 上接受唯一合法 pending 尾块，同时仍对额外
或缺失 allocation fail closed。

052 的已知边界是：

```text
prompt tokens          = 33
cutover output tokens  = 160
all known tokens       = 193
computed tokens        = 192
pending tokens         = 1
block size             = 16
snapshot blocks        = 12
allocated blocks       = 13
```

### 7.2 操作

- 使用与旧 D3 一致的 model、dtype、greedy contract、trigger、cutover 和 comparison
  budget；
- 在新目录中只运行 `d3-formal-052`；
- 保存 source snapshot、target allocation、restore metadata、四 rank receipts、takeover
  state、source/target response 和日志；
- 明确标记为 `postfix_<commit>` engineering regression。

### 7.3 通过标准

- 不再出现 `Target block allocation differs from live snapshot`；
- restore metadata 只覆盖 12 个 snapshot prefix blocks；
- 第 13 个 pending 尾块仍由 scheduler 所有，不被 payload 覆盖；
- 四 rank ready/readback 完整；
- migration 和 control 均正常完成；
- 没有双 owner、重复或缺失 client-visible stream。

### 7.4 证明边界

通过后只证明 052 机械边界修复成立。它不增加旧 D 的同质样本数，不改变旧 49 条统计，也
不证明任务准确率或正式数值非劣效。

## 8. C0/C1：no-migration guard 标定

### 8.1 要证明什么

证明 TP1 source guard 是从当前机器、当前 server args 和相同 workload shape 的因果
telemetry 中得到的可解释工程阈值，而不是未来信息、错误容量量纲或逐轮调参。

### 8.2 C0 bring-up

先以 `--dry-run` 启动 controller，使 telemetry、risk 和审计逻辑工作，但禁止任何 actuator。
验证：

- source/target metrics 可持续采集；
- free KV tokens 根据实际 total blocks 和 block size 正确计算；
- current/legacy preemption counter 与 server log 一致；
- background job 的 request ID 不会匹配 anchor prefix；
- controller 未读取 background manifest；
- `--dry-run` 下没有 Shadow、stager receipt、cleanup 或 takeover。

### 8.3 C1 正式标定

使用同一 frozen workload shape 至少独立重复三次。每次重启服务并使用新目录，记录：

- 每 tick free KV tokens、KV usage、running、waiting；
- preemptions/recompute counter；
- free-token EWMA decline rate；
- 距候选 guard 的 time-to-guard；
- 请求到达、结束和自然容量释放时间；
- 是否出现不可恢复 overload。

标定规则必须在查看 held-out CAP-0 场景前写下。优先选择首次 preemption 之前、留有工程
反应时间的保守 free-token 水位；若三次均无 preemption，则使用最近点和最坏下降率形成
保守 guard，并记录这是 closest-approach 标定而非 preemption-boundary 标定。

### 8.4 通过标准

- 至少三次完整运行，指标和日志一致可解释；
- preemption metric 不得与 server log 静默矛盾；
- 选出的 guard 为正且小于实际 total KV tokens；
- guard 选择规则、数值、适用 server args 和 workload shape 已冻结；
- 后续三个 CAP-0 场景不得逐轮回调 guard。

### 8.5 证明边界

标定只得到 CAP-0 工程 guard，不是 `p_oom`、`p_cap` 或跨 workload/hardware 校准的风险
模型，也不证明 target reservation。

## 9. P0：CAP-0 No-op 场景

### 9.1 要证明什么

证明 source headroom 压力不会绕过 target current guard；当 TP4 当前不可接受时，
controller 能安全选择不迁移。

### 9.2 场景条件

- TP1 capacity signal 达到 `ENTER/HOLD`；
- TP4 的 `kv_usage_frac` 超过 current guard，或 `num_waiting` 超过上限；
- 使用已冻结的 source guard；
- CAP-0 exclusive path 开启。

### 9.3 执行

先完成一次 bring-up，随后至少三次独立重复。每次重启全部相关服务、使用新 run directory，
并保持 workload shape、guard 和配置冻结。

### 9.4 通过标准

- audit 出现 active capacity signal；
- `capacity_pilot_decision=STAY`；
- 原因为 target fails current guard；
- 不进入 Shadow，不产生 staging manifest 或 rank receipt；
- 不调用 cleanup/takeover；
- anchor 和 background 请求均按各自 pool 正常结束或按冻结 workload contract 结束；
- 没有 request-ID 串号。

### 9.5 能证明和不能证明什么

它证明 CAP-0 不会在明显不安全的当前 target 状态下盲目迁移。它不证明 TP4 current guard
等价于完整 target reservation，也不证明该 guard 在所有 workload 下最优。

## 10. P1：CAP-0 Rescue reachability 场景

### 10.1 要证明什么

证明在真实多请求 scheduler contention 下，一个显式 TP1 anchor 能由纯因果 headroom
信号触发，并沿既有 Phase 1--8 数据面完成四 rank progressive takeover。

### 10.2 场景条件

- TP1 capacity signal active；
- TP4 初始可被 background 工作占用，随后在信号仍 active 时变为 current-guard
  admissible；
- anchor prefix 精确匹配且只匹配一个 source request；
- 使用已冻结 guard，不读取未来 background schedule。

### 10.3 执行

先完成一次 bring-up，随后至少三次独立重复。每次保存完整 stager、rank、takeover、proxy、
source 和 target 工件。

### 10.4 通过标准

- audit 记录 `trigger_path=CAPACITY_PILOT`；
- trigger 发生时 signal active 且 TP4 通过 current guard；
- 四 rank receipt、digest、restore 和 readback 完整；
- 不允许缺 rank receipt 时 commit；
- 状态合法地经过 Shadow/Handoff/Committed；
- commit 后 target 成为唯一 client-visible owner；
- source 不再继续产生 owner stream；
- background 请求没有被选为 anchor，没有 cross-request contamination；
- 无 deadlock、重复 token、缺失 stream 或未解释异常终态。

### 10.5 能证明和不能证明什么

它证明单 anchor 迁移机制在真实 contention 下可由容量信号触达并完成。它不证明迁移一定
赶得上正式 capacity deadline、commit 后一定释放足够 blocks、总 goodput 改善，或多个
候选时选择最优请求。

## 11. P2：CAP-0 Safe abandon 场景

### 11.1 要证明什么

证明 Shadow 开始后，如果 source 压力在 cutover 前消退，controller 可以只撤销迁移，
保留 TP1 请求及其唯一 ownership，并安全停止 source mirror worker。

### 11.2 场景条件

- capacity signal 先 `ENTER` 并触发 Shadow；
- 在 commit/cutover 前恢复到 `CLEAR`；
- cleanup 显式使用 `abort_source=false`；
- TP1 anchor 仍有足够输出预算继续 decode。

### 11.3 执行

先完成一次 bring-up，随后至少三次独立重复。场景必须通过 workload 自然形成或预先冻结的
负载时序形成，不能观察结果后临时改变 guard。

### 11.4 通过标准

- audit 先记录 `CAPACITY_PILOT` Shadow，后记录 signal `CLEAR` 和 abandon；
- cleanup request 中 `abort_source=false`；
- receipt 中 `source_abort_dispatched=false`、`source_continues_on_tp1=true`；
- 不发生 commit，target 不成为 owner；
- staging 和四 rank worker 被完整 drain/停止；
- cleanup 后继续 TP1 decode 不向已停止 queue 入队；
- TP1 anchor 正常生成到 EOS 或 frozen max_tokens；
- 没有 source loss、deadlock、重复 stream 或 background 串号。

### 11.5 能证明和不能证明什么

它证明 pre-commit migration abandonment 对用户请求是非破坏性的，并验证 mirror
terminal/drain 竞态修复。它不证明 post-commit rollback，也不覆盖所有 cancel/EOS race。

## 12. G0：跨运行工件审计与 CAP-0 go/no-go

### 12.1 每轮必备工件

- `phase9_audit.jsonl`；
- `source_progress.json`；
- `session_manifest.json`；
- Shadow 场景的 `staging_manifest.json` 和四 rank receipts；
- `takeover_state.json`；
- `source_cleanup_receipt.json`；
- `response_proxy_stats.json`；
- source/target response；
- background manifest、events、summary；
- source/target/stager/controller 启动日志和命令行；
- environment、git revision、dirty patch、dtype、model revision、block counts、GPU
  topology；
- metrics scrape 或时序 CSV；
- run directory 的 `SHA256SUMS`。

### 12.2 跨运行检查

- 所有报告运行使用同一 CAP-0 commit；
- guard 和 workload 版本在 held-out 场景中保持冻结；
- 每个场景有一次 bring-up 和至少三次合格重复；
- request ID、migration ID 和 run directory 一一对应；
- No-op 无 Shadow 工件；Rescue 有完整 commit 工件；Safe abandon 有 non-destructive
  cleanup 工件；
- preemption counter 与日志一致；
- 不存在未解释的缺失文件、时间倒序、owner 冲突或 hash 变化。

### 12.3 Go 条件

只有同时满足以下条件才进入 formal capacity controller 开发：

1. 本地完整回归通过；
2. 服务器事实和 commit provenance 完整；
3. 052 post-fix 工程回归通过；
4. guard 由至少三次 no-migration 运行冻结；
5. No-op、Rescue、Safe abandon 各至少三次重复通过；
6. 无 source loss、deadlock、cross-request contamination；
7. 无不完整四 rank receipt commit；
8. 无重复或缺失 owner stream；
9. 所有工件可审计并有 SHA-256。

### 12.4 No-Go 条件和处理

出现任一情况即停止向后推进并先修复：

- 052 block boundary 仍失败；
- anchor prefix 匹配 background 或多个请求；
- target guard 不通过仍启动 Shadow；
- policy abandon 终止 source；
- cleanup 后仍向停止 queue 入队；
- 缺 rank receipt 仍 commit；
- client 观察到双 owner、重复或缺失 stream；
- preemption metric 恒为零但 server log 有 preemption；
- controller 使用了 future workload schedule；
- 关键配置、commit、dtype 或 block counts 无法追溯。

失败运行必须原样保留，不覆盖后重跑；修复后使用新 commit、新目录和新的重复编号。

## 13. CAP-0 通过后的明确结束点

本计划的完成状态应写作：

> `faf16b5` 上的 052 边界修复已在服务器验证；CAP-0 单 anchor 在真实 contention 下完成
> No-op、Rescue reachability 和 Safe abandon 工程门，工件可审计。该结果证明数据面可由
> 因果 headroom signal 安全触达，但 formal multi-request capacity controller 和正式 E
> 尚未完成。

下一代码阶段才实现：

1. RequestRegistry 和多个候选；
2. per-request allocated/released blocks；
3. causal joint source pressure/risk；
4. TP4 current-KV、restore workspace 和 guarded growth reservation；
5. queued/sent/acked bytes 与真实 backlog；
6. frozen commit deadline 和 feasibility；
7. 字典序候选排序；
8. active migration slot accounting；
9. 正式 workload/baseline orchestrator；
10. artifact inspector 和 paired summary。

这些正式 controller 项不属于本文实验计划，也不能用 CAP-0 结果冒充完成。
