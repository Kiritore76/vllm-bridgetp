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
- free-token EWMA decline rate（仅作运行时诊断）；
- 抢占前、`running/waiting/preemptions` 均不变的相邻样本瞬时下降率及其 p95；
- 距候选 guard 的 time-to-guard；
- 请求到达、结束和自然容量释放时间；
- 是否出现不可恢复 overload。

标定规则必须在查看 held-out CAP-0 场景前写下。`F_i` 取 preemption counter 首次上升前的
最后一个 telemetry 样本，而不是抢占释放 KV 后的样本。请求入场、退出或 recompute 可能在
单次 scrape 间隔内离散分配/释放数千 KV token；不得把这种阶跃当作持续生成速率外推。
每轮只在首次抢占前、相邻样本的 `num_running`、`num_waiting` 和
`preemptions_total` 均不变时计算正向瞬时下降率，取其 nearest-rank p95 为 `D_i`；正式三轮
再取 `max_i(D_i)`。v2 默认协议要求每轮都有 preemption，不使用 closest approach。

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

## 14. 当前机制冻结：这些实验为何可以在 Shadow/Bridge 方案未定时先跑

本轮服务器命令验证的是现有实现 `M0_FULL_RESTORE`，不是尚未确定的论文最终机制：

```text
mechanism = M0_FULL_RESTORE
shadow    = HISTORY_PRECOPY_PLUS_DELTA
bridge    = NOT_IMPLEMENTED
handoff   = FULL_KV_BEFORE_TARGET_EXECUTION
```

这里的 Shadow 会先复制 anchor 截止边界以前的历史 KV，再传边界后的增量；TP4 在完整 restore、
四 rank ready 和 takeover commit 前不为该 anchor 执行远程 attention，也不存在“TP4 计算端 +
TP1 远程 attention”的 Bridge 中间态。因此，E0、052、guard 标定和 CAP-0 三场景可以继续，
因为它们回答的是现有 M0 的边界修复、因果触发、可达性和安全放弃。它们不会替未来的
Shadow 历史复制方向或 Bridge remote-attention 方案作结论。

正式 E 仍需等两件事冻结后再规划：论文最终状态机（是否以及如何引入 Bridge）和对应的数据
传输/ownership contract。若未来实现改为 newest-first、boundary-outward 或 remote-attention
Bridge，需要使用新 mechanism 名、新 commit 和新结果目录重新做机制相关门；本轮 M0 结果
仍可保留为 full-restore baseline 和工程回归。

## 15. 服务器命令的使用约定

以下命令面向 Linux 服务器，仓库和环境默认是：

```text
统一 checkout：/root/autodl-tmp/bridgetp/vllm_bridge
venv：          /root/autodl-tmp/bridgetp/.venv_bridge
model：         /root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
GPU：           GPU0=TP1，GPU1--4=TP4
端口：          TP1=8001，TP4=8200，snapshot=29800，delta=29900，delivery=30000
```

如果服务器实际路径不同，只改本节导出的路径变量，不改实验语义。所有命令都应保存退出码；
出现非零退出码时保留目录并停止向后推进，不在原目录覆盖重跑。

本文每个可复制的 Bash 命令块都显式以固定绝对路径 `cd` 开始，并重新激活 venv。不要依赖
前一个命令块留下的当前目录或 Conda/venv 状态。跨命令块所需变量只从手册明确生成的
`phase9_cap0_common.env`、`phase9_cap0_geometry.env` 或
`phase9_cap0_active.env` 加载；`cd` 本身不再依赖导出的 `$BRIDGE_REPO`。

远端 `origin/bridgetp/phase9-cap0-pilot` 已包含 CAP-0。服务器在现有
`vllm_bridge` checkout 内取证、获取该分支并安全切换，不再创建第二个 worktree；这样会继续
使用该 checkout 里与当前 vLLM 构建匹配的本地 CUDA 扩展。CAP-0 代码锚点是：

```text
73b03bbe61c0398704dc77548e615d0c92c05c0a  Add BridgeTP Phase 9 CAP-0 pilot
```

文档提交可位于该锚点之后，因此服务器验证规则是“该提交必须是 HEAD 的祖先”，而不是要求
HEAD 永远等于 `73b03bb`。

## 16. S0：安全部署、E0 盘点和 CPU 回归的完整命令

### 16.1 现有 checkout 切换前取证

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

# 交互终端不要继承此前脚本留下的 errexit/nounset/pipefail；失败由每步退出码显式判断。
set +e
set +u
set +o pipefail

export BRIDGE_BASE=/root/autodl-tmp/bridgetp/vllm_bridge
export BRIDGE_REPO=/root/autodl-tmp/bridgetp/vllm_bridge
export BRIDGE_PY=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
export BRIDGE_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
export OMP_NUM_THREADS=1
export CAP0_RESULTS=/root/autodl-tmp/bridgetp/results/phase9_cap0
export CAP0_MANIFEST_ROOT=/root/autodl-tmp/bridgetp/phase9_cap0_manifests
export E0_ID="e0_$(date -u +%Y%m%dT%H%M%SZ)"
export E0_DIR="$CAP0_RESULTS/e0/$E0_ID"
mkdir -p "$E0_DIR"

printf '%s\n' \
  "export BRIDGE_BASE=$BRIDGE_BASE" \
  "export BRIDGE_REPO=$BRIDGE_REPO" \
  "export BRIDGE_PY=$BRIDGE_PY" \
  "export BRIDGE_MODEL=$BRIDGE_MODEL" \
  "export OMP_NUM_THREADS=$OMP_NUM_THREADS" \
  "export CAP0_RESULTS=$CAP0_RESULTS" \
  "export CAP0_MANIFEST_ROOT=$CAP0_MANIFEST_ROOT" \
  "export E0_ID=$E0_ID" \
  "export E0_DIR=$E0_DIR" \
  > /root/autodl-tmp/bridgetp/phase9_cap0_common.env

git -C "$BRIDGE_BASE" status --short --branch | tee "$E0_DIR/base_git_status.txt"
git -C "$BRIDGE_BASE" branch --show-current | tee "$E0_DIR/base_git_branch.txt"
git -C "$BRIDGE_BASE" rev-parse HEAD | tee "$E0_DIR/base_git_head.txt"
git -C "$BRIDGE_BASE" log -5 --oneline | tee "$E0_DIR/base_git_log.txt"
git -C "$BRIDGE_BASE" remote -v | tee "$E0_DIR/base_git_remote.txt"
git -C "$BRIDGE_BASE" diff --binary > "$E0_DIR/base_dirty.patch"
git -C "$BRIDGE_BASE" ls-files --others --exclude-standard \
  | tee "$E0_DIR/base_untracked_files.txt"
```

这里不运行 `reset`、`clean`、`checkout --` 或覆盖性复制。`base_dirty.patch` 只记录已跟踪
改动；未跟踪文件只列名，不擅自打包或移动。

### 16.2 在现有 checkout 获取并安全切换 CAP-0

以下命令不使用 `reset`、`clean` 或强制 checkout。Git 若发现现有已跟踪或未跟踪文件会被
覆盖，将拒绝切换并保持原工作树不变；此时停止并人工检查，不得强行覆盖。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

git -C "$BRIDGE_BASE" fetch origin \
  refs/heads/bridgetp/phase9-cap0-pilot:refs/remotes/origin/bridgetp/phase9-cap0-pilot
git -C "$BRIDGE_REPO" switch --detach origin/bridgetp/phase9-cap0-pilot
```

核验代码锚点并保存切换后的工作树状态。允许切换前已经存在且不与 CAP-0 冲突的用户文件
继续留在目录中；不要把这些文件加入 CAP-0 提交：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

git merge-base --is-ancestor 73b03bbe61c0398704dc77548e615d0c92c05c0a HEAD
git status --short --branch | tee "$E0_DIR/cap0_git_status.txt"
git rev-parse HEAD | tee "$E0_DIR/cap0_git_head.txt"
git log -5 --oneline | tee "$E0_DIR/cap0_git_log.txt"
git remote -v | tee "$E0_DIR/cap0_git_remote.txt"
```

### 16.3 环境和 GPU 取证

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

"$BRIDGE_PY" -c 'import sys,torch,vllm; print("python",sys.version); print("torch",torch.__version__); print("torch_cuda",torch.version.cuda); print("vllm",vllm.__version__); print("vllm_path",vllm.__file__)' \
  | tee "$E0_DIR/environment.txt"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader | tee "$E0_DIR/gpu_inventory.txt"
nvidia-smi topo -m | tee "$E0_DIR/gpu_topology.txt"
sha256sum "$BRIDGE_MODEL/config.json" "$BRIDGE_MODEL/generation_config.json" \
  | tee "$E0_DIR/model_config_SHA256SUMS"
```

### 16.4 CPU 回归

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

"$BRIDGE_PY" -m py_compile \
  tools/bridge_tp/run_phase9_capacity_background.py \
  tools/bridge_tp/run_phase9_controller.py \
  vllm/bridge_tp/controller/action_adapter.py \
  vllm/bridge_tp/controller/anchor_selector.py \
  vllm/bridge_tp/controller/capacity_signal.py \
  vllm/bridge_tp/controller/config.py \
  vllm/bridge_tp/controller/events.py \
  vllm/bridge_tp/controller/policy.py \
  vllm/bridge_tp/controller/state_machine.py \
  vllm/bridge_tp/controller/telemetry.py \
  vllm/bridge_tp/kv_stream.py \
  vllm/bridge_tp/phase8_source.py \
  vllm/bridge_tp/takeover_api.py \
  2>&1 | tee "$E0_DIR/py_compile.txt"

"$BRIDGE_PY" -m unittest \
  tests.bridge_tp.test_phase9_capacity_pilot \
  tests.bridge_tp.test_phase9_predictor_policy \
  tests.bridge_tp.test_phase9_state_and_proxy \
  tests.bridge_tp.test_phase9_online_integration \
  tests.bridge_tp.test_phase9_d3_batch \
  tests.bridge_tp.test_phase9_telemetry_control \
  2>&1 | tee "$E0_DIR/cap0_targeted_tests.txt"

"$BRIDGE_PY" -m unittest discover -s tests -t . -p 'test_phase9_*.py' \
  2>&1 | tee "$E0_DIR/all_phase9_tests.txt"

git diff --check | tee "$E0_DIR/git_diff_check.txt"
echo "git_diff_check_rc=${PIPESTATUS[0]}" | tee -a "$E0_DIR/git_diff_check.txt"
git status --short --branch | tee "$E0_DIR/post_test_git_status.txt"

# 不再断言整个工作树为空：切换前已存在且不与 CAP-0 冲突的用户文件允许保留。
# 只要上面的 CAP-0 commit 祖先核验、Python 编译和测试通过，就不会因这些用户文件退出终端。
```

预期基线是 targeted 121 tests 通过、全部 `test_phase9_*.py` 为 174 tests 通过且 4 skipped；
若服务器依赖版本导致计数变化，必须解释新增/缺失测试，不能只记录“PASS”。

## 17. 读取真实 TP1/TP4 KV block 数

先确认 8001/8200 未被旧进程占用；若有占用，人工确认其归属并正常停止，不用 `pkill -f`：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

ss -ltnp | grep -E ':(8001|8200)\b' || true
```

启动一次与后续相同参数的 clean geometry probe。两个 server 都以前台方式运行，分别占用
一个终端；不要关闭终端，也不要在命令末尾添加 `&`。

终端 G1：TP4，监听 8200：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export GEOM_DIR="$E0_DIR/geometry_probe"
mkdir -p "$GEOM_DIR"

CUDA_VISIBLE_DEVICES=1,2,3,4 "$BRIDGE_PY" \
  -m vllm.entrypoints.openai.api_server \
  --model "$BRIDGE_MODEL" --served-model-name bridgetp-model \
  --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.88 --port 8200 \
  --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  2>&1 | tee "$GEOM_DIR/target_tp4.log"
```

终端 G2：TP1，监听 8001：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export GEOM_DIR="$E0_DIR/geometry_probe"
mkdir -p "$GEOM_DIR"

CUDA_VISIBLE_DEVICES=0 "$BRIDGE_PY" \
  -m vllm.entrypoints.openai.api_server \
  --model "$BRIDGE_MODEL" --served-model-name bridgetp-model \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.88 --port 8001 \
  --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  2>&1 | tee "$GEOM_DIR/source_tp1.log"
```

终端 G3：等待健康并读取日志：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export GEOM_DIR="$E0_DIR/geometry_probe"

for url in http://127.0.0.1:8001/health http://127.0.0.1:8200/health; do
  for attempt in $(seq 1 450); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "READY: $url"
      break
    fi
    if [ "$attempt" -eq 450 ]; then
      echo "TIMEOUT: $url"
      exit 1
    fi
    sleep 2
  done
done

grep -Ei 'GPU KV cache size|GPU blocks|num_gpu_blocks' \
  "$GEOM_DIR/source_tp1.log" "$GEOM_DIR/target_tp4.log" \
  | tee "$GEOM_DIR/kv_block_lines.txt"
```

日志中的 `GPU KV cache size` 单位是 token，而后续参数单位是 KV block。当前配置的
`block_size=16 tokens`，必须先验证能够整除 16，再换算成 block 数。TP4 日志值已经对应
该 TP4 实例的 per-rank 容量，换算后不要再乘 4：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export GEOM_DIR="$E0_DIR/geometry_probe"
export KV_BLOCK_SIZE=16
export TP1_KV_TOKENS=31488
export TP4_KV_TOKENS=571824
test $((TP1_KV_TOKENS % KV_BLOCK_SIZE)) -eq 0
test $((TP4_KV_TOKENS % KV_BLOCK_SIZE)) -eq 0
export TP1_BLOCKS=$((TP1_KV_TOKENS / KV_BLOCK_SIZE))
export TP4_BLOCKS=$((TP4_KV_TOKENS / KV_BLOCK_SIZE))
test "$TP1_BLOCKS" -gt 0
test "$TP4_BLOCKS" -gt 0
printf 'export KV_BLOCK_SIZE=%s\nexport TP1_KV_TOKENS=%s\nexport TP4_KV_TOKENS=%s\nexport TP1_BLOCKS=%s\nexport TP4_BLOCKS=%s\n' \
  "$KV_BLOCK_SIZE" "$TP1_KV_TOKENS" "$TP4_KV_TOKENS" \
  "$TP1_BLOCKS" "$TP4_BLOCKS" \
  | tee /root/autodl-tmp/bridgetp/phase9_cap0_geometry.env

test "$TP1_BLOCKS" -eq 1968
test "$TP4_BLOCKS" -eq 35739
```

环境文件写好后，分别回到 G2、G1 按 `Ctrl+C` 正常停止 TP1、TP4。确认 8001/8200 已释放后
再运行 052。若任一 server 失败，traceback 会直接显示在对应前台终端并同时保存在日志中。

## 18. S1：`d3-formal-052` post-fix 工程回归完整命令

052 必须在 CAP-0 进程全部停止、8001/8200 空闲时运行。批处理器不支持在 formal 模式选择
单条，因此生成一个保持原 50 条 frozen prompts 不变的工程 manifest，只把 052 的 prompt
复制到第一个 smoke slot，并使用不与 formal ID 重叠的新 ID。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

source /root/autodl-tmp/bridgetp/phase9_cap0_geometry.env
unset BRIDGETP_STREAM_SOURCE_REQUEST_ID_PREFIX

export POSTFIX_COMMIT="$(git rev-parse --short=12 HEAD)"
export POSTFIX_ID="d3-formal-052-postfix_${POSTFIX_COMMIT}"
export D3_ROOT="/root/autodl-tmp/bridgetp/results/phase9_d3_postfix_052/${POSTFIX_ID}"
export D3_MANIFEST="$D3_ROOT/d3_052_engineering_manifest.json"
mkdir -p "$D3_ROOT"

"$BRIDGE_PY" - \
  experiments/phase9/manifests/d3_prompts_50.json \
  "$D3_MANIFEST" "$POSTFIX_ID" <<'PY'
import json, sys
from pathlib import Path

source, target, request_id = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
data = json.loads(source.read_text(encoding='utf-8'))
prompt = next(x for x in data['prompts'] if x['request_id'] == 'd3-formal-052')
data['smoke_prompts'][0] = {**prompt, 'request_id': request_id}
target.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
PY

sha256sum experiments/phase9/manifests/d3_prompts_50.json "$D3_MANIFEST" \
  | tee "$D3_ROOT/manifest_SHA256SUMS"

"$BRIDGE_PY" tools/bridge_tp/run_phase9_d3_batch.py validate \
  --manifest "$D3_MANIFEST" --out-root "$D3_ROOT" --mode smoke

"$BRIDGE_PY" tools/bridge_tp/run_phase9_d3_batch.py migrate \
  --manifest "$D3_MANIFEST" --out-root "$D3_ROOT" --mode smoke --limit 1 \
  --python-bin "$BRIDGE_PY" --model-path "$BRIDGE_MODEL" \
  --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS" \
  --dtype bfloat16 --max-attempts 2 \
  2>&1 | tee "$D3_ROOT/migrate_console.txt"

"$BRIDGE_PY" tools/bridge_tp/run_phase9_d3_batch.py controls \
  --manifest "$D3_MANIFEST" --out-root "$D3_ROOT" --mode smoke --limit 1 \
  --python-bin "$BRIDGE_PY" --model-path "$BRIDGE_MODEL" \
  --dtype bfloat16 --vllm-commit "$(git rev-parse HEAD)" \
  2>&1 | tee "$D3_ROOT/controls_console.txt"

"$BRIDGE_PY" tools/bridge_tp/run_phase9_d3_batch.py status \
  --manifest "$D3_MANIFEST" --out-root "$D3_ROOT" --mode smoke --limit 1 \
  2>&1 | tee "$D3_ROOT/status_console.txt"
```

工程验收：`batch_progress.json` 中该 ID 的 migration/control 均为 `COMPLETE`；成功 attempt
中 `pending_known_tokens=1`，四 rank receipt 全部 exact readback，`takeover_state.json` 为
`COMMITTED`，日志不再出现 `Target block allocation differs from live snapshot`。不要运行
formal summarize，也不要把该目录合并进旧 49 条 D 结果。

## 19. 生成 no-migration/CAP-0 workload manifest

### 19.1 冻结 controller predictor 输入

controller 即使在 capacity disabled 和 `--dry-run` 下也会初始化 `SurvivalTable`。先从既有
M1 trace 确定性生成 CAP-0 predictor 输入，并保存 trace/table hash；这不是重跑 C，也不能
替代正式 E 的 predictor provenance：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export CAP0_TRACE=/root/autodl-tmp/bridgetp/experiments4/data/traces/qwen_traceA_e1.csv
export CAP0_INPUT_ROOT=/root/autodl-tmp/bridgetp/phase9_cap0_inputs
export CAP0_SURVIVAL_TABLE="$CAP0_INPUT_ROOT/survival_table_m1_v1.json"
test -f "$CAP0_TRACE"
mkdir -p "$CAP0_INPUT_ROOT"

"$BRIDGE_PY" tools/bridge_tp/build_survival_table.py \
  --trace "$CAP0_TRACE" --output-field output_len --time-field timestamp \
  --train-frac 0.7 --expected-total-rows 43058 \
  --expected-train-rows 30140 \
  --out "$CAP0_SURVIVAL_TABLE"

"$BRIDGE_PY" -c 'from vllm.bridge_tp.controller.predictor import SurvivalTable; import sys; t=SurvivalTable.load(sys.argv[1]); print("VALID",sys.argv[1],"max_observed_length",t.max_observed_length)' \
  "$CAP0_SURVIVAL_TABLE"
sha256sum "$CAP0_TRACE" "$CAP0_SURVIVAL_TABLE" \
  | tee "$CAP0_INPUT_ROOT/predictor_inputs.sha256"
test "$(sha256sum "$CAP0_TRACE" | awk '{print $1}')" = \
  443ad43e5264ba9c48e984e999f723ab3e73dcec53a46d7a2e1240514d393314
test "$(sha256sum "$CAP0_SURVIVAL_TABLE" | awk '{print $1}')" = \
  031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a
printf 'export CAP0_TRACE=%s\nexport CAP0_INPUT_ROOT=%s\nexport CAP0_SURVIVAL_TABLE=%s\n' \
  "$CAP0_TRACE" "$CAP0_INPUT_ROOT" "$CAP0_SURVIVAL_TABLE" \
  > /root/autodl-tmp/bridgetp/phase9_cap0_predictor.env
```

### 19.2 生成并冻结 workload manifest

先建立独立 manifest 目录。manifest 永远传给 workload generator，不传给 controller，也不
放进 controller 的 run directory：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export CAP0_MANIFEST_ROOT=/root/autodl-tmp/bridgetp/phase9_cap0_manifests
mkdir -p "$CAP0_MANIFEST_ROOT/working" "$CAP0_MANIFEST_ROOT/frozen"
```

以下生成器用于 bring-up。通过环境变量调节并发数、长度和到达时刻；它会把所有 model alias
冻结为 server 实际暴露的 `bridgetp-model`：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export LOAD_NAME=calibration_working
export TARGET_JOBS=1
export TARGET_TOKENS=512
export TARGET_START_S=0.0
export SOURCE_JOBS=2
export SOURCE_TOKENS=1024
export SOURCE_START_S=8.0

"$BRIDGE_PY" - "$CAP0_MANIFEST_ROOT/working/$LOAD_NAME.json" <<'PY'
import json, os, sys
from pathlib import Path

jobs = []
for i in range(int(os.environ['TARGET_JOBS'])):
    jobs.append({
        'job_id': f'target_{i:03d}', 'pool': 'target',
        'start_after_s': float(os.environ['TARGET_START_S']) + i * 0.2,
        'request': {'model': 'bridgetp-model',
                    'prompt': 'Explain queueing delay in a distributed inference service.',
                    'max_tokens': int(os.environ['TARGET_TOKENS']),
                    'ignore_eos': True}})
for i in range(int(os.environ['SOURCE_JOBS'])):
    jobs.append({
        'job_id': f'source_{i:03d}', 'pool': 'source',
        'start_after_s': float(os.environ['SOURCE_START_S']) + i * 0.2,
        'request': {'model': 'bridgetp-model',
                    'prompt': 'Write a detailed systems design review for a capacity-limited scheduler.',
                    'max_tokens': int(os.environ['SOURCE_TOKENS']),
                    'ignore_eos': True}})
out = Path(sys.argv[1])
payload = {'format_version': 1,
           'note': 'workload-generator input only; never controller input',
           'jobs': jobs}
out.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
print(out)
PY

"$BRIDGE_PY" tools/bridge_tp/run_phase9_capacity_background.py \
  --manifest "$CAP0_MANIFEST_ROOT/working/$LOAD_NAME.json" \
  --out-dir "/tmp/${LOAD_NAME}_validate" --validate-only
```

只允许在 bring-up 阶段改上述六个 workload 参数。下面是旧的 v1 冻结记录；它现在仅用于
说明版本化流程，不得重新作为有效 calibration 输入：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export LOAD_NAME=calibration_working
export FROZEN_NAME=calibration_v1
cp "$CAP0_MANIFEST_ROOT/working/$LOAD_NAME.json" \
  "$CAP0_MANIFEST_ROOT/frozen/$FROZEN_NAME.json"
sha256sum "$CAP0_MANIFEST_ROOT/frozen/$FROZEN_NAME.json" \
  | tee "$CAP0_MANIFEST_ROOT/frozen/$FROZEN_NAME.sha256"
```

`calibration_v1`、`noop_v1`、`rescue_v1` 和 `abandon_v1` 分别版本化。三个报告重复不能再改
对应 manifest；若必须改，版本号递增并重新做 bring-up，旧失败目录保留。

2026-09-05 的 `calibration_v1` 首次运行虽然工程验收通过，但只达到 2.44% TP1 KV usage，
controller 的 12.9 秒观测窗口也没有覆盖 33.6 秒 background 生命周期。因此 v1 被保留为
无效 bring-up 证据，禁止用于 guard。替代输入是仓库内冻结的
`experiments/phase9/manifests/cap0_calibration_v2.json`；smoke 和 formal 必须使用它的同一
SHA-256，不能在两阶段之间调参。

## 20. 每一轮通用的目录、config 和服务启动命令

本节保留给 held-out 场景和人工诊断。calibration v2 不得直接照搬本节的 512-token
`request_long.json`；它必须使用第 21 节 runner 生成的 8000-token anchor，否则 controller
会再次早于 background 结束。

### 20.1 创建全新运行身份

每轮先设置 `CAP0_CLASS`、`CAP0_REP`、manifest 和 guard：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

source /root/autodl-tmp/bridgetp/phase9_cap0_geometry.env
source /root/autodl-tmp/bridgetp/phase9_cap0_predictor.env

export CAP0_CLASS=calibration
export CAP0_REP=01
export CAP0_ID="cap0-${CAP0_CLASS}-r${CAP0_REP}-$(date -u +%Y%m%dT%H%M%SZ)"
export CAP0_CONTROLLER_ROOT="$CAP0_RESULTS/controller_runs"
export CAP0_BACKGROUND_ROOT="$CAP0_RESULTS/background_runs"
export CAP0_PROVENANCE_ROOT="$CAP0_RESULTS/provenance"
export CAP0_DIR="$CAP0_CONTROLLER_ROOT/$CAP0_ID"
export CAP0_BG_DIR="$CAP0_BACKGROUND_ROOT/$CAP0_ID"
export CAP0_PROV_DIR="$CAP0_PROVENANCE_ROOT/$CAP0_ID"
export CAP0_CONFIG="$CAP0_PROV_DIR/controller_config.json"
export CAP0_BG_MANIFEST="$BRIDGE_REPO/experiments/phase9/manifests/cap0_calibration_v2.json"
export CAP0_CAPACITY_ENABLED=0
export CAP0_GUARD=0

test ! -e "$CAP0_DIR"
test ! -e "$CAP0_BG_DIR"
test ! -e "$CAP0_PROV_DIR"
mkdir -p "$CAP0_DIR" "$CAP0_BG_DIR" "$CAP0_PROV_DIR"

printf '%s\n' \
  "export BRIDGE_REPO=$BRIDGE_REPO" \
  "export BRIDGE_PY=$BRIDGE_PY" \
  "export BRIDGE_MODEL=$BRIDGE_MODEL" \
  "export TP1_BLOCKS=$TP1_BLOCKS" \
  "export TP4_BLOCKS=$TP4_BLOCKS" \
  "export CAP0_SURVIVAL_TABLE=$CAP0_SURVIVAL_TABLE" \
  "export CAP0_ID=$CAP0_ID" \
  "export CAP0_DIR=$CAP0_DIR" \
  "export CAP0_BG_DIR=$CAP0_BG_DIR" \
  "export CAP0_PROV_DIR=$CAP0_PROV_DIR" \
  "export CAP0_CONFIG=$CAP0_CONFIG" \
  "export CAP0_BG_MANIFEST=$CAP0_BG_MANIFEST" \
  "export CAP0_CAPACITY_ENABLED=$CAP0_CAPACITY_ENABLED" \
  "export CAP0_GUARD=$CAP0_GUARD" \
  > "/root/autodl-tmp/bridgetp/cap0_${CAP0_ID}.env"

cp "/root/autodl-tmp/bridgetp/cap0_${CAP0_ID}.env" \
  /root/autodl-tmp/bridgetp/phase9_cap0_active.env

git rev-parse HEAD > "$CAP0_PROV_DIR/git_revision.txt"
git status --short --branch > "$CAP0_PROV_DIR/git_status.txt"
sha256sum "$CAP0_BG_MANIFEST" > "$CAP0_PROV_DIR/background_manifest.sha256"
```

为 calibration 设置 `CAP0_CAPACITY_ENABLED=0`、`CAP0_GUARD=0` 并配合 `--dry-run`；
三个 held-out 场景设置 `CAP0_CAPACITY_ENABLED=1` 和已冻结的正 guard：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

"$BRIDGE_PY" - \
  experiments/phase9/configs/cap0_controller.template.json \
  "$CAP0_CONFIG" <<'PY'
import json, os, sys
from pathlib import Path

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
cfg = json.loads(src.read_text(encoding='utf-8'))
cfg['run_dir'] = os.environ['CAP0_DIR']
cfg['tp1_total_kv_blocks'] = int(os.environ['TP1_BLOCKS'])
cfg['tp4_total_kv_blocks'] = int(os.environ['TP4_BLOCKS'])
cfg['survival_table_path'] = os.environ['CAP0_SURVIVAL_TABLE']
cfg['platform_note'] = 'CAP-0 ENGINEERING PILOT; 5x A100 PCIe; exact provenance in run'
cfg['capacity_pilot']['enabled'] = bool(int(os.environ['CAP0_CAPACITY_ENABLED']))
cfg['capacity_pilot']['guard_free_kv_tokens'] = int(os.environ['CAP0_GUARD'])
dst.write_text(json.dumps(cfg, indent=2) + '\n', encoding='utf-8')
PY

"$BRIDGE_PY" -c 'from vllm.bridge_tp.controller.config import ControllerConfig; import sys; ControllerConfig.load(sys.argv[1]); print("VALID",sys.argv[1])' "$CAP0_CONFIG"
```

### 20.2 前台启动 TP4、TP1 和 stager

每个进程使用独立终端前台运行。不要使用 `nohup`、命令末尾的 `&` 或 PID 文件。这样任何
CUDA、模型加载、端口和 connector 异常都会立即显示，同时由 `tee` 保存。

终端 T1：TP4 target，GPU1--4，HTTP 端口 8200：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

export PHASE9_STAGING_MANIFEST="$CAP0_DIR/staging_manifest.json"
export PHASE9_RECEIPTS="$CAP0_DIR/receiver_receipts"
export PHASE9_CONTROL="$CAP0_DIR/takeover_state.json"

CUDA_VISIBLE_DEVICES=1,2,3,4 "$BRIDGE_PY" \
  -m vllm.entrypoints.openai.api_server \
  --model "$BRIDGE_MODEL" --served-model-name bridgetp-model \
  --tensor-parallel-size 4 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.88 --port 8200 \
  --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  --kv-transfer-config "$("$BRIDGE_PY" -c 'import json,os; print(json.dumps({
    "kv_connector":"BridgeTPStreamingConnector",
    "kv_connector_module_path":"vllm.bridge_tp.streaming_connector",
    "kv_role":"kv_consumer","kv_load_failure_policy":"fail",
    "kv_connector_extra_config":{
      "bridgetp_stream_manifest":os.environ["PHASE9_STAGING_MANIFEST"],
      "bridgetp_stream_receipt_dir":os.environ["PHASE9_RECEIPTS"],
      "bridgetp_stream_socket_timeout_s":600,
      "bridgetp_stream_expected_phase":"BridgeTP D3 Phase 8",
      "bridgetp_takeover_control_path":os.environ["PHASE9_CONTROL"],
      "bridgetp_takeover_control_timeout_s":600}}))')" \
  2>&1 | tee "$CAP0_DIR/target_tp4.log"
```

终端 T2：TP1 source，GPU0，HTTP 端口 8001：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

export BRIDGETP_DUMP_ENABLED=0
export BRIDGETP_STREAM_ENABLED=1
export BRIDGETP_STREAM_MIGRATION_ID="$CAP0_ID"
export BRIDGETP_STREAM_RUN_DIR="$CAP0_DIR"
export BRIDGETP_STREAM_HOST=127.0.0.1
export BRIDGETP_STREAM_BASE_PORT=29800
export BRIDGETP_STREAM_TARGET_TP=4
export BRIDGETP_STREAM_HEAD_AXIS=3
export BRIDGETP_STREAM_EXPECTED_KV_HEADS=8
export BRIDGETP_STREAM_AFTER_OUTPUT_TOKENS=128
export BRIDGETP_STREAM_CHUNK_BYTES=1048576
export BRIDGETP_STREAM_RATE_GIB_S=0.50
export BRIDGETP_STREAM_SOCKET_TIMEOUT_S=600
export BRIDGETP_STREAM_PIN_MEMORY=1
export BRIDGETP_STREAM_STRICT=1
export BRIDGETP_STREAM_SOURCE_REQUEST_ID_PREFIX="bridgetp-phase9-$CAP0_ID"
export BRIDGETP_PHASE8_ENABLED=1
export BRIDGETP_PHASE8_CUTOVER_OUTPUT_TOKENS=160
export BRIDGETP_PHASE8_DELTA_HOST=127.0.0.1
export BRIDGETP_PHASE8_DELTA_BASE_PORT=29900
export BRIDGETP_TAKEOVER_ENABLED=1
export BRIDGETP_TAKEOVER_MIGRATION_ID="$CAP0_ID"
export BRIDGETP_TAKEOVER_RUN_DIR="$CAP0_DIR"

CUDA_VISIBLE_DEVICES=0 "$BRIDGE_PY" \
  -m vllm.entrypoints.openai.api_server \
  --model "$BRIDGE_MODEL" --served-model-name bridgetp-model \
  --tensor-parallel-size 1 --dtype bfloat16 --max-model-len 8192 \
  --gpu-memory-utilization 0.88 --port 8001 \
  --no-enable-prefix-caching --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  2>&1 | tee "$CAP0_DIR/source_tp1.log"
```

终端 T6：等待两个 HTTP 服务健康，并保存进程信息：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

for url in http://127.0.0.1:8001/health http://127.0.0.1:8200/health; do
  for attempt in $(seq 1 450); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "READY: $url"
      break
    fi
    if [ "$attempt" -eq 450 ]; then
      echo "TIMEOUT: $url"
      exit 1
    fi
    sleep 2
  done
done

ps -eo pid,ppid,lstart,args \
  | grep -E 'vllm.entrypoints.openai.api_server.*--port (8001|8200)' \
  | grep -v grep \
  | tee "$CAP0_PROV_DIR/processes.txt"
```

只有 T6 报告两个 URL 均 `READY` 后，才打开终端 T3 启动 stager。T3 使用 delta 端口
29900 和 delivery 端口 30000--30003：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

"$BRIDGE_PY" tools/bridge_tp/phase8_stager.py \
  --run-dir "$CAP0_DIR" --delta-host 127.0.0.1 --delta-base-port 29900 \
  --delivery-host 127.0.0.1 --delivery-base-port 30000 --timeout-s 600 \
  2>&1 | tee "$CAP0_DIR/stager.log"
```

每轮都重新核对日志 block 数与冻结值一致：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

grep -Ei 'GPU KV cache size|GPU blocks|num_gpu_blocks' \
  "$CAP0_DIR/source_tp1.log" "$CAP0_DIR/target_tp4.log" \
  | tee "$CAP0_PROV_DIR/kv_block_lines.txt"
```

### 20.3 前台启动独立 workload generator 和 controller

先单独验证 manifest：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

"$BRIDGE_PY" tools/bridge_tp/run_phase9_capacity_background.py \
  --manifest "$CAP0_BG_MANIFEST" \
  --out-dir "$CAP0_BG_DIR" --validate-only
```

终端 T4：先准备好下面的 workload generator 命令，但暂时不要执行。它虽然在代码和文件名中
叫 background，但这里不以 shell 后台进程启动：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

"$BRIDGE_PY" tools/bridge_tp/run_phase9_capacity_background.py \
  --manifest "$CAP0_BG_MANIFEST" \
  --source-url http://127.0.0.1:8001 \
  --target-url http://127.0.0.1:8200 \
  --out-dir "$CAP0_BG_DIR" \
  2>&1 | tee "$CAP0_BG_DIR/background.log"
```

先在终端 T5 启动 controller；看到进程开始运行后立即切到 T4 执行上面的 workload 命令。
这样 controller 的显式 anchor 会先进入 TP1，避免人工切换终端跨过 manifest 中的 source
burst 时刻。calibration/no-migration 在 T5 使用：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

"$BRIDGE_PY" tools/bridge_tp/run_phase9_controller.py \
  --config "$CAP0_CONFIG" --run-dir "$CAP0_DIR" \
  --source-request experiments/phase9/configs/request_long.json \
  --migration-id "$CAP0_ID" --dry-run \
  2>&1 | tee "$CAP0_PROV_DIR/controller_console.txt"
```

No-op、Rescue、Safe abandon 在 T5 使用相同命令但去掉 `--dry-run`：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

"$BRIDGE_PY" tools/bridge_tp/run_phase9_controller.py \
  --config "$CAP0_CONFIG" --run-dir "$CAP0_DIR" \
  --source-request experiments/phase9/configs/request_long.json \
  --migration-id "$CAP0_ID" \
  2>&1 | tee "$CAP0_PROV_DIR/controller_console.txt"
```

等待 T4 workload generator 和 T5 controller 都自然结束并保存输出。然后分别回到 T3、T2、
T1 按 `Ctrl+C`，依次正常停止 stager、TP1、TP4。不要使用 `pkill -f`。下一轮必须重新打开
五个前台进程，并使用新的 run ID 和目录。

## 21. C0/C1：三次 no-migration 标定、分析和 guard 冻结

先用同一 `cap0_calibration_v2.json`、`CAP0_CAPACITY_ENABLED=0`、`CAP0_GUARD=0` 和
`--dry-run` 跑一次不计数 smoke。人工确认 smoke 后，才按同一冻结 contract 执行 `r01`、
`r02`、`r03`。每轮必须重启 TP1、TP4 和 stager。disabled
tracker 仍记录 raw free-KV、EWMA decline、running、waiting 和 preemption；`--dry-run` 保证
即使 legacy performance decision 出现也不会执行 actuator。

### 21.1 第一步：只跑不计数 smoke

首次 v2 smoke `cap0-calibration-smoke-20260905T033220Z` 在 background 启动前失败：嵌套
`smoke/controller` 布局令 controller 生成 `bridgetp-phase9-controller` 请求 ID，而 TP1 selector
错误地使用了包含完整 run ID 的旧前缀。anchor 本身已在 TP1 正常生成，但无法写出
`source_progress.json`。该轮只作失败诊断，不计数；修复后的 runner 从 `controller` 目录名
推导同一请求前缀，并为这个契约增加单测。重跑必须使用新的 smoke ID 和目录。

`run_phase9_cap0_calibration.py` 把第 20 节的五个进程封装成一个前台父进程。它每轮依次
启动 TP4、TP1、stager、dry-run controller 和冻结 workload，健康检查后才进入下一步。
smoke 只运行一次且不计入 r01--r03。它必须同时证明：五个 background jobs 全部完成、
采样到的 TP1 peak KV usage 至少 70%、至少观察到一次 preemption、telemetry 覆盖完整 background
生命周期、最终留在 TP1 且没有任何迁移转换。任一条件不满足都不生成 PASS contract。
70% 是采样下限而非容量阈值：一次 preemption 可瞬时释放最大约 25% TP1 cache，因此不要求
0.2 秒 telemetry 必须恰好捕获释放前的 100%；preemption 单调计数才是确证容量触顶的主门。
controller `run_end` 和最后一条 telemetry 对 background `run_end` 使用同一个显式
`coverage_slack_s=1.0`；容差内的进程完成顺序抖动不判失败，超过容差仍拒绝。该 slack 只处理
采样和进程回收边界，不放宽五个 job 全部完成、preemption、峰值压力及 no-migration 门。

一个终端执行 smoke：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_geometry.env
source /root/autodl-tmp/bridgetp/phase9_cap0_predictor.env

export OMP_NUM_THREADS=1
export CAP0_CODE_REVISION="$(git rev-parse HEAD)"
export CAP0_CAL_MANIFEST="$BRIDGE_REPO/experiments/phase9/manifests/cap0_calibration_v2.json"
export CAP0_CAL_MANIFEST_SHA=7bd5b2f6581610bef15460beb678026c687e84486da9af2ac6c1146f1eced2bf
export SMOKE_ID="cap0-calibration-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
export SMOKE_PARENT="$CAP0_RESULTS/calibration_smoke"
export SMOKE_DIR="$SMOKE_PARENT/$SMOKE_ID"
mkdir -p "$SMOKE_PARENT"
set -o pipefail

"$BRIDGE_PY" tools/bridge_tp/run_phase9_cap0_calibration.py \
  --mode smoke \
  --model-path "$BRIDGE_MODEL" \
  --manifest "$CAP0_CAL_MANIFEST" \
  --survival-table "$CAP0_SURVIVAL_TABLE" \
  --out-root "$SMOKE_DIR" \
  --expected-revision "$CAP0_CODE_REVISION" \
  --expected-manifest-sha256 "$CAP0_CAL_MANIFEST_SHA" \
  --expected-survival-sha256 \
    031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a \
  --tp1-blocks "$TP1_BLOCKS" \
  --tp4-blocks "$TP4_BLOCKS" \
  --anchor-max-tokens 8000 \
  --minimum-peak-kv-usage-frac 0.70 \
  2>&1 | tee "$SMOKE_PARENT/${SMOKE_ID}.console.txt"
```

成功时终端只出现 `SMOKE_COMPLETE`，不会继续跑 r01。将下面三个文件取回并人工审阅：

```text
calibration_smoke/<smoke-id>/
  batch_status.json
  smoke_acceptance.json
  smoke/{controller,background,provenance,status.json}
```

### 21.2 第二步：人工批准 smoke 后跑正式 r01--r03

formal 模式必须显式传入上一步的 `smoke_acceptance.json`。脚本逐字段核对 smoke 与 formal
使用相同 commit、manifest hash、survival hash、TP1/TP4 blocks、anchor 长度和压力门槛；
不一致时在启动 GPU 服务前拒绝运行。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_geometry.env
source /root/autodl-tmp/bridgetp/phase9_cap0_predictor.env

export OMP_NUM_THREADS=1
export CAP0_CODE_REVISION="$(git rev-parse HEAD)"
export CAP0_CAL_MANIFEST="$BRIDGE_REPO/experiments/phase9/manifests/cap0_calibration_v2.json"
export CAP0_CAL_MANIFEST_SHA=7bd5b2f6581610bef15460beb678026c687e84486da9af2ac6c1146f1eced2bf
export SMOKE_ACCEPTANCE=替换为已人工批准的smoke_acceptance.json绝对路径
export CAL_BATCH_ID="cap0-calibration-formal-$(date -u +%Y%m%dT%H%M%SZ)"
export CAL_BATCH_PARENT="$CAP0_RESULTS/calibration_batches"
export CAL_BATCH_DIR="$CAL_BATCH_PARENT/$CAL_BATCH_ID"
mkdir -p "$CAL_BATCH_PARENT"
set -o pipefail

"$BRIDGE_PY" tools/bridge_tp/run_phase9_cap0_calibration.py \
  --mode formal \
  --smoke-acceptance "$SMOKE_ACCEPTANCE" \
  --model-path "$BRIDGE_MODEL" \
  --manifest "$CAP0_CAL_MANIFEST" \
  --survival-table "$CAP0_SURVIVAL_TABLE" \
  --out-root "$CAL_BATCH_DIR" \
  --expected-revision "$CAP0_CODE_REVISION" \
  --expected-manifest-sha256 "$CAP0_CAL_MANIFEST_SHA" \
  --expected-survival-sha256 \
    031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a \
  --tp1-blocks "$TP1_BLOCKS" \
  --tp4-blocks "$TP4_BLOCKS" \
  --anchor-max-tokens 8000 \
  --minimum-peak-kv-usage-frac 0.70 \
  --repetitions 3 \
  2>&1 | tee "$CAL_BATCH_PARENT/${CAL_BATCH_ID}.console.txt"
```

不要预先创建 `$SMOKE_DIR` 或 `$CAL_BATCH_DIR`。正式模式中任一轮失败会立即停止并保留全部
日志；三轮通过后才生成 `guard_candidate.json`，仍需人工审阅，脚本不会冻结 guard。按
`Ctrl+C` 会清理该父进程创建的服务，不要用 `pkill -f`。No-op、Rescue、Safe abandon 不在
此循环中。

三轮完成后输出可审计摘要：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export CAL_BATCH_DIR=替换为正式三轮批次的绝对路径
export CAL_GLOB="$CAL_BATCH_DIR/r*/controller"
"$BRIDGE_PY" - $CAL_GLOB <<'PY'
import json, sys
from pathlib import Path

for run_s in sys.argv[1:]:
    run = Path(run_s)
    rows = [json.loads(x) for x in (run/'phase9_audit.jsonl').read_text().splitlines()]
    tel = [x for x in rows if x.get('kind') == 'telemetry']
    if not tel:
        raise SystemExit(f'no telemetry: {run}')
    initial = int(tel[0]['tp1']['preemptions_total'])
    first_index = next((i for i, x in enumerate(tel)
                        if int(x['tp1']['preemptions_total']) > initial), None)
    first = tel[first_index] if first_index is not None else None
    before = tel[:first_index] if first_index is not None else tel
    free = [int(x['capacity_signal']['free_kv_tokens']) for x in tel]
    decline = [float(x['capacity_signal']['decline_rate_tokens_s']) for x in tel]
    steady = []
    for previous, current in zip(before, before[1:]):
        fields = ('num_running', 'num_waiting', 'preemptions_total')
        if any(int(previous['tp1'][k]) != int(current['tp1'][k]) for k in fields):
            continue
        dt = float(current['tp1']['sampled_unix_s']) - float(previous['tp1']['sampled_unix_s'])
        consumed = (int(previous['capacity_signal']['free_kv_tokens'])
                    - int(current['capacity_signal']['free_kv_tokens']))
        if 0 < dt <= 2.0 and consumed > 0:
            steady.append(consumed / dt)
    steady.sort()
    p95 = steady[max(0, __import__('math').ceil(0.95 * len(steady)) - 1)]
    print(json.dumps({
        'run': run.name,
        'samples': len(tel),
        'minimum_free_kv_tokens': min(free),
        'pre_preemption_free_kv_tokens': (
            int(tel[first_index - 1]['capacity_signal']['free_kv_tokens'])
            if first_index is not None and first_index > 0 else None),
        'post_preemption_free_kv_tokens': (
            int(first['capacity_signal']['free_kv_tokens']) if first else None),
        'steady_decline_sample_count': len(steady),
        'steady_decline_p95_tokens_s': p95,
        'maximum_observed_ewma_decline_tokens_s': max(decline),
        'final_state': next(x for x in reversed(rows) if x.get('kind') == 'run_end')['final_state'],
    }, sort_keys=True))
PY
```

在查看 held-out 三场景前冻结以下事前规则：令每轮 `F_i` 为第一次 preemption counter 上升
之前最后一个 telemetry 样本的 free KV。
v2 默认协议要求 smoke 和三个 formal repetition 都观察到 preemption；任一轮无 preemption
直接失败，不能用 closest approach 代替。`--allow-censored` 只保留给未来显式修订协议，当前
不得使用。令每轮 `D_i` 为首次抢占前、调度状态不变相邻样本正向瞬时下降率的 nearest-rank
p95，`D_max=max_i(D_i)`，`T_enter=8s`。全程最大 EWMA 只保留为诊断量，不能进入 guard
公式，因为请求入场/恢复的离散 KV 分配会污染它。工程 guard 为：

```text
ceil_to_block(max_i(F_i) + D_max * T_enter)
```

并裁剪到 `[block_size, TP1_total_tokens - block_size]`。该规则宁可提前触发；它只是 CAP-0
guard，不是 OOM 概率。把最终整数写入并冻结：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export FROZEN_GUARD=替换为按上述规则计算出的正整数
export TP1_TOTAL_TOKENS=$((TP1_BLOCKS * 16))
test "$FROZEN_GUARD" -gt 0
test "$FROZEN_GUARD" -lt "$TP1_TOTAL_TOKENS"
printf '%s\n' "$FROZEN_GUARD" \
  | tee "$CAP0_MANIFEST_ROOT/frozen/guard_free_kv_tokens.txt"
printf '%s  %s\n' \
  "$(sha256sum "$BRIDGE_REPO/experiments/phase9/manifests/cap0_calibration_v2.json" | awk '{print $1}')" \
  cap0_calibration_v2.json \
  > "$CAP0_MANIFEST_ROOT/frozen/calibration_and_guard_provenance.txt"
```

## 22. P0/P1/P2：三个 held-out 场景的执行矩阵

三个场景均先做不计入重复的 bring-up，冻结 manifest 后做 `r01`、`r02`、`r03`。每轮在
第 20.1 节设置：

| 场景 | `CAP0_CLASS` | manifest | controller | 必须出现 |
|---|---|---|---|---|
| No-op | `noop` | `noop_v1.json` | 非 dry-run | active signal + target guard fail + `STAY` |
| Rescue | `rescue` | `rescue_v1.json` | 非 dry-run | `CAPACITY_PILOT` + 4 ranks + `TAKEOVER` |
| Safe abandon | `abandon` | `abandon_v1.json` | 非 dry-run | `SHADOW` 后 `CLEAR` + `CANCELLED` |

### 22.1 P0 No-op 自动 bring-up

P0 不再手工协调五个终端。`build_phase9_cap0_noop_manifest.py` 生成仅供独立 workload
进程读取的 working manifest；默认使用 72 个 TP4 请求，每个请求由 7000 个确定性 token-ID
prompt 和 1100 个输出 token 构成，总 target context demand 为 583200 tokens，高于实测 TP4
容量 571824 tokens。72 个 target 请求在 1 秒内形成 burst，并早于四个 TP1 source-pressure
请求；TP1 仍使用与 calibration 相同的 4x7000 output shape 和独立 8000-token anchor。

`run_phase9_cap0_noop.py` 是前台父进程，拥有并清理 TP4、TP1、stager、controller 和
workload。它启用 frozen guard、关闭 performance trigger 的竞争路径，并使用非 dry-run
controller。bring-up 只有同时满足以下条件才 PASS：

- source capacity signal 至少一次 active；
- active 时 TP4 `kv_usage_frac>0.85` 或 `num_waiting>4`，并记录明确 `STAY`；
- 从未记录 `START_SHADOW`，也没有 Shadow/Handoff/Takeover transition；
- 不存在 staging、takeover 或 cleanup 工件；
- 8000-token anchor 全部由 TP1 发出，五类进程和全部 background jobs 正常结束；
- 输出状态只能是 `BRINGUP_COMPLETE`，不得自动冻结 manifest 或启动正式重复。

默认 target shape 只是预注册的首次 bring-up 候选，不是成功保证。如果它没有在 source
signal active 的同时形成 target guard failure，保留失败目录并只修改下一版本 working
manifest；不得降低 frozen source guard、读取 controller 未来状态或人工取消请求。

生成器与 runner 的服务器命令在代码提交后以该提交对应的命令块为准。bring-up 输出至少包括
`status.json`、`provenance/noop_acceptance.json`、输入哈希、完整 controller/background 日志和
实际展开后的 background manifest。人工审阅 PASS 后才允许把确切 jobs 冻结为
`noop_v1.json` 并开发正式重复入口。

代码提交并在服务器切换到明确 revision 后，首次 bring-up 使用一个前台终端：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_geometry.env
source /root/autodl-tmp/bridgetp/phase9_cap0_predictor.env

export OMP_NUM_THREADS=1
export CAP0_CODE_REVISION="$(git rev-parse HEAD)"
export CAP0_GUARD_FILE="$CAP0_MANIFEST_ROOT/frozen/guard_free_kv_tokens.txt"
export CAP0_GUARD_SHA=0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b
export NOOP_ID="cap0-noop-bringup-$(date -u +%Y%m%dT%H%M%SZ)"
export NOOP_MANIFEST="$CAP0_MANIFEST_ROOT/working/${NOOP_ID}.json"
export NOOP_ROOT="$CAP0_RESULTS/noop_bringup/$NOOP_ID"

mkdir -p "$CAP0_MANIFEST_ROOT/working" "$CAP0_RESULTS/noop_bringup"

"$BRIDGE_PY" tools/bridge_tp/build_phase9_cap0_noop_manifest.py \
  --out "$NOOP_MANIFEST"
export NOOP_MANIFEST_SHA="$(sha256sum "$NOOP_MANIFEST" | awk '{print $1}')"

"$BRIDGE_PY" tools/bridge_tp/run_phase9_cap0_noop.py \
  --validate-only \
  --model-path "$BRIDGE_MODEL" --manifest "$NOOP_MANIFEST" \
  --survival-table "$CAP0_SURVIVAL_TABLE" --guard-file "$CAP0_GUARD_FILE" \
  --out-root "$NOOP_ROOT" --expected-revision "$CAP0_CODE_REVISION" \
  --expected-manifest-sha256 "$NOOP_MANIFEST_SHA" \
  --expected-survival-sha256 \
    031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a \
  --expected-guard 8448 --expected-guard-sha256 "$CAP0_GUARD_SHA" \
  --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS"

set -o pipefail
"$BRIDGE_PY" tools/bridge_tp/run_phase9_cap0_noop.py \
  --model-path "$BRIDGE_MODEL" --manifest "$NOOP_MANIFEST" \
  --survival-table "$CAP0_SURVIVAL_TABLE" --guard-file "$CAP0_GUARD_FILE" \
  --out-root "$NOOP_ROOT" --expected-revision "$CAP0_CODE_REVISION" \
  --expected-manifest-sha256 "$NOOP_MANIFEST_SHA" \
  --expected-survival-sha256 \
    031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a \
  --expected-guard 8448 --expected-guard-sha256 "$CAP0_GUARD_SHA" \
  --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS" \
  2>&1 | tee "$CAP0_RESULTS/noop_bringup/${NOOP_ID}.console.txt"
echo "NOOP_RC=${PIPESTATUS[0]}"
echo "NOOP_ROOT=$NOOP_ROOT"
```

### 22.2 冻结 P0 No-op workload 并运行正式重复

只有 22.1 的 bring-up 输出 `NOOP_BRINGUP_COMPLETE`，且人工核验每一个 active
capacity decision 都是 target-guarded `STAY` 后，才允许冻结。冻结器读取实际展开后的
`background_manifest.json`，逐项核对 bring-up status、acceptance、inputs、代码 revision、
survival table、guard 和原始 manifest 哈希，然后生成不可覆盖的 `noop_v1.json`、checksum
和 provenance。不能手工编辑冻结文件，也不能把重新调用 builder 生成的近似 workload 当作
已验证 workload。

以下命令中的 bring-up ID 和原始哈希对应首次通过的 P0 bring-up。正式代码 revision 必须在
服务器拉取正式 runner 提交后固定：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_geometry.env
source /root/autodl-tmp/bridgetp/phase9_cap0_predictor.env

export OMP_NUM_THREADS=1
export CAP0_CODE_REVISION="$(git rev-parse HEAD)"
export BRINGUP_REVISION=613daee714e2632b3b225765badcaa72fab44c61
export BRINGUP_ID=cap0-noop-bringup-20260905T071651Z
export BRINGUP_ROOT="$CAP0_RESULTS/noop_bringup/$BRINGUP_ID"
export EXECUTED_MANIFEST="$BRINGUP_ROOT/background/background_manifest.json"
export ORIGIN_MANIFEST_SHA=e3310a97b3ae7d524d681ddfbd26f114f0563684575b245dd85e1cfd572140b8
export CAP0_GUARD_FILE="$CAP0_MANIFEST_ROOT/frozen/guard_free_kv_tokens.txt"
export CAP0_GUARD_SHA=0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b
export CAP0_SURVIVAL_SHA=031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a
export FROZEN_NOOP="$CAP0_MANIFEST_ROOT/frozen/noop_v1.json"

"$BRIDGE_PY" tools/bridge_tp/freeze_phase9_cap0_noop.py \
  --bringup-root "$BRINGUP_ROOT" \
  --working-manifest "$EXECUTED_MANIFEST" \
  --out "$FROZEN_NOOP" \
  --expected-bringup-revision "$BRINGUP_REVISION" \
  --expected-working-sha256 "$ORIGIN_MANIFEST_SHA" \
  --expected-survival-sha256 "$CAP0_SURVIVAL_SHA" \
  --expected-guard-sha256 "$CAP0_GUARD_SHA" \
  --expected-guard 8448 \
  --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS"

export FROZEN_NOOP_PROVENANCE="$CAP0_MANIFEST_ROOT/frozen/noop_v1.provenance.json"
export FROZEN_NOOP_SHA="$(awk '{print $1}' "$CAP0_MANIFEST_ROOT/frozen/noop_v1.sha256")"
test "$(sha256sum "$FROZEN_NOOP" | awk '{print $1}')" = "$FROZEN_NOOP_SHA"

export NOOP_BATCH_ID="cap0-noop-formal-$(date -u +%Y%m%dT%H%M%SZ)"
export NOOP_BATCH_ROOT="$CAP0_RESULTS/noop_batches/$NOOP_BATCH_ID"

"$BRIDGE_PY" tools/bridge_tp/run_phase9_cap0_noop_formal.py \
  --validate-only \
  --model-path "$BRIDGE_MODEL" \
  --manifest "$FROZEN_NOOP" \
  --manifest-provenance "$FROZEN_NOOP_PROVENANCE" \
  --bringup-root "$BRINGUP_ROOT" \
  --survival-table "$CAP0_SURVIVAL_TABLE" \
  --guard-file "$CAP0_GUARD_FILE" \
  --out-root "$NOOP_BATCH_ROOT" \
  --expected-revision "$CAP0_CODE_REVISION" \
  --expected-manifest-sha256 "$FROZEN_NOOP_SHA" \
  --expected-survival-sha256 "$CAP0_SURVIVAL_SHA" \
  --expected-guard-sha256 "$CAP0_GUARD_SHA" \
  --expected-guard 8448 \
  --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS" \
  --repetitions 3

set -o pipefail
"$BRIDGE_PY" tools/bridge_tp/run_phase9_cap0_noop_formal.py \
  --model-path "$BRIDGE_MODEL" \
  --manifest "$FROZEN_NOOP" \
  --manifest-provenance "$FROZEN_NOOP_PROVENANCE" \
  --bringup-root "$BRINGUP_ROOT" \
  --survival-table "$CAP0_SURVIVAL_TABLE" \
  --guard-file "$CAP0_GUARD_FILE" \
  --out-root "$NOOP_BATCH_ROOT" \
  --expected-revision "$CAP0_CODE_REVISION" \
  --expected-manifest-sha256 "$FROZEN_NOOP_SHA" \
  --expected-survival-sha256 "$CAP0_SURVIVAL_SHA" \
  --expected-guard-sha256 "$CAP0_GUARD_SHA" \
  --expected-guard 8448 \
  --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS" \
  --repetitions 3 \
  2>&1 | tee "$CAP0_RESULTS/noop_batches/${NOOP_BATCH_ID}.console.txt"
echo "NOOP_FORMAL_RC=${PIPESTATUS[0]}"
echo "NOOP_BATCH_ROOT=$NOOP_BATCH_ROOT"
```

每轮必须独立重启五类进程、使用新目录，并满足 `active_capacity_decisions ==
blocked_stay_decisions > 0`、`START_SHADOW=0`、migration transition 为 0、Anchor 完整由
TP1 发出、76/76 background jobs 完成。三轮全部通过时，批次状态为 `FORMAL_COMPLETE`，
控制台打印 `NOOP_FORMAL_COMPLETE`；任一轮失败则保留现场并停止，不得覆盖后重跑。

P0 已于 2026-09-05 在 revision
`2861cfefd7f7fc552f647f0462715099b0ce7b40` 完成三轮正式重复；三轮均为 PASS，且
`START_SHADOW=0`、migration transition 为 0、76/76 background jobs 完成。该结果只关闭
No-op 门，不代替下面的 Rescue。

### 22.3 P1 Rescue reachability 自动 bring-up

P1 先只跑一次不计数 bring-up，不立即冻结 `rescue_v1`，也不运行正式重复。其目标不是比较
性能，而是证明同一个因果信号能从“因 TP4 当前 guard 不满足而保持 TP1”走到“TP4 自然
恢复后允许迁移”，并真实贯通数据面。

默认 working manifest 包含 48 个 TP4 target jobs 和 4 个 TP1 source-pressure jobs，共 52
个 background jobs。每个 target job 使用 7000 个显式 token IDs 加 1100 output tokens，
target 总 context demand 为 388800 tokens：高于 50% contention floor，但低于实测 TP4
571824-token KV 容量。这样 target burst 应在 signal active 时先令 `waiting > 4`，随后有限
作业自然完成并降至 `waiting <= 4`；这只是事前候选，不能把一次通过当作参数已冻结。

bring-up 必须同时证明：

- active capacity signal 下先出现至少一次由 TP4 guard 阻挡的 `STAY`；
- 后续仅出现一次满足当前 KV/waiting guard 的 `START_SHADOW`，触发路径为
  `CAPACITY_PILOT`；
- 状态严格经过 `SHADOW -> HANDOFF -> TAKEOVER`，且只有四个 rank 全部 ready 才 commit；
- 四份 sender/receiver receipt 的 migration/request ID、payload bytes、SHA-256 一致，四个
  receiver 均为 `OWNERSHIP_COMMITTED` 且 `exact_readback=true`；
- commit 后 source abort 已下发、target 成为唯一 owner，统一输出恰好 8000 tokens，source
  prefix 与 target suffix 都非空，index 连续且 JSONL/token IDs 完全一致；
- 52/52 background jobs 全部成功，且 anchor 的 source/target request ID 不与 background
  request 串号。

服务器拉取含 Rescue bring-up 的 revision 后，在一个前台终端执行。父进程会依次启动并拥有
TP4、TP1、stager、controller 和 background workload；结束或 `Ctrl+C` 时由父进程清理，
不要另开终端预启动服务，也不要使用 `nohup`：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

set +e
set +u
set +o pipefail

export CAP0_CODE_REVISION="$(git rev-parse HEAD)"
export CAP0_SURVIVAL_SHA=031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a
export CAP0_GUARD_FILE="$CAP0_MANIFEST_ROOT/frozen/guard_free_kv_tokens.txt"
export CAP0_GUARD_SHA=0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b

test "$(sha256sum "$CAP0_SURVIVAL_TABLE" | awk '{print $1}')" = "$CAP0_SURVIVAL_SHA"
test "$(sha256sum "$CAP0_GUARD_FILE" | awk '{print $1}')" = "$CAP0_GUARD_SHA"
test "$(cat "$CAP0_GUARD_FILE")" = 8448

export RESCUE_ID="cap0-rescue-bringup-$(date -u +%Y%m%dT%H%M%SZ)"
export RESCUE_MANIFEST="$CAP0_MANIFEST_ROOT/working/${RESCUE_ID}.json"
export RESCUE_ROOT="$CAP0_RESULTS/rescue_bringup/$RESCUE_ID"
mkdir -p "$CAP0_MANIFEST_ROOT/working" "$CAP0_RESULTS/rescue_bringup"

"$BRIDGE_PY" tools/bridge_tp/build_phase9_cap0_rescue_manifest.py \
  --out "$RESCUE_MANIFEST"
export RESCUE_MANIFEST_SHA="$(sha256sum "$RESCUE_MANIFEST" | awk '{print $1}')"

"$BRIDGE_PY" tools/bridge_tp/run_phase9_cap0_rescue.py \
  --validate-only \
  --model-path "$BRIDGE_MODEL" \
  --manifest "$RESCUE_MANIFEST" \
  --survival-table "$CAP0_SURVIVAL_TABLE" \
  --guard-file "$CAP0_GUARD_FILE" \
  --out-root "$RESCUE_ROOT" \
  --expected-revision "$CAP0_CODE_REVISION" \
  --expected-manifest-sha256 "$RESCUE_MANIFEST_SHA" \
  --expected-survival-sha256 "$CAP0_SURVIVAL_SHA" \
  --expected-guard-sha256 "$CAP0_GUARD_SHA" \
  --expected-guard 8448 \
  --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS"
export VALIDATE_RC=$?

if [ "$VALIDATE_RC" -eq 0 ]; then
  set -o pipefail
  "$BRIDGE_PY" tools/bridge_tp/run_phase9_cap0_rescue.py \
    --model-path "$BRIDGE_MODEL" \
    --manifest "$RESCUE_MANIFEST" \
    --survival-table "$CAP0_SURVIVAL_TABLE" \
    --guard-file "$CAP0_GUARD_FILE" \
    --out-root "$RESCUE_ROOT" \
    --expected-revision "$CAP0_CODE_REVISION" \
    --expected-manifest-sha256 "$RESCUE_MANIFEST_SHA" \
    --expected-survival-sha256 "$CAP0_SURVIVAL_SHA" \
    --expected-guard-sha256 "$CAP0_GUARD_SHA" \
    --expected-guard 8448 \
    --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS" \
    2>&1 | tee "$CAP0_RESULTS/rescue_bringup/${RESCUE_ID}.console.txt"
  export RESCUE_RC=${PIPESTATUS[0]}
else
  export RESCUE_RC=98
fi
set +o pipefail

echo "VALIDATE_RC=$VALIDATE_RC"
echo "RESCUE_RC=$RESCUE_RC"
echo "RESCUE_ROOT=$RESCUE_ROOT"
echo "RESCUE_MANIFEST=$RESCUE_MANIFEST"
test "$VALIDATE_RC" -eq 0
test "$RESCUE_RC" -eq 0
```

成功标志为 `RESCUE_BRINGUP_COMPLETE`，机器验收文件是
`$RESCUE_ROOT/provenance/rescue_acceptance.json`。失败时保留整个目录并停止；先分析
`capacity_pilot_decision` 的 waiting/KV 时间序列和五类日志，不能事后改 guard。只有一次
bring-up 完整通过并人工确认工件后，下一步才是把该 exact jobs manifest 冻结成
`rescue_v1.json`，再实现和运行至少三轮正式重复。

### 22.4 冻结 P1 Rescue workload 并运行正式重复

`cap0-rescue-bringup-20260905T102948Z` 已在 revision `2c2fe7f` 完成标准 bring-up：
52/52 background jobs 成功，170 次 guarded `STAY` 后仅一次 `START_SHADOW`，四 rank
readback、commit、source abort 和 8000-token unified stream 全部通过。其 working/expanded
manifest SHA-256 均为
`5199f9502e3ffdde889823ea798d399ce05159472064a414ca25e25f81d59a27`。

先拉取包含 Rescue freezer/formal runner 的明确 revision。以下命令冻结 bring-up 实际执行的
exact jobs；冻结器同时要求原 working manifest、运行目录 expanded manifest 和 provenance
三方哈希相同，不会依据当前代码重新生成 workload：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

set +e
set +u
set +o pipefail

export BRINGUP_ID=cap0-rescue-bringup-20260905T102948Z
export BRINGUP_REVISION=2c2fe7f981db511eb911a08b902c48aeedb74606
export BRINGUP_ROOT="$CAP0_RESULTS/rescue_bringup/$BRINGUP_ID"
export ORIGIN_RESCUE="$CAP0_MANIFEST_ROOT/working/${BRINGUP_ID}.json"
export ORIGIN_RESCUE_SHA=5199f9502e3ffdde889823ea798d399ce05159472064a414ca25e25f81d59a27
export CAP0_SURVIVAL_SHA=031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a
export CAP0_GUARD_FILE="$CAP0_MANIFEST_ROOT/frozen/guard_free_kv_tokens.txt"
export CAP0_GUARD_SHA=0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b
export FROZEN_RESCUE="$CAP0_MANIFEST_ROOT/frozen/rescue_v1.json"

test "$(sha256sum "$ORIGIN_RESCUE" | awk '{print $1}')" = "$ORIGIN_RESCUE_SHA"
test "$(sha256sum "$BRINGUP_ROOT/background/background_manifest.json" | awk '{print $1}')" = "$ORIGIN_RESCUE_SHA"

"$BRIDGE_PY" tools/bridge_tp/freeze_phase9_cap0_rescue.py \
  --bringup-root "$BRINGUP_ROOT" \
  --working-manifest "$ORIGIN_RESCUE" \
  --out "$FROZEN_RESCUE" \
  --expected-bringup-revision "$BRINGUP_REVISION" \
  --expected-working-sha256 "$ORIGIN_RESCUE_SHA" \
  --expected-survival-sha256 "$CAP0_SURVIVAL_SHA" \
  --expected-guard-sha256 "$CAP0_GUARD_SHA" \
  --expected-guard 8448 \
  --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS"
export FREEZE_RC=$?

export FROZEN_RESCUE_PROVENANCE="$CAP0_MANIFEST_ROOT/frozen/rescue_v1.provenance.json"
export FROZEN_RESCUE_SHA="$(awk '{print $1}' "$CAP0_MANIFEST_ROOT/frozen/rescue_v1.sha256")"

echo "FREEZE_RC=$FREEZE_RC"
echo "FROZEN_RESCUE_SHA=$FROZEN_RESCUE_SHA"
test "$FREEZE_RC" -eq 0
test "$(sha256sum "$FROZEN_RESCUE" | awk '{print $1}')" = "$FROZEN_RESCUE_SHA"
```

冻结成功后，在同一前台终端先做 formal contract 静态核验，再连续运行 r01--r03。每轮由父
runner 重新启动 TP4、TP1、stager、controller 和 background workload，使用独立目录；
任一轮失败会停止批次并保留现场：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

set +e
set +u
set +o pipefail

export CAP0_CODE_REVISION="$(git rev-parse HEAD)"
export BRINGUP_ID=cap0-rescue-bringup-20260905T102948Z
export BRINGUP_ROOT="$CAP0_RESULTS/rescue_bringup/$BRINGUP_ID"
export CAP0_SURVIVAL_SHA=031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a
export CAP0_GUARD_FILE="$CAP0_MANIFEST_ROOT/frozen/guard_free_kv_tokens.txt"
export CAP0_GUARD_SHA=0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b
export FROZEN_RESCUE="$CAP0_MANIFEST_ROOT/frozen/rescue_v1.json"
export FROZEN_RESCUE_PROVENANCE="$CAP0_MANIFEST_ROOT/frozen/rescue_v1.provenance.json"
export FROZEN_RESCUE_SHA="$(awk '{print $1}' "$CAP0_MANIFEST_ROOT/frozen/rescue_v1.sha256")"
export RESCUE_BATCH_ID="cap0-rescue-formal-$(date -u +%Y%m%dT%H%M%SZ)"
export RESCUE_BATCH_ROOT="$CAP0_RESULTS/rescue_batches/$RESCUE_BATCH_ID"
mkdir -p "$CAP0_RESULTS/rescue_batches"

"$BRIDGE_PY" tools/bridge_tp/run_phase9_cap0_rescue_formal.py \
  --validate-only \
  --model-path "$BRIDGE_MODEL" \
  --manifest "$FROZEN_RESCUE" \
  --manifest-provenance "$FROZEN_RESCUE_PROVENANCE" \
  --bringup-root "$BRINGUP_ROOT" \
  --survival-table "$CAP0_SURVIVAL_TABLE" \
  --guard-file "$CAP0_GUARD_FILE" \
  --out-root "$RESCUE_BATCH_ROOT" \
  --expected-revision "$CAP0_CODE_REVISION" \
  --expected-manifest-sha256 "$FROZEN_RESCUE_SHA" \
  --expected-survival-sha256 "$CAP0_SURVIVAL_SHA" \
  --expected-guard-sha256 "$CAP0_GUARD_SHA" \
  --expected-guard 8448 \
  --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS" \
  --repetitions 3
export VALIDATE_RC=$?

if [ "$VALIDATE_RC" -eq 0 ]; then
  set -o pipefail
  "$BRIDGE_PY" tools/bridge_tp/run_phase9_cap0_rescue_formal.py \
    --model-path "$BRIDGE_MODEL" \
    --manifest "$FROZEN_RESCUE" \
    --manifest-provenance "$FROZEN_RESCUE_PROVENANCE" \
    --bringup-root "$BRINGUP_ROOT" \
    --survival-table "$CAP0_SURVIVAL_TABLE" \
    --guard-file "$CAP0_GUARD_FILE" \
    --out-root "$RESCUE_BATCH_ROOT" \
    --expected-revision "$CAP0_CODE_REVISION" \
    --expected-manifest-sha256 "$FROZEN_RESCUE_SHA" \
    --expected-survival-sha256 "$CAP0_SURVIVAL_SHA" \
    --expected-guard-sha256 "$CAP0_GUARD_SHA" \
    --expected-guard 8448 \
    --tp1-blocks "$TP1_BLOCKS" --tp4-blocks "$TP4_BLOCKS" \
    --repetitions 3 \
    2>&1 | tee "$CAP0_RESULTS/rescue_batches/${RESCUE_BATCH_ID}.console.txt"
  export RESCUE_FORMAL_RC=${PIPESTATUS[0]}
  set +o pipefail
else
  export RESCUE_FORMAL_RC=98
fi

echo "VALIDATE_RC=$VALIDATE_RC"
echo "RESCUE_FORMAL_RC=$RESCUE_FORMAL_RC"
echo "RESCUE_BATCH_ROOT=$RESCUE_BATCH_ROOT"
test "$VALIDATE_RC" -eq 0
test "$RESCUE_FORMAL_RC" -eq 0
```

成功标志为 `RESCUE_FORMAL_COMPLETE`，且 `batch_status.json` 必须为
`FORMAL_COMPLETE`、`formal_acceptance.json` 必须为 3/3 PASS。正式轮仍只证明 CAP-0
Rescue reachability 的重复工程门，不把 13 秒级 handoff stall 或其他延迟写成正式 E 性能
结论。

三者共同设置：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env

export CAP0_CAPACITY_ENABLED=1
export CAP0_GUARD="$(cat "$CAP0_MANIFEST_ROOT/frozen/guard_free_kv_tokens.txt")"
```

bring-up 时只通过第 19 节 workload 参数形成条件：

- No-op：增加/延长 target jobs，确保 source signal active 时 TP4 持续超过
  `max_target_kv_usage_frac=0.85` 或 `max_target_waiting=4`，直到 anchor 完成；
- Rescue：target 初始忙，随后自然结束；TP4 变得 admissible 时 source signal 仍 active；
- Safe abandon：让 source burst 在历史复制完成前自然结束，使 signal `CLEAR`；必要时调整
  workload 时序或长度，不改 frozen guard，不注入 controller future information。

如果只靠当前 load generator 无法稳定形成某场景，停止并记录 `NO-GO`，不要用 sleep、手工
取消请求或观察结果后改 guard 把场景“做出来”。这意味着需要补 workload orchestrator，
而不是 CAP-0 已通过。

## 23. 单轮自动验收命令

运行下面的只读 checker，`EXPECTED` 取 `calibration`、`noop`、`rescue` 或 `abandon`：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

export EXPECTED=替换为场景名
"$BRIDGE_PY" - "$CAP0_DIR" "$EXPECTED" <<'PY'
import json, sys
from pathlib import Path

run, expected = Path(sys.argv[1]), sys.argv[2]
rows = [json.loads(x) for x in (run/'phase9_audit.jsonl').read_text().splitlines()]
end = next(x for x in reversed(rows) if x.get('kind') == 'run_end')
transitions = [x.get('to') for x in rows if x.get('kind') == 'transition']
cap = [x for x in rows if x.get('kind') == 'capacity_pilot_decision']

if expected == 'calibration':
    assert end['final_state'] == 'COMPLETED_ON_TP1'
    assert 'SHADOW' not in transitions
elif expected == 'noop':
    assert any(x.get('action') == 'STAY' and x.get('signal', {}).get('active') for x in cap)
    assert 'SHADOW' not in transitions
    assert not (run/'staging_manifest.json').exists()
elif expected == 'rescue':
    assert any(x.get('action') == 'START_SHADOW' for x in cap)
    assert end.get('trigger_path') == 'CAPACITY_PILOT'
    assert transitions[-1] == 'TAKEOVER'
    state = json.loads((run/'takeover_state.json').read_text())
    assert state['state'] == 'COMMITTED'
    assert state['source_abort_dispatched'] is True
    receipts = list((run/'receiver_receipts').glob('*/tp_rank_*.json'))
    assert len(receipts) == 4
    assert all(json.loads(x.read_text()).get('exact_readback') is True for x in receipts)
elif expected == 'abandon':
    assert 'SHADOW' in transitions and transitions[-1] == 'CANCELLED'
    assert any(x.get('kind') == 'abandon' and 'headroom recovered' in x.get('reason','') for x in rows)
    cleanup = json.loads((run/'cleanup_request.json').read_text())
    state = json.loads((run/'takeover_state.json').read_text())
    assert cleanup['abort_source'] is False
    assert state['state'] == 'CANCELLED'
    assert state['source_abort_dispatched'] is False
    assert state['source_continues_on_tp1'] is True
else:
    raise SystemExit(f'unknown EXPECTED={expected}')
print('PASS', expected, run, end['final_state'])
PY
```

checker 只做必要条件检查，不能代替人工检查 source/target/stager 日志、request-ID 串号、
client stream 完整性和 background summary。

## 24. 每轮封存和 SHA-256

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
source /root/autodl-tmp/bridgetp/phase9_cap0_common.env
source /root/autodl-tmp/bridgetp/phase9_cap0_active.env

git -C "$BRIDGE_REPO" rev-parse HEAD > "$CAP0_PROV_DIR/git_revision_after.txt"
git -C "$BRIDGE_REPO" status --short --branch > "$CAP0_PROV_DIR/git_status_after.txt"
cp "$CAP0_CONFIG" "$CAP0_PROV_DIR/controller_config.frozen.json"
cp "$CAP0_BG_MANIFEST" "$CAP0_PROV_DIR/background_manifest.frozen.json"

find "$CAP0_DIR" "$CAP0_BG_DIR" "$CAP0_PROV_DIR" \
  -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum \
  > "$CAP0_PROV_DIR/SHA256SUMS"
sha256sum -c "$CAP0_PROV_DIR/SHA256SUMS"
```

失败轮同样封存。不得在生成 `SHA256SUMS` 后编辑被覆盖文件；如需增加分析文件，生成新的
`SHA256SUMS.v2` 并保留旧文件。

## 25. 最终执行顺序和停止点

```text
1. 等远端分支可见；独立 worktree 部署，不触碰服务器旧 checkout 的 dirty changes。
2. S0/E0 盘点 + CPU 回归。
3. 用相同 server args 测真实 TP1/TP4 KV blocks。
4. S1 单独运行 052 post-fix migrate + controls；通过后封存，绝不并入旧49条。
5. C0 smoke，验证并冻结 calibration_v2 workload contract。
6. C1 no-migration r01/r02/r03；按事前规则计算并冻结 guard。
7. P0 No-op bring-up，冻结 noop_v1，再跑 r01/r02/r03。
8. P1 Rescue bring-up，冻结 rescue_v1，再跑 r01/r02/r03。
9. P2 Safe abandon bring-up，冻结 abandon_v1，再跑 r01/r02/r03。
10. 逐轮 checker、人工审计、SHA-256 和跨轮 go/no-go。
11. 到此停止；不启动 formal E。
```

本轮做完应证明：M0 full-restore 的 052 尾块修复成立；单 anchor 在 contention 下能被因果
KV headroom 信号安全地“不迁移、迁移完成、迁移前安全放弃”。本轮不能证明：哪种历史 KV
复制顺序最好、是否应有 Bridge remote attention、正式 capacity controller 已完成，或系统
goodput/SLO 优于 baseline。
