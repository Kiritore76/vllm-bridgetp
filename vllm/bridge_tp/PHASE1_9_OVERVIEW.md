# BridgeTP Phase 1–9 总览与证明边界

## 1. 文档目的

本文统一说明BridgeTP在线原型从Phase 1到Phase 8完成了什么、每一阶段新增了哪一层
能力、对应证据能够支持什么结论，以及Phase 9在线controller应当完成什么。

Phase 1–3最初作为一个连续开发包交付，目标是定位vLLM真实KV cache并完成请求级导出。
本文为了说明工程演进，将其中的模块隔离、导出器和真实decode hook分别列为Phase 1、
Phase 2和Phase 3。Phase 4以后按当前仓库正式编号描述。

所有已回收的在线验证均来自同一类A100 PCIe平台。不同GPU平台的数据不能直接合并，
带宽阈值、迁移时间和尾延迟结论也不能跨平台外推。

---

## 2. Phase 1：BridgeTP模块与安全开关

### 做了什么

- 在vLLM源码中建立独立的`vllm/bridge_tp/`目录；
- 通过环境变量显式启用BridgeTP调试逻辑；
- 默认关闭，不改变普通vLLM请求路径；
- 将实验代码与`worker.py`、`scheduler.py`等上游核心文件尽量隔离。

### 证明了什么

BridgeTP可以作为可选扩展接入固定版本vLLM源码，不需要直接修改安装环境中的
`site-packages/vllm`，也不需要把全部逻辑散落在上游worker和scheduler中。

### 没有证明什么

本阶段没有导出KV tensor，没有做TP转换，也没有迁移请求。

---

## 3. Phase 2：请求级KV导出器

### 做了什么

- 实现请求级KV cache exporter；
- 记录模型层名、tensor shape、dtype、block axis和物理block ID；
- 记录请求的prompt token、output token和computed/pending边界；
- 支持metadata-only与真实tensor dump，避免一开始就复制大量数据；
- 对输出目录、最大导出字节数和单次触发进行限制。

### 证明了什么

BridgeTP能够把某一个真实请求占用的KV block与该请求的token历史对应起来，而不是
导出整个GPU KV pool或依赖估算出来的synthetic KV大小。

### 没有证明什么

本阶段的结果仍是源端磁盘文件，没有目标TP布局、在线传输或续写验证。

---

## 4. Phase 3：真实decode hook与token/KV边界

### 做了什么

- 在vLLM V1 GPU model runner的真实decode iteration边界调用exporter；
- 要求关闭async scheduling、prefix cache、speculative decoding和hybrid KV manager，
  固定第一版证据边界；
- 区分`num_computed_tokens`和已经采样但尚未写入KV的pending token；
- 在TP1真实Qwen2.5-14B-Instruct请求上导出48层BF16 KV。

### 关键实测

```text
单层请求KV布局：[9, 2, 16, 8, 128]
block axis：0
token axis：2
KV-head axis：3
已写入KV：137 token
已知但pending：1 token
```

### 证明了什么

BridgeTP找到了真实执行路径和正确的token/KV一致性边界，能够回答“哪些token已经由
当前KV表示，哪个最新token仍需目标端计算”。这是后续恢复正确性的必要前提。

### 没有证明什么

本阶段仍是TP1源端dump，不包含TP1→TP4布局转换或目标恢复。

---

## 5. Phase 4：TP1→TP4离线KV reshard

### 做了什么

对已验证的TP1请求KV沿KV-head轴进行无损拆分：

```text
TP1：8个KV heads
  ├── TP4 rank 0：heads [0:2]
  ├── TP4 rank 1：heads [2:4]
  ├── TP4 rank 2：heads [4:6]
  └── TP4 rank 3：heads [6:8]
```

每个rank生成独立分片；验证器按rank顺序回拼所有分片并与源tensor逐元素比较。

### 关键实测

- 48层全部完成拆分；
- 14,155,776个元素回拼后逐元素一致；
- head axis和源KV-head数量作为显式schema，不当成跨模型常数。

### 证明了什么

对于当前Qwen2.5-14B布局，TP1 KV可以无损转换成TP4每rank所需的KV-head分片。

### 没有证明什么

这是离线CPU布局转换，没有经过目标scheduler分配，也没有写入TP4 GPU KV pool。

正式记录见`PHASE4_VALIDATION.md`。

---

## 6. Phase 5：TP4 scheduler分配、文件式恢复与续写

### 做了什么

- 使用vLLM KV Connector生命周期接入TP4 target；
- 由TP4 scheduler正规分配目标逻辑block table；
- 每个TP4 worker只加载自己的rank分片；
- 写入各rank GPU本地KV pool后立即精确读回；
- 以完整token history作为目标prompt，在pending边界继续生成；
- 使用相同模型、相同prompt和greedy sampling的干净TP1作为control。

### 关键实测

- 四rank GPU exact readback全部通过；
- target block table由目标allocator产生，不要求等于源物理block ID；
- 恢复后的TP4继续生成32个token，与干净TP1逐token一致。

### 证明了什么

Phase 4分片不只在离线文件中数值正确，还能写入真实TP4 worker的scheduler-owned KV
block，并从正确pending边界继续生成相同的greedy token。

### 没有证明什么

KV仍通过共享文件传递；没有在线网络传输、source cancellation或ownership takeover。
Phase 5归档中两次成功调用的debug timing不能混成同一轮性能bundle。

正式记录见`PHASE5_VALIDATION.md`。

---

## 7. Phase 6：五卡实时TCP传输

### 做了什么

```text
GPU0：live TP1 source
  │
  ├── iteration-boundary request KV snapshot
  ├── TP1 KV-head reshard为四份
  └── 四路framed TCP + frame/full-payload SHA256
          ↓
GPU1–4：TP4 scheduler分配 → 各rank接收 → GPU注入/读回 → shadow continuation
```

- 使用唯一run目录和migration ID；
- 每个TP4 worker只通过TCP接收自己的rank payload；
- 支持pinned CPU staging、chunking和速率限制；
- target在全部校验通过后返回`READY`。

### 关键实测

- 同节点5×NVIDIA A100-PCIE-40GB；
- 快照边界：128 output、147 computed、1 pending；
- 48层BF16、10 blocks；
- raw tensor总量31,457,280 bytes；
- 四rank sender/receiver的字节数和SHA256一致；
- 四rank GPU exact readback通过；
- live TP1与恢复TP4的后续32个greedy token逐个一致。

### 证明了什么

真实TP1请求KV可以在decode期间完成请求级快照、四rank在线字节传输、TP4恢复和正确
shadow continuation，KV tensor不再依赖共享分片文件。

### 没有证明什么

source仍然是唯一owner，TP4只是shadow continuation；没有abort source、commit、
rollback或客户端响应切换。

正式记录见`PHASE6_VALIDATION.md`。

---

## 8. Phase 7：应用级原子接管与回滚

### 做了什么

引入正常进程条件下的两阶段ownership handoff：

```text
TP4四rank exact readback
        ↓
四rank停在TARGET_READY
        ↓
controller验证sender/receiver
        ↓
PREPARING → COMMITTING
        ↓
source engine_client.abort()
        ↓
COMMITTED
        ↓
TP4越过barrier并接管剩余generation budget
```

同时实现commit前rollback：写入`ROLLED_BACK`，不abort source，TP1继续完成请求。

### 关键实测

Commit run：

- 四rank最终全部`OWNERSHIP_COMMITTED`；
- source `finish_reason=abort`；
- 快照前128 token + TP4后64 token与干净TP1的192 token逐个一致。

Rollback run：

- takeover state为`ROLLED_BACK`；
- `source_abort_dispatched=false`；
- 四rank全部`ROLLED_BACK`；
- TP1继续完成192 token并与control一致。

### 证明了什么

在API、source和target进程正常运行的条件下，BridgeTP能保持单一owner顺序，完成
`target ready → source abort → target continue`，并能在commit前安全回滚。

### 没有证明什么

这不是跨节点共识协议；不证明API在`COMMITTING`期间崩溃后的自动恢复，也不迁移任意
sampling RNG状态。

正式记录见`PHASE7_VALIDATION.md`。

---

## 9. Phase 8：后台old-KV、新KV双写与取消清理

### 做了什么

Phase 8不在更晚边界重新复制完整KV，而是把历史搬运和decode并行：

```text
第128个输出token：
TP1 ──完整old-KV──> CPU stager（四rank后台传输）
 │
 └──继续decode，并把每轮新增computed KV双写到CPU stager

第160个输出token：
stager验证四rank delta无缺口
  → 组装最终KV
  → TP4注入/精确读回
  → 复用Phase 7 commit barrier
```

另有pre-cutover cancellation路径：controller在old-KV和部分delta到达后abort source，
source delta mirror和CPU stager分别排空队列、释放四rank buffer并写`CLEANED` receipt。

### Dual-write commit关键实测

- 初始边界：147 computed；
- cutover边界：160 output、179 computed、1 pending；
- 32个new-KV token / 32 batches；
- 四rank delta均连续覆盖`[147,179)`；
- old-KV和new-KV传输时间重叠；
- 四rank staged delivery和GPU exact readback全部通过；
- takeover state为`COMMITTED`；
- source在191 output时被abort，cutover后多算的31 token被丢弃；
- cutover前160 token + TP4后128 token与干净TP1的288 token逐个一致。

原controller最终汇总错误地从普通completion入口读取规范化后的streaming TP4响应，
所以原始运行未生成最终result。原始293个归档文件SHA256全部通过；本地工具从不可变
raw JSON和receipt离线复算得到`PASS`。派生结果显式标记
`evidence_origin=offline_reconstruction`，没有伪造无法恢复的controller-local timing。

### Cancellation关键实测

- takeover state为`CANCELLED`；
- source `finish_reason=abort`；
- source mirror与CPU stager均`CLEANED`；
- 排空2个delta token并释放四rank buffer；
- 没有target request，没有takeover commit；
- 原始inspector直接得到`PASS`。

### 证明了什么

在同节点单请求greedy条件下，BridgeTP能让历史KV搬运与TP1 decode重叠，用token粒度
new-KV mirror保持stager状态追上source，并在cutover后正确接管；迁移在commit前取消时
也能清理source和stager资源。

### 没有证明什么

- CPU staging不是GPU P2P/NVLink或跨节点RDMA；
- 没有验证自然EOS的全部竞争时序；
- 没有统一的客户端响应流；
- 没有多请求并发、正式handoff stall分布或在线策略决策。

正式记录见`PHASE8_VALIDATION.md`。

---

## 10. Phase 1–8最终证明了什么

### 10.1 核心结论

Phase 1–8形成了一条连续、逐层收紧证据边界的正确性链：

```text
真实TP1请求KV可定位和导出
        ↓
TP1 KV可无损转换成TP4四rank布局
        ↓
TP4 scheduler-owned GPU blocks可精确恢复
        ↓
恢复后的TP4可从computed/pending边界正确续写
        ↓
KV可在live decode期间通过四路TCP传输
        ↓
TP4 ready后可原子abort TP1并接管ownership
        ↓
old-KV后台搬运期间可持续双写new-KV
        ↓
commit前取消可释放source/stager资源
```

因此，在以下严格限定范围内，可以得出结论：

> 对同节点A100 PCIe上的Qwen2.5-14B-Instruct单个greedy请求，BridgeTP已经实现并
> 验证了真实request-scoped KV从TP1到TP4的在线转换、传输、恢复、token连续续写、
> 正常进程条件下的应用级原子ownership接管、pre-commit rollback，以及后台old-KV
> 搬运期间的new-KV增量同步和显式取消清理。

这已经不再只是synthetic traffic或离线KV文件实验，而是一个真实vLLM执行路径上的
窄范围在线Bridge正确性原型。

### 10.2 前八阶段没有证明的结论

仍然不能声称：

- 已支持temperature、top-p等任意sampling RNG状态迁移；
- 已支持多请求并发和所有scheduler交错；
- 已实现生产级统一客户端API和无缝stream response代理；
- 已实现进程/节点崩溃期间的分布式共识与自动恢复；
- 已实现GPU P2P、NVLink、NIXL、RDMA或跨节点高速传输；
- 已证明controller能正确选择迁移对象和迁移时机；
- 已测得正式handoff stall、TPOT spike、E2E收益或系统吞吐收益；
- A100 PCIe上的0.4–0.7 GiB/s、128或160 token边界是跨平台常数；
- 已实现TP pool在线无损重配置。

前八阶段回答的是“能不能正确迁移”，Phase 9才回答“什么时候迁、迁谁、以多快速度
迁，以及迁移后是否真的改善线上指标”。

---

## 11. Phase 9：在线fast/slow controller

## 11.1 目标

Phase 9在Phase 8迁移机制之上加入闭环策略和客户端数据面。它不重新实现KV搬运，
而是调用已经通过验证的precopy、dual-write、commit、rollback和cancel动作。

Phase 7/8已有的是机制控制面：收到明确动作后安全执行。Phase 9要增加的是策略控制器：

```text
请求进度 + 长请求风险 + TP1/TP4性能模型
       + TP4容量/排队/KV风险 + 迁移干扰
                         ↓
                  Phase 9 controller
                 /         |          \
             保持TP1    启动/限速迁移    cancel/commit
                             ↓
                    Phase 8 migration engine
                             ↓
                    Phase 7 atomic takeover
```

## 11.2 Controller必须包含的组件

### A. Telemetry

采集并统一时间戳：

- 请求prompt长度、已生成token、decode速率和存活时间；
- source KV大小、old-KV剩余字节、delta backlog；
- TP1/TP4 running、waiting、KV cache usage和preemption；
- 当前迁移速率、目标rank状态和transfer error；
- source最后一个可见token、TARGET_READY、commit、target第一个可见token。

`*_created`指标是counter创建时间戳，不能当成风险计数；preemption必须使用total
counter或可靠的增量。

### B. Request risk / remaining-work predictor

使用E1已经修正的survival-conditioned risk，而不是只看固定token阈值。输出至少包括：

- 请求在当前checkpoint后成为长请求的条件概率；
- 预计剩余output token；
- 预测不确定性；
- 当前是否仍位于“信息足够但加速机会尚未耗尽”的窗口。

640–768附近只是在既有trace上的观察，不得硬编码成跨workload阈值。

### C. Fast per-request controller

对每个eligible请求计算：

```text
expected_benefit
  = predicted_remaining_time_on_tp1
  - predicted_remaining_time_on_tp4

expected_cost
  = old_kv_copy_time
  + delta_catchup_time
  + target_restore_and_commit_time
  + interference_penalty
  + duplicated_post_cutover_compute
  + safety_margin
```

只有在目标TP4已保留容量、`expected_benefit > expected_cost`且风险约束满足时，才允许
从`STAY_TP1`进入`PRECOPY`。在迁移期间持续重新评估；收益消失、target风险过高、EOS
或错误发生时执行cancel/rollback，而不是强行commit。

### D. Migration state machine

建议状态：

```text
ADMITTED_TP1
  → PRECOPY_OLD_KV
  → DUAL_WRITE
  → TARGET_READY
  → COMMITTING
  → TP4_OWNER
```

失败/停止状态：

```text
NOT_ELIGIBLE
CANCELLED
ROLLED_BACK
FAILED
COMPLETED_ON_TP1
```

状态转换必须幂等、带migration ID并落盘审计。controller不得绕过Phase 7的四rank
exact-readback和单一owner条件。

### E. Migration rate controller

根据target实时风险选择old-KV速率：

```text
target风险低且有收益     → 提高速率
target tail开始恶化       → 降速
delta backlog接近上限     → 在安全范围内加速或提前cutover
target容量/风险不允许迁入 → 暂停或cancel
```

P2在当前A100 PCIe上观察到0.4–0.7 GiB/s相对平滑、1.37–2.16 GiB/s干扰明显，但
controller必须把这些值作为可校准初值，而不是硬编码的系统常数。

### F. Unified response proxy

真实客户端只能看到一个external request ID和一条连续响应流：

```text
commit前：转发TP1可见token
commit时：冻结已确认cutover边界，丢弃source post-cutover重复结果
commit后：从正确offset开始转发TP4 token
```

必须保证无token缺口、无重复、顺序一致，且客户端不会看到内部source/target两个请求。
Phase 8只做了离线token拼接；统一response proxy是Phase 9成为完整在线系统的必要条件。

### G. Slow TP-pool controller

slow controller观察持续风险而不是单次burst，使用EWMA、迟滞和最小保持时间决定是否
让GPU1–4保持warm TP4，或在更长时间尺度重新配置TP1 pool/TP4 pool。

E4可作为初始参数来源：

```text
EWMA alpha：0.08
slow high：0.65
slow low：0.45
persistent windows：150
recovery windows：60
minimum hold：120 s
```

五卡环境中`4×TP1 ↔ 1×TP4`会涉及停止server、模型装载和warm-up，不能当作瞬时动作。
Phase 9必须显式记录reconfiguration latency；如果第一版只保持warm TP4，则应标记为
fast-controller MVP，不能声称已经实现物理TP pool动态重配置。

## 11.3 推荐代码结构

建议在同一个Phase 9分支内按职责拆分，而不是把策略塞进model runner：

```text
vllm/bridge_tp/controller/
├── telemetry.py
├── predictor.py
├── policy.py
├── state_machine.py
├── action_adapter.py
├── response_proxy.py
└── audit.py

tools/bridge_tp/
├── run_phase9_controller.py
├── inspect_phase9_run.py
└── summarize_phase9_runs.py
```

`action_adapter.py`只调用Phase 7/8已有API；KV tensor操作继续留在migration engine中。

## 11.4 推荐开发顺序

在统一分支`bridgetp/d3-phase9-online-controller`中按以下顺序推进：

1. 定义controller配置、event schema和状态机；
2. 接入source/target实时telemetry；
3. 实现只支持greedy请求的fast policy；
4. 将policy动作接到Phase 8 precopy/cancel和Phase 7 commit/rollback；
5. 实现统一客户端response proxy；
6. 增加迁移速率动态调节；
7. 接入slow persistent-risk policy；
8. 运行单请求正确性、故障路径和多请求性能实验；
9. 汇总同一GPU平台上的多轮统计。

第一版继续限制greedy、单模型和显式eligible请求。不要在同一提交中同时扩展任意sampling、
跨节点传输、crash consensus和物理pool重配置。

## 11.5 Phase 9必须记录的指标

### 正确性

- 客户端token无缺口、无重复、顺序正确；
- 迁移输出与相同greedy control逐token一致；
- 任意时刻只有一个owner向客户端产生可见token；
- cancel/rollback后source继续或终止语义正确；
- 所有rank receipt、状态转换和migration ID一致。

### Handoff

```text
handoff_stall
  = target第一个客户端可见token时间
  - source最后一个客户端可见token时间
```

还应记录：

- decision latency；
- precopy duration；
- delta catch-up duration/backlog；
- TARGET_READY→COMMITTED；
- source post-cutover discarded token数；
- target restore/readback时间。

### 系统性能

- migrated与non-migrated请求的TTFT、TPOT、E2E；
- TP1/TP4 request throughput和output throughput；
- migration期间同机TP4请求的P95/P99 TPOT与最大iteration spike；
- copy rate、总迁移字节、完成时间；
- target KV usage、waiting和preemption增量；
- controller触发率、commit率、cancel率、rollback率和误迁移率。

### Slow controller

- transient burst是否被正确过滤；
- persistent shift触发时间；
- topology hold time和reconfiguration latency；
- resize前后capacity、queue和tail latency变化。

## 11.6 Phase 9通过条件

Phase 9不能仅凭“controller发出了commit”判为通过。至少要求：

1. **正确性门槛**：统一客户端流无缺口/重复，greedy control逐token一致；
2. **安全门槛**：四rank未ready时禁止commit，cancel/rollback清理通过；
3. **策略门槛**：每次决策都有输入、收益、成本和最终动作审计记录；
4. **Fast controller门槛**：对选定workload表现出可解释的迁移/不迁移分界；
5. **Slow controller门槛**：transient不resize，persistent在迟滞条件满足后触发；
6. **性能门槛**：在同一GPU平台、相同workload、至少三轮下报告handoff stall、TPOT
   spike、E2E和throughput，不能只报告一次成功请求；
7. **证据边界**：如仍只支持warm TP4、greedy或单节点，必须在结论中明确说明。

Phase 9的目标不是保证所有请求都迁移后变快，而是证明controller能够在收益大于成本且
目标安全时迁移，在收益不足或风险过高时不迁移/取消，并以可重复数据说明这一选择优于
无策略的固定阈值迁移。

---

## 12. 当前项目状态

```text
Phase 1–8：窄范围在线迁移机制与正确性链已完成
Phase 9：待开发在线fast/slow controller、统一响应代理和正式性能实验
```

Phase 8结果的正式说明见`PHASE8_VALIDATION.md`。开始Phase 9前不需要重新设计或重跑
Phase 1–7；Phase 8也不需要推倒重来。后续Phase 9性能运行必须使用修复后的controller
并重新保存完整时间戳，不能从Phase 8离线重建文件推算handoff timing。
