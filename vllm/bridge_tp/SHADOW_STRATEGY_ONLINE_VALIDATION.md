# Shadow策略在线vLLM对照实验

## 1. 目标

本实验在真实TP1/TP4 vLLM服务上对照两种Shadow传输时序：

- `S_NEW`：Shadow期间只发送每轮新增KV；到固定Bridge边界后，才启动历史KV传输；
- `S_NEW_OLD`：Shadow开始时立即传历史KV，同时继续优先镜像新增KV。

两种策略使用相同请求、固定trigger/cutover、相同目标负载和相同TCP限速。每一轮都重新启动
TP1、TP4和stager，交替策略顺序，避免上一次迁移状态或固定执行顺序污染结果。

## 2. 真实执行范围

本实验执行：

1. TP1真实paged KV定位、D2H和TP1到TP4四rank拆分；
2. Shadow期间逐token新增KV传输；
3. 按策略在Shadow或Bridge启动历史KV传输；
4. CPU stager组装、TP4 scheduler-owned block恢复和四rank精确读回；
5. Phase 7原子takeover和Phase 9统一响应流；
6. TP4真实前台请求在`PRE_SHADOW/SHADOW/BRIDGE/POST_COMMIT`窗口中的TPOT。

当前在线decode仍要等完整历史KV恢复后才让TP4续写。本实验没有在vLLM model runner中执行
远端attention，因此不能单独宣称已经得到“带在线远端attention的端到端净收益”。它回答的
是更窄但必要的问题：真实在线系统中，历史预拷贝是否缩短Bridge等待/客户端handoff stall，
以及它给TP4真实请求造成多少干扰。

## 3. 通过条件

每个策略必须同时满足：

- 固定边界触发`SHADOW -> HANDOFF -> TAKEOVER`；
- `S_NEW`历史传输不得在Bridge前开始；
- `S_NEW_OLD`历史传输必须随Shadow开始；
- 四rank sender/receiver一致且GPU exact readback全部通过；
- takeover为`COMMITTED`，统一响应token连续、无缺口、无重复；
- TP4背景任务全部完成；
- `PRE_SHADOW`、`SHADOW`、`BRIDGE`每个窗口达到最小TPOT样本数。

## 4. 主要指标

- `shadow_duration_ms`；
- `bridge_to_commit_ms`；
- `handoff_stall_ms`；
- 四个阶段的TP4 `tpot_p50_ms/p95_ms/p99_ms`；
- source/target可见token数；
- 历史传输实际启动时间和阶段；
- 四rank exact readback与统一响应连续性。

决策时主要比较配对轮次中的：

```text
S_NEW bridge_to_commit - S_NEW_OLD bridge_to_commit
S_NEW handoff_stall    - S_NEW_OLD handoff_stall
S_NEW_OLD Shadow TP4 tail inflation - S_NEW Shadow TP4 tail inflation
```

如果前两项稳定为正、第三项较小，才支持“高提交概率时提前传历史KV”。若请求经常取消，
还必须把取消时的无效历史字节加入后续期望收益模型。

## 5. 执行顺序

先运行一对smoke。只有smoke的两个策略均PASS，才冻结同一target-only manifest并运行至少
三对正式实验。正式轮次自动交替策略顺序。

默认在线负载为8个TP4请求，每个请求4096 prompt token和2048 output token。runner会等待
至少两个TP4任务已经产生首token，再启动迁移anchor，从而保证迁移窗口落在真实decode负载
中，而不是落在TP4 server启动或prefill之前。
