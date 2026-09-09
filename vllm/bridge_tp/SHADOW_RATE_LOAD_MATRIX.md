# Shadow固定速率与TP4负载矩阵

## 1. 要证明什么

上一轮在线实验已经证明真实KV历史预拷贝可以运行，但Bridge耗时的轮间噪声较大，且原来的
速率控制器因TP4原生P99遥测为零而持续升到上限。本矩阵把传输速率和TP4负载改成显式实验
自变量，回答：

1. 不同TP4负载下，历史预拷贝对TP4请求TPOT的影响如何随速率变化；
2. 提高速率能否稳定减少进入Bridge后尚未完成的工作和handoff stall；
3. 每个负载下可接受的最大Shadow低优先级速率是多少；
4. `S_NEW_OLD`相对`S_NEW`的收益是否在相同速率、相同负载的配对中保持为正。

本实验仍使用真实vLLM KV导出、TCP传输、CPU stager、TP4恢复、精确读回和takeover，但不
执行在线远端attention。因此它校准的是Shadow拷贝速率，不单独证明完整Bridge净收益。

## 2. 固定速率如何生效

在线runner的`--fixed-rate-gib-s`同时固定控制器的最小、最大、初始和硬上限，并设置源端
启动默认值。发送线程每个chunk重新读取运行时控制值，再按四个TP4 rank均分聚合带宽。
验收会检查audit中的每个速率记录均等于指定值。`0`沿用传输协议的unlimited约定。

## 3. 默认矩阵

Smoke只检查边界组合：

- TP4负载：2和16个并发请求；
- 聚合限速：0.4和1.2 GiB/s；
- 每格1对策略，共8次在线运行。

Formal覆盖：

- TP4负载：低2、中8、高24个并发请求；
- 聚合限速：0.2、0.4、0.8、1.2 GiB/s和unlimited；
- 每格4对交错顺序策略，共120次在线运行。

每个TP4请求默认使用4096 prompt token和2048 output token。每格会等待所有目标请求都已
产生首token，确保迁移发生在decode负载而非server启动或prefill阶段。

## 4. 输出与恢复

每个cell保留原在线实验的`acceptance.json`、`measurements.csv`和
`paired_comparisons.csv`。矩阵根目录额外生成：

- `contract.json`：冻结的负载、速率、重复次数和代码版本；
- `batch_status.json`：逐cell进度；
- `matrix_measurements.csv`：全部单次运行；
- `matrix_paired_comparisons.csv`：全部配对差值；
- `acceptance.json`：矩阵级完整性验收。

长时间正式实验可用`--resume`继续。工具只跳过已有且状态为PASS的cell；对不完整或失败的
cell会停止，避免把残缺结果误当成成功结果。

## 5. 决策规则

先用`S_NEW`同负载基线计算TP4 TPOT增量，再比较`S_NEW_OLD`。候选Shadow速率必须同时满足：

- TP4 P99 TPOT增量不超过预先规定的SLO预算；
- 历史预拷贝的Bridge/stall收益在多数配对中为正；
- 收益分布不能由少量极端轮次主导；
- 实际传输吞吐随配置速率变化，否则说明瓶颈已转移到序列化、stager或TCP。

得到固定速率响应曲线后，再把真实TP4 TPOT接入闭环控制器。Shadow使用可暂停的低优先级
速率；进入不可反悔的Bridge后，再根据剩余历史量和时限提高到更高上限。
