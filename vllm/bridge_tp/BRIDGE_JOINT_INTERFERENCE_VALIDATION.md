# BridgeTP G3-J Bridge联合干扰验证

更新时间：2026-09-09

## 1. 要证明的问题

G3-I已经分别测量Shadow复制和Bridge复制对TP4合成负载的影响。G3-J进一步在同一个Bridge
step中联合执行：

- TP1到TP4的远端Attention查询与统计量回传；
- 一次全层新KV加一个16-token历史block的复制；
- TP4本地CUDA GEMM负载。

每个factorial cell固定比较三个处理臂：

- `ATTENTION_ONLY`：只有远端Attention；
- `COPY_ONLY`：只有Bridge KV复制；
- `JOINT`：远端Attention与KV复制同时发生。

`JOINT - ATTENTION_ONLY - COPY_ONLY`是交互项。正值表示两种Bridge工作叠加后对TP4的干扰
大于简单相加，负值表示资源重叠或计时摊薄。

本实验为原计划G4事件驱动模拟提供联合干扰实测表，不替代G4。它也不是在线vLLM实验。

## 2. 实际执行内容

- rank 0代表TP1，rank 1–4代表TP4；
- 远端Attention执行Qwen2.5-14B的40个Q head、8个KV head和128 head dim；
- 每个逻辑step真实执行48层Attention通信与计算，不用单层延迟乘48代替；
- KV复制每个TP4 rank发送835,584 bytes，对应一个新token加一个16-token历史block的
  Qwen全层BF16 KV；
- TP4本地负载在独立CUDA stream上与Bridge工作并发；
- 每个处理step按`control-before -> treatment -> control-after`执行，无Bridge control使用
  相同GEMM次数；
- copy payload在TP4逐rank验证，Attention输出与完整GQA Attention比较。

## 3. 输出

- `measurements.csv`：每个cell和处理臂的汇总；
- `step_measurements.csv`：每个step的前后control、treatment、delta、harm和Bridge path时间；
- `factorial_interactions.csv`：每个cell的Attention、Copy、Joint slowdown与交互项；
- `acceptance.json`：正确性、完整性和原始step复算验收；
- `provenance.json`：版本、参数、GPU和源码哈希。

主要指标：

- `target_signed_slowdown_frac`：`sum(treatment-control) / sum(control)`；
- `target_harm_ms`：逐step的`max(0, treatment-control)`累加；
- `bridge_path_p50_ms/p95_ms`：TP1主导的Bridge处理路径时间；
- `joint_interaction_slowdown_frac`：联合干扰中的非加性部分。

## 4. 证据边界

实验使用真实CUDA、NCCL、48层Attention算术和Qwen KV字节量，但Q/K/V、paged-KV布局和TP4
本地工作负载仍是合成的。它不能声称真实请求TPOT、P99或goodput已经得到验证。在线结论必须
由后续真实vLLM target-native请求实验给出。
