# Experiment A 基线能力审计（P0）

分支：`bridgetp/experiment-a-runtime-upgrade`

基线提交：`3eb69fae9f939e2cd27cbd38c95c84061b6e05ef`

## 已复用的真实能力

- TP1→TP4 的真实 KV 导出、按 TP4 四 rank 重分片和 TCP 传输。
- Shadow-only 的历史 KV 头部顺序预拷贝、新 KV 增量追赶、TP4 GPU block 预分配和逐块精确回读。
- 四 rank 最终 watermark/READY 门、统一响应代理和原子 owner 切换。
- Bridge Remote Attention 保留为后续历史基线，不进入本次第 7 节四模式 smoke。

## 第 7 节前必须补齐的缺口

- `Always TP1` 与 `Always TP4` 的同拓扑静态基线。
- 真正的请求级 Stop-and-Copy：只冻结 anchor，保留源 KV，完整复制后再接管。
- 区分“已发出 source abort”和“KVCacheManager.free 已实际返回”的证据。
- 每进程同时记录 wall clock 与 monotonic clock 的时间线，并在实验结束时合并校验。
- 四模式、三重复、交错顺序的一键 smoke gate 和统一 CSV。

## 证据边界

第 7 节 smoke 只验证四种机制在一套 5-GPU vLLM 拓扑中可运行、响应连续、KV 精确恢复、时间线完整。它不用于得出性能优劣结论，也不启动 A1–A5。

Stop-and-Copy 的 scheduler gate 是单请求级；同一 TP1 的其他请求不会被 gate 移出调度。但当前 KV 快照仍沿用现有 model-worker 导出实现，D2H/序列化可能争用 GPU/CPU。第 7 节无源侧 peer，因此“peer 无停顿”必须在后续 A 实验中单独量化，不能由本 smoke 声称。
