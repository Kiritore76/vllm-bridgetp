# BridgeTP D3 Phase 7 验证记录

Phase 7 已在同一台五卡 NVIDIA A100-PCIE-40GB 服务器上完成 commit 与 rollback
两轮独立实验。两轮环境均为 Python 3.12.13、PyTorch 2.11.0+cu130、CUDA 13.0、
vLLM 0.23，代码提交 `94256817cb1be4df2f02e92335212254bf9c7659`。

## Commit run

- migration ID：`0ffb7714-a3af-4a1b-81bf-cd98cab4644a`
- 归档：`phase7_0ffb7714-a3af-4a1b-81bf-cd98cab4644a.tar.gz`
- 归档 SHA256：`3bc549bd3b922563ddc7d42adf88f899be37794a85ac65b71d62a1519edb6340`
- inspector：`PASS`
- takeover state：`COMMITTED`
- 四个 receiver：全部 `OWNERSHIP_COMMITTED` 且 exact readback
- source：`finish_reason=abort`
- `快照前缀128 token + TP4续写64 token` 与干净 TP1 的192 token逐个一致

## Rollback run

- migration ID：`bba4bd1d-94c7-4649-8e6f-0ab3f34aae00`
- 归档：`phase7_bba4bd1d-94c7-4649-8e6f-0ab3f34aae00.tar.gz`
- 归档 SHA256：`5a1ff319c8de527255e40ea924ff87ecccef84f3669a4518e01f5e0c04835cc3`
- inspector：`PASS`
- takeover state：`ROLLED_BACK`
- `source_abort_dispatched=false`
- 四个 receiver：全部 `ROLLED_BACK`
- TP1继续完成192 token，输出前缀与干净 TP1 control 一致

两个归档内部的 `SHA256SUMS` 均重新计算通过。Phase 7 因而只证明正常进程条件下
的应用级原子接管和 commit 前回滚；不证明进程崩溃期间的分布式共识。
