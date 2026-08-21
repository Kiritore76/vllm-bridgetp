# BridgeTP D3 Phase 8 验证记录

## 结论

Phase 8在同一台五卡NVIDIA A100-PCIE-40GB服务器完成了两轮独立验证：

1. `dualwrite_commit`：原始迁移、接管和连续性证据通过；controller最终汇总时因
   TP4 streaming响应token入口错误而未写出原始`phase8_result.json`，现已通过保持
   原始归档不变的离线重建得到`PASS`；
2. `pre_cutover_controller_cancellation`：服务器原始inspector直接得到`PASS`。

两轮代码均为`a9a003c81107f9bad0faed3f980385d3d1abc6b2`，环境均为Python
3.12.13、PyTorch 2.11.0+cu130、CUDA 13.0、vLLM 0.23.0。GPU0–4的型号、UUID和
driver 595.71.05逐项一致，没有混合不同GPU平台。

## Dual-write commit run

- migration ID：`adf1ba96-426e-4705-8e97-d340dd6255bb`
- 原始归档：`phase8_adf1ba96-426e-4705-8e97-d340dd6255bb.tar.gz`
- 原始归档SHA256：`dff8686212ddceacadbb039c8436c3a0fb1d30bcb32aa6d705c46ae34a44d143`
- 原始`SHA256SUMS`：293/293文件本地复算通过
- 初始old-KV边界：147 computed token
- cutover边界：160 output、179 computed、1 pending
- new-KV：32 token / 32 batches；四rank均连续覆盖`[147,179)`
- old-KV/new-KV时间重叠：通过
- 四rank staged delivery、GPU exact readback：全部通过
- 四rank最终状态：全部`OWNERSHIP_COMMITTED`
- takeover：`COMMITTED`，`source_abort_dispatched=true`
- source：`finish_reason=abort`，abort前生成191 token
- cutover后source多算并丢弃：31 token
- TP4接管：128 token
- `cutover前160 + TP4后128`与干净TP1 control的288 token逐个一致

原controller在读取目标结果时错误复用了普通completion入口
`choices[0].token_ids`；Phase 8 streaming helper实际将token规范化到顶层
`response["token_ids"]`。因此原始`controller_output.txt`和`inspection.json`为空，
且没有原始`phase8_result.json`。这不是迁移或连续性失败。

本地工具`tools/bridge_tp/reconstruct_phase8_result.py`先验证原始293个文件，再只从
归档原始JSON和receipt复算判据，生成：

```text
phase8_result.offline.json
inspection.offline.json
OFFLINE_RECONSTRUCTION.json
SHA256SUMS.offline
```

所有派生文件显式标记`evidence_origin=offline_reconstruction`，不覆盖原始文件。
`target_ready_to_commit_response_ms`、`commit_api_ms`和
`commit_to_target_first_token_ms`只存在于原controller进程内存，无法从归档恢复，
因此保持`null`，不得补造或作为本轮性能结论。

## Pre-cutover cancellation run

- migration ID：`3d4a51e9-d35d-4ea9-959f-48937c99cd60`
- 原始归档：`phase8_3d4a51e9-d35d-4ea9-959f-48937c99cd60.tar.gz`
- 原始归档SHA256：`e7ae6e2a8f2c394c64c24e8b35ca92749acd9d5184780e0cbff04a589ed20e11`
- 原始`SHA256SUMS`：43/43文件本地复算通过
- inspector：`PASS`
- takeover state：`CANCELLED`
- source：`finish_reason=abort`
- source delta mirror和CPU stager：均`CLEANED`
- 已排空2个new-KV token；释放四rank buffer
- 没有target request，没有takeover commit

## 证据边界

Phase 8证明的是同节点、单请求、greedy sampling、CPU staging条件下的后台old-KV
搬运、token粒度new-KV mirror、TP4原子接管和显式取消清理。它不证明：

- GPU P2P/NVLink或跨节点RDMA数据通路；
- 任意temperature/top-p请求的RNG状态迁移；
- 多请求并发与正式handoff stall分布；
- API/节点崩溃期间的分布式共识；
- 自然EOS的全部竞争时序；
- 生产级统一客户端响应代理。

正式handoff timing和线上干扰测量属于Phase 9，必须使用修复后的controller重新产生
带完整时间戳的新运行，不能从本次离线重建推算。
