# BridgeTP真实KV传输与远端Attention验证

## 1. 本分支现在验证什么

原来的Shadow策略脚本根据C1/C2/C3输入计算传输时间，属于解析模拟。本分支已移除该模拟，
改为使用五张真实CUDA GPU和NCCL执行：

1. GPU0生成Qwen2.5-14B形状的Q/K/V；
2. GPU0把历史KV的远端suffix按8个KV heads切成四份；
3. GPU0通过NCCL把每份真实tensor发送给GPU1–4；
4. 每个TP4 rank接收10个query heads对应的Q shard；
5. TP1和TP4分别计算各自token区间的`m/l/o`；
6. TP4把局部统计传回TP1；
7. TP1执行全局online-softmax合并，并与完整Attention比较；
8. 可选加入每step真实NCCL copy竞争和TP4 CUDA GEMM负载。

这不是传输时间模拟。所有报告的staging和step latency都来自真实CUDA/NCCL执行。

## 2. 当前证据边界

当前runner是G1/G2级微基准，尚未接入vLLM在线decode：

- K/V内容是固定seed生成的Qwen形状tensor，不是模型运行时捕获值；
- 测量一层split-KV Attention，并给出48层串行投影；
- target load是可控CUDA GEMM，不是真实TP4 vLLM workload；
- `copy_bytes_per_rank_step`是真实NCCL链路流量，但不是在线KV manager产生的block；
- 尚未执行Bridge中的scheduler、逐块释放、response proxy和最终Takeover。

因此它可以证明协议数学正确性、真实张量可传输和五卡成本范围，不能单独证明完整BridgeTP
带来系统收益。

runner支持`--tensor-bundle`。捕获包必须由可信的本地工具生成，内容为：

```text
q: [40, 128]
k: [8, max_context_tokens, 128]
v: [8, max_context_tokens, 128]
```

`q/k`必须是应用RoPE之后、进入Attention之前的值，`v`是对应层的cache值；现有vLLM分页KV
布局必须先规范化为上述连续逻辑token顺序。捕获包只表示一个确定layer和一个decode step，
其layer、request、token边界和模型revision还必须在旁路provenance中冻结。

加载使用`torch.load(..., weights_only=True)`。同时提供`--expected-tensor-sha256`时会在初始化
NCCL前核验。没有捕获包的运行明确标记为`GPU_SYNTHETIC_KV_TRANSFER`；使用捕获包才标记为
`GPU_CAPTURED_MODEL_TENSORS`。两者都还不是在线vLLM Bridge结果。

## 3. 输出

每轮使用全新目录，输出：

- `measurements.csv`：每个context、远端KV比例、target load和copy竞争cell的实测数据；
- `provenance.json`：commit、CLI参数、五卡identity、PyTorch/CUDA/NCCL和源码hash；
- `acceptance.json`：case数量、数值误差、有限值、实际/预期传输字节和正延迟检查。

`acceptance.json`不是论文系统结论，只是进入vLLM Bridge实现前的工程门。

## 4. 服务器拉取后CPU/静态检查

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

"$VIRTUAL_ENV/bin/python" -m py_compile \
  vllm/bridge_tp/remote_attention_protocol.py \
  tools/bridge_tp/run_remote_attention_validation.py \
  tests/bridge_tp/test_remote_attention_protocol.py \
  tests/bridge_tp/test_remote_attention_math.py

"$VIRTUAL_ENV/bin/python" -m unittest \
  tests.bridge_tp.test_remote_attention_protocol \
  tests.bridge_tp.test_remote_attention_math

"$VIRTUAL_ENV/bin/python" \
  tools/bridge_tp/run_remote_attention_validation.py \
  --validate-only \
  --out-dir /tmp/remote_attention_not_created \
  --expected-revision "$(git rev-parse HEAD)" \
  --context-tokens 128 1024 \
  --remote-fractions 0.25 0.75
```

## 5. 五卡smoke

运行前确认五张GPU没有其他服务占用。该命令在一个前台终端内启动全部五个进程。

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export RA_HEAD="$(git rev-parse HEAD)"
export RA_ID="remote-attention-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
export RA_OUT="/root/autodl-tmp/bridgetp/results/remote_attention/$RA_ID"
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
"$VIRTUAL_ENV/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node=5 \
  tools/bridge_tp/run_remote_attention_validation.py \
  --out-dir "$RA_OUT" \
  --expected-revision "$RA_HEAD" \
  --context-tokens 128 1024 \
  --remote-fractions 0.25 0.75 \
  --target-load-repeats 0 \
  --copy-bytes-per-rank-step 0 \
  --warmup-steps 2 \
  --measured-steps 5

echo "RA_OUT=$RA_OUT"
cat "$RA_OUT/acceptance.json"
```

## 6. G2测量批次

smoke的`acceptance.json`为`PASS`后，再运行较完整矩阵：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export RA_HEAD="$(git rev-parse HEAD)"
export RA_ID="remote-attention-g2-$(date -u +%Y%m%dT%H%M%SZ)"
export RA_OUT="/root/autodl-tmp/bridgetp/results/remote_attention/$RA_ID"
export OMP_NUM_THREADS=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4 \
"$VIRTUAL_ENV/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node=5 \
  tools/bridge_tp/run_remote_attention_validation.py \
  --out-dir "$RA_OUT" \
  --expected-revision "$RA_HEAD" \
  --context-tokens 128 512 1024 2048 4096 8192 \
  --remote-fractions 0.10 0.25 0.50 0.75 0.90 \
  --target-load-repeats 0 1 4 \
  --copy-bytes-per-rank-step 0 1048576 \
  --background-gemm-size 1024 \
  --warmup-steps 5 \
  --measured-steps 20

echo "RA_OUT=$RA_OUT"
cat "$RA_OUT/acceptance.json"
```

矩阵为6个context×5个远端比例×3个计算负载×2个copy条件，共180个实测cell。第一次运行可
先把`context-tokens`限制为`128 1024 4096`做pilot，再决定是否执行全部180个cell。

## 7. 结果解释

需要分别报告：

- `staging_gib_s`：一层历史KV suffix实际传给四个rank的聚合带宽；
- `step_p50_ms/step_p95_ms`：一层Q发送、两侧Attention、统计返回和合并的端到端时间；
- `projected_all_layers_p50_ms`：按48层串行调用的投影，只用于筛选，不作为实测token TPOT；
- `max_abs_error/mean_abs_error`：split与完整Attention的数值差异；
- target load和copy竞争造成的paired latency delta。

若数值验收失败，停止性能解释。若48层投影已经不可接受，应优先修改通信聚合或执行粒度，
而不是直接接入vLLM。

## 8. 通过后下一步

G2通过后才进入真实在线Bridge：

1. 用模型运行时捕获的Q/K/V替换synthetic tensor；
2. 把远端Attention接入一个隔离的vLLM decode请求；
3. 实现Bridge block ownership与ACK后TP1释放；
4. 与现有Phase 8 transfer、takeover和response proxy连接；
5. 完成零/部分/完整远端KV三个bring-up；
6. 最后运行`S_NEW`与`S_NEW_OLD`的正式配对实验。
