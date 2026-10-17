# BridgeTP：开发与运行说明

更新：2026-10-17。本目录只维护这一份模块说明。阶段计划、旧实验结果和旧版指标文档已从代码目录移除；历史版本可通过Git查看。实验输入、JSON配置与必需的运行契约仍位于`experiments/phase9/`，分析报告及结果包不放进源码目录。

## 1. 当前系统与代码入口

BridgeTP在vLLM之上实现TP1/TP4实例间分流，以及TP1请求的滚动KV搬运、目标接管与源端释放。实例内部排队、prefill、decode和抢占仍由vLLM调度。当前压力控制器不增加“预留完整未来输出后才允许请求进入”的外部准入门槛。

当前开发分支为`bridgetp/predictor-nonuniform84-20261010`。旧`bridgetp/m5-predictor-integration`等分支保留的是历史控制器；不能因为看到同名README就认为该checkout具备最新压力入口。切换前先保存并核对工作区，禁止覆盖未提交修改。

| 功能 | 当前入口 |
|---|---|
| 云端统一运行、进度和打包 | `tools/bridge_tp/run_pressure_pilot.sh` |
| 四模式配对实验 | `tools/bridge_tp/run_pressure_comparison.py` |
| 原生请求/KV观测 | `vllm/bridge_tp/pressure_inventory.py` |
| 工作量压力与分流 | `controller/pressure_workload.py`、`controller/pressure_policy.py` |
| 分流减负后的确认窗口 | `controller/pressure_relief.py` |
| 协调、请求流、库存读取 | `controller/pressure_runtime.py`、`pressure_ingress.py`、`pressure_inventory_process.py` |
| 滚动迁移及多会话 | `tools/bridge_tp/pressure_session.py`、`pressure_sessions.py` |
| 可逆阶段取消 | `controller/pressure_cancel.py` |
| 冻结、所有权、响应拼接 | `request_freeze.py`、`controller/state_machine.py`、`controller/response_proxy.py` |
| 历史单锚控制器及离线回放 | `tools/bridge_tp/run_phase9_controller.py`、`replay_policy.py` |

上表仓库根路径从本README所在目录向上两级解析。历史分支可能尚未包含部分压力模块，应以实际检出的代码为准。

## 2. 压力、分流和迁移规则

- 在线被动读取两端原生请求库存、当前上下文、未完成prefill、空闲KV及增长、输出间隔；不额外发送探测请求。
- 工作量参考来自已有重复测量中达到该上下文档实测峰值goodoutput至少90%的最小Nd。参考中心3072/5120/7168 token，TP1 Nd=8/6/3，TP4 Nd=16/24/16；在支持范围内对`1/Nd`插值，范围外保守回退。这是实测工作点，不是最大服务容量保证。
- 新请求先估计加入两端后的工作量、prefill和KV压力。TP1仍在预算内时优先TP1，否则比较两端投影压力；选中端prompt放不下而另一端可放下时改投另一端。两端暂时都放不下仍进入原生队列，参数/上下文是否合法另行检查。
- 尚未被库存看见的已分发请求由账本临时计入，原生库存出现后对账。普通请求账本不是完整输出KV预分配；已经分配的prompt块不能重复扣减。
- 普通迁移先要求分流减负已安装、旧prefill排空，再以新鲜原生prefill计数和库存确认独立质量窗口；源端仍有压力、TP4接收后仍可承受、候选可行才启动。当前持续确认3秒，普通冷却2秒。新prefill、新源端准入或证据过期会重置确认。
- KV优先路径按空闲块、未分配prompt及增长估计风险，当前安全余量125块、块大小16 token、释放前瞻16.864101秒。KV路径可绕过普通确认/冷却，但不能绕过目标容量、候选有效性和通道可用性；它不是OOM保证。
- 普通候选优先预计剩余工作较长、源端减负更多且目标成本可承受的请求；KV候选优先能及时释放足够块并存活到切换的请求。预测器输出剩余分布；`输出停止上限−已生成数量`只是剩余合法范围的截断，不是对预测剩余量重复相减。
- 历史KV发送时源端继续decode，随后补增量，在最终边界冻结单条请求并接管。只有匹配的目标提交、所有权与源端free返回证据才能结算；证据不明时保留承诺，不能伪装成安全清理。
- shadow只能在可逆阶段取消，恢复条件要持续成立；不能因一个短暂低压样本就取消。后续会话可在前次释放通道后继续，当前比较入口不限制只能迁一条。
- 当前pressure pilot使用FULL_SPEED：速率0表示不限速，history pacing关闭。最终边界合并等待与带宽限速是不同机制，不得静默恢复限速。
- 旧health参考UNKNOWN不等于新workload压力不可用；新库存缺失/过期时必须记录并按归一化最少在途回退。缺少迁移证据时不启动实际迁移。

配置数值以[`health_service_policy.json`](../../experiments/phase9/pressure_balanced_v2/health_service_policy.json)和对应运行包为准；参考推导、场景契约见[`HEALTH_SUSTAINED.md`](../../experiments/phase9/pressure_balanced_v2/HEALTH_SUSTAINED.md)。修改被SHA清单覆盖的代码、输入或配置后，要同步更新清单并完成CPU检查。

## 3. 运行与结果收集

云端沿用现有checkout、环境、模型与预测器路径。确认分支正确且工作区改动已保存后，在仓库根目录运行已有入口，例如：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
git pull --ff-only origin bridgetp/predictor-nonuniform84-20261010 &&
bash tools/bridge_tp/run_pressure_pilot.sh --pool-compare
```

| 参数 | 范围 |
|---|---|
| 无参数 | 轻载smoke |
| `--full` | 四个功能场景 |
| `--workload-pilot` | 第一个种子的压力STAY/MIGRATE配对 |
| `--pool-compare` | 三个种子、四种分流/迁移模式，共12场 |
| `--pool-compare-r1` | 仅第一个种子的四模式复测 |
| `--pool-compare-r1-migrate` | 仅第一个种子的MIGRATE单场复测 |
| `--health-kv-only` | 固定落点的双波KV压力机制专项，STAY/MIGRATE各一场 |

四种模式为实测能力加权轮询、归一化最少在途、压力分流但禁迁移、压力分流加迁移。固定落点专项只验证机制，不应当作自动分流系统对照；单臂复测没有同期STAY，不能生成配对收益结论。

launcher在启动GPU前检查checkout、输入/模型/预测器SHA、实际serving/stager/controller参数及CPU回归。发现未提交工作会备份并停止，不reset/clean/stash；发现其他GPU计算进程会停止，不自动杀进程。

进度显示启动、warmup、提交/完成、原生waiting、分流减负阶段与迁移状态。每场完整收尾，结果写到`/root/autodl-tmp/bridgetp/results/pressure-balanced-git-v2-<模式>-<时间戳>/`并生成同名`.tar.gz`。打包成功不代表运行通过，需分别检查runner退出码、场景状态、到达有效性和覆盖情况。

GPU残留进程清理前必须核对PID、父进程、进程组、命令及本轮运行身份，只清理确认归属本轮的进程组；不能依据“进程名含vllm”批量终止。现有运行目录或会话不支持无证据自动恢复，使用新身份，保留旧故障证据。

## 4. 评测规则与证据边界

当前压力实验定义集中在[`EVALUATION_STANDARD_CN.md`](../../experiments/phase9/pressure_balanced_v2/EVALUATION_STANDARD_CN.md)，不得混用历史v1至v6的TTFT/E2E/50ms门槛。

- 每场公共窗口300秒，吞吐只计窗口内客户端实际输出token，不计prompt与窗口外drain；完整收尾质量仍作用于窗口内token。
- 软权重`w=min(1,0.01/f,1/m)`，f为完整请求超过100ms的输出间隔比例，m为最大间隔秒数；分母为0的项按1处理。软goodoutput是窗口内token乘权重后除以时间。
- 硬输出流SLO为`f<=1% 且 m<=1秒`。同时报告全部/decode/单token请求达标率、窗口内达标完成请求/s，以及达标请求窗口输出token/s。
- TTFT、TPOT、E2E、慢间隔比例和最大间隔单独报告；TTFT/排队暂没有硬门槛。TPOT均值不等于尾部ITL，迁移接缝不得删掉。
- handoff与START到源KV释放分别报告；空闲块净变化不是确切释放块数。真实源/目标交付来源与初始路由分组分开统计。
- 当前到达误差上限1秒；负载覆盖要求至少95%窗口有在途请求且每个10秒桶有输出。无效到达、故障、STAY及覆盖不足均保留，不能混入有效对照平均。
- 自然EOS下不同TP/批次/分流可能改变输出长度。固定输入与种子不保证最终token一致，性能比较应报告输出量，固定逐请求长度主对照和自然EOS补充要区分。
- 无实际迁移不证明搬运性能；单次正收益不证明统计优势；一次复测正常不证明上次共同停顿根因已修复。

## 5. 仍须保留的特殊约束

### KV、所有权与响应连续性

区分已计算prefix和已采样但尚未进入KV的pending token，目标只能把确切prefix标成computed。目标placeholder映射、剩余输出预算、四rank APPLIED/ready和源端冻结边界必须一致。不能用预留上限代替实际边界，不能把未应用delta的ACK当已写回。源端实际释放或匹配的完整清理回执是回收承诺的依据。

请求级KV块和共享refcount必须从原生manager取证；零块的等待请求仍属于库存，不能丢失其prompt需求。多KV组、speculative、async scheduling及其他不受支持组合应由对应入口明确拒绝或标未知，不能沿用旧单组布局假设。

### 数值一致性与历史D3实验

机械接管成功、KV exact readback、自然输出token完全相同是不同证据。跨TP的数值差异不能仅凭首个分歧判为传输损坏，也不能据此宣称迁移没有额外误差。D0 logits、固定prefix的D1 provenance、精度探索D2与成对D3必须分别保存请求、模型、精度、GPU、输入和采样契约。

历史D3中052的pending-tail边界修复补跑不能直接与旧HEAD的49条混成同版本正式50条队列；需要恢复旧手册或结论时使用其对应Git版本。单点agreement、合成CUDA/NCCL或远端Attention实验不能外推为真实请求SLO、最大容量或跨平台统计结论。

logits诊断需显式核对请求ID过滤、捕获索引集合与global offset，分别识别control和迁移流；不能沿用旧实验的固定token ID、K、KV块数或本机trace路径。拓扑probe对共同prefix的重算不等同于迁移请求在原数值状态下的现场logits。

旧收益模型入口仍需真实`calibration_source`并回放检查边界；压力入口不会把旧收益模型参数伪装成当前压力指标。保留旧代码用于复现，不继续使用旧阶段文档作为现行配置来源。

### 请求级导出诊断

导出hook默认关闭。需要单独定位KV时先metadata-only，并限制请求身份；不要把调试导出配置复制到正常性能实验。

```bash
export BRIDGETP_DUMP_ENABLED=1
export BRIDGETP_DUMP_AFTER_OUTPUT_TOKENS=128
export BRIDGETP_DUMP_DIR=/root/autodl-tmp/bridgetp/bridge_dumps
export BRIDGETP_DUMP_TENSORS=0
export BRIDGETP_DUMP_STRICT=1
```

确认metadata后才按明确需求启用`BRIDGETP_DUMP_TENSORS=1`与字节上限。单请求诊断使用同步调度，默认TP1；TP4 rank-local导出需显式选择允许world sizes。检查工具为`tools/bridge_tp/inspect_dump.py`；离线reshard工具为`reshard_kv_dump.py`和`inspect_reshard.py`。这些工具各自的通过只证明对应环节。

## 6. 文档与历史维护

后续修改直接更新本README和必要的实验契约，不再在Python包中新增按日期/阶段命名的计划、验证结果或指标修订副本。结果报告保留在工作区`control/`，大数据、模型、checkpoint与压缩结果不提交Git。

旧文件覆盖了Phase1至9、M0、GoodOutput各版、Shadow与remote-attention研究路线；删除的是这些历史说明，不是相应Python实现或实验输入。需要查阅删除前版本时，在仓库中执行：

```bash
git log --all -- vllm/bridge_tp
git show cb1476f:vllm/bridge_tp/PHASE1_9_OVERVIEW.md
```

具体结果必须回到对应归档和SHA，不从历史文档的“已通过”“尚未执行”等文字推断当前系统状态。未提交工作清理前先比对并备份；有用说明合并到本文件后再删除原稿。
