# 批次性能观测4-arm功能pilot

本批仅验证两端观测。目标idle、高prefill各一对START/STAY，固定配方后再采集；
不读取最终测试，不重训预测器，不修改迁移决策或冻结计划。

## 固定设置

- 四层76桶冻结预测器，SHA256为
  `dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7`。
- guard 2000；300秒窗口及完整请求drain；slow1pct_soft评分不变。
- 概率门槛0，U在[0,0.5]，目标状态门槛沿用当前配方。0门槛用于尽早取得
  工程观测，不代表部署阈值，也不以固定生成token启动迁移。
- 两对使用新seed 20262400、20262401；源端3请求×4096 prompt。
- idle无目标背景请求；高prefill目标8请求×6144 prompt、输出上限256，
  首次源输出后1秒注入，后续周期30秒。生成自然结束，不强制长输出。
- 两对均标记diagnostic_only，不能自动加入正式训练、选择或最终测试集。
- 第1–2对技术失败仍继续；本批只有两对，执行完汇总，不自动扩量。

## 观测

使用`BRIDGETP_PERFORMANCE_TRACE=1`启用，runner为每个arm分别配置源、目标目录
与finalize token。默认关闭，不在共享服务模式使用。

调度器在推进computed计数之前保存query、已计算上下文、prompt、已知总长度及
抢占次数。区分prompt prefill、生成历史重算、正常decode；speculative样本单独标记。
累计调度计数在目标观测开启时也记录，不再依赖源端freeze gate。

非空batch每8步采样一次；含prefill或重算的batch全部采样。采样策略不同，
离线计算不能将所有记录等权当作每步总体分布。空batch不计时。

CUDA事件包围forward及logits，当前流上记录，后续batch仅query完成事件。
它不是所有GPU kernel的纯执行时间，也不包括输入准备、采样与异步独立写回流；
流内等待/干扰会体现在跨度内。TP4各rank使用同一step id，离线取rank最大跨度，
不在每步增加allreduce或强制同步。另存提交到模型/采样结果返回的墙钟延迟，
其中可能含异步队列等待，不能冒充纯GPU计算耗时。

事件缓冲64、有界异步写盘队列2048；溢出或CUDA计时异常写入receipt并导致
观测验收失败，不为了日志阻塞decode。结束后的认证路由只允许调用指定finalizer，
引擎非空则拒绝；空闲时才等待剩余事件（最多5秒）及写盘，保留完整footer。

## 输出与验收

每个arm的`controller/performance_trace/{source,target}/`含engine.jsonl及各rank
gpu_rankN.jsonl。所有行有时间、step id和实际批组成，含warmup与300秒后drain，
分析必须用时间戳分段。迁移阶段可与phase9_audit、proxy时间戳对齐。

`verify_performance_trace.py`检查两端engine/GPU批次集合、组成、提交标识一致，
footer无丢失、计时为正、rank集合正确及finalization确认。另检查首个源输出后
300秒内有源decode、高prefill组有真实目标prefill，避免用warmup冒充负载证据。
正常STAY idle目标在测量期无工作是允许的，不把它当零耗时训练点。

GPU运行前额外检查SchedulerOutput的安全序列化往返、HEAD、运行文件SHA、
模型配置与冻结参数SHA、5张A100-PCIE-40GB。GPU UUID仅记录。
只打一个证据包，保留原始性能行与既有收益/特征证据，不嵌套复制包。

本批不能证明记录开销足够小，也不能证明完整性能曲面有效；开销开/关对照和
独立收益验证在本批结果通过后再决定。Windows CPU测试使用最小包stub，
真实CUDA与多rank仍由服务器pilot验收。
