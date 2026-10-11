# 四层76桶与目标工作相位：V4采集／离线拟合

固定checkpoint SHA：`dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7`。
guard2000、H与arrival_window均300秒、slow1pct_soft、beta1。
迁移保持现有历史确认后自适应冻结选点：基础提前量128，最多调整2次；
target reservation仍512，是空间预留上界，不能与基础冻结提前量混为一谈。

## 四层接入与观测

运行时按checkpoint构造四层，严格验证结构／输入缩放／温度／模型配置SHA；
旧两层保持兼容。采集指定`--predictor-profile risk76_mlp4`，不能用旧文件名绕过校验。

目标队列未处理prompt token另计，包含排队及执行中未处理部分，按请求去重，
不按KV block取整；不改变源端guard/U容量公式。
控制器只读取实际原生到达与流式ITL，读不到未来配方。
到达强度窗口10秒，decode压力窗口3秒，是ITL/当前TPOT参考100ms的p95；
至少20个间隔且最新≤2秒。繁忙prefill数据按引擎时间戳≤2秒核验，
新鲜HTTP scrape不能刷新陈旧工作数据。
真正空闲且无新到达时，旧空闲引擎报告可以继续有效；中性decode压力为1，
状态明确为IDLE。其余缺失标MISSING，不填0。

预测每20token一次，异步保存各源请求的实际hidden、位置、概率与身份；
独立磁盘线程与有限队列，收尾写receipt和每批SHA。候选事件包含feature_id。
队列溢出／写盘错误／缺receipt不能当完整数据。
每对之后CPU重放全部保存状态，检查在线概率误差≤1e-5与候选关联。
这增加了特征复制和保存负担，必须通过GPU pilot衡量，CPU一致性不代表GPU开销通过。

实验收尾必须在controller和背景请求全部完成后，通过源端专用finalize接口
排空预测与写盘队列，确认receipt和保存批次数，再停止服务进程。
不能依赖API server的abort/强制退出完成异步保存。该操作位于窗口和请求排空之后，
不进入冻结／最终增量确认路径；接口只接受本轮secret，不开放任意开发RPC。
重复finalize／模型shutdown不会再次向已退出的写盘线程投递终止标记。

## 分阶段采集

`run_benefit_v4_collection.py --stage pilot`只运行六格训练replicate0，12arm。
两档源：实测U≤0.5、有余量；U≥0.8、紧急（允许物理可行的U>1）。
三档目标：真正空闲、原生decode繁忙、prefill队列积压。
人数／prompt／到达周期只是构造旋钮，候选必须符合实际观测门槛。

pilot探索门槛p_lower=0；decode压力下界暂0.25，代表100ms参考下25ms的p95，
并要求prefill已排空；prefill积压下界暂1024 token。
这两个是待pilot检查的工程配方参数，不是校准后的“忙”分类线，也没有修改SLO。
源常规3个同伴×4096prompt；紧急5个同伴×4992prompt；anchor2048prompt。
decode目标4个同伴×2048prompt、40秒波；prefill目标8个×6144prompt、10秒波。
自然EOS，不强制长度，窗口后排空。未达到状态或未跨门槛保留，不能假称覆盖完成。

后续stage：train-rest18对；validation6对；test6对。总36对／72arm，
pilot协议不变则计入24对训练；改协议时pilot转诊断，不静默跨版本混用。
训练后两轮门槛暂0.1／0.3，验证／测试暂0.1；必须先据pilot不看收益的
状态覆盖检查其可达性，不能默认运行全部stage或用收益结果选择触发门槛。
preregistration.json写出全部角色，commands.json保存命令与每个pair的完整配方。
请求记录root隔离；所有split沿用同类构造任务模板，当前不声称跨模板泛化。
固定同配方START/STAY随机顺序，记录候选匹配差异。技术错误打包后停止，
OOM／抢占／超时等服务结果按现有评分保留。

## 离线F8／F10

`fit_benefit_v4.py --collections <pilot> <train-rest> <validation> --out-dir <new-fit>`
只读取训练与选择验证结果，不读取test标签。
F8的目标到达强度改为候选前实际10秒prompt tokens/s；F10额外增加未处理prefill和
3秒原生decode压力。两项有效才能进入F10，不用0插补。
每对一个ΔQ样本；ridge网格0.1/1/10/100，训练折标准化，按请求家族及负载块留出。
选择验证比较工程余量0/5/10；先看有害START损失，再看regret。
预测超出训练特征范围标UNKNOWN。报告完整域与F8/F10共同支持域，
UNKNOWN按STAY计算机会损失；不因全部弃权就称正确。

模型冻结后：`--collections <test> --frozen-models <fit/result.json> --out-dir <new-test>`。
报告目标三档分层指标、负收益误START、正收益漏迁、MAE/RMSE、regret和覆盖。
近零等价带max(5质量加权token/s, 2%STAY Q)。至少2对有意义正例与2对负例
才能讨论初步方向验收；样本少时只决定下一轮补采，不生成validated_LCB或自动部署。

服务器用原仓库、SSH拉固定HEAD、激活原环境，核对5张A100-PCIE-40GB和各文件SHA；
UUID仅记录。已有未提交工作先备份并保留，使用return而不关闭终端。
