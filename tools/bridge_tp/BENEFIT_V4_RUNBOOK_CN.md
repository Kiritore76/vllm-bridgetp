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

原生事件使用65536条有限队列、最多512条批量写盘和50ms定时刷新，
后台生成回调仍只做非阻塞提交。发生丢失或写入错误仍拒绝完整性验收；
background_summary先保存逐请求诊断与native_evidence_integrity，再报工程失败。

到达版本为FIXED_EVENT_EPOCH_DUE_TIME_DISPATCH_V2：共享审计中事件发生的unix_s，
映射到固定monotonic epoch，调度器只在到达时派发HTTP任务；响应线程不提前占位睡眠。
最多512个在途HTTP请求；客户端超限显式失败，不在采集器内悄悄排队。
每条结果保存epoch、scheduled_unix_s和真实schedule_lag_s。
发生于引擎内部的排队／抢占仍属于服务结果，不能通过改事件锚点隐藏采集端晚发。

## 分阶段采集

`run_benefit_v4_collection.py --stage pilot`只运行六格训练replicate0，12arm。

`--stage repair-pilot`仅运行有余量源的decode／prefill两对，4arm；独立seed
20261251／20261252，diagnostic_only=true，不使用后续选择验证／测试记录。
decode配方为12请求×2048prompt、40秒周期，保留25ms压力档；prefill配方为
4请求×6144prompt、30秒一波同时到达、最多256输出token。两者都按真实观测选候选。
新配方尚需GPU可达性验收；未达到状态不能称已覆盖，不能进入正式模型选择。
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

读取probability pilot的`probability_outcomes.json`，校验其专用状态契约。
`paired_outcomes.json`是另一个paired-only协议产物，不能复制或重命名替代。
无干预对保留为eligible=false，不变成效果标签；缺重放证据的臂继续拒绝。
诊断可显式选择已完成pair；训练CLI仍要求所选阶段全部结果，不静默跳过失败对。

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

## 2026-10-12：到达验收与两臂调度复验

新增 `--stage dispatch-pilot`：只重复 repair decode 配方/种子20261251的一对START/STAY；diagnostic_only，不能进入模型选择。负载仍目标12请求、2048 prompt、40秒周期、4096输出上限；源端3同伴，guard2000，冻结预测器/SLO/KV迁移机制。没有增大请求或改固定token触发。

300秒窗口评分现在也应用预注册max_arrival_lag_s（默认2秒）。迟发不删除原始horizon_score或服务失败结果，只令fixed_horizon_eligible=false，并保存arrival_schedule_missed。先验收发送协议，再分析效应；不得以收益符号决定资格。

background_events改为AsyncJournal，关闭后核对footer及summary完整性；每条请求记录dispatch、worker_entry、HTTP客户端调用边界与各段耗时。HTTP时间是调用边界，不是socket发出或服务器收到的时刻。collector_resources.jsonl每2秒读取Linux CPU/内存压力与cgroup原始计数，无GPU调用；缺文件标为unavailable，不补零。

若复验仍迟发，先据分段时序与资源记录定位，不能据CPU模拟已通过就认定GPU/硬件有问题。CPU检查233项通过并不代替服务器性能验收。

## 2026-10-12：回传包精简

采集层子进程使用--no-archive，上层仅打包一次；外层模板也给stage传--no-archive，并使用tools/bridge_tp/result_archive.py。保留预测特征、概率/状态快照、原始输出与时间戳、SLO/窗口结果、日志、receipt、输入与唯一源码包；不因收益符号或运行失败删除记录。移除有对应完整采集目录/契约的重复内层tar和Python缓存，并附archive_contents_manifest.json。没有对应原目录的唯一tar包（例如runtime_source.tar.gz）保留。

打包不修改原结果目录，原子替换输出；失败保留旧的完整包。此次真实6-arm包750.73MB可精简为192.46MB（减少74.36%），15970个特征文件校验及31份核心原始文件字节一致检查通过。239项CPU回归通过；精简是包装层变化，不改预测器、p/U、SLO、窗口、KV迁移或实验标签。
