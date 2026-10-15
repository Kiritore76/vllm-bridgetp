# 四层剩余长度回归对照

固定入口：`bash tools/bridge_tp/run_predictor_length_regression.sh`。
只使用保存的特征和标签，单卡训练小 MLP，不启动 Qwen，不改在线预测器。
默认物理 GPU 0，可通过 `BRIDGETP_PREDICTOR_GPU_INDEX` 指定空闲 A100 40GB。
正在运行的迁移 pilot 结束后再启动；GPU 忙时脚本停止，不清理其他进程。

## 复用范围

- 旧5000条：`results/length_predictor/oasst1-long5000-v1-decoder31/capture-5000`。
- 长输出300条：根据原 recipe SHA 唯一解析已完成的 collection。
- 首60条：根据原 recipe SHA 解析，保留原600计划的已完成60条。
- 当前分类参考：`results/predictor-risk76-mlp2-mlp4-v1-20261011T093049Z-1353/train-mlp4-50pct/predictor_distribution.pt`。

可以用既有环境变量 `BRIDGETP_PREDICTOR_LEGACY_CAPTURE`、
`BRIDGETP_PREDICTOR_LONG300_BATCH`、`BRIDGETP_PREDICTOR_FIRST60_BATCH`
指定原目录；不复制特征到输出，不需要新输入包，不重新采集。
同类目录不唯一时停止，避免静默选错数据。数据的 recipe、revision、
输入、特征/索引/标签一致性及 source-tree 划分沿用原审计，另记录数据库 SHA。
tail6、tail12、targeted4、longwriter3 不加入训练，保留其原诊断角色。

## 固定试验设置

输入为 decoder:31 hidden+residual 的5120维特征和生成位置；输出为剩余长度标量。
结构 `5121 → 256 → 512 → 256 → 1`，三个 GELU/dropout 隐藏层，四个 Linear。
随机初始化，不继承分类器参数；只用训练状态计算归一化。

预测目标是 `min(自然剩余token,8192)`。剩余长度标签已经是剩余量，不再扣已生成量。
该有界目标沿用当前8192范围，不声称自然输出不会超过8192。
在8192内直接拟合连续数值，推理裁剪到0..8192；不分桶，也不保证逐token命中。
自然完成样本使用 Huber（拐点1024 token）；提前截断且下界小于8192的样本
仅在预测低于已知下界时施加单侧 Huber，不把上限误标为自然EOS。
下界已经达到8192时，有界目标确认为8192。

沿用原训练/验证/测试请求，不按状态随机重切。旧/新增损失各50%，
每请求prefill20%，其余decode按早/中/晚30/40/30；训练请求为原4000+210+56。
AdamW学习率0.0003、weight_decay0.01、dropout0.1、batch256、seed42、
最多30轮、patience6。旧/新验证来源分别按原哈希拆分selection/calibration半集；
回归版只按selection半集的旧/新各50%请求等权MAE选轮次，校准半集保留，
不调整回归输出温度，不用测试数据选模型。

## 比较与产物

冻结分类器 SHA 为 `dbe02d968e19c32d69deaa9a37748e9ca68f506807b220c114d22a4e2845c8a7`。
用它原有的归一化、温度和四层网络，在同一批状态上重算概率；
同时比较桶均值和中位桶中点，开放尾质量保留在8192。
分类器按NLL选模，回归器按长度MAE选模，这个目标差异在解释结果时保留。

主指标为请求等权MAE，另看RMSE、偏差、误差≤128/256/512/1024比例和
样本级P90绝对误差。分别报告来源、完整输出长度、prefill和decode四个进度段。
自然EOS样本作为三种点估计的相同评估集合；截断样本不冒充准确长度测试。
已有测试集被看过，这是开发对照，不是新的盲测，也不能用MAE替代候选排序准确率。
不因回归版不输出概率就自动删除线上生存概率检查。

输出到 `results/predictor-length-regression-mlp4-<UTC>-<pid>`，包含：

- `data_audit.json`：来源、SHA、划分与实际训练权重。
- `train/predictor_regression.pt`：含归一化和目标定义的最佳回归权重。
- `report.json`、`TRAINING_RESULT_CN.md`：同样本比较与训练历史。
- `predictions.npz`：各状态的真实剩余量/截断下界、两种分类点估计和回归输出。
- 运行身份、日志、真实返回码；外层生成单份 `.tar.gz`，不打包原特征副本。

在线参数与权重保持原样。结果回来后再判断是否值得做在线接入和候选排序对照。
