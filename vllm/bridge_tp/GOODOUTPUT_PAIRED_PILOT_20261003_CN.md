# GoodOutput STAY/MIGRATE 配对接线验收

本批次先验证反事实实验的代码路径，再决定迁移收益。运行脚本为 `tools/bridge_tp/run_goodoutput_pair_a100.sh`，当前只做一次 STAY→MIGRATE 诊断对，不给出显著性结论。

两臂使用同一台五 A100、同一 HEAD、模型配置、预测器权重、survival table、guard、同一个生成的 A4-P manifest、相同源锚请求和 M1–M5 配置。所有输入路径及 SHA 在 GPU 启动前写入 `preflight.txt`。背景源请求的到达以各臂锚请求首个输出为事件锚；实际时间戳仍需在拿回后核查。

STAY 臂运行原来的 M1/M5 观测并记录自然 M1 START 决策，但 `--paired-stay` 抑制所有迁移动作；验收要求锚请求的 1024 个 token 全部由 TP1 输出，状态为 `COMPLETED_ON_TP1`，不存在迁移会话或 TP4 锚输出。MIGRATE 臂沿用现有 M1–M4 物理路径及 M5 shadow，要求现有在线验收通过并有 TP4 锚输出。两个臂都保留所有 TP1/TP4 背景请求。

完成后 `compare_goodoutput_pair.py` 检查版本、manifest SHA、survival/guard/checkpoint SHA、锚输出上限、请求数和输出 token 总数，按 [v5 尾部门禁](GOODOUTPUT_TPOT_TAIL_GUARD_V5_20261003_CN.md)计算整池的 `GoodOutput(MIGRATE) - GoodOutput(STAY)`，同时保留 v4 历史对照。结果标记 `PILOT_COMPARABLE_NOT_STATISTICAL`。只看一对的正负不允许调整在线收益函数；后续应同负载交错重复、检查顺序和随机波动，再做 W1–W4 的自然 EOS 工作负载。

本批次复用已经通过的高压 A4-P 输入，因此属于 W2 风格的接线验收；`ignore_eos=true` 的 1024-token 合成输出不能用于拟合长度预测的真实迁移收益。当前 GoodOutput 指标不设独立 TTFT 门槛，仍报告 TPOT 尾部、E2E、handoff 与 OOM。
