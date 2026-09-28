# Migration Manager M0 实现与回放

M0 是现有 Phase 9 控制器的旁路观察层。它记录标准化快照、现有策略的提议和
`WOULD_START`／`WOULD_SET_RATE`／`WOULD_COMMIT`／`WOULD_CANCEL`／`WOULD_WAIT`
结果；当前 live 提议覆盖 Start／Rate／Cancel，Commit 仍待 M3 定义，不能把
诊断 EARLIEST_READY 直接当成收益驱动 Commit。M0 本身不会调用迁移动作。当前 live 接入点在
`vllm_bridge/tools/bridge_tp/run_phase9_controller.py`，需显式加
`--manager-m0-shadow`。原有控制器仍按既有参数运行，旁路结果写入同一
`phase9_audit.jsonl` 的 `manager_m0_shadow` 记录。
外层 `run_shadow_strategy_online_validation.py` 也接受同名参数并传给每轮
controller；只有显式传入时才启用旁路记录。

## 代码

- `vllm_bridge/vllm/bridge_tp/controller/manager_m0.py`：状态快照、字段缺失与
  时效校验、只读决策、单目标活跃 session 的进程内 registry、审计适配器。
- `vllm_bridge/tools/bridge_tp/replay_manager_m0.py`：离线回放既有审计，输出
  独立 JSONL 与含原始文件 SHA256 的覆盖报告。
- `vllm_bridge/tests/bridge_tp/test_manager_m0.py`：确定性、缺失证据、通道占用与
  回放测试。

## 本地回放

从 `vllm_bridge` 目录运行：

```powershell
.\.venv\Scripts\python.exe tools\bridge_tp\replay_manager_m0.py `
  --audit "<原始 controller\phase9_audit.jsonl>" `
  --out "<新的输出目录\m0_replay.jsonl>"
```

回放仅使用审计文件内可见的时点证据，不读取后来的 receipt 来填补过去。
历史 TODO7／TODO9 多数是诊断固定边界：它们能验证快照解析与 Rate 旁路，
不能当成自动 Start／Commit 的反事实收益证据。旧审计没有完整的 history
resident 字节、rank armed 和请求上下文快照；对应字段保持 `null`。

## 已执行的本地验证

- 标准库单元测试：4 个通过。
- `py_compile`：三个新增／修改的运行模块通过。
- TODO7 4090D／7B 的 O128 formal `r05`：10 个 tick，4 个
  `WOULD_SET_RATE`、6 个 `WOULD_WAIT`。
- TODO9 4090D／7B 的 high `r01`：11 个 tick，全部 `WOULD_WAIT`；
  TP4 p99 TPOT 在该审计窗口缺有效样本，因此没有提议升降速。

回放输出在 `control/m0_validation/`，每份 `.summary.json` 包含输入 SHA、
动作计数和缺失字段计数。原始审计没有修改。

## 下一阶段的明确边界

M0 没有自主 Start／Rate／Commit 策略。其提议来自现有控制器，且历史 trace
缺少部分迁移进度证据。M1 才将 Start 接入执行；M2 校准三档速率；M3 解决
目标请求固定候选边界与 `READY_NOT_COMMITTED` 的关系，再加入收益驱动 Commit。
新服务器实验前须按交接文档重新核对 GPU、HEAD、模型、manifest、输入 SHA、
guard、端口和原始结果路径。
