# Experiment A 第 7 节：四模式 smoke 运行手册

## 要证明什么

这一关只证明实验基础设施可信并且四种机制都能走通：

1. `ALWAYS_TP1`：请求始终由 TP1 完成。
2. `ALWAYS_TP4`：请求始终由 TP4 完成。
3. `STOP_AND_COPY`：在 O64 完成后冻结单个 anchor，源 KV 保留；复制完整前缀并由四 rank 精确回读后，原子切到 TP4；源 KV 的释放必须发生在 `KVCacheManager.free()` 返回之后。
4. `SHADOW_ONLY`：在 O64 开始从请求头部预拷贝历史 KV，同时 TP1 继续生成并传增量；O160 做最终同步，四 rank ready 后直接 `SHADOW→TAKEOVER`。

每个模式运行 3 次，采用三组不同顺序降低热机/漂移偏差。smoke 通过后立即停止，不运行 A1–A5。

## 服务器运行前提

- Linux，GPU0 给 TP1，GPU1–4 给 TP4。
- 当前分支必须干净并固定到本次提交。
- 模型、survival table、冻结 guard 和 TP1/TP4 block 数必须填写真实值。
- 每条命令从 `cd` 和虚拟环境激活开始，不依赖上一终端残留的 export。

## 一键执行

提交推送后，以仓库实际提交号替换 `<EXPECTED_REVISION>`，模型路径若不同也必须修改：

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export MODEL_PATH=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B
export SURVIVAL_TABLE=/root/autodl-tmp/bridgetp/phase9_cap0_inputs/survival_table_m1_v1.json
export GUARD_FILE=/root/autodl-tmp/bridgetp/phase9_cap0_manifests/frozen/guard_free_kv_tokens.txt
export EXPECTED_REVISION=<EXPECTED_REVISION>
export RUN_ID="experiment-a-section7-smoke-$(date -u +%Y%m%dT%H%M%SZ)"
export OUT_ROOT="/root/autodl-tmp/bridgetp/results/experiment_a/${RUN_ID}"
mkdir -p "$(dirname "$OUT_ROOT")"

python tools/bridge_tp/run_experiment_a_four_mode_smoke.py \
  --validate-only \
  --model-path "$MODEL_PATH" \
  --survival-table "$SURVIVAL_TABLE" \
  --guard-file "$GUARD_FILE" \
  --out-root "$OUT_ROOT" \
  --expected-revision "$EXPECTED_REVISION" \
  --expected-survival-sha256 031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a \
  --expected-guard-sha256 0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b \
  --expected-guard 8448 \
  --tp1-blocks 1968 \
  --tp4-blocks 35739
export VALIDATE_RC=$?

if [ "$VALIDATE_RC" -eq 0 ]; then
  set -o pipefail
  python tools/bridge_tp/run_experiment_a_four_mode_smoke.py \
    --model-path "$MODEL_PATH" \
    --survival-table "$SURVIVAL_TABLE" \
    --guard-file "$GUARD_FILE" \
    --out-root "$OUT_ROOT" \
    --expected-revision "$EXPECTED_REVISION" \
    --expected-survival-sha256 031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a \
    --expected-guard-sha256 0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b \
    --expected-guard 8448 \
    --tp1-blocks 1968 \
    --tp4-blocks 35739 \
    2>&1 | tee "${OUT_ROOT}.console.txt"
  export SMOKE_RC=${PIPESTATUS[0]}
else
  export SMOKE_RC=98
fi

echo "VALIDATE_RC=$VALIDATE_RC"
echo "SMOKE_RC=$SMOKE_RC"
echo "OUT_ROOT=$OUT_ROOT"
test "$SMOKE_RC" -eq 0
```

## 通过条件与需要带回的文件

最终必须出现：

```text
EXPERIMENT_A_FOUR_MODE_SMOKE_COMPLETE: ...
SMOKE_RC=0
```

至少带回整个 `OUT_ROOT` 以及旁边的 `.console.txt`。总验收看：

- `acceptance.json`：`status=PASS`、`recorded_rows=12`；
- `four_mode_measurements.csv`：4 模式 × 3 重复；
- 每个迁移 run 的 `controller/timeline.jsonl` 与 `timeline_acceptance.json`；
- Stop-and-Copy 的 `request_frozen_receipt.json` 和 `source_kv_release_receipt.json`；
- 每个 run 的 `process_lifetimes.json`、服务日志和原始 acceptance。

这批数据确认后，才规划并启动 A1–A5。
