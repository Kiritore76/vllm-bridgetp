#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：G1 三卡兼容验收（多实例入口首次上GPU）。
# 固定一对：GPU2=TP1源，GPU0/1=TP2目标，GPU3空闲；双32768/.88；与10-30深长突发同配方。
#   1) 深长MIGRATE 1场（--long-burst-memory-verify）：须真实提交/源释放、双rank回执、handoff<=1s
#   2) 深长STAY 1场（--long-burst-deep-stay）：完整完成且无迁移
# 两场各约8-10分钟。任一场技术失败不影响另一场执行；最后统一打包。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_g1_three_card

cd "$repo" || exit 1
source "$venv/bin/activate" || exit 1
[ -z "$(git status --porcelain)" ] || { echo '代码有未提交改动，停止并保留。'; exit 1; }
[ "$(git branch --show-current)" = "$branch" ] || { echo '分支不匹配，停止。'; exit 1; }
git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch" || exit 1
head=$(git rev-parse HEAD)
[ "$head" = "$(git rev-parse "origin/$branch")" ] || {
    echo '本地不是远端最新，先执行 git pull --ff-only 再运行。'; exit 1;
}
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" = 4 ] || {
    echo '需要4张GPU，停止。'; exit 1;
}

run=/root/autodl-tmp/bridgetp/results/${task}_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir -p "$run" || exit 1
echo "$head" > "$run/revision.txt"
echo "代码：$head"
echo "本批目录：$run"

# 增量预编译：只编译改动过的源码，未改的直接跳过（缩短引擎启动）。
"$py" -m compileall -q -j 0 "$repo/vllm" "$repo/tools" >"$run/compileall.log" 2>&1

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1

status=0
for mode in long-burst-memory-verify long-burst-deep-stay; do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
        echo "GPU上有计算进程，未停止任何进程，跳过 $mode。"; status=1; continue;
    }
    echo "=== 开始：$mode ==="
    "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
        "--$mode" --current-machine --repetitions 1 \
        --model /root/autodl-tmp/models/Qwen3-8B \
        --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
        --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
        --output "$run/$mode" --expected-revision "$head" 2>&1 | tee "$run/$mode.console.log"
    code=${PIPESTATUS[0]}
    echo "$mode 退出码：$code" | tee -a "$run/exit_codes.txt"
    [ "$code" = 0 ] || status=1
done

# 两场各自已打成 compact 包；外层只收这些包和日志，不重复原始目录。
tar -C "$(dirname "$run")" -czf "$run.tar.gz" \
    --exclude="$(basename "$run")/long-burst-memory-verify" \
    --exclude="$(basename "$run")/long-burst-deep-stay" \
    "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo '=== 退出码 ==='
cat "$run/exit_codes.txt"
echo "最终退出码：$status"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
