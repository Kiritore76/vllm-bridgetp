#!/usr/bin/env bash
# 当前任务：同TP迁移冒烟第二场，TP2→TP2（GPU2,3→GPU0,1），无背景负载，手动触发。
# 只验证机制，不是负载收益。TP1→TP1的结果回收前不能当作前提已满足。
set -uo pipefail
same_tp=2
repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
cd "$repo" || exit 1
source "$venv/bin/activate" || exit 1
[ -z "$(git status --porcelain)" ] || { echo '代码有未提交改动，停止并保留。'; exit 1; }
[ "$(git branch --show-current)" = "$branch" ] || { echo '分支不匹配。'; exit 1; }
# GitHub认证偶尔被拒（11-04实测约一半），重试；始终拿不到远端就停止。
fetched=0
for attempt in $(seq 1 20); do
    if git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch"; then
        fetched=1
        break
    fi
    sleep 3
done
[ "$fetched" = 1 ] || { echo '20次都拿不到远端，停止。'; exit 1; }
revision=$(git rev-parse HEAD)
[ "$revision" = "$(git rev-parse "origin/$branch")" ] || {
    echo '先执行 git pull --ff-only 再运行。'; exit 1;
}
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" = 4 ] || {
    echo '需要4张GPU。'; exit 1;
}
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
    echo 'GPU上有计算进程，停止，未清理任何进程。'; exit 1;
}
mkdir -p /root/autodl-tmp/bridgetp/results || exit 1
run=/root/autodl-tmp/bridgetp/results/qwen3_same_tp${same_tp}_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
printf '%s\n' "$revision" > "$run/revision.txt"
echo "本批目录：$run"
echo "同TP迁移冒烟：TP${same_tp}→TP${same_tp}，一条2048输入、6144输出的请求，输出到2048个token时手动迁移。"
echo '迁移完成后，同一请求再跑一遍不迁移的版本，逐token对比，不一致判失败。'
echo '约10–15分钟，上限20分钟；不自动重试，不删除原始目录。'
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1
"$py" -m compileall -q -j 0 "$repo/vllm" "$repo/tools" > "$run/compileall.log" 2>&1 || exit 1
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv,noheader -lms 1000 > "$run/gpu_samples.csv" &
smi=$!
trap 'kill "$smi" 2>/dev/null || true' EXIT
"$py" -B tools/bridge_tp/run_tp2_migration_smoke.py \
    --platform qwen3 --same-tp "$same_tp" \
    --model-path "$model" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --out-root "$run/same-tp${same_tp}" \
    --expected-revision "$revision" \
    --history-tokens 2048 --timeout-s 1200 --no-archive 2>&1 | tee "$run/same-tp${same_tp}.console.log"
final=${PIPESTATUS[0]}
echo "same-tp${same_tp} 退出码：$final" | tee -a "$run/exit_codes.txt"
kill "$smi" 2>/dev/null || true
wait "$smi" 2>/dev/null || true
tar -C "$(dirname "$run")" -czf "$run.tar.gz" "$(basename "$run")" || exit 1
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256" || exit 1
cat "$run/same-tp${same_tp}/result.json" 2>/dev/null
echo "总退出码：$final"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$final"
