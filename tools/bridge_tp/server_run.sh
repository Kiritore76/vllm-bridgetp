#!/usr/bin/env bash
# 当前任务：仅一场显式四次转换验证；不是自动决策/性能对比。
# cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
set -uo pipefail
repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
cd "$repo" || exit 1
source "$venv/bin/activate" || exit 1
[ -z "$(git status --porcelain)" ] || { echo '代码有未提交改动，停止并保留。'; exit 1; }
[ "$(git branch --show-current)" = "$branch" ] || { echo '分支不匹配。'; exit 1; }
git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch" || exit 1
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
run=/root/autodl-tmp/bridgetp/results/qwen3_conversion_cycle_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
avail=$(df --output=avail -k "$run" | tail -1)
[ "$avail" -ge 8388608 ] || { echo '磁盘可用不足8GB。'; exit 1; }
printf '%s\n' "$revision" > "$run/revision.txt"
echo "本批目录：$run"
echo 'A={GPU2,GPU3}，B={GPU0,GPU1}；双32768/.88，四卡。'
echo 'L1A→L2→L1A→L0→L1A；计划触发60/180/480/720秒，串行等待完成。'
echo '1080秒短请求背景，两批各6条16384输入/8192输出长请求。'
echo '保持原分流/迁移策略；显式转换不使用新决策规则。'
echo '约18–25分钟，阻塞时可能更长；不自动重试，不删除原始目录。'
echo '返回L1A后的两个窗口均须真实迁移提交；零迁移记覆盖不足。'
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1
"$py" -m compileall -q -j 0 "$repo/vllm" "$repo/tools" > "$run/compileall.log" 2>&1 || exit 1
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv,noheader -lms 1000 > "$run/gpu_samples.csv" &
smi=$!
trap 'kill "$smi" 2>/dev/null || true' EXIT
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    --conversion-cycle --current-machine --repetitions 1 \
    --model "$model" \
    --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --output "$run/trial" --expected-revision "$revision" 2>&1 | tee "$run/console.log"
status=${PIPESTATUS[0]}
printf '%s\n' "$status" > "$run/exit_code.txt"
kill "$smi" 2>/dev/null || true
wait "$smi" 2>/dev/null || true
# 有有效内层无损包时，外层只收该包与日志，不重复原始trial；原始目录仍保留。
excludes=()
if (cd "$run" && sha256sum -c trial.tar.gz.sha256 >/dev/null 2>&1) \
    && tar -tzf "$run/trial.tar.gz" >/dev/null 2>&1; then
    excludes+=(--exclude="$(basename "$run")/trial")
else
    echo '内层包未通过校验，外层包含原始目录用于诊断。'
fi
tar -C "$(dirname "$run")" "${excludes[@]}" -czf "$run.tar.gz" "$(basename "$run")" || exit 1
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256" || exit 1
echo "退出码：$status（0通过，2覆盖不足，其余查看故障证据）"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
