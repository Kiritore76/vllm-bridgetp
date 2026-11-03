#!/usr/bin/env bash
# 当前任务：保留队列下补测布局差距，第1场：L1A merge_deep（共7场，确认后放其余）。
# 负载同qwen3_layout_boundary的merge_deep，用来和已有L2数据比较；不转换，无决策器。
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
run=/root/autodl-tmp/bridgetp/results/qwen3_retest_l1a_merge_deep_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
printf '%s\n' "$revision" > "$run/revision.txt"
echo "本批目录：$run"
echo 'A={GPU2,GPU3}，B={GPU0,GPU1}；双32768/.88，四卡。'
echo '静态L1A，开迁移和短请求分流，不转换，无决策器。'
echo '请求留在控制器：每引擎未开始8条；遥测确认KV不够时继续等待；只有遥测失效时长请求等满300秒才发出。源端持续过载时不等队列清空就迁移长请求。'
echo '负载merge_deep：短0.5条/s+长0.05条/s共420秒，再加60秒收尾。'
echo '长请求16384输入/8192输出；约15分钟，控制器上限20分钟；不自动重试，不删除原始目录。'
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1
"$py" -m compileall -q -j 0 "$repo/vllm" "$repo/tools" > "$run/compileall.log" 2>&1 || exit 1
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv,noheader -lms 1000 > "$run/gpu_samples.csv" &
smi=$!
trap 'kill "$smi" 2>/dev/null || true' EXIT
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    --phase-shift --phase-scenario merge_deep --static-layout L1A \
    --kv-short-exempt --hold-queue --overload-migrate \
    --expanded-context --current-machine --repetitions 1 \
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
echo "退出码：$status"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
