#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：加快布局转换的两项改动的GPU检查，两场，约30分钟。
#   prestart-roll：同S5a第6次（L1A，110s转换L1A-L2，辅助排空ROLL，深长突发），另开预启动：
#     排空一开始就启动新TP2进程（import、起服务、起工作进程），工作进程在碰GPU之前停在
#     启动闸门上，旧引擎释放GPU后才放行。对照S5a第6次：停止接入→恢复路由79.6s、重建约37s。
#   tilt-record：同上次决策器只记录的分阶段负载（L1A，P1短请求高峰、P2长请求、P3平静），
#     另开路由预倾斜：决策器某个候选一满足，就不再把新的长请求派给那一组（短请求照常），
#     候选不满足30s后恢复。决策器仍只记录、不转换。看提议时那一组上剩下的长请求是否明显变少。
# A={2,3}，B={0,1}；双32768/.88。全程每秒采样一次GPU显存与利用率。
# 一场技术失败（退出码不是0或2）就停。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_fast_conversion

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
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
    echo 'GPU上有计算进程，未停止任何进程，请先确认。'; exit 1;
}

run=/root/autodl-tmp/bridgetp/results/${task}_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir -p "$run" || exit 1
echo "$head" > "$run/revision.txt"
echo "代码：$head"
echo "本批目录：$run"

# 增量预编译：未改动的源码直接跳过。
"$py" -m compileall -q -j 0 "$repo/vllm" "$repo/tools" >"$run/compileall.log" 2>&1

# GPU显存/利用率，每秒一次（时间戳为nvidia-smi本地时间）。
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv,noheader -lms 1000 > "$run/gpu_samples.csv" &
smi=$!
trap 'kill "$smi" 2>/dev/null' EXIT

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1

common=(
    --current-machine --repetitions 1
    --model "$model"
    --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt"
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json"
    --expected-revision "$head"
)
status=0
for name in prestart-roll tilt-record; do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
        echo "上一场后GPU仍有计算进程，停止于 $name 之前。"; status=1; break;
    }
    if [ "$name" = prestart-roll ]; then
        mode=(--long-burst-memory-verify --layout-l1a-channels --conversion L1A-L2
              --conversion-trigger-s 110 --conversion-assist --conversion-prestart)
    else
        mode=(--phase-shift --static-layout L1A --conversion-decider record
              --decider-pre-tilt --expanded-context)
    fi
    echo "=== 开始：$name（$(date +%T)）==="
    date +%s.%N > "$run/$name.started_unix_s"
    "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py "${mode[@]}" "${common[@]}" \
        --output "$run/$name" 2>&1 | tee "$run/$name.console.log"
    status=${PIPESTATUS[0]}
    echo "$name 退出码：$status" | tee -a "$run/exit_codes.txt"
    [ "$status" = 0 ] || [ "$status" = 2 ] || break
done

kill "$smi" 2>/dev/null
# 每场的运行包已是compact包；外层只收它们和日志，不重复原始目录。
tar -C "$(dirname "$run")" -czf "$run.tar.gz" \
    --exclude="$(basename "$run")/prestart-roll" \
    --exclude="$(basename "$run")/tilt-record" \
    "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo '=== 退出码 ==='
cat "$run/exit_codes.txt"
echo "最终退出码：$status"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
