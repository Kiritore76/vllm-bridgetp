#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：S5a GPU检查，同负载同触发的一对：DRAIN（纯排空）对 ROLL（辅助排空），
# 每场约8-10分钟，共约20分钟。
# 布局与负载同S4：L1A，A={2,3}，B={0,1}；TP1@2、TP1@3、TP2@0,1；双32768/.88；
# 深长突发MIGRATE臂（短请求每2s一条，90-100s到达6条16384/8192长请求）。
# 两场都在控制器启动后110s请求转换L1A-L2，都先经过启动闸门（TP2空闲才停止接入，
# 等满180s后只看KV余量）和安全阀（TP2 KV持续3s压到安全余量就恢复接入）。
#   drain：只等长请求自然结束（fast策略原有的迁移照常）。
#   roll ：另外按方案A辅助迁出长请求（剩余>3倍迁移时间、TP2空闲才迁、每次转换最多2条）。
# 全程每秒采样一次GPU显存与利用率。一场技术失败（退出码不是0或2）就停。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_conversion_assist

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

status=0
for arm in drain roll; do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
        echo "上一场后GPU仍有计算进程，停止于 $arm 之前。"; status=1; break;
    }
    extra=()
    [ "$arm" = roll ] && extra=(--conversion-assist)
    echo "=== 开始：L1A-L2 110s，$arm（$(date +%T)）==="
    date +%s.%N > "$run/$arm.started_unix_s"
    "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
        --long-burst-memory-verify --layout-l1a-channels --conversion L1A-L2 \
        --conversion-trigger-s 110 "${extra[@]}" \
        --current-machine --repetitions 1 \
        --model "$model" \
        --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
        --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
        --output "$run/$arm" --expected-revision "$head" 2>&1 | tee "$run/$arm.console.log"
    status=${PIPESTATUS[0]}
    echo "$arm 退出码：$status" | tee -a "$run/exit_codes.txt"
    [ "$status" = 0 ] || [ "$status" = 2 ] || break
done

kill "$smi" 2>/dev/null
# 每场的运行包已是compact包；外层只收它们和日志，不重复原始目录。
tar -C "$(dirname "$run")" -czf "$run.tar.gz" \
    --exclude="$(basename "$run")/drain" \
    --exclude="$(basename "$run")/roll" \
    "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo '=== 退出码 ==='
cat "$run/exit_codes.txt"
echo "最终退出码：$status"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
