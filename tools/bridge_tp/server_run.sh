#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：S4 SHADOW规划器GPU检查（1场，约10-12分钟）。
# 布局与负载同G2/S3b-3：L1A，A={2,3}，B={0,1}；TP1@2（迁移源，通道1）、TP1@3（通道3）、
# TP2@0,1；双32768/.88；深长突发MIGRATE臂（短请求每2s一条，90-100s到达6条16384/8192长请求）。
# 与S3b-3不同：控制器启动后110s才触发L1A-L2（A的两个TP1合并成TP2@2,3），此时A上
# 正有长请求在跑。DRAIN等它们自然结束（fast策略原有的迁移照常可以把请求迁出到TP2），
# 排空期间SHADOW规划器每秒给出“迁哪些能缩短排空”的建议，只记录不执行。
# 控制器会一直服务到转换结束才退出（负载跑完也等）。
# 全程每秒采样一次GPU显存与利用率。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_conversion_shadow

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

edge=L1A-L2
echo "=== 开始：长请求在跑时触发 $edge（110s），SHADOW规划器（$(date +%T)）==="
date +%s.%N > "$run/$edge.started_unix_s"
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    --long-burst-memory-verify --layout-l1a-channels --conversion "$edge" \
    --conversion-trigger-s 110 \
    --current-machine --repetitions 1 \
    --model "$model" \
    --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --output "$run/$edge" --expected-revision "$head" 2>&1 | tee "$run/$edge.console.log"
status=${PIPESTATUS[0]}
echo "$edge 退出码：$status" | tee "$run/exit_codes.txt"

kill "$smi" 2>/dev/null
# 运行包已是compact包；外层只收它和日志，不重复原始目录。
tar -C "$(dirname "$run")" -czf "$run.tar.gz" \
    --exclude="$(basename "$run")/$edge" \
    "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo "最终退出码：$status"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
