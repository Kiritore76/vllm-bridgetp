#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：S5b GPU检查——转换后新实例加入迁移（通道对账）。两场，每场约8-10分钟。
#   L0-L1A：以L0启动（四个TP1都带迁移源配置，目标列表为空），30s时把B组TP1@0、TP1@1
#           合成TP2(0,1)；新TP2接替迁移对目标（端口、会话目录不变，此时才启动stager），
#           保留的TP1@2、TP1@3在运行中连上它。
#   L2-L1A：以L2启动（保留的TP2(0,1)带接收端配置，通道计划为空，stager从开始就在），
#           30s时把A组TP2(2,3)拆成TP1@2、TP1@3；两个新TP1作为迁移源连上TP2(0,1)。
# 两场都是：退役实例的通道先关；新布局的每条TP1→TP2通道两端都留下预连回执后才发布路由。
# A={2,3}，B={0,1}；双32768/.88；深长突发MIGRATE臂（短请求每2s一条，90-100s到达6条
# 16384/8192长请求），这些长请求落在转换后的布局上，看能否真实迁往TP2。
# 全程每秒采样一次GPU显存与利用率。一场技术失败（退出码不是0或2）就停。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_conversion_join
edges="L0-L1A L2-L1A"

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
for edge in $edges; do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
        echo "上一场后GPU仍有计算进程，停止于 $edge 之前。"; status=1; break;
    }
    echo "=== 开始：$edge 30s（$(date +%T)）==="
    date +%s.%N > "$run/$edge.started_unix_s"
    "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
        --long-burst-memory-verify --conversion "$edge" \
        --conversion-trigger-s 30 \
        --current-machine --repetitions 1 \
        --model "$model" \
        --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
        --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
        --output "$run/$edge" --expected-revision "$head" 2>&1 | tee "$run/$edge.console.log"
    status=${PIPESTATUS[0]}
    echo "$edge 退出码：$status" | tee -a "$run/exit_codes.txt"
    [ "$status" = 0 ] || [ "$status" = 2 ] || break
done

kill "$smi" 2>/dev/null
# 每场的运行包已是compact包；外层只收它们和日志，不重复原始目录。
excludes=()
for edge in $edges; do
    excludes+=(--exclude="$(basename "$run")/$edge")
done
tar -C "$(dirname "$run")" -czf "$run.tar.gz" "${excludes[@]}" "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo '=== 退出码 ==='
cat "$run/exit_codes.txt"
echo "最终退出码：$status"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
