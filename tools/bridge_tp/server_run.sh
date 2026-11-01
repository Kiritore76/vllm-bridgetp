#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：转换决策器第一轮——只记录、不执行（一场，约20分钟）。
# 以L1A启动（A={2,3}：TP1@2、TP1@3；B={0,1}：TP2(0,1)），双32768/.88，fast策略照常迁移。
# 分阶段负载：
#   P1 短请求高峰（0-360s）：512/128，前60s 4 req/s，之后9 req/s（约L1A短请求容量的70%）
#      → 预期决策器提议“拆B组TP2”（L1A→L0）；
#   P2 长请求阶段（360-720s）：短请求0.5 req/s + 6144/2048长请求0.4 req/s（约占44%）
#      → 预期提议“合A组TP1”（L1A→L2）；
#   P3 平静尾段（720-840s）：512/128 3 req/s → 不应提议。
# 决策器每秒判断一次，全部写入conversion_decider.jsonl；不会真的转换。
# 规则与初值：窗口120s、持续3分钟（紧急60s）、收益≥3倍代价、同一方向冷却10分钟。
# 全程每秒采样一次GPU显存与利用率。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_decider_record

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

name=phase-shift-record
echo "=== 开始：$name（$(date +%T)）==="
date +%s.%N > "$run/$name.started_unix_s"
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    --phase-shift --static-layout L1A --conversion-decider record \
    --expanded-context --current-machine --repetitions 1 \
    --model "$model" \
    --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --output "$run/$name" --expected-revision "$head" 2>&1 | tee "$run/$name.console.log"
status=${PIPESTATUS[0]}
echo "$name 退出码：$status" | tee -a "$run/exit_codes.txt"

kill "$smi" 2>/dev/null
# 运行包已是compact包；外层只收它和日志，不重复原始目录。
tar -C "$(dirname "$run")" -czf "$run.tar.gz" \
    --exclude="$(basename "$run")/$name" "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo '=== 退出码 ==='
cat "$run/exit_codes.txt"
echo "最终退出码：$status"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
