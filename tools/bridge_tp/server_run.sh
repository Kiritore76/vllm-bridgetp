#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：G2 四卡L1A双源通道验收（S1-d2）。
# 布局L1A，A={2,3}，B={0,1}：TP1@GPU2（迁移对源）、TP1@GPU3（第二个迁移源）、TP2@GPU0/1。
# 两个TP1都在启动时预连到TP2（各自端口块与通道代际），全局同时只迁一组。
# 负载：深长突发MIGRATE 1场（--long-burst-memory-verify），与10-30及G1同配方；双32768/.88。
# 约10-12分钟。不训练、不重装；技术失败即停并打包。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_g2_l1a_channels

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
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv > "$run/gpu_memory_before.csv"

# 增量预编译：未改动的源码直接跳过。
"$py" -m compileall -q -j 0 "$repo/vllm" "$repo/tools" >"$run/compileall.log" 2>&1

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1

mode=long-burst-memory-verify
echo "=== 开始：G2（$mode + L1A双源通道）==="
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    "--$mode" --layout-l1a-channels --current-machine --repetitions 1 \
    --model /root/autodl-tmp/models/Qwen3-8B \
    --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --output "$run/g2" --expected-revision "$head" 2>&1 | tee "$run/g2.console.log"
status=${PIPESTATUS[0]}
echo "G2 退出码：$status" | tee "$run/exit_codes.txt"

# 运行包已是 compact 包；外层只收它和日志，不重复原始目录。
tar -C "$(dirname "$run")" -czf "$run.tar.gz" \
    --exclude="$(basename "$run")/g2" \
    "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo "最终退出码：$status"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
