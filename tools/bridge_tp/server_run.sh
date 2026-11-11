#!/usr/bin/env bash
# 当前任务：两卡机器上的vLLM Sleep Mode（level 1）往返小实验：两个TP1休眠↔TP2，纯vLLM，不接BridgeTP；参数同生产（bf16/32K/0.88/eager/16槽/2048/块16/无前缀缓存）。
set -uo pipefail
repo=${REPO:-/root/autodl-tmp/bridgetp/vllm_bridge}
venv=${VENV:-/root/autodl-tmp/bridgetp/.venv_bridge}
py=$venv/bin/python
model=${MODEL:-/root/autodl-tmp/models/Qwen3-8B}
results=${RESULTS:-/root/autodl-tmp/bridgetp/results}
branch=bridgetp/drain-recompute
cd "$repo" || exit 1
[ -x "$py" ] || { echo "找不到虚拟环境：$py（可用 VENV=... 覆盖）"; exit 1; }
source "$venv/bin/activate" || exit 1
[ -z "$(git status --porcelain)" ] || { echo '工作树有改动，停止并保留。'; exit 1; }
[ "$(git branch --show-current)" = "$branch" ] || { echo '分支不匹配，停止。'; exit 1; }
git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch" || exit 1
revision=$(git rev-parse HEAD)
[ "$revision" = "$(git rev-parse "origin/$branch")" ] || { echo '请先git pull --ff-only。'; exit 1; }
[ -f "$model/config.json" ] || { echo "找不到模型：$model（可用 MODEL=... 覆盖）"; exit 1; }
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" = 2 ] || { echo '本实验需要恰好两张GPU。'; exit 1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo 'GPU忙，未清理其他进程。'; exit 1; }
mkdir -p "$results" || exit 1
run=$results/qwen3_sleep_probe_2gpu_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
printf '%s\n' "$revision" > "$run/revision.txt"
printf 'vLLM Sleep Mode level-1 round trip on two GPUs: two TP1 asleep <-> one TP2, plain vLLM without the BridgeTP connector, production serving flags plus --enable-sleep-mode; timings, residual GPU memory, host memory, KV size and greedy output equality; no BridgeTP performance claim.\n' > "$run/scope.txt"
echo "本批目录：$run"
echo '流程：两个TP1冷启动→休眠→TP2在旁冷启动（0.88起不来则依次降到0.80）→3轮往返（TP2休眠/TP1唤醒/TP1休眠/TP2唤醒）→全部停止后TP2单独冷启动对照。'
echo '预计15–30分钟（含三次冷启动）；level 1会把每个TP1约15.3GiB、TP2每卡约7.6GiB权重放进主机pinned内存。'
{
    nvidia-smi
    nvidia-smi --query-gpu=index,name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv
    free -g
    nproc
    "$py" -c 'import torch, vllm; print("torch", torch.__version__, "cuda", torch.version.cuda); print("vllm", vllm.__version__)'
} > "$run/environment.txt" 2>&1
cat "$run/environment.txt" | tail -n 12
smi=
pack_result() {
    local status=$1
    if [ -n "$smi" ]; then kill "$smi" 2>/dev/null || true; wait "$smi" 2>/dev/null || true; smi=; fi
    printf '%s\n' "$status" > "$run/exit_code.txt"
    tar -C "$(dirname "$run")" -czf "$run.tar.gz" "$(basename "$run")" || return 1
    sha256sum "$run.tar.gz" > "$run.tar.gz.sha256" || return 1
    echo "请拿回：$run.tar.gz"
    echo "以及：$run.tar.gz.sha256"
}
trap 'pack_result 130; exit 130' INT TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 VLLM_USE_V2_MODEL_RUNNER=0
task_tmp=$(mktemp -d /root/autodl-tmp/bt-tmp.XXXXXX 2>/dev/null || mktemp -d) || { pack_result 1; exit 1; }
printf '%s\n' "$task_tmp" > "$run/temporary_directory.txt"
export TMPDIR="$task_tmp" TEMP="$task_tmp" TMP="$task_tmp"
"$py" -B -m py_compile tools/bridge_tp/sleep_mode_two_gpu_probe.py || { pack_result 1; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader -lms 500 > "$run/gpu_samples.csv" &
smi=$!
timeout 3600 "$py" -B tools/bridge_tp/sleep_mode_two_gpu_probe.py \
    --model "$model" --python "$py" --output "$run/probe" --rounds 3 2>&1 | tee "$run/console.log"
status=${PIPESTATUS[0]}
free -g > "$run/free_after.txt" 2>&1
echo '结果见probe/probe_result.json：各步耗时、每卡显存、主机内存与进程RSS、KV大小、唤醒后贪心输出是否与冷启动一致。'
pack_result "$status" || exit 1
exit "$status"
