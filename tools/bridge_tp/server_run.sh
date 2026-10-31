#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：S3a 四边带载DRAIN（生命周期测试夹具，不接fast控制器、不迁移）。
# A={2,3}，B={0,1}；所有引擎32768/.88；边顺序 L0->L1A、L1A->L2、L2->L1A、L1A->L0。
# 每条边：0s起每4s一条请求，持续120s（30条），prompt 2048、输出固定4096 token，
# 请求轮流发到当时ACTIVE的实例；60s时触发转换：被转换bundle停止接新请求，
# 等在途请求自然跑完后释放、重建、恢复路由。全程每秒采样一次GPU显存与利用率。
# 每条边约5-7分钟，总计约25分钟。任一边技术失败即停，已完成部分照常打包。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_loaded_drain

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

"$py" -m compileall -q -j 0 "$repo/vllm" "$repo/tools" >"$run/compileall.log" 2>&1

# GPU显存/利用率，每秒一次（时间戳为nvidia-smi本地时间）。
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv,noheader -lms 1000 > "$run/gpu_samples.csv" &
smi=$!
trap 'kill "$smi" 2>/dev/null' EXIT

export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1

status=0
for edge in L0-L1A L1A-L2 L2-L1A L1A-L0; do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
        echo "上一场后GPU仍有计算进程，停止于 $edge 之前。"; status=1; break;
    }
    echo "=== 开始带载转换：$edge（$(date +%T)）==="
    date +%s.%N > "$run/$edge.started_unix_s"
    "$py" -B tools/bridge_tp/run_tp_bundle_drain.py \
        --edge "$edge" --platform qwen3 --expanded-context \
        --bundle-gpus 2,3,0,1 \
        --model-path "$model" \
        --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
        --expected-revision "$head" \
        --out-root "$run/loaded-$edge" \
        --trigger-s 60 --arrival-duration-s 120 --arrival-interval-s 4 \
        --output-tokens 4096 \
        --plan-timeout-s 600 --request-timeout-s 900 \
        --base-port 8700 --no-archive 2>&1 | tee "$run/$edge.console.log"
    status=${PIPESTATUS[0]}
    echo "$edge 退出码：$status" | tee -a "$run/exit_codes.txt"
    [ "$status" = 0 ] || break
done

kill "$smi" 2>/dev/null
tar -C "$(dirname "$run")" -czf "$run.tar.gz" "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo '=== 退出码 ==='
cat "$run/exit_codes.txt"
echo "最终退出码：$status"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
