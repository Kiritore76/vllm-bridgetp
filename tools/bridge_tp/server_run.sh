#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：入口优化后的高速率短请求扫描（不转换）。上次扫描控制器入口在约11 req/s
# 成为瓶颈（每条到达都把整个历史账本写两遍进审计）；本版只记录仍在跑的条目。
# 本次看：①名义速率是否真正到达（派发滞后应接近0）；②L0（4×TP1）的真实吞吐上限。
# A={2,3}，B={0,1}；双32768/.88；每个引擎max-num-seqs 16（L0共64并发，L2共32）。
# 负载：512输入/128输出（固定输出）的短请求，12、16、20、24 req/s各持续60s，然后排空。
# L0、L2各4场；L1A另加8 req/s共5场（新的排队感知路由：首选实例在排队而另一个排队更少、
# 且提示放得下、KV未告警时，改发给另一个）。共13场，约65分钟。
# 一场失败不中断其余布局；只有GPU未空闲时才停止。全程每秒采样GPU显存与利用率。
# 每格只有1场，只作初筛。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_rate_sweep_hi
layouts="L0 L1A L2"

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

final=0
for layout in $layouts; do
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
        echo "上一场后GPU仍有计算进程，停止于 $layout 之前。"; final=1; break;
    }
    name="ratesweep-$layout"
    rates=12,16,20,24
    [ "$layout" = L1A ] && rates=8,12,16,20,24
    echo "=== 开始：$name，速率 $rates（$(date +%T)）==="
    date +%s.%N > "$run/$name.started_unix_s"
    "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
        --rate-sweep --sweep-rates "$rates" \
        --expanded-context --static-layout "$layout" \
        --current-machine --repetitions 1 \
        --model "$model" \
        --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
        --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
        --output "$run/$name" --expected-revision "$head" 2>&1 | tee "$run/$name.console.log"
    status=${PIPESTATUS[0]}
    echo "$name 退出码：$status" | tee -a "$run/exit_codes.txt"
    [ "$status" = 0 ] || final=$status
done

kill "$smi" 2>/dev/null
# 每个布局的运行包已是compact包；外层只收它们和日志，不重复原始目录。
excludes=()
for layout in $layouts; do
    excludes+=(--exclude="$(basename "$run")/ratesweep-$layout")
done
tar -C "$(dirname "$run")" -czf "$run.tar.gz" "${excludes[@]}" "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo '=== 退出码 ==='
cat "$run/exit_codes.txt"
echo "最终退出码：$final"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$final"
