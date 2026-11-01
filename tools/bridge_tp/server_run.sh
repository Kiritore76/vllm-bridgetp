#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：四卡静态布局初筛（不转换）。看不同负载下三种布局是否各有优势。
# A={2,3}，B={0,1}；双32768/.88。直接以布局启动：
#   L0 ：TP1@0、TP1@1、TP1@2、TP1@3（不能迁移）
#   L1A：TP1@2、TP1@3、TP2@0,1（两个TP1都有到TP2的通道，fast策略照常迁移）
#   L2 ：TP2@2,3、TP2@0,1（不能迁移）
# 负载（与之前实验相同，固定输出）：
#   routine：routine_mixed、moderate_decode、transient_burst（上下文都在8K以内）
#   longburst：deep_long_burst（短请求每2s一条，90-100s到达6条16384/8192长请求）
# 每个布局：routine 3场 + longburst 1场；共12场，约1.5-2小时。
# 一场失败不中断其余布局；只有GPU未空闲时才停止。全程每秒采样GPU显存与利用率。
# 每格只有1场，只作初筛，不是正式布局比较（正式的放六卡）。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_static_layout_screen

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
for family in routine longburst; do
    for layout in L0 L1A L2; do
        [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
            echo "上一场后GPU仍有计算进程，停止于 $family-$layout 之前。"; final=1; break 2;
        }
        if [ "$family" = routine ]; then
            mode=(--routine-four-mode --expanded-context)
        else
            mode=(--long-burst-pair)
        fi
        name="$family-$layout"
        echo "=== 开始：$name（$(date +%T)）==="
        date +%s.%N > "$run/$name.started_unix_s"
        "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
            "${mode[@]}" --static-layout "$layout" \
            --current-machine --repetitions 1 \
            --model "$model" \
            --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
            --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
            --output "$run/$name" --expected-revision "$head" 2>&1 | tee "$run/$name.console.log"
        status=${PIPESTATUS[0]}
        echo "$name 退出码：$status" | tee -a "$run/exit_codes.txt"
        [ "$status" = 0 ] || final=$status
    done
done

kill "$smi" 2>/dev/null
# 每场的运行包已是compact包；外层只收它们和日志，不重复原始目录。
excludes=()
for family in routine longburst; do
    for layout in L0 L1A L2; do
        excludes+=(--exclude="$(basename "$run")/$family-$layout")
    done
done
tar -C "$(dirname "$run")" -czf "$run.tar.gz" "${excludes[@]}" "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo '=== 退出码 ==='
cat "$run/exit_codes.txt"
echo "最终退出码：$final"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$final"
