#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：L1A有没有必要存在（用户11-02：只比L0、L1A、L2），补齐突发负载两行，3场，约35分钟。
# burst_short6 已有L1A（soft 482.4）、L2（402.7），缺L0；burst9_short6 已有L1A（473.5），缺L2、L0。
# 负载：短6 req/s，120s起每2s一条长请求（16384/8192），共6或9条，420s，再加60s收尾。
# 都直接以静态布局启动（A={2,3}，B={0,1}，双32768/.88），无转换、无决策器。
# 看：全部请求完成、退出码0；L1A是否同时胜过L0和L2。L0放不下这么多长请求，预计有长中断。
# 这批之后按结果选：短3 req/s的突发，或 short_mid、split_trickle。
# 磁盘可用不足8GB或GPU未空闲就停；一场失败不中断其余场。
# 每场的无损compact包校验通过后删除该场原始目录，所以磁盘占用不随场数累积；
# 没有得到有效compact包的场保留原始目录。全程每秒采样GPU显存与利用率。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_layout_boundary
# 每场：负载:布局；L1A-stay是同引擎同通道、不执行迁移的对照
scenes="burst_short6:L0 burst9_short6:L2 burst9_short6:L0"

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
excludes=()
for scene in $scenes; do
    scenario=${scene%%:*}
    layout=${scene##*:}
    extra=()
    [ "$layout" = L1A-stay ] && extra=(--static-no-migrate)
    name="$scenario-$layout"
    excludes+=(--exclude="$(basename "$run")/$name")
    avail=$(df --output=avail -k "$run" | tail -1)
    [ "$avail" -ge 8388608 ] || {
        echo "磁盘可用不足8GB，停止于 $name 之前。"; final=1; break;
    }
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
        echo "上一场后GPU仍有计算进程，停止于 $name 之前。"; final=1; break;
    }
    echo "=== 开始：$name（$(date +%T)）==="
    date +%s.%N > "$run/$name.started_unix_s"
    "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
        --phase-shift --phase-scenario "$scenario" --static-layout "${layout%-stay}" "${extra[@]}" \
        --expanded-context --current-machine --repetitions 1 \
        --model "$model" \
        --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
        --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
        --output "$run/$name" --expected-revision "$head" 2>&1 | tee "$run/$name.console.log"
    status=${PIPESTATUS[0]}
    echo "$name 退出码：$status" | tee -a "$run/exit_codes.txt"
    [ "$status" = 0 ] || final=$status
    # 压缩包完整才删原始目录；sidecar哈希和tar目录都要通过。
    if (cd "$run" && sha256sum -c "$name.tar.gz.sha256" >/dev/null 2>&1)             && tar -tzf "$run/$name.tar.gz" >/dev/null 2>&1; then
        rm -rf -- "${run:?}/$name"
        echo "$name 原始目录已删除（compact包已校验）"
    else
        echo "$name 没有有效compact包，保留原始目录"
    fi
done

kill "$smi" 2>/dev/null
# 每场的运行包已是compact包；外层只收它们和日志，不重复原始目录。
tar -C "$(dirname "$run")" -czf "$run.tar.gz" "${excludes[@]}" "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo '=== 退出码 ==='
cat "$run/exit_codes.txt"
echo "最终退出码：$final"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$final"
