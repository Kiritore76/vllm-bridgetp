#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：决策器真正执行转换（每场最多一次），看短时倾斜有没有优势。约2–2.5小时。
# 都从L1A启动（A={2,3}：TP1@2、TP1@3；B={0,1}：TP2(0,1)），双32768/.88，fast策略照常迁移。
# 四种负载，每种两场，只差倾斜：
#   -tilt  ：提议前60s起（接收方空闲时）不再把新的长请求派给要转换的组，最多90s；
#   -notilt：不倾斜。
# 两场都：提议前60s预启动新引擎（闸门挡在碰GPU之前），合并时用辅助迁移。
#   merge_heavy   ：短0.5 + 长0.4 req/s（6144/2048），420s；长请求多于辅助迁移上限。
#   merge_moderate：短0.5 + 长0.25 req/s（6144/2048），420s；TP2在倾斜时还有余量。
#   merge_deep    ：短0.5 + 长0.05 req/s（16384/8192），420s；靠KV压力而不是长请求比例触发。
#   split_trickle ：短9 + 长0.1 req/s（6144/2048），360s；TP2上一直有零星长请求。
# 磁盘可用不足8GB（单场原始数据实测最多约5.4GB）或GPU未空闲就停；一场失败不中断其余场。
# 每场的无损compact包校验通过后删除该场原始目录，所以磁盘占用不随场数累积；
# 没有得到有效compact包的场保留原始目录。全程每秒采样GPU显存与利用率。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_decided_conversion
scenarios="merge_heavy merge_moderate merge_deep split_trickle"

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
for scenario in $scenarios; do
    for arm in tilt notilt; do
        name="$scenario-$arm"
        excludes+=(--exclude="$(basename "$run")/$name")
        avail=$(df --output=avail -k "$run" | tail -1)
        [ "$avail" -ge 8388608 ] || {
            echo "磁盘可用不足8GB，停止于 $name 之前。"; final=1; break 2;
        }
        [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
            echo "上一场后GPU仍有计算进程，停止于 $name 之前。"; final=1; break 2;
        }
        extra=()
        [ "$arm" = tilt ] && extra=(--decider-pre-tilt)
        echo "=== 开始：$name（$(date +%T)）==="
        date +%s.%N > "$run/$name.started_unix_s"
        "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
            --phase-shift --phase-scenario "$scenario" --static-layout L1A \
            --conversion-decider execute --decider-assist --conversion-prestart \
            "${extra[@]}" \
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
