#!/usr/bin/env bash
# 当前任务：混合负载静态L1A、L2各一场，作自动转换qwen3_auto_20261104_152748_19235的同代码对照。
# 负载mixed_p1_10；保留队列只看prompt，L1A迁移目标只看当前长度；不转换，无决策器。
# cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
set -uo pipefail
scenes='mixed_p1_10:L1A mixed_p1_10:L2'
repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
cd "$repo" || exit 1
source "$venv/bin/activate" || exit 1
[ -z "$(git status --porcelain)" ] || { echo '代码有未提交改动，停止并保留。'; exit 1; }
[ "$(git branch --show-current)" = "$branch" ] || { echo '分支不匹配。'; exit 1; }
git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch" || exit 1
revision=$(git rev-parse HEAD)
[ "$revision" = "$(git rev-parse "origin/$branch")" ] || {
    echo '先执行 git pull --ff-only 再运行。'; exit 1;
}
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" = 4 ] || {
    echo '需要4张GPU。'; exit 1;
}
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
    echo 'GPU上有计算进程，停止，未清理任何进程。'; exit 1;
}
mkdir -p /root/autodl-tmp/bridgetp/results || exit 1
run=/root/autodl-tmp/bridgetp/results/qwen3_static_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
printf '%s\n' "$revision" > "$run/revision.txt"
echo "本批目录：$run"
echo 'A={GPU2,GPU3}，B={GPU0,GPU1}；双32768/.88，四卡。'
echo "场次：$scenes"
echo '请求留在控制器：每引擎未开始8条；只看prompt是否放得下，不看输出长度，也不为增长预留。'
echo '静态布局，不转换；L1A开迁移、短请求分流和过载迁移，迁移目标只看当前长度。'
echo '混合负载：短10+少量长→短1+长0.05/s→短8+两次6条突发，到达共1980秒；长请求16384输入/8192输出。'
echo '依次两场，共约1.5小时，每场控制器上限60分钟；L2短请求会排队较久，属本场要看的代价；不自动重试，不删除原始目录。'
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1
"$py" -m compileall -q -j 0 "$repo/vllm" "$repo/tools" > "$run/compileall.log" 2>&1 || exit 1
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu \
    --format=csv,noheader -lms 1000 > "$run/gpu_samples.csv" &
smi=$!
trap 'kill "$smi" 2>/dev/null || true' EXIT
final=0
excludes=()
for scene in $scenes; do
    scenario=${scene%%:*}
    layout=${scene##*:}
    name="$scenario-$layout"
    extra=()
    static=$layout
    case "$layout" in
        L1A) extra=(--kv-short-exempt --overload-migrate --migrate-current-length) ;;
        auto)
            static=L1A
            extra=(--kv-short-exempt --overload-migrate --migrate-current-length
                --conversion-decider execute --decider-multi --decider-pre-tilt
                --decider-assist --conversion-prestart)
            ;;
    esac
    [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || {
        echo "上一场后GPU仍有计算进程，停止于 $name 之前。"; final=1; break;
    }
    echo "=== 开始：$name（$(date +%T)）==="
    "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
        --phase-shift --phase-scenario "$scenario" --static-layout "$static" \
        --hold-queue "${extra[@]}" \
        --expanded-context --current-machine --repetitions 1 \
        --model "$model" \
        --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
        --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
        --output "$run/$name" --expected-revision "$revision" 2>&1 | tee "$run/$name.console.log"
    status=${PIPESTATUS[0]}
    echo "$name 退出码：$status" | tee -a "$run/exit_codes.txt"
    [ "$status" = 0 ] || final=$status
    # 有有效内层无损包时，外层只收该包与日志，不重复原始目录；原始目录仍保留。
    if (cd "$run" && sha256sum -c "$name.tar.gz.sha256" >/dev/null 2>&1) \
        && tar -tzf "$run/$name.tar.gz" >/dev/null 2>&1; then
        excludes+=(--exclude="$(basename "$run")/$name")
    else
        echo "$name 内层包未通过校验，外层包含原始目录用于诊断。"
    fi
done
kill "$smi" 2>/dev/null || true
wait "$smi" 2>/dev/null || true
tar -C "$(dirname "$run")" "${excludes[@]}" -czf "$run.tar.gz" "$(basename "$run")" || exit 1
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256" || exit 1
cat "$run/exit_codes.txt" 2>/dev/null
echo "总退出码：$final"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$final"
