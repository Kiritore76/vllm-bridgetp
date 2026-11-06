#!/usr/bin/env bash
# 当前任务：六卡主机上，GPU4 TP1到GPU0/1 TP2显式迁移一场。
set -uo pipefail
repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/tp2-4090d-qwen3-8b
cd "$repo" || exit 1
[ -x "$py" ] && [ -f "$model/config.json" ] || { echo '虚拟环境或模型不存在，请先核对部署路径。'; exit 1; }
source "$venv/bin/activate" || exit 1
[ -z "$(git status --porcelain)" ] || { echo '代码有未提交改动，停止并保留。'; exit 1; }
[ "$(git branch --show-current)" = "$branch" ] || { echo '分支不匹配。'; exit 1; }
fetched=0
for attempt in $(seq 1 20); do
    if git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch"; then
        fetched=1
        break
    fi
    sleep 3
done
[ "$fetched" = 1 ] || { echo '拿不到远端，停止。'; exit 1; }
revision=$(git rev-parse HEAD)
[ "$revision" = "$(git rev-parse "origin/$branch")" ] || { echo '请先git pull --ff-only。'; exit 1; }
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" = 6 ] || { echo '需要六张GPU。'; exit 1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo 'GPU有计算进程，停止；未清理任何进程。'; exit 1; }
mkdir -p /root/autodl-tmp/bridgetp/results || exit 1
run=/root/autodl-tmp/bridgetp/results/qwen3_six_migration_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
avail=$(df --output=avail -k "$run" | tail -1)
[ "$avail" -ge 8388608 ] || { echo '磁盘可用不足8GB。'; exit 1; }
printf '%s\n' "$revision" > "$run/revision.txt"
echo "本批目录：$run"
echo '仅一场显式迁移：GPU4=TP1源，GPU0/1=TP2目标，GPU2/3/5空闲。'
echo '双32768/.88；2048输入/6144输出，生成达到2048后显式触发，目标无背景。'
echo '复用真实迁移、原生分配和提交/源KV释放检查；不运行压力策略或转换。'
echo '只验证隔离链路，不代表五实例L1AC并发迁移验证；handoff要求仍为1秒，结果单独评价。'
echo '预计约5–10分钟加启动和打包；失败不自动重试，不删除结果。'
pack_result() {
    local status=$1
    printf '%s\n' "$status" > "$run/exit_code.txt"
    local excludes=()
    if (cd "$run" && sha256sum -c trial.tar.gz.sha256 >/dev/null 2>&1) \
        && tar -tzf "$run/trial.tar.gz" >/dev/null 2>&1; then
        excludes+=(--exclude="$(basename "$run")/trial")
    fi
    tar -C "$(dirname "$run")" "${excludes[@]}" -czf "$run.tar.gz" "$(basename "$run")" || return 1
    sha256sum "$run.tar.gz" > "$run.tar.gz.sha256" || return 1
    echo "退出码：$status（0机制完成，其余查看故障证据；1秒handoff标准单独分析）"
    echo "请拿回：$run.tar.gz"
    echo "以及：$run.tar.gz.sha256"
}
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export VLLM_USE_V2_MODEL_RUNNER=0 OMP_NUM_THREADS=1
# multiprocessing/forkserver binds AF_UNIX sockets below the temp directory.
# The timestamped result path is too long for Linux's 108-byte socket path.
# Keep temporary files on the data disk, using a short path for all children.
task_tmp=$(mktemp -d /root/autodl-tmp/bt-tmp.XXXXXX) || { pack_result 1; exit 1; }
printf '%s\n' "$task_tmp" > "$run/temporary_directory.txt"
export TMPDIR="$task_tmp" TEMP="$task_tmp" TMP="$task_tmp"
echo '[CPU前置] 运行本次迁移与通道相关CPU回归检查；不执行GPU计算。'
# 与本地CPU检查相同，导入源码控制逻辑，避免执行vLLM顶层的GPU平台探测。
"$py" -B -c 'import sys,types,unittest; from pathlib import Path; p=types.ModuleType("vllm"); p.__path__=[str(Path.cwd()/"vllm")]; sys.modules["vllm"]=p; s=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:]); r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(not r.wasSuccessful())' \
    tests.bridge_tp.test_migration_smoke_placement \
    tests.bridge_tp.test_same_tp_migration \
    tests.bridge_tp.test_gpu_direct_targets tests.bridge_tp.test_source_channels \
    tests.bridge_tp.test_4090d_qwen3 \
    > "$run/cpu_checks.log" 2>&1
status=$?
if [ "$status" != 0 ]; then
    echo '[CPU前置] 未通过，不启动GPU实验；请拿回诊断包。'
    tail -n 30 "$run/cpu_checks.log"
    pack_result "$status" || exit 1
    exit "$status"
fi
echo '[CPU前置] 通过；准备单场GPU冒烟。'
echo '[预编译] 数据盘短临时路径；最多4个编译进程。'
"$py" -m compileall -q -j 4 "$repo/vllm" "$repo/tools" > "$run/compileall.log" 2>&1 || {
    echo '[预编译] 失败，尚未启动GPU服务；请拿回诊断包。'
    tail -n 30 "$run/compileall.log"
    pack_result 1
    exit 1
}
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { pack_result 1; exit 1; }
# 先确认本机支持查询字段；失败时在GPU实验之前停止。
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader > "$run/gpu_before.csv" || { pack_result 1; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current \
    --format=csv,noheader -lms 1000 > "$run/gpu_samples.csv" &
smi=$!
trap 'kill "$smi" 2>/dev/null || true' EXIT
"$py" -B tools/bridge_tp/run_tp2_migration_smoke.py \
    --platform qwen3 --cards 6 --source-gpus 4 --target-gpus 0,1 \
    --expanded-context --history-tokens 2048 --timeout-s 900 --no-archive \
    --model-path "$model" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --out-root "$run/trial" --expected-revision "$revision" 2>&1 | tee "$run/console.log"
status=${PIPESTATUS[0]}
kill "$smi" 2>/dev/null || true
wait "$smi" 2>/dev/null || true
pack_result "$status" || exit 1
exit "$status"
