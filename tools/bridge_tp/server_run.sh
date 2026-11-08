#!/usr/bin/env bash
# 当前任务：六卡轮换v2，11阶段3900秒，静态L1AC首场正式基线。
set -uo pipefail
repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/unified-scheduler
cd "$repo" || exit 1
[ -x "$py" ] || { echo '找不到现有虚拟环境，请核对路径。'; exit 1; }
source "$venv/bin/activate" || exit 1
[ -z "$(git status --porcelain)" ] || { echo '工作树有改动，停止并保留。'; exit 1; }
[ "$(git branch --show-current)" = "$branch" ] || { echo '分支不匹配，停止。'; exit 1; }
git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch" || exit 1
revision=$(git rev-parse HEAD)
[ "$revision" = "$(git rev-parse "origin/$branch")" ] || { echo '请先git pull --ff-only。'; exit 1; }
mkdir -p /root/autodl-tmp/bridgetp/results || exit 1
run=/root/autodl-tmp/bridgetp/results/qwen3_six_rotation_l1ac_v2_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
printf '%s\n' "$revision" > "$run/revision.txt"
printf 'Six-card rotation v2: static L1AC first baseline only; no conversion or retry.\n' > "$run/scope.txt"
printf '3900s, seed6, 11 phases, 36371 requests (36257 short/114 long), 5574784 output tokens.\n' >> "$run/scope.txt"
printf 'INPUT_SHA256=128e6a69f8f9b30d6b38edc7a4b2533d0b22331ab5bc01decaef27a5b1c6e1ec\n' >> "$run/scope.txt"
printf '5400s controller budget; 32K/.88/slots16/guard125/unified mean32ms; original policy thresholds.\n' >> "$run/scope.txt"
printf 'Future static L2C/L3 and auto share this registered input; they are NOT run by this task.\n' >> "$run/scope.txt"
printf 'Zero migration is valid baseline coverage; SLO failures, dispatch/CPU limits and drain remain visible.\n' >> "$run/scope.txt"
echo "本批目录：$run"
echo 'Linux完整CPU前置失败/错误/跳过即停止；通过后仅运行静态L1AC首场。'
echo '65分钟到达，另有启动、排空和打包；不自动重试、不删结果、不清理其他进程。'
smi=
pack_result() {
    local status=$1
    if [ -n "$smi" ]; then kill "$smi" 2>/dev/null || true; wait "$smi" 2>/dev/null || true; smi=; fi
    printf '%s\n' "$status" > "$run/exit_code.txt"
    local excludes=() edge
    for edge in L1AC; do
        # Never omit unarchived failure data. Keep its raw trial if no inner pack.
        if [ -f "$run/$edge/trial.tar.gz" ] && [ -f "$run/$edge/trial.tar.gz.sha256" ] &&
           (cd "$run/$edge" && sha256sum -c trial.tar.gz.sha256 > /dev/null 2>&1); then
            excludes+=(--exclude="$(basename "$run")/$edge/trial")
        fi
    done
    tar "${excludes[@]}" -C "$(dirname "$run")" -czf "$run.tar.gz" "$(basename "$run")" || return 1
    sha256sum "$run.tar.gz" > "$run.tar.gz.sha256" || return 1
    echo "请拿回：$run.tar.gz"
    echo "以及：$run.tar.gz.sha256"
}
trap 'pack_result 130; exit 130' INT TERM
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 VLLM_USE_V2_MODEL_RUNNER=0
task_tmp=$(mktemp -d /root/autodl-tmp/bt-tmp.XXXXXX) || { pack_result 1; exit 1; }
printf '%s\n' "$task_tmp" > "$run/temporary_directory.txt"
export TMPDIR="$task_tmp" TEMP="$task_tmp" TMP="$task_tmp"
"$py" -B -c 'import json,sys,types,unittest; from pathlib import Path; p=types.ModuleType("vllm"); p.__path__=[str(Path.cwd()/"vllm")]; sys.modules["vllm"]=p; s=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[2:]); r=unittest.TextTestRunner(verbosity=2).run(s); ok=r.wasSuccessful() and not r.skipped; Path(sys.argv[1]).write_text(json.dumps(dict(scope="CPU only; no CUDA/NCCL", tests=r.testsRun, passed=ok, failures=[t.id() for t,_ in r.failures], errors=[t.id() for t,_ in r.errors], skipped=[t.id() for t,_ in r.skipped]),indent=2)+"\n"); raise SystemExit(not ok)' \
    "$run/cpu_summary.json" \
    tests.bridge_tp.test_six_conversion tests.bridge_tp.test_conversion_channels \
    tests.bridge_tp.test_multi_target tests.bridge_tp.test_pressure_sessions \
    tests.bridge_tp.test_migration_receivers tests.bridge_tp.test_six_card_gate \
    tests.bridge_tp.test_six_card_decider tests.bridge_tp.test_conversion_decider \
    tests.bridge_tp.test_conversion_sequence tests.bridge_tp.test_conversion_safety \
    tests.bridge_tp.test_conversion_planner tests.bridge_tp.test_unified_scheduler \
    tests.bridge_tp.test_pressure_conversion tests.bridge_tp.test_runtime_instance_ids \
    tests.bridge_tp.test_migration_plan tests.bridge_tp.test_qwen3_six_card \
    tests.bridge_tp.test_pressure_launcher \
    tests.bridge_tp.test_source_channels tests.bridge_tp.test_channel_runtime \
    tests.bridge_tp.test_tcp_ports tests.bridge_tp.test_gpu_direct_history_lifecycle \
    tests.bridge_tp.test_six_split_probe \
    tests.bridge_tp.test_pressure_cancel tests.bridge_tp.test_unified_cancel \
    tests.bridge_tp.test_six_edge_batch tests.bridge_tp.test_six_single_class \
    tests.bridge_tp.test_first_target_startup tests.bridge_tp.test_gpu_direct_targets \
    tests.bridge_tp.test_six_rotation \
    > "$run/cpu_checks.log" 2>&1
status=$?
tail -n 12 "$run/cpu_checks.log"
if [ -f "$run/cpu_summary.json" ]; then
    cat "$run/cpu_summary.json"
fi
[ "$status" = 0 ] || { pack_result "$status"; exit "$status"; }
"$py" -B -c 'import json,sys; from pathlib import Path; from tools.bridge_tp.pressure_cpu import load; Path(sys.argv[1]).write_text(json.dumps(load("tcp_ports").environment_snapshot(),indent=2)+"\n")' "$run/port_environment.json" || { pack_result 1; exit 1; }
[ -f "$model/config.json" ] || { echo '找不到Qwen3-8B模型。'; pack_result 1; exit 1; }
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" = 6 ] || { echo '需要六卡。'; pack_result 1; exit 1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo 'GPU忙，未清理其他进程。'; pack_result 1; exit 1; }
avail=$(df --output=avail -k "$run" | tail -1)
[ "$avail" -ge 33554432 ] || { echo '单场长时数据至少预留32GiB，当前不足；未启动GPU。'; pack_result 1; exit 1; }
"$py" -m compileall -q -j 4 "$repo/vllm" "$repo/tools" > "$run/compileall.log" 2>&1 || { tail -n 30 "$run/compileall.log"; pack_result 1; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader > "$run/gpu_before.csv" || { pack_result 1; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader -lms 1000 > "$run/gpu_samples.csv" &
smi=$!
"$py" -B -m tools.bridge_tp.six_rotation plan --directory "$run" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    > "$run/plan.log" 2>&1 || { cat "$run/plan.log"; pack_result 1; exit 1; }
cat "$run/plan.log"
case_dir="$run/L1AC"
mkdir "$case_dir" || { pack_result 1; exit 1; }
echo '开始静态L1AC：11阶段3900秒，保留迁移，整场不转换；先回收此正式基线再决定后续三组。'
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    --cards 6 --static-layout L1AC --phase-shift --phase-scenario mixed_rotation_six_v2 \
    --scheduler unified --hold-queue --kv-short-exempt --overload-migrate --migrate-current-length \
    --expanded-context --current-machine --repetitions 1 \
    --model "$model" \
    --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --output "$case_dir/trial" --expected-revision "$revision" 2>&1 | tee "$case_dir/console.log"
runner_status=${PIPESTATUS[0]}
printf '%s\n' "$runner_status" > "$case_dir/runner_exit_code.txt"
"$py" -B -m tools.bridge_tp.six_rotation assess --directory "$case_dir/trial" \
    > "$case_dir/baseline_check.log" 2>&1
check_status=$?
cat "$case_dir/baseline_check.log"
printf '%s\n' "$check_status" > "$case_dir/check_exit_code.txt"
status=$runner_status
if [ "$status" = 0 ]; then status=$check_status; fi
echo '本场只验完整基线/收尾；SLO、派发和CPU表现回收后分析，不把退出0称为性能达标。'
pack_result "$status" || exit 1
exit "$status"
