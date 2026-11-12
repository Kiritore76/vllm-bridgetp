#!/usr/bin/env bash
# 当前任务：standby加重挂起复现——4轮A合并/拆回；每次合并前2条14336/8192长请求固定到A-g1-tp1（保证在唤醒的TP2上重算），合并前5秒到后70秒每秒14条短请求形成积压；带零依赖挂起诊断；机制与稳定性核对，不是性能结论。
set -uo pipefail
repo=${REPO:-/root/autodl-tmp/bridgetp/vllm_bridge}
venv=${VENV:-/root/autodl-tmp/bridgetp/.venv_bridge}
py=$venv/bin/python
model=${MODEL:-/root/autodl-tmp/models/Qwen3-8B}
results=${RESULTS:-/root/autodl-tmp/bridgetp/results}
branch=bridgetp/sleep-reuse
cd "$repo" || exit 1
[ -x "$py" ] || { echo "找不到虚拟环境：$py（可用 VENV=... 覆盖）"; exit 1; }
source "$venv/bin/activate" || exit 1
[ -z "$(git status --porcelain)" ] || { echo '工作树有改动，停止并保留。'; exit 1; }
[ "$(git branch --show-current)" = "$branch" ] || { echo "分支不匹配（需要 $branch），停止。"; exit 1; }
git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch" || exit 1
revision=$(git rev-parse HEAD)
[ "$revision" = "$(git rev-parse "origin/$branch")" ] || { echo '请先git pull --ff-only。'; exit 1; }
mkdir -p "$results" || exit 1
run=$results/qwen3_six_standby_heavy_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
printf '%s\n' "$revision" > "$run/revision.txt"
printf 'Six-card engine-reuse hang reproduction, heavy (the 11-12 stress run left each woken TP2 nearly idle): --conversion-engine-reuse standby, four explicit A merge/split rounds (merge at 110+400k s, split 280 s later); before every merge two 14336/8192 long requests pinned to the pair source A-g1-tp1, still running at the 45 s drain-recompute check, so both are recomputed on the woken TP2; 14 short requests per second from 5 s before to 70 s after every merge (1 per second otherwise) so a backlog waits for the woken TP2. Same hang diagnostics (faulthandler on SIGUSR1, hang_watch kernel stacks and nvidia-smi, NCCL_DEBUG=WARN). Mechanism and stability check only, no performance claim.\n' > "$run/scope.txt"
echo "本批目录：$run"
echo '先Linux CPU检查；通过后只跑一场standby加重复现：4轮A合并→拆回；每次合并前2条14336/8192长请求固定在A-g1-tp1，合并前后每秒14条短请求，唤醒的TP2一接流就要重算2条长请求并承接积压。'
echo '合并时刻110/510/910/1310秒，拆回在合并后280秒；共8步，全部是唤醒。挂起时hang_watch会抓各进程线程栈（写进各自日志）、内核栈和nvidia-smi到hang_diag/。'
echo '共9个引擎进程，启动约3分钟；场景约29分钟（到达1650秒）加排空，总计约40–50分钟，加CPU检查和打包。不自动重试。'
smi=
pack_result() {
    local status=$1
    if [ -n "$smi" ]; then kill "$smi" 2>/dev/null || true; wait "$smi" 2>/dev/null || true; smi=; fi
    if [ -n "${watch:-}" ]; then kill "$watch" 2>/dev/null || true; wait "$watch" 2>/dev/null || true; watch=; fi
    free -g > "$run/free_after.txt" 2>&1
    printf '%s\n' "$status" > "$run/exit_code.txt"
    local excludes=() edge
    for edge in sleep_smoke; do
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
# Hang diagnostics: faulthandler stacks on SIGUSR1 in every Python process.
export PY_STACK_DUMP_ON_USR1=1 NCCL_DEBUG=WARN
export PYTHONPATH="$repo/tools/bridge_tp/stack_dump${PYTHONPATH:+:$PYTHONPATH}"
task_tmp=$(mktemp -d /root/autodl-tmp/bt-tmp.XXXXXX 2>/dev/null || mktemp -d) || { pack_result 1; exit 1; }
printf '%s\n' "$task_tmp" > "$run/temporary_directory.txt"
export TMPDIR="$task_tmp" TEMP="$task_tmp" TMP="$task_tmp"
"$py" -B -c 'import json,sys,types,unittest; from pathlib import Path; p=types.ModuleType("vllm"); p.__path__=[str(Path.cwd()/"vllm")]; sys.modules["vllm"]=p; s=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[2:]); r=unittest.TextTestRunner(verbosity=2).run(s); ok=r.wasSuccessful() and not r.skipped; Path(sys.argv[1]).write_text(json.dumps(dict(scope="CPU only; no CUDA/NCCL", tests=r.testsRun, passed=ok, failures=[t.id() for t,_ in r.failures], errors=[t.id() for t,_ in r.errors], skipped=[t.id() for t,_ in r.skipped]),indent=2)+"\n"); raise SystemExit(not ok)' \
    "$run/cpu_summary.json" \
    tests.bridge_tp.test_six_sleep_reuse tests.bridge_tp.test_six_sleep_smoke \
    tests.bridge_tp.test_six_standby_launch \
    tests.bridge_tp.test_hang_diagnostics \
    tests.bridge_tp.test_tp2_reconfiguration tests.bridge_tp.test_pressure_multi \
    tests.bridge_tp.test_drain_recompute tests.bridge_tp.test_pressure_ingress \
    tests.bridge_tp.test_six_conversion tests.bridge_tp.test_conversion_channels \
    tests.bridge_tp.test_multi_target tests.bridge_tp.test_pressure_sessions \
    tests.bridge_tp.test_migration_receivers tests.bridge_tp.test_six_card_gate \
    tests.bridge_tp.test_six_card_decider tests.bridge_tp.test_conversion_decider \
    tests.bridge_tp.test_six_conversion_budget tests.bridge_tp.test_six_gap_rules \
    tests.bridge_tp.test_conversion_sequence tests.bridge_tp.test_conversion_safety \
    tests.bridge_tp.test_conversion_planner tests.bridge_tp.test_unified_scheduler \
    tests.bridge_tp.test_pressure_conversion tests.bridge_tp.test_pressure_sustained \
    tests.bridge_tp.test_drain_freshness \
    tests.bridge_tp.test_runtime_instance_ids \
    tests.bridge_tp.test_migration_plan tests.bridge_tp.test_qwen3_six_card \
    tests.bridge_tp.test_pressure_launcher \
    tests.bridge_tp.test_source_channels tests.bridge_tp.test_channel_runtime \
    tests.bridge_tp.test_tcp_ports tests.bridge_tp.test_gpu_direct_history_lifecycle \
    tests.bridge_tp.test_six_split_probe \
    tests.bridge_tp.test_pressure_cancel tests.bridge_tp.test_unified_cancel \
    tests.bridge_tp.test_six_edge_batch tests.bridge_tp.test_six_single_class \
    tests.bridge_tp.test_first_target_startup tests.bridge_tp.test_gpu_direct_targets \
    tests.bridge_tp.test_six_rotation tests.bridge_tp.test_six_simple \
    tests.bridge_tp.test_six_return \
    tests.bridge_tp.test_six_guard \
    tests.bridge_tp.test_pressure_result_export tests.bridge_tp.test_six_auto \
    tests.bridge_tp.test_six_assist tests.bridge_tp.test_six_auto_smoke \
    tests.bridge_tp.test_six_current_assist_smoke \
    tests.bridge_tp.test_pressure_pause_diagnostics \
    tests.bridge_tp.test_pressure_inventory_process.InventoryProcessTests.test_gc_callback_records_outside_collection_and_restores_callbacks \
    > "$run/cpu_checks.log" 2>&1
status=$?
tail -n 12 "$run/cpu_checks.log"
if [ -f "$run/cpu_summary.json" ]; then
    cat "$run/cpu_summary.json"
fi
[ "$status" = 0 ] || { pack_result "$status"; exit "$status"; }
"$py" -B -c 'import json,sys; from pathlib import Path; from tools.bridge_tp.pressure_cpu import load; Path(sys.argv[1]).write_text(json.dumps(load("tcp_ports").environment_snapshot(),indent=2)+"\n")' "$run/port_environment.json" || { pack_result 1; exit 1; }
[ -f "$model/config.json" ] || { echo "找不到模型：$model（可用 MODEL=... 覆盖）"; pack_result 1; exit 1; }
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" = 6 ] || { echo '需要六卡。'; pack_result 1; exit 1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo 'GPU忙，未清理其他进程。'; pack_result 1; exit 1; }
"$py" -m compileall -q -j 4 "$repo/vllm" "$repo/tools" > "$run/compileall.log" 2>&1 || { tail -n 30 "$run/compileall.log"; pack_result 1; exit 1; }
free -g > "$run/free_before.txt" 2>&1
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader > "$run/gpu_before.csv" || { pack_result 1; exit 1; }
# Residual memory of sleeping engines and host memory over the run.
( while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader
    printf 'host %s %s\n' "$(date +%s)" "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
    sleep 1
  done ) > "$run/gpu_samples.csv" 2>&1 &
smi=$!
case_dir="$run/sleep_smoke"
mkdir "$case_dir" || { pack_result 1; exit 1; }
"$py" -B tools/bridge_tp/hang_watch.py --root "$run/sleep_smoke" --output "$run/hang_diag" > "$run/hang_watch.log" 2>&1 &
watch=$!
echo '开始加重复现：8步，事件SLEPT/WOKEN记录每次休眠与唤醒；若某引擎60秒不返回，hang_watch抓诊断。'
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    --cards 6 --static-layout L1AC --long-burst-pair \
    --scheduler unified --hold-queue --kv-short-exempt --overload-migrate --migrate-current-length \
    --conversion L1AC-L2C --conversion-trigger-s 110 --conversion-prestart --conversion-recompute \
    --conversion-engine-reuse standby --six-sleep-smoke --six-sleep-profile heavy \
    --expanded-context --current-machine --repetitions 1 \
    --model "$model" \
    --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --output "$case_dir/trial" --expected-revision "$revision" 2>&1 | tee "$case_dir/console.log"
runner_status=${PIPESTATUS[0]}
printf '%s\n' "$runner_status" > "$case_dir/runner_exit_code.txt"
"$py" -B -m tools.bridge_tp.six_sleep_smoke --directory "$case_dir/trial" > "$case_dir/sleep_check.log" 2>&1
check_status=$?
cat "$case_dir/sleep_check.log"
printf '%s\n' "$check_status" > "$case_dir/check_exit_code.txt"
status=$runner_status
if [ "$status" = 0 ]; then status=$check_status; fi
echo '检查码：0=8步全部是唤醒，没有冷启动和回退，被唤醒的引擎接到了重算或迁移；2=有效但覆盖缺失；1=技术失败或证据缺失。若失败，先看hang_diag/与各引擎日志里的Thread栈（Current thread / Thread 0x...）。'
pack_result "$status" || exit 1
exit "$status"
