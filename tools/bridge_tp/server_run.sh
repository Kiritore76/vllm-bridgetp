#!/usr/bin/env bash
# 当前任务：六卡启动时预建备用形态（standby）小场——启动时每组另一种形态先冷启动并预热休眠，醒着的形态也预热休眠一次；L1AC起步，显式A合并(110s)→拆回(240s)→再合并(440s)全部为唤醒，无冷启动；两段长突发；机制核对，不是性能结论。
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
run=$results/qwen3_six_standby_smoke_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
printf '%s\n' "$revision" > "$run/revision.txt"
printf 'Six-card engine-reuse smoke with startup standby: --conversion-engine-reuse standby (vLLM sleep mode level 1). Before the workload the other form of each group (A TP2, B 2xTP1, C TP2) is cold-started first on empty GPUs, slept, woken and slept again (pinned backup pre-allocated); the awake L1AC engines then start, sleep and wake once, and only then open their channels. Explicit A merge (110s), split (240s), merge again (440s): every step wakes a sleeping form, no cold start. Prestart and drain recompute on. 300 short + two long bursts (6x16384/8192 at 90-100s, 6x16384/4096 at 420-430s). Mechanism check only, no performance claim.\n' > "$run/scope.txt"
echo "本批目录：$run"
echo '先Linux CPU检查；通过后只跑一场：启动时先建并预热休眠备用形态（A的TP2、B的两个TP1、C的TP2），再启动L1AC醒着的引擎并预热休眠一次，之后才发布迁移通道。'
echo '110秒A合并（唤醒备用TP2）→240秒拆回（唤醒原来的两个TP1）→440秒再合并（再唤醒同一个TP2），三步都没有冷启动。'
echo '负载：600秒内每2秒一条短请求；90–100秒6条16384/8192长请求，420–430秒6条16384/4096长请求；合并时开启预启动和受限重算。'
echo '共9个引擎进程，启动比平时多约2–3分钟（备用冷启动约1分钟，两轮预热休眠各约15秒）；主机pinned内存约190GiB。场景与收尾约15–25分钟，加上CPU检查和打包。不自动重试。'
smi=
pack_result() {
    local status=$1
    if [ -n "$smi" ]; then kill "$smi" 2>/dev/null || true; wait "$smi" 2>/dev/null || true; smi=; fi
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
task_tmp=$(mktemp -d /root/autodl-tmp/bt-tmp.XXXXXX 2>/dev/null || mktemp -d) || { pack_result 1; exit 1; }
printf '%s\n' "$task_tmp" > "$run/temporary_directory.txt"
export TMPDIR="$task_tmp" TEMP="$task_tmp" TMP="$task_tmp"
"$py" -B -c 'import json,sys,types,unittest; from pathlib import Path; p=types.ModuleType("vllm"); p.__path__=[str(Path.cwd()/"vllm")]; sys.modules["vllm"]=p; s=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[2:]); r=unittest.TextTestRunner(verbosity=2).run(s); ok=r.wasSuccessful() and not r.skipped; Path(sys.argv[1]).write_text(json.dumps(dict(scope="CPU only; no CUDA/NCCL", tests=r.testsRun, passed=ok, failures=[t.id() for t,_ in r.failures], errors=[t.id() for t,_ in r.errors], skipped=[t.id() for t,_ in r.skipped]),indent=2)+"\n"); raise SystemExit(not ok)' \
    "$run/cpu_summary.json" \
    tests.bridge_tp.test_six_sleep_reuse tests.bridge_tp.test_six_sleep_smoke \
    tests.bridge_tp.test_six_standby_launch \
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
echo '开始小场：启动预热记录在standby_prewarm.json；110/240/440秒三步，事件SLEPT/WOKEN记录每次休眠与唤醒耗时和残留显存。'
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    --cards 6 --static-layout L1AC --long-burst-pair \
    --scheduler unified --hold-queue --kv-short-exempt --overload-migrate --migrate-current-length \
    --conversion L1AC-L2C --conversion-trigger-s 110 --conversion-prestart --conversion-recompute \
    --conversion-engine-reuse standby --six-sleep-smoke \
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
echo '检查码：0=三步都是唤醒（合并1唤醒备用TP2），没有冷启动和回退，被唤醒的引擎接到了重算或迁移；2=有效，但没有重算或迁移落到被唤醒的引擎上（覆盖缺失）；1=技术失败或证据缺失。耗时只作机制记录，不是性能结论。'
pack_result "$status" || exit 1
exit "$status"
