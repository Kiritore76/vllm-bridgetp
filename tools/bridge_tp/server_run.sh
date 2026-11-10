#!/usr/bin/env bash
# 当前任务：重算小场——六卡L1AC起步，110秒显式A合并为TP2（自然排空+受限重算+预启动），长突发小负载；确认重算功能后再跑40分钟大场。
set -uo pipefail
repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/drain-recompute
cd "$repo" || exit 1
[ -x "$py" ] || { echo '找不到现有虚拟环境，请核对路径。'; exit 1; }
source "$venv/bin/activate" || exit 1
[ -z "$(git status --porcelain)" ] || { echo '工作树有改动，停止并保留。'; exit 1; }
[ "$(git branch --show-current)" = "$branch" ] || { echo '分支不匹配，停止。'; exit 1; }
git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch" || exit 1
revision=$(git rev-parse HEAD)
[ "$revision" = "$(git rev-parse "origin/$branch")" ] || { echo '请先git pull --ff-only。'; exit 1; }
mkdir -p /root/autodl-tmp/bridgetp/results || exit 1
run=/root/autodl-tmp/bridgetp/results/qwen3_six_recompute_smoke_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
printf '%s\n' "$revision" > "$run/revision.txt"
printf 'Small six-card drain recompute smoke: L1AC start, explicit A merge to TP2 at 110s (DRAIN, prestart, recompute after 45s drain if <=3 requests and <=65536 context tokens); long-burst workload 150 short + 6 long 16384/8192 at 90-100s; mechanism check only, no performance claim.\n' > "$run/scope.txt"
echo "本批目录：$run"
echo '先Linux CPU检查；通过后只跑一场小场：L1AC起步，110秒显式把A合并为TP2，开启预启动和受限重算；不自动重试。'
echo '150条短请求+90–100秒6条长请求（16384/8192）；关门后排空满45秒、A上≤3条且上下文≤65536 token时中止源端并在新TP2续算。'
echo 'A上长请求超过3条时不会触发重算，会如实记为覆盖缺失（检查码2），不算功能失败。预计含CPU/启动/预热/打包15–30分钟。'
smi=
pack_result() {
    local status=$1
    if [ -n "$smi" ]; then kill "$smi" 2>/dev/null || true; wait "$smi" 2>/dev/null || true; smi=; fi
    printf '%s\n' "$status" > "$run/exit_code.txt"
    local excludes=() edge
    for edge in L1AC-L2C_recompute; do
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
[ -f "$model/config.json" ] || { echo '找不到Qwen3-8B模型。'; pack_result 1; exit 1; }
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" = 6 ] || { echo '需要六卡。'; pack_result 1; exit 1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo 'GPU忙，未清理其他进程。'; pack_result 1; exit 1; }
"$py" -m compileall -q -j 4 "$repo/vllm" "$repo/tools" > "$run/compileall.log" 2>&1 || { tail -n 30 "$run/compileall.log"; pack_result 1; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader > "$run/gpu_before.csv" || { pack_result 1; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader -lms 1000 > "$run/gpu_samples.csv" &
smi=$!
case_dir="$run/L1AC-L2C_recompute"
mkdir "$case_dir" || { pack_result 1; exit 1; }
echo '开始小场：110秒A关门；预计约155秒请求重算，新A接流后续算到完整输出。'
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    --cards 6 --static-layout L1AC --long-burst-pair \
    --scheduler unified --hold-queue --kv-short-exempt --overload-migrate --migrate-current-length \
    --conversion L1AC-L2C --conversion-trigger-s 110 --conversion-prestart --conversion-recompute \
    --expanded-context --current-machine --repetitions 1 \
    --model "$model" \
    --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --output "$case_dir/trial" --expected-revision "$revision" 2>&1 | tee "$case_dir/console.log"
runner_status=${PIPESTATUS[0]}
printf '%s\n' "$runner_status" > "$case_dir/runner_exit_code.txt"
"$py" -B -m tools.bridge_tp.drain_recompute_smoke --directory "$case_dir/trial" > "$case_dir/recompute_check.log" 2>&1
check_status=$?
cat "$case_dir/recompute_check.log"
printf '%s\n' "$check_status" > "$case_dir/check_exit_code.txt"
status=$runner_status
if [ "$status" = 0 ]; then status=$check_status; fi
echo '检查码：0=A合并完成且重算续算完整无缝；2=有效但未触发重算（覆盖缺失）；1=技术失败或证据缺失。时间线只作机制核对，不是性能结论。'
pack_result "$status" || exit 1
exit "$status"
