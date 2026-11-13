#!/usr/bin/env bash
# 当前任务：guard40自动场（mixed_guard_six_v1）＋standby休眠复用＋迁移历史直写KV（--history-direct-to-kv，每秒按rank记显存）＋挂起诊断；其余参数与guard40 standby场（b0aff5a2a，两次在首个迁移进唤醒TP2时挂起）及同版stall场相同；同输入；与stall/split5/静态L1AC对比。
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
[ "$revision" = "$(git rev-parse "origin/$branch")" ] || { echo '本地落后于远端：请先 git merge --ff-only refs/remotes/origin/'"$branch"'。'; exit 1; }
mkdir -p "$results" || exit 1
run=$results/qwen3_six_guard40_standby_directkv_auto_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
printf '%s\n' "$revision" > "$run/revision.txt"
printf 'Six-card 2400s automatic trial (mixed_guard_six_v1, seed6 input 96a46bc7) with --conversion-engine-reuse standby and --history-direct-to-kv: identical to the 11-12 guard40 standby runs (b0aff5a2a; both hung when the first migration into a just-woken TP2 allocated an out-of-KV full-history staging buffer of about 1.5 GB per rank next to the sleeping residual) except that migrated history is now received in bounded 16-slot ring chunks straight into the admitted request KV blocks (validated 11-13 in qwen3_six_standby_directkv_20261113_000826_3743: four commits into the woken TP2, zero readback mismatches, handoff at most 0.68 s, +44 MiB peak), plus --gpu-memory-log-s 1 per-rank memory logs. Also includes since b0aff5a2a: pinned-arrival wait in the dispatcher, the direct-to-KV admission gate, no preemption of the migrating source request (start deferral refused after 10 s, post-start preemption refuses the session) and channel-generation rotation counting. All other options identical to the same-version stall run (a6a2ebf18): unified scheduler, hold queue, decider execute/multi/assist, prestart, drain recompute, split confirm 5s/30s, unlimited conversions, util 0.88. Rerun after the 11-13 run (qwen3_six_guard40_standby_directkv_auto_20261113_094302_30844, no hang, 7 direct-to-KV migrations) whose short-pulse KEEP coverage could not be judged: the record-only service-budget estimate required exact profile KV blocks (TP1 2325, TP2 11712) and sleep mode leaves TP1 2343 / TP2 11747, so every estimate was unsupported; it now accepts up to 2% more blocks (packing keeps the profile blocks). Decisions are unchanged (gap rules do not use the estimate). Compare with stall, split5 and static L1AC; one run, no repetition statistics.\n' > "$run/scope.txt"
echo "本批目录：$run"
echo '先Linux CPU检查；通过后只跑一场同输入2400秒自动场：参数与guard40 standby场（b0aff5a2a）相同，另加 --history-direct-to-kv --gpu-memory-log-s 1；不自动重试。'
echo '本场是11-13直写大场的重跑：上一场没有挂起、7次迁移都走直写，但短峰KEEP覆盖项无法判定（sleep mode下KV块数为2343/11747，预算标定要求2325/11712完全相等）；现已允许多出2%以内的块数，决策规则不变。'
echo '与上一场guard40 standby的区别：迁移历史分段直接写进目标请求的KV块，不再申请整段暂存（上两场在首个迁移进唤醒TP2时挂起）；另含直写小场验证过的修复：固定路由等待、直写准入门槛、迁移源不被抢占、按通道代次轮换。'
echo '带挂起诊断：若某引擎60秒不返回，hang_watch抓各进程线程栈、内核栈和nvidia-smi到hang_diag/。9个引擎，启动多约1分钟，主机pinned内存约214GiB。'
echo '引擎库存快照已精简：不再发布逐块列表（读取端从未使用），只保留块数；满载TP2每次快照省约2.9ms（构建与JSON编码），每行由约292KiB降到约5KiB。'
echo 'GPU启动前先做CPU空闲检查：CPU压力（PSI some avg10）须连续两次低于5%，最多等180秒；不通过就停止并打包cpu_gate/（含占用CPU最多的进程）。上一场重跑开始前PSI已有25%，性能因此不可用。'
echo '40分钟到达；含CPU/启动/预热/排空/打包，预计55–70分钟，慢机器或积压可能更久。'
smi=
pack_result() {
    local status=$1
    if [ -n "$smi" ]; then kill "$smi" 2>/dev/null || true; wait "$smi" 2>/dev/null || true; smi=; fi
    if [ -n "${watch:-}" ]; then kill "$watch" 2>/dev/null || true; wait "$watch" 2>/dev/null || true; watch=; fi
    free -g > "$run/free_after.txt" 2>&1
    printf '%s\n' "$status" > "$run/exit_code.txt"
    local excludes=() edge
    for edge in auto_from_L1AC; do
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
    tests.bridge_tp.test_hang_diagnostics tests.bridge_tp.test_six_sleep_migrate \
    tests.bridge_tp.test_direct_history tests.bridge_tp.test_direct_receive \
    tests.bridge_tp.test_direct_connector tests.bridge_tp.test_direct_prebind \
    tests.bridge_tp.test_direct_config tests.bridge_tp.test_pressure_dispatch \
    tests.bridge_tp.test_direct_admission_gate tests.bridge_tp.test_source_preemption \
    tests.bridge_tp.test_phase9_online_integration tests.bridge_tp.test_pressure_observation \
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
# Host CPU must be free of outside work (11-13 rerun: PSI 25% before start).
bash tools/bridge_tp/cpu_idle_gate.sh "$run/cpu_gate" || { pack_result 1; exit 1; }
free -g > "$run/free_before.txt" 2>&1
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader > "$run/gpu_before.csv" || { pack_result 1; exit 1; }
# GPU memory (sleep residuals) and host MemAvailable every second.
( while true; do
    nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv,noheader
    printf 'host %s %s\n' "$(date +%s)" "$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
    sleep 1
  done ) > "$run/gpu_samples.csv" 2>&1 &
smi=$!
case_dir="$run/auto_from_L1AC"
mkdir "$case_dir" || { pack_result 1; exit 1; }
"$py" -B tools/bridge_tp/hang_watch.py --root "$run/auto_from_L1AC" --output "$run/hang_diag" > "$run/hang_watch.log" 2>&1 &
watch=$!
echo '开始：启动时预建备用形态并预热休眠（standby_prewarm.json），然后2400秒自动场；25:00起10秒短峰应保持，27:00起持续短压应拆回。'
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    --cards 6 --static-layout L1AC --phase-shift --phase-scenario mixed_guard_six_v1 \
    --scheduler unified --hold-queue --kv-short-exempt --overload-migrate --migrate-current-length \
    --expanded-context --current-machine --repetitions 1 \
    --conversion-decider execute --decider-multi --decider-assist --conversion-prestart --conversion-recompute \
    --conversion-engine-reuse standby \
    --history-direct-to-kv --gpu-memory-log-s 1 \
    --migration-targets all --decider-max-conversions 0 \
    --model "$model" \
    --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --output "$case_dir/trial" --expected-revision "$revision" 2>&1 | tee "$case_dir/console.log"
runner_status=${PIPESTATUS[0]}
printf '%s\n' "$runner_status" > "$case_dir/runner_exit_code.txt"
"$py" -B -m tools.bridge_tp.six_rotation assess --arm auto_from_L1AC --scenario mixed_guard_six_v1 --directory "$case_dir/trial" > "$case_dir/auto_check.log" 2>&1
check_status=$?
cat "$case_dir/auto_check.log"
printf '%s\n' "$check_status" > "$case_dir/check_exit_code.txt"
status=$runner_status
if [ "$status" = 0 ]; then status=$check_status; fi
echo '每次转换的休眠/唤醒耗时与残留显存见conversion_events.jsonl的SLEPT/WOKEN；SLEEP_FALLBACK表示回退为停机。迁移凭证gpu_initial_receipts的history_path应为DIRECT_TO_KV；各rank显存见controller/gpu_memory_*.jsonl。'
echo '须真实完成长段A合并和持续短段A拆回；短峰及恢复须有效L2C/KEEP，零转换/中止/证据缺失不算覆盖通过；完整输出/ACK/trace/空闲/收尾须通过。'
pack_result "$status" || exit 1
exit "$status"
