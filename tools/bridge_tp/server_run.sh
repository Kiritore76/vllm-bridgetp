#!/usr/bin/env bash
# 当前任务：修复新源监听重试后，CPU前置通过再重跑L2C到L1AC一场；保留rank级通道进度/错误回执，不绕过预热。
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
run=/root/autodl-tmp/bridgetp/results/qwen3_six_conversion_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
printf '%s\n' "$revision" > "$run/revision.txt"
printf 'six-card unified; explicit L2C to L1AC (split A) at 60s; DRAIN natural; gate<0.9; no TP2 migration.\n' > "$run/scope.txt"
printf 'retry after pending-listener fix; per-rank connection receipts; warmup proof remains mandatory.\n' >> "$run/scope.txt"
echo "本批目录：$run"
echo 'Linux CPU前置失败/错误/跳过即停止；通过后仅运行L2C到L1AC一场GPU冒烟。'
echo '失败即打包，不自动重试、不删除结果；不清理其他进程。'
smi=
pack_result() {
    local status=$1
    if [ -n "$smi" ]; then kill "$smi" 2>/dev/null || true; wait "$smi" 2>/dev/null || true; smi=; fi
    printf '%s\n' "$status" > "$run/exit_code.txt"
    if [ -f "$run/trial.tar.gz" ]; then
        tar --exclude="$(basename "$run")/trial" -C "$(dirname "$run")" -czf "$run.tar.gz" "$(basename "$run")" || return 1
    else
        tar -C "$(dirname "$run")" -czf "$run.tar.gz" "$(basename "$run")" || return 1
    fi
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
    > "$run/cpu_checks.log" 2>&1
status=$?
tail -n 12 "$run/cpu_checks.log"
if [ -f "$run/cpu_summary.json" ]; then
    cat "$run/cpu_summary.json"
fi
[ "$status" = 0 ] || { pack_result "$status"; exit "$status"; }
[ -f "$model/config.json" ] || { echo '找不到Qwen3-8B模型。'; pack_result 1; exit 1; }
[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)" = 6 ] || { echo '需要六卡。'; pack_result 1; exit 1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo 'GPU忙，未清理其他进程。'; pack_result 1; exit 1; }
avail=$(df --output=avail -k "$run" | tail -1)
[ "$avail" -ge 12582912 ] || { echo '数据盘不足12GiB。'; pack_result 1; exit 1; }
"$py" -m compileall -q -j 4 "$repo/vllm" "$repo/tools" > "$run/compileall.log" 2>&1 || { tail -n 30 "$run/compileall.log"; pack_result 1; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader > "$run/gpu_before.csv" || { pack_result 1; exit 1; }
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader -lms 1000 > "$run/gpu_samples.csv" &
smi=$!
echo '唯一GPU场：L2C起点，60秒提议A拆回两个TP1；A停止接新请求和新迁移，已有请求自然结束后退役，源入口改回B；B、C保留服务，新TP1加入路由后可作为源迁往B。'
"$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
    --cards 6 --static-layout L2C --long-burst-pair \
    --scheduler unified --hold-queue --kv-short-exempt --overload-migrate --migrate-current-length \
    --conversion L2C-L1AC --conversion-trigger-s 60 \
    --expanded-context --current-machine --repetitions 1 \
    --model "$model" \
    --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
    --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
    --output "$run/trial" --expected-revision "$revision" 2>&1 | tee "$run/console.log"
status=${PIPESTATUS[0]}
if [ "$status" = 0 ]; then
    "$py" -B - "$run/trial" > "$run/mechanism_check.log" 2>&1 <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
paths=list(root.rglob("conversion_sequence_acceptance.json"))
assert len(paths)==1, paths
proof=json.loads(paths[0].read_text())
assert proof["passed"] and proof["all_steps_completed"],proof
assert proof.get("late_dispatches")==[],proof.get("late_dispatches")
sequence=json.loads((paths[0].parent/"conversion_sequence_result.json").read_text())
assert len(sequence["steps"])==1,sequence
step=sequence["steps"][0]
assert step["request"]["bundle"]=="A" and step["request"]["target_mode"]=="TP1",step
assert step["layout"]==dict(A=1,B=2,C=1),step
phase=proof["phases"][-1]
# Report the split's own timings; the acceptance above is the gate.
print(json.dumps(dict(passed=True,layout=step["layout"],
    drain_s=phase.get("drain_s"),rebuild_s=phase.get("rebuild_s"),
    admission_closed_s=phase.get("admission_closed_s"),
    trigger_to_route_s=phase.get("trigger_to_route_s"),
    migration_coverage=proof.get("migration_coverage")),indent=2))
PY
    status=$?
    cat "$run/mechanism_check.log"
fi
echo '仅验机制，另报告排空/重建/转换耗时、期间服务和迁移；零迁移不算失败，不作收益结论。'
pack_result "$status" || exit 1
exit "$status"
