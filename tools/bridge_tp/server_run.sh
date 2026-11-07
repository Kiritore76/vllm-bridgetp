#!/usr/bin/env bash
# 当前任务：六卡阶段4转换阈值的容量重测（用户11-07同意）：L1AC统一调度长请求容量一场，
# 负载完整mixed_long_ramp（与旧调度L2C/L3长容量同负载）；四布局短请求能力已测（L0 20、L1AC 18、L2C 16、L3 12 req/s）。
scenes='L1AC:mixed_long_ramp'
set -uo pipefail
repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
model=/root/autodl-tmp/models/Qwen3-8B
branch=bridgetp/unified-scheduler
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
run=/root/autodl-tmp/bridgetp/results/qwen3_unified_longcap_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
avail=$(df --output=avail -k "$run" | tail -1)
[ "$avail" -ge 16777216 ] || { echo '磁盘可用不足16GB。'; exit 1; }
printf '%s\n' "$revision" > "$run/revision.txt"
echo "本批目录：$run"
echo "场次 $scenes（布局:负载）；A={2,3}、B={0,1}、C={4,5}；L1AC=A、C为四个TP1，B为TP2；槽位均16。"
echo '统一调度：准入=引擎排队≤8且提示放得下（保护125块）；选实例=满度最低（槽位、KV、平滑后token间隔均值÷32ms取最大）；迁移=TP1→B，KV≥90%或约8秒耗尽立即迁，满度≥0.9持续10秒且高出目标0.3才按压力迁；规则与阈值不变。'
echo '负载mixed_long_ramp（种子6）：0–60秒短4/s；60–1020秒短8/s，16384/8192深长每240秒一档0.02/0.04/0.06/0.08/s（3/10/18/17条）；1020–1080秒短1/s；8045请求、1416832输出token。双32768/.88、guard125、hold-queue、kv-short-exempt/overload-migrate/migrate-current-length。'
echo '用途：看L1AC在多少条同时进行的深长请求、多少块占用时开始撑不住（长TPOT越35ms、KV用尽、抢占），核对按块数换算的L1AC→L2C阈值（约12600块）；不改调度、不转换、不训练。'
echo '有无迁移都不算覆盖不足；请求/token不全、槽位或调度器不符、清理不完整即停，不自动重试。'
echo '约25–30分钟含启动/收尾/打包；不删除服务器结果。'
printf 'scene\trunner_exit\tbatch_exit\n' > "$run/scenes.tsv"
pack_result() {
    local status=$1
    printf '%s\n' "$status" > "$run/exit_code.txt"
    local excludes=()
    for scene in $scenes; do
        local name=${scene%%:*}
        if (cd "$run" && sha256sum -c "$name.tar.gz.sha256" >/dev/null 2>&1) \
            && tar -tzf "$run/$name.tar.gz" >/dev/null 2>&1; then
            excludes+=(--exclude="$(basename "$run")/$name")
        fi
    done
    tar -C "$(dirname "$run")" "${excludes[@]}" -czf "$run.tar.gz" "$(basename "$run")" || return 1
    sha256sum "$run.tar.gz" > "$run.tar.gz.sha256" || return 1
    echo "退出码：$status（0服务与收尾完整，其余查看故障证据；容量单独分析）"
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
echo '[CPU前置] 运行本次相关的回归检查（含统一调度各项）；不执行GPU计算。'
# 与本地CPU检查相同，导入源码控制逻辑，避免执行vLLM顶层的GPU平台探测。
"$py" -B -c 'import sys,types,unittest; from pathlib import Path; p=types.ModuleType("vllm"); p.__path__=[str(Path.cwd()/"vllm")]; sys.modules["vllm"]=p; s=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:]); r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(not r.wasSuccessful() or bool(r.skipped))' \
    tests.bridge_tp.test_migration_receivers tests.bridge_tp.test_migration_plan \
    tests.bridge_tp.test_runtime_instance_ids tests.bridge_tp.test_pressure_conversion \
    tests.bridge_tp.test_qwen3_six_card tests.bridge_tp.test_static_layout \
    tests.bridge_tp.test_pressure_launcher tests.bridge_tp.test_pressure_multi \
    tests.bridge_tp.test_gpu_direct_targets tests.bridge_tp.test_source_channels \
    tests.bridge_tp.test_4090d_qwen3 tests.bridge_tp.test_qwen3_stay_migrate \
    tests.bridge_tp.test_unified_scheduler tests.bridge_tp.test_pressure_multi \
    tests.bridge_tp.test_pressure_policy tests.bridge_tp.test_pressure_dispatch \
    tests.bridge_tp.test_multi_target tests.bridge_tp.test_pressure_sessions \
    tests.bridge_tp.test_same_tp_evacuation \
    > "$run/cpu_checks.log" 2>&1
status=$?
if [ "$status" != 0 ]; then
    echo '[CPU前置] 未通过，不启动GPU实验；请拿回诊断包。'
    tail -n 30 "$run/cpu_checks.log"
    pack_result "$status" || exit 1
    exit "$status"
fi
echo '[CPU前置] 通过（含本地权限受限的链接测试，Linux不允许跳过）；准备L1AC统一调度长请求容量一场。'
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
batch_status=0
for scene in $scenes; do
    layout=${scene%%:*}
    scenario=${scene#*:}
    avail=$(df --output=avail -k "$run" | tail -1)
    if [ "$avail" -lt 8388608 ] || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
        echo "[停止] $layout 前磁盘不足8GB或GPU有计算进程；未清理任何进程。"
        batch_status=1
        break
    fi
    options=(--hold-queue --kv-short-exempt --overload-migrate --migrate-current-length)
    echo "[场次] $layout 开始：统一调度长请求容量，负载 $scenario"
    "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
        --cards 6 --static-layout "$layout" --tp2-max-num-seqs 16 --scheduler unified \
        --phase-shift --phase-scenario "$scenario" \
        "${options[@]}" --expanded-context --current-machine --repetitions 1 \
        --model "$model" \
        --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
        --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
        --output "$run/$layout" --expected-revision "$revision" 2>&1 | tee "$run/$layout.console.log"
    status=${PIPESTATUS[0]}
    runner_status=$status
    if [ "$status" = 0 ] || [ "$status" = 2 ]; then
        # A capacity run need not migrate: a migration gap is not a failure here.
        "$py" -B - "$run/$layout" "$scenario" > "$run/$layout.long_check.log" 2>&1 <<'PY'
import collections
import json
import sys
from pathlib import Path

from tools.bridge_tp import qwen3_phase_shift as phase

root, scenario = Path(sys.argv[1]), sys.argv[2]


def unique(name):
    paths = list(root.rglob(name))
    assert len(paths) == 1, (name, paths)
    return paths[0], json.loads(paths[0].read_text())


_, acceptance = unique("six_card_acceptance.json")
assert acceptance["all_instances_active"] and acceptance["requests_complete"]
_, shutdown = unique("owned_shutdown_receipt.json")
assert shutdown["all_owned_groups_ended"], "process cleanup incomplete"
argv_path, argv = unique("effective_argv.json")
engines = dict(target=argv.get("target_argv"), source=argv.get("source_argv"))
engines.update(argv.get("extra_argv", {}))
slots = {k: v[v.index("--max-num-seqs") + 1] for k, v in engines.items()
         if v and "--max-num-seqs" in v}
assert set(slots.values()) == {"16"}, ("slots", slots)
controller = argv_path.parent / "controller"
inputs = json.loads(Path(
    "experiments/platforms/qwen3_pressure_observation/inputs.json").read_text())
planned = phase.jobs(inputs, scenario)
requests = json.loads((controller / "pressure_request_results.json").read_text())
assert len(requests) == len(planned), (len(requests), len(planned))
tokens = sum(len(r["response"]["emitted"]) for r in requests)
assert tokens == sum(j["request"]["max_tokens"] for j in planned), tokens
schedulers, outcomes, starts = set(), collections.Counter(), 0
with (controller / "pressure_audit.jsonl").open() as handle:
    for line in handle:
        row = json.loads(line)
        kind = row.get("kind")
        if kind == "pressure_admission":
            schedulers.add(row.get("scheduler"))
        elif kind == "pressure_settlement":
            outcomes[row.get("outcome")] += 1
        elif kind == "pressure_start_published":
            starts += 1
assert schedulers == {"unified"}, schedulers
summary = dict(layout=root.name, scenario=scenario, requests=len(requests),
               output_tokens=tokens, slots=slots, migration_starts=starts,
               settlements=dict(outcomes))
(root / "long_capacity_check.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary))
PY
        check=$?
        if [ "$check" != 0 ]; then
            status=1
        else
            status=0
        fi
        cat "$run/$layout.long_check.log"
    fi
    printf '%s\t%s\t%s\n' "$layout" "$runner_status" "$status" >> "$run/scenes.tsv"
    if [ "$status" != 0 ]; then
        echo "[场次] $layout 技术失败或核对失败，停止余下场次。"
        batch_status=$status
        break
    fi
done
status=$batch_status
kill "$smi" 2>/dev/null || true
wait "$smi" 2>/dev/null || true
pack_result "$status" || exit 1
exit "$status"
