#!/usr/bin/env bash
# 当前任务：统一调度速度基准由35ms改为28ms（用户11-06，TP1与TP2统一），L2C:16一场。
# 负载只跑长请求递增的前三档（mixed_long_ramp_r3）：各档起点与种子不变，到达与此前L2C-16两场逐条相同，可直接对比r1–r3。
# 对照：旧调度qwen3_six_longcap_20261106_153723_9226的L2C-s16，统一调度35ms qwen3_unified_20261106_205522_4987。
scenes='L2C:16:unified'
scenario=mixed_long_ramp_r3
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
run=/root/autodl-tmp/bridgetp/results/qwen3_unified_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
avail=$(df --output=avail -k "$run" | tail -1)
[ "$avail" -ge 16777216 ] || { echo '磁盘可用不足16GB。'; exit 1; }
printf '%s\n' "$revision" > "$run/revision.txt"
echo "本批目录：$run"
echo "统一调度：场次 $scenes（布局:TP2槽位:调度方式）；A={2,3}、B={0,1}、C={4,5}；TP1槽位16。"
echo '统一调度：准入=引擎排队≤8且提示放得下（保护125块）；选实例=满度最低（槽位、KV、平滑ITL÷28ms取最大）；迁移=KV≥90%或约8秒耗尽立即迁，满度≥0.9持续10秒且高出目标0.3才按压力迁；挑请求沿用旧调度评估，目标仍只有B。'
echo '负载：预热60秒短4/s；60–780秒短8/s，深长16384/8192每240秒一档0.02/0.04/0.06/s（3/10/18条）；收尾60秒短1/s；种子6。'
echo '双32768/.88、guard125、hold-queue；L2C沿用kv-short-exempt/overload-migrate/migrate-current-length，仍只迁向B。'
echo '不转换、不训练；旧调度代码保留在开关后面；性能负结果和零迁移保留；技术故障/请求或清理不完整即停，不自动重试。'
echo '约18–25分钟含启动/收尾/打包；不删除服务器结果。'
printf 'scene\trunner_exit\tbatch_exit\n' > "$run/scenes.tsv"
pack_result() {
    local status=$1
    printf '%s\n' "$status" > "$run/exit_code.txt"
    local excludes=()
    for scene in $scenes; do
        local rest=${scene#*:}
        local name=${scene%%:*}-s${rest%%:*}-${rest#*:}
        if (cd "$run" && sha256sum -c "$name.tar.gz.sha256" >/dev/null 2>&1) \
            && tar -tzf "$run/$name.tar.gz" >/dev/null 2>&1; then
            excludes+=(--exclude="$(basename "$run")/$name")
        fi
    done
    tar -C "$(dirname "$run")" "${excludes[@]}" -czf "$run.tar.gz" "$(basename "$run")" || return 1
    sha256sum "$run.tar.gz" > "$run.tar.gz.sha256" || return 1
    echo "退出码：$status（0机制覆盖通过，2覆盖不足，其余查看故障证据；性能单独分析）"
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
echo '[CPU前置] 运行本次相关的回归检查（含统一调度20项）；不执行GPU计算。'
# 与本地CPU检查相同，导入源码控制逻辑，避免执行vLLM顶层的GPU平台探测。
"$py" -B -c 'import sys,types,unittest; from pathlib import Path; p=types.ModuleType("vllm"); p.__path__=[str(Path.cwd()/"vllm")]; sys.modules["vllm"]=p; s=unittest.defaultTestLoader.loadTestsFromNames(sys.argv[1:]); r=unittest.TextTestRunner(verbosity=2).run(s); raise SystemExit(not r.wasSuccessful())' \
    tests.bridge_tp.test_qwen3_six_card tests.bridge_tp.test_static_layout \
    tests.bridge_tp.test_pressure_launcher tests.bridge_tp.test_pressure_multi \
    tests.bridge_tp.test_gpu_direct_targets tests.bridge_tp.test_source_channels \
    tests.bridge_tp.test_4090d_qwen3 tests.bridge_tp.test_qwen3_stay_migrate \
    tests.bridge_tp.test_unified_scheduler tests.bridge_tp.test_pressure_multi \
    tests.bridge_tp.test_pressure_policy tests.bridge_tp.test_pressure_dispatch \
    > "$run/cpu_checks.log" 2>&1
status=$?
if [ "$status" != 0 ]; then
    echo '[CPU前置] 未通过，不启动GPU实验；请拿回诊断包。'
    tail -n 30 "$run/cpu_checks.log"
    pack_result "$status" || exit 1
    exit "$status"
fi
echo '[CPU前置] 通过；准备长请求容量场次。'
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
    rest=${scene#*:}
    slots=${rest%%:*}
    scheduler=${rest#*:}
    name=$layout-s$slots-$scheduler
    avail=$(df --output=avail -k "$run" | tail -1)
    if [ "$avail" -lt 8388608 ] || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
        echo "[停止] $name 前磁盘不足8GB或GPU有计算进程；未清理任何进程。"
        batch_status=1
        break
    fi
    options=(--hold-queue)
    if [ "$layout" = L2C ]; then
        options+=(--kv-short-exempt --overload-migrate --migrate-current-length)
    fi
    echo "[场次] $name 开始"
    "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
        --cards 6 --static-layout "$layout" --tp2-max-num-seqs "$slots" --scheduler "$scheduler" \
        --phase-shift --phase-scenario "$scenario" \
        "${options[@]}" --expanded-context --current-machine --repetitions 1 \
        --model "$model" \
        --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
        --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
        --output "$run/$name" --expected-revision "$revision" 2>&1 | tee "$run/$name.console.log"
    status=${PIPESTATUS[0]}
    runner_status=$status
    if [ "$status" = 0 ] || [ "$status" = 2 ]; then
        # Coverage-only gaps can continue; incomplete service/cleanup cannot.
        "$py" -B -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); a=list(p.rglob("six_card_acceptance.json")); c=list(p.rglob("owned_shutdown_receipt.json")); assert len(a)==len(c)==1, "missing scene receipts"; assert all(json.loads(f.read_text())["all_instances_active"] and json.loads(f.read_text())["requests_complete"] for f in a), "incomplete static service"; assert all(json.loads(f.read_text())["all_owned_groups_ended"] for f in c), "process cleanup incomplete"' "$run/$name" \
            > "$run/$name.receipt_check.log" 2>&1 || status=1
        # Every TP2 engine must really have run with the requested slot count.
        "$py" -B - "$run/$name" "$slots" > "$run/$name.slots_check.log" 2>&1 <<'PY' || status=1
import json
import sys
from pathlib import Path

root, wanted = Path(sys.argv[1]), sys.argv[2]
found = list(root.rglob("effective_argv.json"))
assert len(found) == 1, f"effective_argv.json files: {len(found)}"
argv = json.loads(found[0].read_text())
engines = dict(target=argv["target_argv"], source=argv["source_argv"])
engines.update(argv.get("extra_argv", {}))
slots = {}
for engine, command in engines.items():
    if "--max-num-seqs" in command:
        slots[engine] = command[command.index("--max-num-seqs") + 1]
tp2 = {k: v for k, v in slots.items() if k == "target" or "tp2" in k}
tp1 = {k: v for k, v in slots.items() if k == "source" or "tp1" in k}
print(json.dumps(dict(tp2=tp2, tp1=tp1)))
assert tp2 and set(tp2.values()) == {wanted}, "TP2 slots differ from the request"
assert set(tp1.values()) <= {"16"}, "TP1 slots changed"
PY
        cat "$run/$name.slots_check.log"
    fi
    printf '%s\t%s\t%s\n' "$name" "$runner_status" "$status" >> "$run/scenes.tsv"
    if [ "$status" != 0 ] && [ "$status" != 2 ]; then
        echo "[场次] $name 技术失败或核对失败，停止余下场次。"
        batch_status=$status
        break
    fi
    if [ "$status" = 2 ]; then
        batch_status=2
        echo "[场次] $name 覆盖不足，保留结果；不修改参数或自动重试。"
    fi
done
status=$batch_status
kill "$smi" 2>/dev/null || true
wait "$smi" 2>/dev/null || true
pack_result "$status" || exit 1
exit "$status"
