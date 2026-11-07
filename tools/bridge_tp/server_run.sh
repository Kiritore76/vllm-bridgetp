#!/usr/bin/env bash
# 当前任务：六卡阶段4转换阈值的容量重测（用户11-07同意）：统一调度下短请求细扫，
# 本场L3（预注册8–16 req/s每2一档，每档60秒512/128）；已测L0 20、L1AC 18、L2C 16 req/s；L1AC长请求容量随后下发。
scenes='L3:8,10,12,14,16'
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
run=/root/autodl-tmp/bridgetp/results/qwen3_unified_sweep_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
avail=$(df --output=avail -k "$run" | tail -1)
[ "$avail" -ge 16777216 ] || { echo '磁盘可用不足16GB。'; exit 1; }
printf '%s\n' "$revision" > "$run/revision.txt"
echo "本批目录：$run"
echo "场次 $scenes（布局:速率列表）；A={2,3}、B={0,1}、C={4,5}；TP1与TP2槽位均16。"
echo '统一调度：准入=引擎排队≤8且提示放得下（保护125块）；选实例=满度最低（槽位、KV、平滑后token间隔均值÷32ms取最大）；迁移规则同阶段4第1场，纯短请求通常不触发。'
echo '负载：每档固定512输入/128输出短请求，均匀到达60秒后停止并排空；种子20261101；每档单独启动一遍布局。双32768/.88、guard125、hold-queue；L1AC/L2C沿用kv-short-exempt/overload-migrate/migrate-current-length；L0、L3不是混合布局，不加这三项。'
echo '用途：量各布局短请求能力（严格≥95%且TTFT不持续上升的最高速率），定转换决策器短请求阈值；不改调度、不转换、不训练。'
echo '纯短请求零迁移不算覆盖不足；请求/token不全、槽位或调度器不符、清理不完整即停，不自动重试。'
echo '约30–40分钟含每档启动/收尾/打包；不删除服务器结果。'
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
    echo "退出码：$status（0全部档位完整，其余查看故障证据；容量单独分析）"
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
echo '[CPU前置] 通过（含本地权限受限的链接测试，Linux不允许跳过）；准备统一调度短请求细扫（L3）。'
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
    rates=${scene#*:}
    count=$(printf '%s\n' "$rates" | tr ',' '\n' | grep -c .)
    avail=$(df --output=avail -k "$run" | tail -1)
    if [ "$avail" -lt 8388608 ] || [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
        echo "[停止] $layout 前磁盘不足8GB或GPU有计算进程；未清理任何进程。"
        batch_status=1
        break
    fi
    options=(--hold-queue)
    if [ "$layout" = L1AC ] || [ "$layout" = L2C ]; then
        options+=(--kv-short-exempt --overload-migrate --migrate-current-length)
    fi
    echo "[场次] $layout 开始：统一调度短请求细扫 $rates req/s"
    "$py" -B tools/bridge_tp/run_qwen3_stay_migrate.py \
        --cards 6 --static-layout "$layout" --tp2-max-num-seqs 16 --scheduler unified \
        --rate-sweep --sweep-rates "$rates" \
        "${options[@]}" --expanded-context --current-machine --repetitions 1 \
        --model "$model" \
        --checkpoint "$repo/experiments/platforms/qwen3_pressure_observation/artifacts/predictor_distribution.pt" \
        --inputs "$repo/experiments/platforms/qwen3_pressure_observation/inputs.json" \
        --output "$run/$layout" --expected-revision "$revision" 2>&1 | tee "$run/$layout.console.log"
    status=${PIPESTATUS[0]}
    runner_status=$status
    if [ "$status" = 0 ] || [ "$status" = 2 ]; then
        # Pure short traffic need not migrate: a migration gap is not a failure here.
        "$py" -B - "$run/$layout" "$count" > "$run/$layout.sweep_check.log" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

root, count = Path(sys.argv[1]), int(sys.argv[2])
acceptance = list(root.rglob("six_card_acceptance.json"))
shutdown = list(root.rglob("owned_shutdown_receipt.json"))
assert len(acceptance) == len(shutdown) == count, "missing per-rate receipts"
for path in acceptance:
    value = json.loads(path.read_text())
    assert value["all_instances_active"] and value["requests_complete"], path
for path in shutdown:
    assert json.loads(path.read_text())["all_owned_groups_ended"], path
cells = []
for argv_path in sorted(root.rglob("effective_argv.json")):
    argv = json.loads(argv_path.read_text())
    engines = dict(target=argv.get("target_argv"), source=argv.get("source_argv"))
    engines.update(argv.get("extra_argv", {}))
    slots = {k: v[v.index("--max-num-seqs") + 1] for k, v in engines.items()
             if v and "--max-num-seqs" in v}
    assert set(slots.values()) <= {"16"}, ("slots", argv_path, slots)
    controller = argv_path.parent / "controller"
    audit = controller / "pressure_audit.jsonl"
    schedulers = set()
    with audit.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("kind") == "pressure_admission":
                schedulers.add(row.get("scheduler"))
    assert schedulers == {"unified"}, ("scheduler", argv_path, schedulers)
    requests = json.loads((controller / "pressure_request_results.json").read_text())
    tokens = sum(len(r["response"]["emitted"]) for r in requests)
    assert tokens == sum(r["request"]["max_tokens"] for r in requests), argv_path
    cells.append(dict(cell=argv_path.parent.name, requests=len(requests),
                      output_tokens=tokens, slots=slots))
assert len(cells) == count, cells
print(json.dumps(dict(layout=root.name, cells=cells)))
PY
        check=$?
        if [ "$check" != 0 ]; then
            status=1
        else
            status=0
        fi
        cat "$run/$layout.sweep_check.log"
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
