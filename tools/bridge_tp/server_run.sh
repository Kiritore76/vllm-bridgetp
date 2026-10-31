#!/usr/bin/env bash
# BridgeTP 服务器当前任务（唯一入口，每次下发新任务时覆盖本文件）。
# 运行：
#   cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
#
# 当前任务：vLLM 导入计时（仅 CPU）。
# 比较 vllm 源码无字节码缓存与 compileall 预编译后的导入耗时，判断重建实例慢是否因每个进程重新编译源码。
# 不启动服务、不加载模型、不申请显存；只新增 __pycache__（已被 .gitignore 忽略，可随时删除）。

repo=/root/autodl-tmp/bridgetp/vllm_bridge
venv=/root/autodl-tmp/bridgetp/.venv_bridge
py=$venv/bin/python
branch=bridgetp/tp2-4090d-qwen3-8b
task=qwen3_import_timing

cd "$repo" || exit 1
source "$venv/bin/activate" || exit 1
[ -z "$(git status --porcelain)" ] || { echo '代码有未提交改动，停止并保留。'; exit 1; }
[ "$(git branch --show-current)" = "$branch" ] || { echo '分支不匹配，停止。'; exit 1; }
git fetch origin "refs/heads/$branch:refs/remotes/origin/$branch" || exit 1
head=$(git rev-parse HEAD)
[ "$head" = "$(git rev-parse "origin/$branch")" ] || {
    echo '本地不是远端最新，先执行 git pull --ff-only 再运行。'; exit 1;
}

run=/root/autodl-tmp/bridgetp/results/${task}_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir -p "$run" || exit 1
echo "$head" > "$run/revision.txt"
echo "代码：$head"
echo "本批目录：$run"
echo '只测导入时间，不碰GPU。约3-6分钟。'

site=$("$py" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || exit 1
mods='import vllm.entrypoints.openai.api_server, vllm.v1.engine.core, vllm.v1.executor.multiproc_executor'

# 统计 .py 有无对应的 .pyc（当前解释器版本）
count_pyc() {
    "$py" -B -c 'import importlib.util as u, os, sys; [print(r, "py=%d" % len(p), "missing_pyc=%d" % sum(not os.path.exists(u.cache_from_source(x)) for x in p)) for r in sys.argv[1:] for p in [[os.path.join(d, f) for d, _, fs in os.walk(r) for f in fs if f.endswith(".py")]]]' "$@"
}
# 每次在全新进程中导入一次，记录墙钟秒数；与现有指令同为 -B + PYTHONDONTWRITEBYTECODE=1
time_import() {
    local label="$1" i s e
    for i in 1 2 3; do
        s=$(date +%s.%N)
        PYTHONDONTWRITEBYTECODE=1 "$py" -B -c "$mods" >/dev/null 2>>"$run/import_stderr.log" || {
            echo "$label 第$i次导入失败，见 import_stderr.log"; return 1;
        }
        e=$(date +%s.%N)
        echo "$label run$i $(awk -v a="$s" -v b="$e" 'BEGIN{printf "%.3f", b-a}')" | tee -a "$run/timing.txt"
    done
}

{
    echo "python: $("$py" -V 2>&1)"
    echo "site-packages: $site"
    echo "nproc: $(nproc)"
    echo "--- 预编译前"
    count_pyc "$repo/vllm" "$site/torch" "$site/transformers"
} | tee "$run/pyc_before.txt"

status=0
echo '1) 现状：无预编译'
time_import BEFORE || status=1

if [ "$status" = 0 ]; then
    echo '2) 预编译 vllm 源码及 site-packages（只新增 __pycache__）'
    cs=$(date +%s.%N)
    "$py" -m compileall -q -j 0 "$repo/vllm" "$site" >"$run/compileall.log" 2>&1
    echo "compileall 退出码 $?，耗时 $(awk -v a="$cs" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f", b-a}')s" | tee -a "$run/timing.txt"
    { echo "--- 预编译后"; count_pyc "$repo/vllm" "$site/torch" "$site/transformers"; } | tee "$run/pyc_after.txt"

    echo '3) 预编译后，运行方式不变（-B 只禁止写缓存，仍读已有缓存）'
    time_import AFTER || status=1
fi

echo "git status（应为0）：$(git status --porcelain | wc -l) 行" | tee -a "$run/timing.txt"
tar -C "$(dirname "$run")" -czf "$run.tar.gz" "$(basename "$run")"
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256"
echo '=== 结果 ==='
cat "$run/timing.txt"
echo "退出码：$status"
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
exit "$status"
