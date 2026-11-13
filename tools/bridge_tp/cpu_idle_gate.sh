#!/usr/bin/env bash
# Refuse a GPU run while the host CPU is busy with work outside the experiment.
#
# 11-13 guard40 rerun: CPU pressure (PSI "some avg10") was already 25% and 28
# processes were runnable before the workload started (9 in the previous run);
# the CPU-bound sustained-short phase lost half its strict requests. This gate
# waits until CPU pressure is low for two consecutive samples, records every
# sample and a top-CPU process snapshot, and fails (exit 1) after the wait.
#
# Usage: cpu_idle_gate.sh <output dir>
# Env: CPU_PSI_MAX (percent, default 5), CPU_IDLE_WAIT_S (default 180),
#      CPU_IDLE_INTERVAL_S (default 5), CPU_PSI_FILE (default /proc/pressure/cpu).
set -u
out=${1:?usage: cpu_idle_gate.sh <output dir>}
max=${CPU_PSI_MAX:-5}
wait_s=${CPU_IDLE_WAIT_S:-180}
interval=${CPU_IDLE_INTERVAL_S:-5}
psi_file=${CPU_PSI_FILE:-/proc/pressure/cpu}
mkdir -p "$out" || exit 1
samples="$out/cpu_gate_samples.txt"
: > "$samples"

snapshot() {
    {
        echo "# $(date '+%F %T') nproc=$(nproc 2>/dev/null || echo ?) loadavg=$(cat /proc/loadavg 2>/dev/null)"
        ps aux --sort=-%cpu 2>/dev/null | head -n 40 || ps aux 2>/dev/null | head -n 40
    } > "$out/cpu_gate_processes.txt" 2>&1
}

running() {
    awk '/^procs_running/ {print $2}' /proc/stat 2>/dev/null
}

if [ ! -r "$psi_file" ]; then
    snapshot
    printf 'NO_PSI %s\n' "$psi_file" > "$out/cpu_gate.txt"
    echo "CPU空闲检查：内核未提供 $psi_file，只记录进程快照，不阻止运行。"
    exit 0
fi

start=$(date +%s)
passes=0
while true; do
    avg10=$(awk '/^some/ {for (i = 1; i <= NF; i++) if ($i ~ /^avg10=/) {sub("avg10=", "", $i); print $i}}' "$psi_file")
    now=$(date +%s)
    printf '%s psi_some_avg10=%s loadavg=%s procs_running=%s\n' \
        "$now" "${avg10:-?}" "$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null)" "$(running)" >> "$samples"
    if [ -n "$avg10" ] && awk -v v="$avg10" -v m="$max" 'BEGIN {exit !(v < m)}'; then
        passes=$((passes + 1))
    else
        passes=0
    fi
    if [ "$passes" -ge 2 ]; then
        snapshot
        printf 'PASS psi_some_avg10=%s max=%s waited_s=%s\n' "$avg10" "$max" "$((now - start))" > "$out/cpu_gate.txt"
        echo "CPU空闲检查通过：CPU压力 avg10=${avg10}% < ${max}%（等了$((now - start))秒）。"
        exit 0
    fi
    if [ $((now - start)) -ge "$wait_s" ]; then
        snapshot
        printf 'FAIL psi_some_avg10=%s max=%s waited_s=%s\n' "$avg10" "$max" "$((now - start))" > "$out/cpu_gate.txt"
        echo "CPU空闲检查未通过：等了$((now - start))秒，CPU压力 avg10 仍为 ${avg10}%（阈值 ${max}%）。主机上有实验之外的CPU负载，停止，不启动GPU。"
        echo '占用CPU最多的进程（完整列表见 cpu_gate_processes.txt）：'
        head -n 16 "$out/cpu_gate_processes.txt"
        exit 1
    fi
    sleep "$interval"
done
