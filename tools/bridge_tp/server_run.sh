#!/usr/bin/env bash
# 当前任务：六卡机器只读采集拓扑、NUMA、PCIe、CPU和内存，用来定A/B/C三组（六卡规划第0阶段）。
# 不启动GPU计算，不改任何文件，只在结果目录写文本并打包。
# cd /root/autodl-tmp/bridgetp/vllm_bridge && git pull --ff-only && bash tools/bridge_tp/server_run.sh
set -uo pipefail
repo=/root/autodl-tmp/bridgetp/vllm_bridge
branch=bridgetp/tp2-4090d-qwen3-8b
cd "$repo" || exit 1
[ "$(git branch --show-current)" = "$branch" ] || { echo '分支不匹配。'; exit 1; }
mkdir -p /root/autodl-tmp/bridgetp/results || exit 1
run=/root/autodl-tmp/bridgetp/results/topology_$(date +%Y%m%d_%H%M%S)_${RANDOM}
mkdir "$run" || exit 1
git rev-parse HEAD > "$run/revision.txt"
echo "本批目录：$run"
echo '只读采集：GPU列表、拓扑矩阵、P2P、PCIe链路、NUMA、CPU、内存、磁盘。'
# 每条命令单独记录，失败不影响其余项。
probe() {
    local name=$1; shift
    { echo "\$ $*"; "$@"; echo "[退出码 $?]"; } > "$run/$name.txt" 2>&1
    echo "== $name =="; cat "$run/$name.txt"
}
probe gpu_list nvidia-smi -L
probe gpu_status nvidia-smi
probe topo_matrix nvidia-smi topo -m
probe p2p_read nvidia-smi topo -p2p r
probe p2p_write nvidia-smi topo -p2p w
probe pcie nvidia-smi --query-gpu=index,name,pci.bus_id,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,memory.total,memory.used,driver_version --format=csv
probe compute_apps nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
probe numa_nodes bash -c 'for d in /sys/bus/pci/devices/*; do [ "$(cat "$d/vendor" 2>/dev/null)" = 0x10de ] && [ "$(cat "$d/class" 2>/dev/null | cut -c1-6)" = 0x0300 -o "$(cat "$d/class" 2>/dev/null | cut -c1-6)" = 0x0302 ] && echo "$(basename "$d") numa=$(cat "$d/numa_node")"; done'
probe numactl bash -c 'command -v numactl >/dev/null && numactl -H || echo "numactl未安装"'
probe lscpu lscpu
probe cpu_limit bash -c 'nproc; cat /sys/fs/cgroup/cpu.max 2>/dev/null || cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2>/dev/null'
probe memory free -h
probe disk df -h /root/autodl-tmp
probe system bash -c 'uname -a; cat /etc/os-release | head -3; nvcc --version 2>/dev/null | tail -2'
cd "$(dirname "$run")" || exit 1
tar -czf "$run.tar.gz" "$(basename "$run")" || exit 1
sha256sum "$run.tar.gz" > "$run.tar.gz.sha256" || exit 1
echo "请拿回：$run.tar.gz"
echo "以及：$run.tar.gz.sha256"
