#!/usr/bin/env bash
# Run only after SSH fast-forward and exact HEAD verification.
# Returns on preflight failure without exiting the interactive terminal.

run_predictor_ablation() {
  cd /root/autodl-tmp/bridgetp/vllm_bridge || return 1
  source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate || return 1

  local ablation_machine=autodl-container-6eb54ead64-389e397a
  local ablation_revision="${BRIDGETP_EXPECTED_REVISION:-}"
  local ablation_python=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
  local ablation_raw=/root/autodl-tmp/bridgetp/length_predictor/inputs/2023-04-12_oasst_prompts.messages.jsonl.gz
  local ablation_model=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
  local ablation_data=/root/autodl-tmp/bridgetp/results/length_predictor/oasst1-staged-v2-decoder31
  local ablation_capture="$ablation_data/capture-2000"
  local ablation_baseline="$ablation_data/trained-2000-20260930T235401283970Z/report.json"
  local ablation_gpu_names ablation_gpu_pids ablation_root ablation_log

  if [[ "$(hostname)" != "$ablation_machine" ]]; then
    echo "机器不符；预期：$ablation_machine；实际：$(hostname)"
    return 1
  fi
  if [[ ! "$ablation_revision" =~ ^[0-9a-f]{40}$ ]] ||
     [[ "$(git rev-parse HEAD)" != "$ablation_revision" ]]; then
    echo "请设置完整 BRIDGETP_EXPECTED_REVISION；HEAD 不符，未运行"
    git rev-parse HEAD
    return 1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "工作区仍有未提交文件，未运行"
    git status --short --branch
    return 1
  fi
  if pgrep -af 'tools/bridge_tp/(run_predictor_(large_a100|capture|layer_probe_a100|ablation_a100)|train_predictor_distribution)\.py'; then
    echo "已有预测器任务，等其完成后再运行"
    return 1
  fi
  ablation_gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\r') || return 1
  if [[ "$ablation_gpu_names" != 'NVIDIA A100-PCIE-40GB' ]]; then
    echo "GPU 名称或数量不符：$ablation_gpu_names"
    return 1
  fi
  ablation_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || return 1
  if [[ -n "$ablation_gpu_pids" ]]; then
    nvidia-smi
    echo "GPU 已有计算进程，未启动第二个任务"
    return 1
  fi
  hostname
  git rev-parse HEAD
  nvidia-smi -L || return 1
  echo "原始数据：$ablation_raw"
  echo "复用采集目录：$ablation_capture"
  echo "旧训练报告：$ablation_baseline"
  echo "621ccd86a6ef320ca4e24c137121bd4b39bcc7a0df839f0897fcc965ef2076ed  $ablation_raw" | sha256sum -c - || return 1
  echo "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e  $ablation_model/config.json" | sha256sum -c - || return 1
  echo "8dc77e63c2b93d8448466f9ee2fbaa4f24489678c9049c52e8d1fd81c3c9424b  $ablation_capture/input_requests.jsonl" | sha256sum -c - || return 1
  echo "a759790101695639160b17ef8e32570c908e71c69deab6916419607658c51163  $ablation_baseline" | sha256sum -c - || return 1
  if [[ ! -s "$ablation_capture/features/features.sqlite3" ]]; then
    echo "缺少第32层原始隐藏状态，未运行；不能用训练结果包替代采集包"
    return 1
  fi
  env OMP_NUM_THREADS=4 "$ablation_python" -m unittest \
    tests.bridge_tp.test_predictor_distribution \
    tests.bridge_tp.test_predictor_ablation -q || return 1

  ablation_root="/root/autodl-tmp/bridgetp/results/length_predictor/decoder31-ablation2000-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  ablation_log="${ablation_root}.launcher.log"
  mkdir -p "$(dirname "$ablation_root")" || return 1
  nohup env OMP_NUM_THREADS=4 bash -c '
    ablation_root=$1
    ablation_log=$2
    ablation_python=$3
    ablation_revision=$4
    ablation_machine=$5
    ablation_capture=$6
    ablation_baseline=$7
    ablation_raw=$8
    ablation_model=$9
    "$ablation_python" -u tools/bridge_tp/run_predictor_ablation_a100.py \
      --expected-revision "$ablation_revision" \
      --expected-hostname "$ablation_machine" \
      --run-dir "$ablation_capture" --baseline-report "$ablation_baseline" \
      --source "$ablation_raw" --model "$ablation_model" \
      --out-dir "$ablation_root"
    ablation_rc=$?
    echo "ablation_exit_code=$ablation_rc"
    mkdir -p "$ablation_root"
    printf "%s\n" "$ablation_rc" > "$ablation_root/runner_exit_code.txt"
    cp "$ablation_log" "$ablation_root/runner.log"
    tar -czf "${ablation_root}.tar.gz" -C "$(dirname "$ablation_root")" "$(basename "$ablation_root")"
    ablation_archive_rc=$?
    echo "archive_exit_code=$ablation_archive_rc"
    if [[ "$ablation_archive_rc" -eq 0 ]]; then
      echo "archive_to_retrieve=${ablation_root}.tar.gz"
    fi
  ' ablation "$ablation_root" "$ablation_log" "$ablation_python" \
    "$ablation_revision" "$ablation_machine" "$ablation_capture" \
    "$ablation_baseline" "$ablation_raw" "$ablation_model" \
    > "$ablation_log" 2>&1 < /dev/null &
  echo "后台进程 PID：$!"
  echo "结果目录：$ablation_root"
  echo "实时查看：tail -n 60 -F '$ablation_log'"
  echo "完成后拿回：${ablation_root}.tar.gz"
}

run_predictor_ablation
