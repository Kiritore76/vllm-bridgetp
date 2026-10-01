#!/usr/bin/env bash
# Use on the A100 machine holding decoder31's merged 2000-request capture.
# BRIDGETP_EXPECTED_REVISION must be the full pushed commit SHA.

run_predictor_regularization() {
  cd /root/autodl-tmp/bridgetp/vllm_bridge || return 1
  source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate || return 1

  local expected_machine=autodl-container-6eb54ead64-389e397a
  local expected_revision="${BRIDGETP_EXPECTED_REVISION:-}"
  local runner_python=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
  local raw_source=/root/autodl-tmp/bridgetp/length_predictor/inputs/2023-04-12_oasst_prompts.messages.jsonl.gz
  local model_path=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
  local data_root=/root/autodl-tmp/bridgetp/results/length_predictor/oasst1-staged-v2-decoder31
  local capture_dir="$data_root/capture-2000"
  local baseline_report="$data_root/trained-2000-20260930T235401283970Z/report.json"
  local gpu_names gpu_pids result_root runner_log

  if [[ "$(hostname)" != "$expected_machine" ]]; then
    echo "机器不符；预期：$expected_machine；实际：$(hostname)"
    return 1
  fi
  if [[ ! "$expected_revision" =~ ^[0-9a-f]{40}$ ]] ||
     [[ "$(git rev-parse HEAD)" != "$expected_revision" ]]; then
    echo "请设置完整 BRIDGETP_EXPECTED_REVISION；HEAD 不符，未运行"
    git rev-parse HEAD
    return 1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "工作区仍有未提交文件，未运行"
    git status --short --branch
    return 1
  fi
  if pgrep -af 'tools/bridge_tp/(run_predictor_(large_a100|capture|layer_probe_a100|ablation_a100|regularization_a100)|train_predictor_distribution)\.py'; then
    echo "已有预测器任务，等其完成后再运行"
    return 1
  fi
  gpu_names=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\r') || return 1
  if [[ "$gpu_names" != NVIDIA\ A100-PCIE-40GB ]]; then
    echo "GPU 名称或数量不符：$gpu_names"
    return 1
  fi
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || return 1
  if [[ -n "$gpu_pids" ]]; then
    nvidia-smi
    echo "GPU 正忙，未启动新任务"
    return 1
  fi
  hostname
  git rev-parse HEAD
  nvidia-smi -L || return 1
  echo "原始数据：$raw_source"
  echo "采集目录：$capture_dir"
  echo "旧基线报告：$baseline_report"
  echo "621ccd86a6ef320ca4e24c137121bd4b39bcc7a0df839f0897fcc965ef2076ed  $raw_source" | sha256sum -c - || return 1
  echo "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e  $model_path/config.json" | sha256sum -c - || return 1
  echo "8dc77e63c2b93d8448466f9ee2fbaa4f24489678c9049c52e8d1fd81c3c9424b  $capture_dir/input_requests.jsonl" | sha256sum -c - || return 1
  echo "a759790101695639160b17ef8e32570c908e71c69deab6916419607658c51163  $baseline_report" | sha256sum -c - || return 1
  if [[ ! -s "$capture_dir/features/features.sqlite3" ]]; then
    echo "原始隐藏状态数据库缺失，未运行"
    return 1
  fi
  env OMP_NUM_THREADS=4 "$runner_python" -m unittest \
    tests.bridge_tp.test_predictor_distribution \
    tests.bridge_tp.test_predictor_ablation \
    tests.bridge_tp.test_predictor_regularization -q || return 1

  result_root="/root/autodl-tmp/bridgetp/results/length_predictor/decoder31-regularization2000-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  runner_log="${result_root}.launcher.log"
  mkdir -p "$(dirname "$result_root")" || return 1
  nohup env OMP_NUM_THREADS=4 bash -c '
    result_root=$1
    runner_log=$2
    runner_python=$3
    expected_revision=$4
    expected_machine=$5
    capture_dir=$6
    baseline_report=$7
    raw_source=$8
    model_path=$9
    "$runner_python" -u tools/bridge_tp/run_predictor_regularization_a100.py \
      --expected-revision "$expected_revision" \
      --expected-hostname "$expected_machine" \
      --run-dir "$capture_dir" --baseline-report "$baseline_report" \
      --source "$raw_source" --model "$model_path" \
      --out-dir "$result_root"
    runner_rc=$?
    echo "regularization_exit_code=$runner_rc"
    mkdir -p "$result_root"
    printf "%s\n" "$runner_rc" > "$result_root/runner_exit_code.txt"
    cp "$runner_log" "$result_root/runner.log"
    tar -czf "${result_root}.tar.gz" -C "$(dirname "$result_root")" "$(basename "$result_root")"
    archive_rc=$?
    echo "archive_exit_code=$archive_rc"
    if [[ "$archive_rc" -eq 0 ]]; then
      echo "archive_to_retrieve=${result_root}.tar.gz"
    fi
  ' regularization "$result_root" "$runner_log" "$runner_python" \
    "$expected_revision" "$expected_machine" "$capture_dir" \
    "$baseline_report" "$raw_source" "$model_path" \
    > "$runner_log" 2>&1 < /dev/null &

  echo "后台进程 PID：$!"
  echo "结果目录：$result_root"
  echo "实时查看：tail -n 60 -F '$runner_log'"
  echo "结束后拿回：${result_root}.tar.gz"
}

run_predictor_regularization
