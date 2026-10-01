#!/usr/bin/env bash
# BRIDGETP_EXPECTED_REVISION must be a complete, already pushed commit SHA.

run_predictor_long5000() {
  cd /root/autodl-tmp/bridgetp/vllm_bridge || return 1
  source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate || return 1
  local machine=autodl-container-6eb54ead64-389e397a
  local revision="${BRIDGETP_EXPECTED_REVISION:-}"
  local runner_python=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
  local raw=/root/autodl-tmp/bridgetp/length_predictor/inputs/2023-04-12_oasst_prompts.messages.jsonl.gz
  local model=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
  local root=/root/autodl-tmp/bridgetp/results/length_predictor/oasst1-long5000-v1-decoder31
  local names pids log
  if [[ "$(hostname)" != "$machine" ]]; then
    echo "机器不符；预期 $machine；实际 $(hostname)"
    return 1
  fi
  if [[ ! "$revision" =~ ^[0-9a-f]{40}$ ]] || [[ "$(git rev-parse HEAD)" != "$revision" ]]; then
    echo "完整HEAD核对失败，未运行"
    return 1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    git status --short --branch
    echo "有未提交文件，请先保护工作区"
    return 1
  fi
  if pgrep -af 'tools/bridge_tp/(run_predictor_(large_a100|capture|layer_probe_a100|ablation_a100|regularization_a100|long5000_a100)|train_predictor_distribution)\.py'; then
    echo "已有预测器任务，未启动"
    return 1
  fi
  names=$(nvidia-smi --query-gpu=name --format=csv,noheader) || return 1
  if [[ "$names" != 'NVIDIA A100-PCIE-40GB' ]]; then
    echo "GPU名称或数量不符：$names"
    return 1
  fi
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader) || return 1
  if [[ -n "$pids" ]]; then
    nvidia-smi
    echo "GPU正忙，未启动"
    return 1
  fi
  hostname
  git rev-parse HEAD
  nvidia-smi -L || return 1
  echo "原始数据：$raw"
  echo "模型：$model"
  echo "新结果目录：$root"
  echo "621ccd86a6ef320ca4e24c137121bd4b39bcc7a0df839f0897fcc965ef2076ed  $raw" | sha256sum -c - || return 1
  echo "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e  $model/config.json" | sha256sum -c - || return 1
  env OMP_NUM_THREADS=4 "$runner_python" -m unittest \
    tests.bridge_tp.test_predictor_long5000 \
    tests.bridge_tp.test_predictor_large \
    tests.bridge_tp.test_predictor_distribution -q || return 1
  mkdir -p "$root" || return 1
  log="$root/launcher-$(date -u +%Y%m%dT%H%M%SZ)-$$.log"
  nohup env OMP_NUM_THREADS=4 bash -c '
    root=$1
    log=$2
    runner_python=$3
    revision=$4
    machine=$5
    raw=$6
    model=$7
    marker="${log%.log}.started"
    touch "$marker"
    "$runner_python" -u tools/bridge_tp/run_predictor_long5000_a100.py \
      --expected-revision "$revision" --expected-hostname "$machine" \
      --source "$raw" --model "$model" --data-root "$root"
    rc=$?
    echo "runner_exit_code=$rc"
    printf "%s\n" "$rc" > "${log%.log}.exit_code.txt"
    for directory in "$root"/trained-5000-*; do
      [[ -d "$directory" ]] || continue
      [[ "$directory" -nt "$marker" ]] || continue
      cp "$log" "$directory/runner.log"
      printf "%s\n" "$rc" > "$directory/runner_exit_code.txt"
      tar -czf "${directory}.tar.gz" -C "$root" "$(basename "$directory")"
      archive_rc=$?
      echo "training_archive_exit_code=$archive_rc"
      if [[ "$archive_rc" -eq 0 ]]; then
        echo "archive_to_retrieve=${directory}.tar.gz"
      fi
    done
    if [[ "$rc" -ne 0 ]]; then
      diagnostic="$root/diagnostic-$(date -u +%Y%m%dT%H%M%SZ)-$$"
      mkdir -p "$diagnostic"
      cp "$log" "$diagnostic/runner.log"
      printf "%s\n" "$rc" > "$diagnostic/runner_exit_code.txt"
      for file in experiment_preflight.json coverage_progress.json inputs-5000/manifest.json; do
        [[ ! -f "$root/$file" ]] || cp "$root/$file" "$diagnostic/$(basename "$file")"
      done
      for shard in "$root"/shard_*; do
        [[ -d "$shard" ]] || continue
        mkdir -p "$diagnostic/$(basename "$shard")"
        for file in preflight.json summary.json labels.jsonl runner.log; do
          [[ ! -f "$shard/$file" ]] || cp "$shard/$file" "$diagnostic/$(basename "$shard")/$file"
        done
      done
      tar -czf "${diagnostic}.tar.gz" -C "$root" "$(basename "$diagnostic")"
      if [[ "$?" -eq 0 ]]; then
        echo "diagnostic_archive_to_retrieve=${diagnostic}.tar.gz"
      fi
    fi
  ' long5000 "$root" "$log" "$runner_python" "$revision" "$machine" "$raw" "$model" \
    > "$log" 2>&1 < /dev/null &
  echo "后台PID：$!"
  echo "实时查看：tail -n 60 -F '$log'"
  echo "完成后拿回：$root/capture-5000.tar.gz 和 $root/trained-5000-*.tar.gz"
}

run_predictor_long5000
