#!/usr/bin/env bash
# Reuse the audited train1000 capture; train the small predictor on one A100.
main() {
  local repo=/root/autodl-tmp/bridgetp/vllm_bridge
  local capture=${BRIDGETP_CAPTURE_DIR:-/root/autodl-tmp/bridgetp/results/length_predictor/oasst1-train1000-20260930T030435Z-1173}
  local expected=${BRIDGETP_EXPECTED_REVISION:-}
  local expected_host=${BRIDGETP_EXPECTED_HOSTNAME:-autodl-container-db401188fa-e7ef73f0}
  if [[ -z "$expected" ]]; then
    echo 'Set BRIDGETP_EXPECTED_REVISION to the full training commit SHA.'
    return 1
  fi
  cd "$repo" || return 1
  if [[ "$(hostname)" != "$expected_host" ]]; then
    echo "Machine differs from expected hostname: $expected_host"
    hostname
    return 1
  fi
  source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate || return 1
  export OMP_NUM_THREADS=4
  hostname
  git rev-parse HEAD
  nvidia-smi -L
  echo "capture_dir=$capture"
  local out="${capture}-distribution-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  local log="${out}.runner.log"
  python tools/bridge_tp/train_predictor_distribution.py \
    --run-dir "$capture" --out-dir "$out" \
    --expected-revision "$expected" \
    --expected-capture-revision f04b2ac32805105b34f8dab525294d512e65ab95 \
    --expected-input-sha256 bfb7e68b6de2ca8e75354e3fe0bca81078acca32e44b38bbe9498939020db017 \
    --expected-gpu-name 'NVIDIA A100-PCIE-40GB' --expected-gpu-count 1 \
    2>&1 | tee "$log"
  local train_rc=${PIPESTATUS[0]}
  mkdir -p "$out" || return 1
  mv "$log" "$out/runner.log" || return 1
  tar -czf "${out}.tar.gz" -C "$(dirname "$out")" "$(basename "$out")"
  local archive_rc=$?
  echo "training_exit_code=$train_rc"
  echo "archive_exit_code=$archive_rc"
  if [[ "$archive_rc" -eq 0 ]]; then
    echo "archive_to_retrieve=${out}.tar.gz"
  fi
  if [[ "$train_rc" -ne 0 ]]; then
    return "$train_rc"
  fi
  return "$archive_rc"
}
main "$@"
