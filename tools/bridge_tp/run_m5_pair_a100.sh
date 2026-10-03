#!/usr/bin/env bash
# Same-host M5 off/on performance smoke; preserves both result archives.

run_m5_pair_a100() {
  local repo=/root/autodl-tmp/bridgetp/vllm_bridge
  local branch=bridgetp/m5-predictor-integration
  local revision="${BRIDGETP_EXPECTED_REVISION:?set BRIDGETP_EXPECTED_REVISION}"
  local checkpoint="${BRIDGETP_M5_CHECKPOINT:-/root/autodl-tmp/bridgetp/m5-seed44.pt}"
  local checkpoint_sha=7d55bec981884ce50aa986693e0e2f687b7f1f4e2ae6f609a833ec50228f506f
  local baseline_rc m5_rc

  cd "$repo" || return 1
  source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate || return 1
  [[ "$(git branch --show-current)" == "$branch" ]] || {
    echo "Wrong branch; no run"
    return 1
  }
  [[ "$(git rev-parse HEAD)" == "$revision" ]] || {
    echo "HEAD differs from expected revision; no run"
    return 1
  }
  [[ -z "$(git status --porcelain)" ]] || {
    echo "Uncommitted work is present; no run"
    git status --short --branch
    return 1
  }
  [[ -f "$checkpoint" ]] || {
    echo "Checkpoint is missing: $checkpoint"
    return 1
  }
  echo "$checkpoint_sha  $checkpoint" | sha256sum -c - || return 1

  echo "=== Baseline: M5 off ==="
  BRIDGETP_EXPECTED_BRANCH="$branch" \
  BRIDGETP_M5_SHADOW=0 \
  BRIDGETP_M2_LOW_GIB_S=0.5 \
  BRIDGETP_M2_REQUIRE_LOW_TO_HIGH=0 \
  BRIDGETP_M3_COMMIT=1 \
  BRIDGETP_M4_CANCEL=1 \
  bash tools/bridge_tp/run_m2_low_to_high_a100_smoke.sh
  baseline_rc=$?
  echo "baseline_exit_code=$baseline_rc"
  [[ "$baseline_rc" -eq 0 ]] || return "$baseline_rc"

  echo "=== M5 predictor shadow on ==="
  BRIDGETP_EXPECTED_BRANCH="$branch" \
  BRIDGETP_M5_SHADOW=1 \
  BRIDGETP_M5_CHECKPOINT="$checkpoint" \
  BRIDGETP_M5_CHECKPOINT_SHA256="$checkpoint_sha" \
  BRIDGETP_M2_LOW_GIB_S=0.5 \
  BRIDGETP_M2_REQUIRE_LOW_TO_HIGH=0 \
  BRIDGETP_M3_COMMIT=1 \
  BRIDGETP_M4_CANCEL=1 \
  bash tools/bridge_tp/run_m2_low_to_high_a100_smoke.sh
  m5_rc=$?
  echo "m5_exit_code=$m5_rc"
  return "$m5_rc"
}

run_m5_pair_a100
