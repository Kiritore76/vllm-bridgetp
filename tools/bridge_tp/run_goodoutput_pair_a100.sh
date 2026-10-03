#!/usr/bin/env bash
# One diagnostic STAY/MIGRATE pair. No shell exit; failures retain an archive.

run_goodoutput_pair_a100() {
  local repo=/root/autodl-tmp/bridgetp/vllm_bridge
  local branch=bridgetp/m5-predictor-integration
  local expected="${BRIDGETP_EXPECTED_REVISION:?set BRIDGETP_EXPECTED_REVISION}"
  local checkpoint="${BRIDGETP_M5_CHECKPOINT:?set BRIDGETP_M5_CHECKPOINT}"
  local checkpoint_sha="${BRIDGETP_M5_CHECKPOINT_SHA256:?set BRIDGETP_M5_CHECKPOINT_SHA256}"
  local model=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
  local base=/root/autodl-tmp/bridgetp/a1d_manifests/working/a1d-full-smoke-20260922T153543Z-output-1024.json
  local survival=/root/autodl-tmp/bridgetp/phase9_cap0_inputs/survival_table_m1_v1.json
  local guard=/root/autodl-tmp/bridgetp/phase9_cap0_manifests/frozen/guard_free_kv_tokens.txt
  local root=/root/autodl-tmp/bridgetp/results/goodoutput_pairs
  local run="$root/a100-w2-pilot-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  local manifest="$run/inputs/paired_pressure.json"
  local manifest_sha arm rc=0

  cd "$repo" || return 1
  source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate || return 1
  export OMP_NUM_THREADS=1
  if [[ "$(git rev-parse HEAD)" != "$expected" ||
        "$(git branch --show-current)" != "$branch" ||
        -n "$(git status --porcelain)" ]]; then
    echo "HEAD、分支或未提交工作不符合预期，未运行"
    git status --short --branch
    return 1
  fi
  echo "机器：$(hostname)"
  nvidia-smi -L || return 1
  [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\r' | grep -cx 'NVIDIA A100-PCIE-40GB')" -eq 5 ]] || {
    echo "不是五张 A100-PCIE-40GB，未运行"
    return 1
  }
  echo "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e  $model/config.json" | sha256sum -c - || return 1
  echo "47e4cf7f4d055eb82f32179fead20f3f1d9f9f061e51c16cbfd5755fae03b2ff  $base" | sha256sum -c - || return 1
  echo "031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a  $survival" | sha256sum -c - || return 1
  echo "0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b  $guard" | sha256sum -c - || return 1
  echo "$checkpoint_sha  $checkpoint" | sha256sum -c - || return 1
  [[ "$(cat "$guard")" == 8448 ]] || return 1
  python -m unittest \
    tests.bridge_tp.test_compare_goodoutput_pair \
    tests.bridge_tp.test_audit_goodoutput \
    tests.bridge_tp.test_shadow_strategy_online_runner.TestOnlineStrategyTiming.test_paired_stay_requires_full_tp1_output_and_suppressed_start \
    tests.bridge_tp.test_phase9_online_integration.TestPairedStayDecision -q || return 1

  mkdir -p "$run/inputs" || return 1
  python tools/bridge_tp/build_experiment_a4_pressure_manifest.py \
    --base-target-manifest "$base" --out "$manifest" \
    --source-jobs 8 --source-prompt-tokens 2304 \
    --source-output-tokens 1024 --source-start-after-s 1.0 \
    --source-start-interval-s 0.05 \
    --source-start-after-anchor-first-output || return 1
  manifest_sha=$(sha256sum "$manifest") || return 1
  manifest_sha=${manifest_sha%% *}
  cp "$base" "$survival" "$guard" "$run/inputs/" || return 1
  {
    hostname
    git rev-parse HEAD
    git status --short --branch
    nvidia-smi -L
    sha256sum "$model/config.json" "$checkpoint" "$base" "$manifest" "$survival" "$guard"
  } > "$run/preflight.txt" || return 1

  for arm in stay migrate; do
    local paired_arg=()
    [[ "$arm" == stay ]] && paired_arg=(--paired-stay)
    echo "=== $arm ==="
    python tools/bridge_tp/run_shadow_strategy_online_validation.py \
      --phase smoke --repetitions 1 \
      --model-path "$model" --manifest "$manifest" \
      --survival-table "$survival" --guard-file "$guard" \
      --out-root "$run/$arm" --expected-revision "$expected" \
      --expected-manifest-sha256 "$manifest_sha" \
      --expected-survival-sha256 031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a \
      --expected-guard-sha256 0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b \
      --expected-guard 8448 --tp1-blocks 1968 --tp4-blocks 35739 \
      --shadow-only-only --gpu-resident-shadow \
      --gpu-direct-history --gpu-direct-history-pacing --gpu-direct-delta \
      --gpu-direct-delta-batch-tokens 16 --gpu-direct-delta-flush-ms 25 \
      --ready-sync-mode STREAM_EVENT --ready-notification-mode UDP \
      --persistent-channel --preconnect-persistent-channel --channel-generation 1 \
      --manager-m0-shadow --manager-m1-auto-start --manager-m2-rate \
      --m1-min-output-tokens 96 --m1-source-release-tail-s 5.0 \
      --manager-m2-force-initial-high --manager-m2-expected-profile HIGH \
      --manager-m2-min-history-byte-frac 0.9 \
      --manager-m3-commit --manager-m4-cancel \
      --manager-m5-predictor-shadow \
      --predictor-checkpoint "$checkpoint" \
      --predictor-checkpoint-sha256 "$checkpoint_sha" \
      --m2-low-gib-s 0.5 --m2-medium-gib-s 2.4 --m2-high-gib-s 8.0 \
      --source-pressure --minimum-ready-source-jobs 0 \
      --trigger-output-tokens 64 --bridge-output-tokens 96 \
      --commit-timing EARLIEST_READY \
      --anchor-prompt-tokens 2048 --anchor-max-tokens 1024 \
      --minimum-ready-target-jobs 2 \
      --tp1-gpu 0 --tp4-gpus 1,2,3,4 \
      --tp1-port 8001 --tp4-port 8200 \
      --snapshot-port 29800 --delta-port 29900 --delivery-port 30000 \
      --gpu-direct-base-port 30400 --ready-notification-port 30500 \
      "${paired_arg[@]}" \
      2>&1 | tee "$run/$arm.console.log"
    rc=${PIPESTATUS[0]}
    echo "$arm exit code: $rc"
    [[ "$rc" -eq 0 ]] || break
  done
  if [[ "$rc" -eq 0 ]]; then
    python tools/bridge_tp/compare_goodoutput_pair.py \
      --stay-root "$run/stay" --migrate-root "$run/migrate" \
      --out-json "$run/pair_comparison.json" \
      > "$run/compare.console.log" 2>&1
    rc=$?
  fi
  tar -czf "$run.tar.gz" -C "$root" "$(basename "$run")" || return 1
  echo "配对退出码：$rc"
  echo "请拿回：$run.tar.gz"
  return "$rc"
}

run_goodoutput_pair_a100
echo "函数退出码：$?"
