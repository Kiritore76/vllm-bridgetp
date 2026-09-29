#!/usr/bin/env bash
# Event-gated M2 rate actuation smoke on the calibrated five-A100 host.

run_m2_low_to_high_a100_smoke() {
  cd /root/autodl-tmp/bridgetp/vllm_bridge || return 1
  source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate || return 1
  export OMP_NUM_THREADS=1
  set -o pipefail

  local expected_revision="${BRIDGETP_EXPECTED_REVISION:?set BRIDGETP_EXPECTED_REVISION}"
  local model=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
  local base=/root/autodl-tmp/bridgetp/a1d_manifests/working/a1d-full-smoke-20260922T153543Z-output-1024.json
  local survival=/root/autodl-tmp/bridgetp/phase9_cap0_inputs/survival_table_m1_v1.json
  local guard=/root/autodl-tmp/bridgetp/phase9_cap0_manifests/frozen/guard_free_kv_tokens.txt
  local root=/root/autodl-tmp/bridgetp/results/migration_manager_m2
  local run="$root/a100-m2-event-low-high-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  local manifest="$run/inputs/event_source_pressure.json"
  local out="$run/online"
  local manifest_sha runner_rc

  [[ "$(git rev-parse HEAD)" == "$expected_revision" ]] || {
    echo "HEAD differs from BRIDGETP_EXPECTED_REVISION; no run"
    return 1
  }
  [[ "$(git branch --show-current)" == bridgetp/runtime-controller ]] || {
    echo "Wrong branch; no run"
    return 1
  }
  [[ -z "$(git status --porcelain)" ]] || {
    echo "Worktree has uncommitted files; no run"
    git status --short --branch
    return 1
  }
  nvidia-smi -L || return 1
  [[ "$(nvidia-smi -L | grep -c A100-PCIE-40GB)" -eq 5 ]] || {
    echo "Expected five A100-PCIE-40GB GPUs; no run"
    return 1
  }
  local file
  for file in "$model/config.json" "$base" "$survival" "$guard"; do
    [[ -f "$file" ]] || { echo "Missing input: $file"; return 1; }
  done
  [[ "$(cat "$guard")" == 8448 ]] || return 1
  [[ "$(sha256sum "$base" | cut -d' ' -f1)" == 47e4cf7f4d055eb82f32179fead20f3f1d9f9f061e51c16cbfd5755fae03b2ff ]] || return 1
  [[ "$(sha256sum "$survival" | cut -d' ' -f1)" == 031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a ]] || return 1
  [[ "$(sha256sum "$guard" | cut -d' ' -f1)" == 0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b ]] || return 1

  python -m unittest discover -s tests/bridge_tp -p test_manager_m2.py || return 1
  mkdir -p "$run/inputs" || return 1
  python tools/bridge_tp/build_experiment_a4_pressure_manifest.py \
    --base-target-manifest "$base" --out "$manifest" \
    --source-jobs 8 --source-prompt-tokens 1920 \
    --source-output-tokens 1024 --source-start-after-s 0 \
    --source-start-interval-s 0.05 \
    --source-start-after-m2-initial || return 1
  python tools/bridge_tp/run_phase9_capacity_background.py \
    --manifest "$manifest" --out-dir "$run/background_validate" \
    --validate-only || return 1
  manifest_sha="$(sha256sum "$manifest" | cut -d' ' -f1)"
  cp "$base" "$survival" "$guard" "$run/inputs/" || return 1
  {
    echo "run=$run"
    echo "model=$model"
    echo "revision=$(git rev-parse HEAD)"
    git status --short --branch
    nvidia-smi -L
    sha256sum "$base" "$manifest" "$survival" "$guard"
    echo "guard=$(cat "$guard")"
  } | tee "$run/preflight.txt"

  # LOW=0.1 only stretches this diagnostic copy. Production LOW remains 0.5.
  python tools/bridge_tp/run_shadow_strategy_online_validation.py \
    --phase smoke --repetitions 1 \
    --model-path "$model" --manifest "$manifest" \
    --survival-table "$survival" --guard-file "$guard" \
    --out-root "$out" --expected-revision "$expected_revision" \
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
    --manager-m2-require-low-to-high \
    --m2-low-gib-s 0.1 --m2-medium-gib-s 2.4 --m2-high-gib-s 8.0 \
    --source-pressure --minimum-ready-source-jobs 0 \
    --trigger-output-tokens 64 --bridge-output-tokens 96 \
    --commit-timing EARLIEST_READY \
    --anchor-prompt-tokens 2048 --anchor-max-tokens 1024 \
    --minimum-ready-target-jobs 2 \
    --tp1-gpu 0 --tp4-gpus 1,2,3,4 \
    --tp1-port 8001 --tp4-port 8200 \
    --snapshot-port 29800 --delta-port 29900 --delivery-port 30000 \
    --gpu-direct-base-port 30400 --ready-notification-port 30500 \
    2>&1 | tee "$run/online.console.txt"
  runner_rc=${PIPESTATUS[0]}
  echo "runner exit code: $runner_rc" | tee "$run/runner.exit.txt"

  python - "$out" >"$run/transition_summary.txt" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
run = out / "r01_shadow_only"
accept = run / "provenance" / "shadow_online_acceptance.json"
audit = run / "controller" / "phase9_audit.jsonl"
sender = run / "controller" / "gpu_direct_sender.json"
if accept.is_file():
    data = json.loads(accept.read_text())
    print("acceptance:", data.get("status"), data.get("errors"))
if audit.is_file():
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    for row in rows:
        if row.get("kind") == "manager_m2_initial_rate":
            print("initial rate:", row.get("decision"))
        elif row.get("kind") == "rate" and (row.get("manager_m2_decision") or {}).get("action") == "SET_RATE":
            print("rate change:", row.get("manager_m2_decision"))
if sender.is_file():
    data = json.loads(sender.read_text())
    ranks = data.get("ranks") or []
    chunks = ranks[0].get("history_pacing_chunks", []) if ranks else []
    by_rate = {}
    for chunk in chunks:
        rate = chunk["requested_rate_gib_s"]
        by_rate[rate] = by_rate.get(rate, 0) + chunk["aggregate_bytes"]
    print("paced history bytes by rate:", by_rate)
PY
  cat "$run/transition_summary.txt"
  tar -czf "$run.tar.gz" -C "$root" "$(basename "$run")" || return 1
  echo "raw result directory: $run"
  echo "archive to retrieve: $run.tar.gz"
  return "$runner_rc"
}

run_m2_low_to_high_a100_smoke
