#!/usr/bin/env bash
# Event-gated M2 rate actuation smoke on the calibrated five-A100 host.

run_m2_low_to_high_a100_smoke() {
  cd /root/autodl-tmp/bridgetp/vllm_bridge || return 1
  source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate || return 1
  export OMP_NUM_THREADS=1
  set -o pipefail

  local expected_revision="${BRIDGETP_EXPECTED_REVISION:?set BRIDGETP_EXPECTED_REVISION}"
  local low_gib_s="${BRIDGETP_M2_LOW_GIB_S:-0.1}"
  local m1_min_output_tokens="${BRIDGETP_M1_MIN_OUTPUT_TOKENS:-32}"
  local require_low_to_high="${BRIDGETP_M2_REQUIRE_LOW_TO_HIGH:-1}"
  local m3_commit="${BRIDGETP_M3_COMMIT:-0}"
  local m4_cancel="${BRIDGETP_M4_CANCEL:-0}"
  local m4_expect_cancel="${BRIDGETP_M4_EXPECT_CANCEL:-0}"
  local anchor_max_tokens=1024 trigger_output_tokens=64
  local source_jobs=8 source_prompt_tokens=1920 source_output_tokens=1024
  local mode=event-low-high
  local m2_requirement=()
  local m3_requirement=()
  if [[ "$require_low_to_high" == 0 ]]; then
    mode=split-capacity
  elif [[ "$require_low_to_high" == 1 ]]; then
    m2_requirement+=(--manager-m2-require-low-to-high)
  else
    echo "BRIDGETP_M2_REQUIRE_LOW_TO_HIGH must be 0 or 1"
    return 1
  fi
  if [[ "$m3_commit" == 1 ]]; then
    m3_requirement=(--manager-m3-commit)
    mode=m3-commit
  elif [[ "$m3_commit" != 0 ]]; then
    echo "BRIDGETP_M3_COMMIT must be 0 or 1"
    return 1
  fi
  if [[ "$m4_cancel" == 1 && "$m3_commit" == 1 ]]; then
    m3_requirement+=(--manager-m4-cancel)
    mode=m4-cancel
  elif [[ "$m4_cancel" != 0 ]]; then
    echo "BRIDGETP_M4_CANCEL requires BRIDGETP_M3_COMMIT=1"
    return 1
  fi
  if [[ "$m4_expect_cancel" == 1 && "$m4_cancel" == 1 ]]; then
    [[ "$require_low_to_high" == 0 ]] || {
      echo "M4 cancellation smoke requires BRIDGETP_M2_REQUIRE_LOW_TO_HIGH=0"
      return 1
    }
    m3_requirement+=(--manager-m4-expect-cancel)
    anchor_max_tokens=128
    trigger_output_tokens=32
    source_jobs=1
    source_prompt_tokens=128
    source_output_tokens=192
  elif [[ "$m4_expect_cancel" != 0 ]]; then
    echo "BRIDGETP_M4_EXPECT_CANCEL requires BRIDGETP_M4_CANCEL=1"
    return 1
  fi
  local model=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
  local base=/root/autodl-tmp/bridgetp/a1d_manifests/working/a1d-full-smoke-20260922T153543Z-output-1024.json
  local survival=/root/autodl-tmp/bridgetp/phase9_cap0_inputs/survival_table_m1_v1.json
  local guard=/root/autodl-tmp/bridgetp/phase9_cap0_manifests/frozen/guard_free_kv_tokens.txt
  local root=/root/autodl-tmp/bridgetp/results/migration_manager_m2
  local run="$root/a100-m2-$mode-$(date -u +%Y%m%dT%H%M%SZ)-$$"
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
  if [[ "$m3_commit" == 1 ]]; then
    python -m unittest tests.bridge_tp.test_manager_m3 || return 1
  fi
  if [[ "$m4_cancel" == 1 ]]; then
    python -m unittest tests.bridge_tp.test_manager_m4 || return 1
  fi
  python -m unittest discover -s tests/bridge_tp \
    -p test_phase9_capacity_pilot.py || return 1
  python -m unittest discover -s tests/bridge_tp \
    -p test_phase9_telemetry_control.py || return 1
  python -m unittest \
    tests.bridge_tp.test_shadow_strategy_online_runner.TestOnlineStrategyTiming \
    || return 1
  python -m unittest \
    tests.bridge_tp.test_phase9_online_integration.TestLazyActionBinding.test_urgent_source_waits_at_prearmed_candidate_for_history \
    tests.bridge_tp.test_phase9_online_integration.TestProxyRecorder.test_target_cleanup_maps_openai_completion_request_id \
    || return 1
  mkdir -p "$run/inputs" || return 1
  python tools/bridge_tp/build_experiment_a4_pressure_manifest.py \
    --base-target-manifest "$base" --out "$manifest" \
    --source-jobs "$source_jobs" --source-prompt-tokens "$source_prompt_tokens" \
    --source-output-tokens "$source_output_tokens" --source-start-after-s 0 \
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
    echo "m2_low_gib_s=$low_gib_s"
    echo "m1_min_output_tokens=$m1_min_output_tokens"
    echo "m2_require_low_to_high=$require_low_to_high"
    echo "m3_commit=$m3_commit"
    echo "m4_cancel=$m4_cancel"
    echo "m4_expect_cancel=$m4_expect_cancel"
    echo "anchor_max_tokens=$anchor_max_tokens"
    echo "trigger_output_tokens=$trigger_output_tokens"
    if [[ "$m3_commit" == 1 ]]; then
      echo "m3_policy=COMMIT_EARLIEST_WHEN_READY"
    fi
  } | tee "$run/preflight.txt"

  # Default LOW=0.1 stretches a diagnostic copy; set the measured 0.5 for
  # the calibrated-profile transition check.
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
    --m1-min-output-tokens "$m1_min_output_tokens" \
    "${m2_requirement[@]}" \
    "${m3_requirement[@]}" \
    --m2-low-gib-s "$low_gib_s" --m2-medium-gib-s 2.4 --m2-high-gib-s 8.0 \
    --source-pressure --minimum-ready-source-jobs 0 \
    --trigger-output-tokens "$trigger_output_tokens" --bridge-output-tokens 96 \
    --commit-timing EARLIEST_READY \
    --anchor-prompt-tokens 2048 --anchor-max-tokens "$anchor_max_tokens" \
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
    print("urgent history wait ms:", data.get("urgent_history_wait_ms"))
models = []
prefill_totals = []
decode_totals = []
if audit.is_file():
    rows = [json.loads(line) for line in audit.read_text().splitlines()]
    for row in rows:
        if row.get("kind") == "telemetry":
            signal = row.get("capacity_signal") or {}
            if signal.get("prefill_scheduled_tokens_total") is not None:
                prefill_totals.append(signal["prefill_scheduled_tokens_total"])
            if signal.get("decode_scheduled_tokens_total") is not None:
                decode_totals.append(signal["decode_scheduled_tokens_total"])
        if row.get("kind") == "manager_m2_initial_rate":
            print("initial rate:", row.get("decision"))
            models.append((row.get("decision") or {}).get("source_capacity_model"))
        elif row.get("kind") == "manager_m3_candidate_decision":
            print("M3 candidate:", row.get("decision"))
            print("M3 TPOT sources:", row.get("source_tpot_source"),
                  row.get("target_tpot_source"))
        elif row.get("kind") == "rate" and (row.get("manager_m2_decision") or {}).get("action") == "SET_RATE":
            print("rate change:", row.get("manager_m2_decision"))
            models.append(row["manager_m2_decision"].get("source_capacity_model"))
print("M2 capacity models:", models)
print("prefill/decode counter ranges:",
      (min(prefill_totals), max(prefill_totals)) if prefill_totals else None,
      (min(decode_totals), max(decode_totals)) if decode_totals else None)
if sender.is_file():
    data = json.loads(sender.read_text())
    ranks = data.get("ranks") or []
    chunks = ranks[0].get("history_pacing_chunks", []) if ranks else []
    by_rate = {}
    for chunk in chunks:
        rate = chunk["requested_rate_gib_s"]
        by_rate[rate] = by_rate.get(rate, 0) + chunk["aggregate_bytes"]
    print("paced history bytes by rate:", by_rate)
if not models or any(
    model != "prefill_reservation_plus_decode_growth" for model in models
):
    raise SystemExit("M2 did not use separated prefill/decode evidence")
if (not prefill_totals or max(prefill_totals) == min(prefill_totals)
        or not decode_totals or max(decode_totals) == min(decode_totals)):
    raise SystemExit("source prefill/decode counters did not both advance")
PY
  local summary_rc=$?
  cat "$run/transition_summary.txt"
  tar -czf "$run.tar.gz" -C "$root" "$(basename "$run")" || return 1
  echo "raw result directory: $run"
  echo "archive to retrieve: $run.tar.gz"
  if [[ "$runner_rc" -eq 0 && "$summary_rc" -ne 0 ]]; then
    return "$summary_rc"
  fi
  return "$runner_rc"
}

run_m2_low_to_high_a100_smoke
