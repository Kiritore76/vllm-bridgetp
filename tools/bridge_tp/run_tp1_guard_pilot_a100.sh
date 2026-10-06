#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run from the existing AutoDL Bash terminal; this script does not exit it.
# The stress phase repeats held-out content to 4096 tokens and is diagnostic,
# not a sample of the natural long-context distribution for formal fitting.

run_tp1_guard_pilot() {
  local repo=/root/autodl-tmp/bridgetp/vllm_bridge
  local py=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
  local input=/root/autodl-tmp/bridgetp/results/length_predictor/oasst1-long5000-v1-decoder31/inputs-5000/requests.jsonl
  local model=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
  local base=/root/autodl-tmp/bridgetp/a1d_manifests/working/a1d-full-smoke-20260922T153543Z-output-1024.json
  local survival=/root/autodl-tmp/bridgetp/phase9_cap0_inputs/survival_table_m1_v1.json
  local guard=/root/autodl-tmp/bridgetp/phase9_cap0_manifests/frozen/guard_free_kv_tokens.txt
  local checkpoint=/root/autodl-tmp/bridgetp/m5-seed44.pt
  local reference=experiments/phase9/slo/slo_v6_a100_tp4_reference_20261004.json
  local output_root=/root/autodl-tmp/bridgetp/results/tp1_guard_pilot
  local expected=${BRIDGETP_EXPECTED_REVISION:-}
  local seed=20261011
  local phase run_dir archive runner_rc risk_rc natural_rc pack_rc
  local -a phase_args

  [[ -n "$expected" ]] || { echo '请设置 BRIDGETP_EXPECTED_REVISION'; return 1; }
  cd "$repo" || return 1
  source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate || return 1
  echo "机器：$(hostname)"
  git status --short --branch || return 1
  [[ -z "$(git status --porcelain --untracked-files=no)" ]] || {
    echo '仓库有未提交的已跟踪改动，未运行'; return 1;
  }
  [[ "$(git rev-parse HEAD)" == "$expected" ]] || {
    echo 'HEAD 与预期不符，未运行'; return 1;
  }
  nvidia-smi -L || return 1
  [[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | tr -d '\r' \
      | grep -cx 'NVIDIA A100-PCIE-40GB')" -eq 5 ]] || {
    echo '不是五张 A100-PCIE-40GB，未运行'; return 1;
  }
  [[ -f "$py" ]] || { echo "Python 不存在：$py"; return 1; }
  export PATH="$(dirname "$py"):$PATH"
  command -v ninja >/dev/null 2>&1 || {
    echo '缺少 ninja，未运行'; return 1;
  }
  "$py" -c 'import pathlib,sys,vllm,vllm.vllm_flash_attn; root=pathlib.Path(sys.argv[1]).resolve(); path=pathlib.Path(vllm.__file__).resolve(); assert path.is_relative_to(root), f"wrong vLLM checkout: {path}"; print("FlashAttention import: OK",path)' "$repo" || return 1
  echo "75cae22e298548b54b6164b3df9adc9ebdc61ced3a85eecfd7df9e96ea7a1be3  $input" | sha256sum -c - || return 1
  echo "0f2085dbbe2ee251bd6a6a0797d84a6ce34436044d629aa3cba793b43d311a9e  $model/config.json" | sha256sum -c - || return 1
  echo "47e4cf7f4d055eb82f32179fead20f3f1d9f9f061e51c16cbfd5755fae03b2ff  $base" | sha256sum -c - || return 1
  echo "031b06b0e7d663d5a4ad9cf71f2a640123b84d8e85eb4c94d94f44baa20aaa4a  $survival" | sha256sum -c - || return 1
  echo "0e86c353044f9610be1b5511ff21e870823b7f259c40ccde24188d84164b545b  $guard" | sha256sum -c - || return 1
  echo "7d55bec981884ce50aa986693e0e2f687b7f1f4e2ae6f609a833ec50228f506f  $checkpoint" | sha256sum -c - || return 1
  echo "c6d81aad4cc9f6c33e0fadbb5cb600f60acdd27924610b9c60b4d61aa9ff78f3  $reference" | sha256sum -c - || return 1

  export OMP_NUM_THREADS=1
  "$py" -m unittest tests.bridge_tp.test_pilot_holdout_audit \
    tests.bridge_tp.test_tp1_capacity_ledger \
    tests.bridge_tp.test_randomized_goodoutput_pilot_a100 \
    tests.bridge_tp.test_migration_risk_audit -q || return 1

  mkdir -p "$output_root" || return 1
  "$py" tools/bridge_tp/audit_pilot_holdout.py \
    --input "$input" \
    --expected-input-sha256 75cae22e298548b54b6164b3df9adc9ebdc61ced3a85eecfd7df9e96ea7a1be3 \
    --seed "$seed" --out-file "$output_root/holdout_seed${seed}.json" || return 1

  for phase in natural stress; do
    case "$phase" in
      natural)
        phase_args=(--cases p00_source1_target2 p02_source3_target8
          --actions stay)
        ;;
      stress)
        phase_args=(--cases p04_source5_target2 p05_source5_target24
          --actions stay now --source-prompt-tokens 4096
          --source-background-max-tokens 1536)
        ;;
    esac
    run_dir="$output_root/a100-${phase}-seed${seed}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    archive="${run_dir}.tar.gz"
    echo "=== ${phase} pilot ==="
    "$py" tools/bridge_tp/run_randomized_goodoutput_pilot_a100.py \
      --expected-revision "$expected" --portable-hardware \
      --model "$model" --input "$input" --base "$base" \
      --survival "$survival" --guard "$guard" \
      --checkpoint "$checkpoint" --reference "$reference" \
      --seed "$seed" --timing-pilot --random-arrivals \
      --risk-observation-shadow \
      --max-model-len 16384 --tp4-max-model-len 32768 \
      --anchor-context-limit --anchor-total-max-tokens 24000 \
      --background-context-limit --evaluation-horizon-s 300 \
      "${phase_args[@]}" --out-dir "$run_dir" \
      2>&1 | tee "${run_dir}.launcher.log"
    runner_rc=${PIPESTATUS[0]}
    risk_rc=1
    natural_rc=1
    pack_rc=0
    if [[ "$runner_rc" -eq 0 && -f "$archive" ]]; then
      "$py" tools/bridge_tp/audit_migration_risk.py \
        --archive "$archive" --out-dir "$run_dir/risk_audit"
      risk_rc=$?
      "$py" - "$run_dir/pilot_summary.json" > "$run_dir/natural_eos_gate.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
failures = []
for case, arms in summary["cases"].items():
    for action, arm in arms.items():
        if arm["background_finish_reasons"] != {
            "stop": arm["background_jobs"]
        }:
            failures.append(f"{case}/{action}: background not natural EOS")
        if action == "stay" and arm["anchor_source_finish_reason"] != "stop":
            failures.append(f"{case}/{action}: anchor not natural EOS")
print(json.dumps({"status": "PASS" if not failures else "FAIL",
                  "failures": failures}, ensure_ascii=False))
if failures:
    raise SystemExit(1)
PY
      natural_rc=$?
    fi
    if [[ -d "$run_dir" ]]; then
      cp "$output_root/holdout_seed${seed}.json" \
        "$run_dir/holdout_audit.json" || pack_rc=1
      cp "${run_dir}.launcher.log" "$run_dir/launcher.log" || pack_rc=1
      tar -czf "$archive" -C "$output_root" "$(basename "$run_dir")" \
        || pack_rc=1
    fi
    echo "${phase}: runner=$runner_rc risk_audit=$risk_rc natural_eos=$natural_rc pack=$pack_rc"
    echo "请拿回完整压缩包：$archive"
    if [[ "$runner_rc" -ne 0 || "$risk_rc" -ne 0 || "$natural_rc" -ne 0 || "$pack_rc" -ne 0 ]]; then
      echo '本阶段未通过，保留诊断包并停止下一阶段'
      return 1
    fi
  done
}

run_tp1_guard_pilot
pilot_rc=$?
echo "函数退出码：$pilot_rc"
[[ "$pilot_rc" -eq 0 ]]
