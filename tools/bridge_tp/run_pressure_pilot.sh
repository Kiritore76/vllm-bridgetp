#!/usr/bin/env bash
# Inputs and launcher are tracked together; pull the branch before running.
# Run from the repository: bash tools/bridge_tp/run_pressure_pilot.sh
# No exit, reset, clean, stash, force push or unrelated process termination.
bridgetp_pressure_pilot_20261015_v2() {
  local base=/root/autodl-tmp/bridgetp
  local repo="$base/vllm_bridge"
  local branch=bridgetp/predictor-nonuniform84-20261010
  local input="$repo/experiments/phase9/pressure_balanced_v2"
  local stamp backup origin batch console rc pack_rc mode launcher_source runner
  local -a run_args
  runner=tools/bridge_tp/launch_pressure_balanced_pilot.py
  case "${1:---smoke-only}" in
    --smoke-only) mode=smoke; run_args=(--smoke-only) ;;
    --full) mode=function; run_args=() ;;
    --calibrate) mode=calibration; run_args=(); runner=tools/bridge_tp/run_pressure_calibration.py ;;
    --pressure-screen) mode=pressure-screen; run_args=(--pressure-screen); runner=tools/bridge_tp/run_pressure_calibration.py ;;
    --efficient-region) mode=efficient-region; run_args=(--efficient-region); runner=tools/bridge_tp/run_pressure_calibration.py ;;
    --efficient-region-last) mode=efficient-region-last; run_args=(--efficient-region-last); runner=tools/bridge_tp/run_pressure_calibration.py ;;
    --health-nd2) mode=health-nd2; run_args=(--health-nd2); runner=tools/bridge_tp/run_pressure_calibration.py ;;
    --compare) mode=comparison; run_args=(); runner=tools/bridge_tp/run_pressure_comparison.py ;;
    --kv-release) mode=kv-release; run_args=(--kv-release); runner=tools/bridge_tp/run_pressure_comparison.py ;;
    --sustained) mode=sustained; run_args=(--sustained); runner=tools/bridge_tp/run_pressure_comparison.py ;;
    --health-screen) mode=health-screen; run_args=(--health-screen); runner=tools/bridge_tp/run_pressure_comparison.py ;;
    --health-sustained) mode=health-sustained; run_args=(--health-sustained); runner=tools/bridge_tp/run_pressure_comparison.py ;;
    --health-kv-only) mode=health-kv-only; run_args=(--health-kv-only); runner=tools/bridge_tp/run_pressure_comparison.py ;;
    --cancel-smoke) mode=cancel-smoke; run_args=(--cancel-smoke); runner=tools/bridge_tp/run_pressure_comparison.py ;;
    --pool-compare) mode=pool-compare; run_args=(--pool-compare); runner=tools/bridge_tp/run_pressure_comparison.py ;;
    --pool-compare-pair) mode=pool-compare-pair; run_args=(--pool-compare-pair); runner=tools/bridge_tp/run_pressure_comparison.py ;;
    --workload-pilot) mode=workload-pilot; run_args=(--workload-pilot); runner=tools/bridge_tp/run_pressure_comparison.py ;;
    *) printf 'Use no argument for smoke; --full, --calibrate, --pressure-screen, --efficient-region, --efficient-region-last, --health-nd2, --compare, --kv-release, --sustained, --health-screen, --health-sustained, --health-kv-only, --cancel-smoke, --pool-compare, --pool-compare-pair or --workload-pilot.\n'; return 2 ;;
  esac
  [ "$#" -le 1 ] || return 2
  launcher_source="$(realpath -- "${BASH_SOURCE[0]}")" || return
  printf '[pilot] mode=%s; checking checkout and inputs\n' "$mode"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$" || return
  cd "$repo" || return
  [ "$(git rev-parse --show-toplevel)" = "$repo" ] || return 1
  origin="$(git remote get-url origin)" || return
  case "$origin" in
    ssh://git@ssh.github.com:443/Kiritore76/vllm-bridgetp.git|ssh://git@github.com:22/Kiritore76/vllm-bridgetp.git|git@github.com:Kiritore76/vllm-bridgetp.git) ;;
    *) printf 'Unexpected repository origin: %s\n' "$origin"; return 1 ;;
  esac
  if [ -n "$(git status --porcelain)" ]; then
    backup="$base/results/pressure-preserve-$stamp"
    mkdir -p "$backup" || return
    git status --porcelain=v1 > "$backup/status.txt" || return
    git rev-parse HEAD > "$backup/head.txt" || return
    git branch --show-current > "$backup/branch.txt" || return
    git diff --binary > "$backup/unstaged.patch" || return
    git diff --cached --binary > "$backup/staged.patch" || return
    git diff --name-only -z --diff-filter=ACMRTUXB HEAD > "$backup/paths.z" || return
    git ls-files --others --exclude-standard -z >> "$backup/paths.z" || return
    tar --null -T "$backup/paths.z" -czf "$backup/dirty-files.tar.gz" || return
    printf 'Existing work preserved at %s; checkout is unchanged. Resolve it before rerunning.\n' "$backup"
    return 1
  fi
  [ "$(git branch --show-current)" = "$branch" ] || {
    printf 'Unexpected branch; preserve checkout and stop.\n'; return 1;
  }
  printf '[pilot] revision=%s; inputs are tracked in Git\n' "$(git rev-parse HEAD)"
  source "$base/.venv_bridge/bin/activate" || return
  local PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  export PYTHONDONTWRITEBYTECODE PYTHONUNBUFFERED
  batch="$base/results/pressure-balanced-git-v2-${mode}-$stamp"
  console="${batch}.console.log"
  mkdir -p "$base/results" || return
  [ ! -e "$batch" ] && [ ! -e "${batch}.tar.gz" ] && [ ! -e "$console" ] || return 1
  # Actual serving/stager parsers and all code/input/model identities, no servers.
  python -B "$runner" \
    --spec "$input/pilot_spec.json" --out-root "$batch" "${run_args[@]}" --validate-only \
    2>&1 | tee "$console"
  rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    printf 'Pre-GPU validation failed rc=%s; no GPU servers started; log=%s\n' "$rc" "$console"
  else
    # Full original rolling CPU regression is a mandatory gate inside the launcher.
    python -B "$runner" \
      --spec "$input/pilot_spec.json" --out-root "$batch" "${run_args[@]}" \
      2>&1 | tee -a "$console"
    rc=${PIPESTATUS[0]}
  fi
  # Include pre-GPU failures and outer stdout in the same single evidence archive.
  mkdir -p "$batch" &&
    cp "$console" "$batch/launcher_console.log" &&
    printf '%s\n' "$rc" > "$batch/launcher_returncode.txt" &&
    git rev-parse HEAD > "$batch/checkout_head.txt" &&
    cp "$input/pilot_spec.json" "$batch/bundle_spec.json" &&
    cp -R -- "$input" "$batch/input_bundle" &&
    cp -- "$launcher_source" "$batch/server_launcher.sh" &&
    python -B tools/bridge_tp/result_archive.py --root "$batch" --output "${batch}.tar.gz"
  pack_rc=$?
  if [ "$pack_rc" -eq 0 ]; then
    wc -c "${batch}.tar.gz"
    sha256sum "${batch}.tar.gz"
  fi
  printf 'Runner return code=%s; pack return code=%s\n' "$rc" "$pack_rc"
  [ "$rc" -eq 0 ] || return "$rc"
  return "$pack_rc"
}
bridgetp_pressure_pilot_20261015_v2 "$@"
