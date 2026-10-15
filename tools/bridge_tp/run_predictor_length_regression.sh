#!/usr/bin/env bash
# Saved features only. This fixed entry point is updated in place.
bridgetp_predictor_length_regression() {
  local repo=/root/autodl-tmp/bridgetp/vllm_bridge
  local results=/root/autodl-tmp/bridgetp/results
  local py=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
  local branch=bridgetp/predictor-nonuniform84-20261010
  local head out stamp rc pack_rc
  cd "$repo" || return
  [ "$(git rev-parse --show-toplevel)" = "$repo" ] || return 1
  [ "$(git branch --show-current)" = "$branch" ] || return 1
  if [ -n "$(git status --porcelain)" ]; then
    printf '存在未提交工作，保留原处并停止；请先处理后重跑。\n'
    return 1
  fi
  head="$(git rev-parse HEAD)" || return
  stamp="$(date -u +%Y%m%dT%H%M%SZ)-$$" || return
  out="$results/predictor-length-regression-mlp4-$stamp"
  [ ! -e "$out" ] && [ ! -e "${out}.tar.gz" ] || return 1
  export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
  printf '[进度] HEAD=%s；仅使用保存特征；选择GPU=%s\n' "$head" "${BRIDGETP_PREDICTOR_GPU_INDEX:-0}"
  mkdir -p "$results" || return
  "$py" -B tools/bridge_tp/run_predictor_length_regression.py \
    --results "$results" --out-dir "$out" --expected-revision "$head" \
    --gpu-index "${BRIDGETP_PREDICTOR_GPU_INDEX:-0}" \
    2>&1 | tee "${out}.console.log"
  rc=${PIPESTATUS[0]}
  mkdir -p "$out" &&
    cp "${out}.console.log" "$out/runner.log" &&
    printf '%s\n' "$rc" > "$out/runner_returncode.txt" &&
    git rev-parse HEAD > "$out/checkout_head.txt" &&
    git status --porcelain > "$out/checkout_status.txt" &&
    "$py" -B tools/bridge_tp/result_archive.py --root "$out" --output "${out}.tar.gz"
  pack_rc=$?
  if [ "$pack_rc" -eq 0 ]; then
    wc -c "${out}.tar.gz"
    sha256sum "${out}.tar.gz"
  fi
  printf 'Runner return code=%s; pack return code=%s\n' "$rc" "$pack_rc"
  [ "$rc" -eq 0 ] || return "$rc"
  return "$pack_rc"
}
bridgetp_predictor_length_regression "$@"
