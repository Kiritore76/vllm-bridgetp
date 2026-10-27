#!/usr/bin/env bash
# Source from the repository root. Only this run's child processes are stopped.
bridgetp_deploy_4090d_main() {
    local repo=/root/autodl-tmp/bridgetp/vllm_bridge
    local env_dir=/root/autodl-tmp/bridgetp/.venv_bridge
    local model=${BRIDGETP_MODEL_PATH:-/root/autodl-tmp/models/Qwen3-8B}
    local provider=${BRIDGETP_MODEL_PROVIDER:-modelscope}
    local run=/root/autodl-tmp/bridgetp/results/4090d_qwen3_deploy_$(date +%Y%m%d_%H%M%S)_${RANDOM}
    local wheel_base=0fc695fc6d1d82e9a5ac6835ac8e4e1c83703665
    local rc=0 revision input_sha gpu_name
    local -a pipeline_codes
    cd "$repo" || return 1
    [ "$(git branch --show-current)" = bridgetp/tp2-4090d-qwen3-8b ] || {
        echo '错误：分支不匹配'; return 1;
    }
    [ -z "$(git status --porcelain)" ] || { echo '错误：代码目录不干净'; return 1; }
    mkdir -p "$run" || return 1
    echo "[部署] 原始记录：$run"
    (
        set -euo pipefail
        export UV_CACHE_DIR=/root/autodl-tmp/bridgetp/.uv-cache
        export HF_HOME=/root/autodl-tmp/models/.hf-cache
        export MODELSCOPE_CACHE=/root/autodl-tmp/models/.modelscope-cache
        # Never inherit an A100 predictor/capture/migration configuration.
        local inherited
        for inherited in ${!BRIDGETP_@}; do unset "$inherited"; done
        echo '[1/7] 核对四张 4090D 和驱动'
        nvidia-smi
        nvidia-smi topo -m
        git rev-parse HEAD > "$run/revision.txt"
        cp experiments/platforms/4090d_qwen3_8b_tp2.json "$run/platform.json"
        if ! command -v uv >/dev/null 2>&1; then
            curl -LsSf https://astral.sh/uv/install.sh -o "$run/install_uv.sh"
            sh "$run/install_uv.sh"
            export PATH="$HOME/.local/bin:$PATH"
        fi
        if [ ! -e "$env_dir" ]; then
            uv venv --python 3.12 "$env_dir"
        fi
        # shellcheck disable=SC1091
        source "$env_dir/bin/activate"
        export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
        echo '[2/7] 安装与本分支 C++ 基线匹配的预编译扩展'
        VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_COMMIT="$wheel_base" \
            uv pip install --python "$env_dir/bin/python" -e . --torch-backend=auto
        uv pip install --python "$env_dir/bin/python" modelscope huggingface_hub numpy
        uv pip freeze --python "$env_dir/bin/python" > "$run/environment_freeze.txt"
        "$env_dir/bin/python" -B - <<'PY'
import json
import torch
import vllm
from tools.bridge_tp.prepare_4090d_qwen3 import PROFILE, read, runtime_inventory
hardware = runtime_inventory(read(PROFILE))
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() == 4, 'four visible GPUs required'
assert all(torch.cuda.get_device_capability(i) == (8, 9) for i in range(4))
print(json.dumps(dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                      vllm_file=vllm.__file__, hardware=hardware), indent=2))
PY
        echo '[3/7] 下载并逐文件冻结 Qwen3-8B 身份（不加载旧预测器）'
        "$env_dir/bin/python" -B tools/bridge_tp/prepare_4090d_qwen3.py \
            --phase download --provider "$provider" --model "$model" \
            --output "$run/model_identity.json" --check-gpus
        "$env_dir/bin/python" -B tools/bridge_tp/prepare_4090d_qwen3.py \
            --phase inputs --model "$model" --output "$run/inputs"
        echo '[4/7] CPU 合同检查'
        "$env_dir/bin/python" -B tools/bridge_tp/run_tp2_cpu_checks.py --output "$run/cpu"
        echo '[5/7] TP1 真实推理、原生 inventory 和 trace 检查'
        "$env_dir/bin/python" -B tools/bridge_tp/run_4090d_qwen3_smoke.py \
            --model "$model" --inputs "$run/inputs" --output "$run/tp1" --tp 1 --port 8600
        echo '[6/7] TP2 真实推理、两个 rank 和 trace 检查'
        "$env_dir/bin/python" -B tools/bridge_tp/run_4090d_qwen3_smoke.py \
            --model "$model" --inputs "$run/inputs" --output "$run/tp2" --tp 2 --port 8600
        echo '[7/7] 单卡 decoder:31 的 Qwen3 特征采集检查（仅两条，非训练集）'
        revision=$(git rev-parse HEAD)
        input_sha=$(sha256sum "$run/inputs/feature_smoke.jsonl" | cut -d' ' -f1)
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n '1p')
        CUDA_VISIBLE_DEVICES=0 "$env_dir/bin/python" -B tools/bridge_tp/run_predictor_capture.py \
            --input "$run/inputs/feature_smoke.jsonl" --model "$model" \
            --out-dir "$run/feature_smoke" --expected-revision "$revision" \
            --expected-input-sha256 "$input_sha" --expected-gpu-name "$gpu_name" \
            --expected-gpu-count 4 --feature-layer decoder:31 --interval 20 \
            --max-tokens 128 --max-model-len 16384 --gpu-memory-utilization 0.85 \
            --thinking-mode disabled --save-token-ids
        "$env_dir/bin/python" -B - "$run/feature_smoke/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1]) as stream:
    summary = json.load(stream)
assert summary['hidden_size'] == 4096, summary
assert summary['phases'].get('PREFILL_COMPLETE') == 2, summary
assert summary['phases'].get('DECODE', 0) > 0, summary
print('Qwen3 hidden4096 / prefill / decode 特征审计通过')
PY
        echo '部署检查通过；预测器尚未训练，KV 容量和迁移性能尚未标定。'
    ) 2>&1 | tee "$run/console.log"
    pipeline_codes=("${PIPESTATUS[@]}")
    rc=${pipeline_codes[0]}
    if [ "${pipeline_codes[1]}" -ne 0 ]; then rc=1; fi
    printf '%s\n' "$rc" > "$run/runner_exit_code.txt"
    # Archive outside the run tree, retaining failures and raw SQLite features.
    if [ -x "$env_dir/bin/python" ]; then
        "$env_dir/bin/python" -B tools/bridge_tp/result_archive.py \
            --root "$run" --output "$run.tar.gz" || return 1
        sha256sum "$run.tar.gz" | tee "$run.tar.gz.sha256"
    else
        tar -czf "$run.tar.gz" -C "$(dirname "$run")" "$(basename "$run")" || return 1
        sha256sum "$run.tar.gz" | tee "$run.tar.gz.sha256"
    fi
    echo "[交回文件] $run.tar.gz"
    echo "[退出码] $rc"
    return "$rc"
}
bridgetp_deploy_4090d_main
