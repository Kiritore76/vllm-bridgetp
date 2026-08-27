# Phase 9 conditional bridge calibration runbook

This runbook implements Section 7 of the Phase 9 next-experiments plan. It is
an A100 PCIe calibration experiment. It does not repeat Phase 1--8, does not
establish D-group migration fidelity, and does not make a cross-platform claim.

## 1. Added executable support

- `record_phase9_calibration.py`: records current vLLM gauges and interval
  (not server-lifetime) TPOT histograms.
- `run_phase9_copy_load.py`: records an equal-duration rate-zero window or
  generates measured P2-D-style sustained P2P traffic.
- `analyze_phase9_calibration.py`: aligns telemetry, copy window, and detailed
  benchmark ITLs; rejects conditions outside their actual KV band or rate.
- `fit_phase9_tick_tpot.py`: fits TP1/TP4 TPOT against instantaneous
  `num_requests_running`.
- `summarize_phase9_calibration.py`: requires exactly 36 accepted interference
  conditions (`3 bands x 4 rates x 3 reps`).

The production telemetry parser accepts both
`vllm:kv_cache_usage_perc` and the older
`vllm:gpu_cache_usage_perc` name.

## 2. Frozen scope

TPOT mapping:

```text
I128/O512
QPS 1, 2, 4
TP1 and TP4
3 repetitions
```

Interference mapping:

```text
platform: NVIDIA A100 PCIe
attainable KV bands: low 0.10-0.25, medium 0.30-0.45, high 0.55-0.65
copy rates: 0, 0.4, 0.7, 1.2 GiB/s
3 repetitions per cell
```

These A100 PCIe TP4 I256/O2048 bands are a protocol amendment frozen before
any nonzero-copy formal condition. Two rate-zero pilots and an isolated QPS
3.0/3.5/4.0 reachability diagnostic showed approximately 256 running requests
while waiting continued to increase; the original 0.15-0.25, 0.45-0.55, and
0.75-0.85 bands were therefore not jointly sustainable. Here `high` means the
highest attainable steady regime in this workload scope, not 75-85% KV usage.
The failed pilots and reachability diagnostic remain diagnostic evidence and
must not be mixed into the formal fit.

The QPS used to hold each KV band must be selected by a rate-zero pilot and
then frozen. Offered QPS is not a substitute for measured KV occupancy.

## 3. Server preflight

Use the same server flags for every condition. Start TP1 on GPU 0 / port 8001
and TP4 on GPUs 1--4 / port 8200. Record `nvidia-smi -L`, topology, git commit,
server logs, model path, dtype, and KV block counts.

Create the shared environment in every terminal:

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export BRIDGE_PY=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
export CAL_ROOT=/root/autodl-tmp/bridgetp/results/phase9_bridge_calibration
export CAL_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
mkdir -p "$CAL_ROOT/server_logs"
```

### Terminal TP1: start the TP1 server for Section 7.1

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
export BRIDGE_PY=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
export CAL_ROOT=/root/autodl-tmp/bridgetp/results/phase9_bridge_calibration
export CAL_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
mkdir -p "$CAL_ROOT/server_logs"

CUDA_VISIBLE_DEVICES=0 "$BRIDGE_PY" \
  -m vllm.entrypoints.openai.api_server \
  --model "$CAL_MODEL" \
  --served-model-name bridgetp-model \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --port 8001 \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  2>&1 | tee "$CAL_ROOT/server_logs/tp1_port8001.log"
```

### Terminal TP4: start the TP4 server for Sections 7.1 and 7.2

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
export BRIDGE_PY=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
export CAL_ROOT=/root/autodl-tmp/bridgetp/results/phase9_bridge_calibration
export CAL_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
mkdir -p "$CAL_ROOT/server_logs"

CUDA_VISIBLE_DEVICES=1,2,3,4 "$BRIDGE_PY" \
  -m vllm.entrypoints.openai.api_server \
  --model "$CAL_MODEL" \
  --served-model-name bridgetp-model \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.88 \
  --port 8200 \
  --no-enable-prefix-caching \
  --disable-hybrid-kv-cache-manager \
  --no-async-scheduling \
  2>&1 | tee "$CAL_ROOT/server_logs/tp4_port8200.log"
```

Do not run a benchmark until these checks pass from a third terminal:

```bash
curl -fsS http://127.0.0.1:8001/v1/models
curl -fsS http://127.0.0.1:8200/v1/models

grep -Ei 'GPU KV cache size|GPU blocks|num_gpu_blocks' \
  "$CAL_ROOT/server_logs/tp1_port8001.log" \
  "$CAL_ROOT/server_logs/tp4_port8200.log" | tail -20
```

Section 7.1 may measure TP1 and TP4 while both servers are available, but only
one benchmark client should run at a time. Before Section 7.2 P2P-copy
conditions, stop the TP1 server with `Ctrl-C` and verify that GPU 0 is free:

```bash
if pgrep -af 'vllm.entrypoints.openai.api_server.*--port 8001'; then
  echo 'STOP: TP1 port 8001 is still running; stop its terminal before 7.2'
else
  echo 'PASS: GPU 0 source is not occupied by the TP1 calibration server'
fi
```

The TP4 server on port 8200 must remain running throughout Section 7.2. The
copy-load tool uses physical GPU 0 as source and physical GPU 1 (TP4 rank 0) as
destination, matching the A100 PCIe P2-D calibration topology.

For the previously validated configuration, the observed per-rank block counts
were TP1=1968 and TP4=35739. Reuse them only if the new server logs reproduce
31,488 and 571,824 KV tokens respectively with block size 16.

Before a run, verify the metrics actually used by the recorder:

```bash
curl -fsS http://127.0.0.1:8200/metrics | grep -E \
  'vllm:(kv_cache_usage_perc|gpu_cache_usage_perc|num_requests_running|num_requests_waiting|request_time_per_output_token_seconds_(bucket|sum|count))' \
  | head -40
```

## 4. One condition, split across terminals

Section 7.1 should normally use the automatic sequential sweep below. The
manual Terminal A/B/C template later in this section is for Section 7.2.

### Automatic Section 7.1 TPOT sweep

Keep both TP1 port 8001 and TP4 port 8200 servers running. From one additional
terminal run:

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export BRIDGE_PY=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
export CAL_ROOT=/root/autodl-tmp/bridgetp/results/phase9_bridge_calibration
export CAL_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
export CAL_TPOT_ROOT="$CAL_ROOT/tpot_sweep_$(date +%Y%m%dT%H%M%S)"

"$BRIDGE_PY" tools/bridge_tp/run_phase9_tpot_sweep.py \
  --out-root "$CAL_TPOT_ROOT" \
  --model "$CAL_MODEL" \
  --served-model-name bridgetp-model \
  --tp1-url http://127.0.0.1:8001 \
  --tp4-url http://127.0.0.1:8200 \
  --tp1-blocks 1968 \
  --tp4-blocks 35739 \
  --block-size 16 \
  --qps 1 2 4 \
  --reps 1 2 3 \
  --input-len 128 \
  --output-len 512 \
  --num-prompts 100 \
  --num-warmups 10 \
  2>&1 | tee "$CAL_ROOT/tpot_sweep_driver.log"
```

The tool runs all 18 conditions sequentially, starts and stops one telemetry
recorder per condition, hashes each condition, and writes:

```text
$CAL_TPOT_ROOT/tick_tpot_candidate.json
$CAL_TPOT_ROOT/SHA256SUMS
```

Verify completion before stopping TP1:

```bash
find "$CAL_TPOT_ROOT" -mindepth 1 -maxdepth 1 \
  -type d -name 'tpot_*' | wc -l

"$BRIDGE_PY" -m json.tool \
  "$CAL_TPOT_ROOT/tick_tpot_candidate.json"
```

The directory count must be 18. After this check, stop the TP1 server with
`Ctrl-C`, verify port 8001 is gone, and keep TP4 port 8200 running for Section
7.2.

### Automatic Section 7.2 load-band pilot

TP1 must be stopped, GPU 0 must be free, and TP4 port 8200 must remain running.
The pilot runs rate-zero conditions sequentially and recommends QPS values only
when the measured mean KV occupancy is inside a band and at least 80% of the
300-second window samples are inside that band.

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export BRIDGE_PY=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
export CAL_ROOT=/root/autodl-tmp/bridgetp/results/phase9_bridge_calibration
export CAL_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
export CAL_PILOT_ROOT="$CAL_ROOT/load_pilot_$(date +%Y%m%dT%H%M%S)"

"$BRIDGE_PY" tools/bridge_tp/run_phase9_interference_sweep.py pilot \
  --out-root "$CAL_PILOT_ROOT" \
  --model "$CAL_MODEL" \
  --served-model-name bridgetp-model \
  --tp4-url http://127.0.0.1:8200 \
  --tp4-blocks 35739 \
  --block-size 16 \
  --source-gpu 0 \
  --target-gpu 1 \
  --candidate-qps 0.7 0.8 0.9 0.95 1.0 1.05 1.1 1.15 1.2 \
  --input-len 256 \
  --output-len 2048 \
  --num-warmups 0 \
  --copy-delay-s 60 \
  --load-settle-timeout-s 300 \
  --stability-window-s 60 \
  --min-band-fraction 0.80 \
  --copy-seconds 300 \
  2>&1 | tee "$CAL_ROOT/load_pilot_driver.log"
```

Inspect, do not silently accept, the selected values:

```bash
"$BRIDGE_PY" -m json.tool \
  "$CAL_PILOT_ROOT/load_pilot_summary.json"
```

The summary must report `status=READY` and non-null selections for all three
bands. If it reports `MORE_QPS_CANDIDATES_REQUIRED`, preserve that pilot and
rerun a new pilot root with an expanded preregistered candidate list.
Candidates that never satisfy the rolling stability gate are preserved with
`stability_status=TIMEOUT`; the automatic pilot continues to the next QPS.

### Reclassify existing pre-formal rate-zero pilots after the attainable-band amendment

Do not repeat already collected rate-zero conditions merely because the load
bands were amended. The reclassifier accepts a historical condition only when
the same telemetry contains a compliant 120-second stability window followed
by a separate compliant 300-second measurement window. It rejects a trace that
only ramps through a band. Pass every relevant rate-zero pilot root and write a
new immutable summary; do not include the isolated QPS 3.0/3.5/4.0 reachability
root because it has only a 300-second diagnostic window.

```bash
export CAL_RECLASS_ROOT="$CAL_ROOT/load_pilot_attainable_reclassified_$(date +%Y%m%dT%H%M%S)"
mkdir -p "$CAL_RECLASS_ROOT"

"$BRIDGE_PY" tools/bridge_tp/reclassify_phase9_load_pilot.py \
  --pilot-roots \
    "$CAL_FIRST_STABLE_PILOT_ROOT" \
    "$CAL_SECOND_STABLE_PILOT_ROOT" \
  --stability-window-s 120 \
  --measurement-window-s 300 \
  --min-band-fraction 0.80 \
  --out "$CAL_RECLASS_ROOT/load_pilot_summary.json"
```

If the result is `READY`, use this reclassified summary as `CAL_PILOT_ROOT`
for the formal sweep. If it reports `MISSING_ATTAINABLE_BAND_EVIDENCE`, run
only the missing band/QPS neighborhood and re-run the deterministic
reclassification over the old and supplemental roots.

### Automatic Section 7.2 formal 36-cell sweep

Running this command is the explicit acceptance of the QPS selections recorded
in the pilot summary. It runs one condition at a time; it does not overlap two
benchmarks or two copy windows.

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate

export BRIDGE_PY=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
export CAL_ROOT=/root/autodl-tmp/bridgetp/results/phase9_bridge_calibration
export CAL_MODEL=/root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master
export CAL_FORMAL_ROOT="$CAL_ROOT/interference_formal_$(date +%Y%m%dT%H%M%S)"

"$BRIDGE_PY" tools/bridge_tp/run_phase9_interference_sweep.py formal \
  --out-root "$CAL_FORMAL_ROOT" \
  --pilot-summary "$CAL_PILOT_ROOT/load_pilot_summary.json" \
  --model "$CAL_MODEL" \
  --served-model-name bridgetp-model \
  --tp4-url http://127.0.0.1:8200 \
  --tp4-blocks 35739 \
  --block-size 16 \
  --source-gpu 0 \
  --target-gpu 1 \
  --input-len 256 \
  --output-len 2048 \
  --num-warmups 0 \
  --copy-delay-s 60 \
  --load-settle-timeout-s 300 \
  --stability-window-s 60 \
  --min-band-fraction 0.80 \
  --copy-seconds 300 \
  2>&1 | tee "$CAL_ROOT/interference_formal_driver.log"
```

Successful completion writes:

```text
$CAL_FORMAL_ROOT/bridge_calibration_summary.json
$CAL_FORMAL_ROOT/SHA256SUMS
```

Verify `status=COMPLETE`, 36 expected conditions, and empty
`missing/unexpected/duplicates/rejected` lists. A rejected condition stops the
automatic sweep and remains on disk; it must not be overwritten or silently
excluded.

Create a unique condition directory and export the same values in every
terminal:

```bash
cd /root/autodl-tmp/bridgetp/vllm_bridge
source /root/autodl-tmp/bridgetp/.venv_bridge/bin/activate
export BRIDGE_PY=/root/autodl-tmp/bridgetp/.venv_bridge/bin/python
export CAL_ROOT=/root/autodl-tmp/bridgetp/results/phase9_bridge_calibration
export CONDITION_ID=medium_rate0.7_r1
export CONDITION_DIR="$CAL_ROOT/$CONDITION_ID"
mkdir -p "$CONDITION_DIR"
```

### Terminal A: interval telemetry

```bash
"$BRIDGE_PY" tools/bridge_tp/record_phase9_calibration.py \
  --base-url http://127.0.0.1:8200 \
  --out "$CONDITION_DIR/telemetry.csv" \
  --interval-s 1 \
  --max-seconds 900 \
  --block-size 16 \
  --total-kv-blocks 35739 \
  --condition-id "$CONDITION_ID" \
  --load-band medium \
  --target-rate-gib-s 0.7 \
  --rep 1 \
  --stop-file "$CONDITION_DIR/telemetry.stop" \
  2>&1 | tee "$CONDITION_DIR/telemetry.log"
```

### Terminal B: TP4 detailed benchmark

Use the QPS frozen by the rate-zero pilot for the requested band. The example
variable is deliberately checked rather than silently defaulted:

```bash
: "${MEDIUM_QPS:?export MEDIUM_QPS from the accepted rate-zero pilot}"

vllm bench serve \
  --backend vllm \
  --base-url http://127.0.0.1:8200 \
  --endpoint /v1/completions \
  --model bridgetp-model \
  --tokenizer /root/autodl-tmp/models/models/Qwen--Qwen2.5-14B-Instruct/snapshots/master \
  --dataset-name random \
  --input-len 256 \
  --output-len 2048 \
  --request-rate "$MEDIUM_QPS" \
  --num-prompts 400 \
  --num-warmups 10 \
  --seed 1 \
  --ignore-eos \
  --percentile-metrics ttft,tpot,itl,e2el \
  --metric-percentiles 50,95,99 \
  --save-result --save-detailed \
  --result-dir "$CONDITION_DIR" \
  --result-filename benchmark.json \
  --label "$CONDITION_ID" \
  2>&1 | tee "$CONDITION_DIR/benchmark.log"
```

### Terminal C: copy or rate-zero window

Start this after the benchmark has been issuing requests for 20 seconds. Rate
zero uses the same command and records a no-copy baseline window.

```bash
"$BRIDGE_PY" tools/bridge_tp/run_phase9_copy_load.py \
  --src 0 \
  --dst 1 \
  --chunk-mib 16 \
  --target-gib-s 0.7 \
  --seconds 300 \
  --out "$CONDITION_DIR/copy_window.json" \
  2>&1 | tee "$CONDITION_DIR/copy_window.log"

touch "$CONDITION_DIR/telemetry.stop"
```

For the baseline cell, change both `--target-rate-gib-s` and
`--target-gib-s` to `0`. Do not run a different-duration baseline.

## 5. Analyze one completed condition

For the medium band example:

```bash
"$BRIDGE_PY" tools/bridge_tp/analyze_phase9_calibration.py \
  --telemetry "$CONDITION_DIR/telemetry.csv" \
  --benchmark-json "$CONDITION_DIR/benchmark.json" \
  --copy-json "$CONDITION_DIR/copy_window.json" \
  --target-rate-gib-s 0.7 \
  --load-min 0.30 \
  --load-max 0.45 \
  --min-band-fraction 0.80 \
  --rate-relative-tolerance 0.05 \
  --out "$CONDITION_DIR/condition_result.json" \
  2>&1 | tee "$CONDITION_DIR/analysis.log"
```

Only `status=ACCEPTED` is eligible for the 36-cell summary. A rejected run must
remain in the archive; adjust the QPS in a new condition ID rather than
overwriting or deleting it.

## 6. TPOT tick fit

After all 18 I128/O512 TPOT conditions finish:

```bash
"$BRIDGE_PY" tools/bridge_tp/fit_phase9_tick_tpot.py \
  --tp1 "$CAL_ROOT"/tpot_tp1_*/telemetry.csv \
  --tp4 "$CAL_ROOT"/tpot_tp4_*/telemetry.csv \
  --input-len 128 \
  --output-len 512 \
  --out "$CAL_ROOT/tick_tpot_candidate.json"
```

This candidate replaces the run-level max-concurrency proxy only after its
input list, environment, and hashes are frozen.

## 7. Complete interference grid

After exactly one accepted result exists for every band/rate/rep cell:

```bash
"$BRIDGE_PY" tools/bridge_tp/summarize_phase9_calibration.py \
  --inputs "$CAL_ROOT"/*/condition_result.json \
  --out "$CAL_ROOT/bridge_calibration_summary.json" \
  2>&1 | tee "$CAL_ROOT/bridge_calibration_summary.log"
```

Expected output:

```text
status: COMPLETE
expected_conditions: 36
missing: []
unexpected: []
duplicates: []
rejected: []
```

Finally hash all accepted evidence:

```bash
find "$CAL_ROOT" -type f \
  \( -name '*.json' -o -name '*.csv' -o -name '*.log' \) \
  -print0 | sort -z | xargs -0 sha256sum \
  > "$CAL_ROOT/SHA256SUMS"
```

Completion of this grid does not authorize formal D-3. The next steps are to
fit and implement the rate-aware interference model, rerun policy replay, and
then freeze the Section 8 preregistration.
