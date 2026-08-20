# BridgeTP D3 Phase 4: offline KV reshard

Phase 4 converts a validated request-scoped TP1 dump into four contiguous
KV-head shards for TP4. It is an offline layout transformation only;
it does not allocate destination vLLM blocks, restore KV into TP4 workers,
transfer data between processes, or change request ownership.

For the validated Qwen2.5-14B TP1 layout:

```text
source: [blocks, K/V, tokens_per_block, 8 KV heads, head_size]
                                      |
                                      +-- head axis = 3

TP4 rank 0: source heads [0:2]
TP4 rank 1: source heads [2:4]
TP4 rank 2: source heads [4:6]
TP4 rank 3: source heads [6:8]
```

The layout facts must be supplied explicitly instead of being treated as
cross-model constants:

```bash
python tools/bridge_tp/reshard_kv_dump.py \
  /path/to/tp1/tp_rank_0 \
  /path/to/phase4_output \
  --target-tp-size 4 \
  --head-axis 3 \
  --expected-source-kv-heads 8
```

The command writes four `kv_shard.pt` files and a `reshard_manifest.json`.
It then concatenates all four shards in rank order and requires bitwise tensor
equality with every source layer.

Run the independent validator with:

```bash
python tools/bridge_tp/inspect_reshard.py \
  /path/to/tp1/tp_rank_0 \
  /path/to/phase4_output
```

`status: PASS` proves lossless offline head sharding. It does not prove that a
live TP4 vLLM worker can import the shards or resume the request; those belong
to a later phase.
