# BridgeTP D3 Phase 4 validation record

## Scope

This record covers an offline TP1-to-TP4 KV-head reshard of one request dump.
The source dump was produced by vLLM v0.23.0 on one NVIDIA A100-PCIE-40GB
using Qwen2.5-14B-Instruct. The reshard and independent reconstruction check
ran on Windows with CPU PyTorch.

This result proves only that the captured TP1 tensor layout can be split into
four lossless contiguous head shards. It does not prove TP4 destination block
allocation, device transfer, KV restore, resumed decoding, or takeover.

## Source identity

```text
request_id: cmpl-bff96b5ee84f93ec-0-bf80dd23
manifest.json SHA256:
  0758643cb4bf6a488a19140d85438aed8fb34771019f25edececc8724ed059d9
generated_tokens.json SHA256:
  2290d869cd4b122b02e8d8b77ac35bdf1026e2e0ba5a5e3d4bbf45e129d853a7
kv_blocks.pt SHA256:
  57b6e69be97e6c4394b433b4b52e0800c92d5dae55d699a7df548907a9316b23
```

The source contains 48 BF16 layers with shape `[9, 2, 16, 8, 128]`.
The nine request-scoped physical blocks are `[1, 2, 3, 4, 5, 6, 7, 8, 9]`.
The raw source tensor payload is 28,311,552 bytes.

## Phase 4 mapping

The model-specific KV-head axis is axis 3. The eight source KV heads are
mapped contiguously in TP-rank order:

```text
rank 0 <- heads [0:2]
rank 1 <- heads [2:4]
rank 2 <- heads [4:6]
rank 3 <- heads [6:8]
```

Each rank has 48 BF16 tensors with shape `[9, 2, 16, 2, 128]`. Each rank's
raw tensor payload is 7,077,888 bytes, so the four ranks conserve all
28,311,552 source bytes.

## Independent validation

`tools/bridge_tp/inspect_reshard.py` reloaded the source and all four shard
files from disk, verified their recorded SHA256 digests and rank metadata,
concatenated each layer along axis 3 in rank order, and compared the result
with `torch.equal`.

```text
status: PASS
layers: 48
reconstructed elements: 14,155,776
raw source bytes: 28,311,552
raw shard bytes: 28,311,552
exact roundtrip: true
```

The final reshard artifact reported about 124.0 ms on this Windows CPU run.
That number includes Python tensor slicing, four `torch.save` operations,
hashing, and an in-memory reconstruction check. It is not an A100 transfer
measurement, an optimized migration result, or a formal performance sample.

## Next evidence target

The next phase must run on four GPUs from the same A100 platform and validate:

1. destination TP4 block allocation and a recorded source-to-destination
   block-table mapping;
2. copying each rank shard into the correct KV-cache layers and block slots;
3. reading the restored tensors back and proving exact equality before any
   decode continuation;
4. only after restore correctness, resuming generation and comparing token
   continuity with a non-migrated control request.
