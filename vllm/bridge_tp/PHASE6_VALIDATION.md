# BridgeTP D3 Phase 6 validation record

## Verdict

`PASS` for the five-GPU live-streaming and shadow-continuation boundary.
`ownership_takeover_proven=false`; this run is the input baseline for Phase 7.

## Immutable identity

```text
migration_id: aa1b6999-b491-425f-af03-821a82265b31
git revision: e565019446a84cada3d99e5fcfb20346d074f34d
archive SHA256: c9810868931fdb1aa4fe04b7d7db6a3b36c8bcaa68bdc9d9ac61c304f9874b8d
source request: cmpl-a8f24f3bafff246f-0-99b1074d
target request: cmpl-81865718c4bd566a-0-88980381
```

Local archive and extraction:

```text
D:\work\bridgetp\vllm\phase6_aa1b6999-b491-425f-af03-821a82265b31.tar.gz
D:\work\bridgetp\vllm\phase6_results\aa1b6999-b491-425f-af03-821a82265b31
```

All entries listed by the archived `SHA256SUMS` passed local recomputation.

## Platform

The run used five `NVIDIA A100-PCIE-40GB` GPUs on one node, all with driver
`595.71.05`. Software was Python 3.12.13, PyTorch 2.11.0+cu130, CUDA 13.0,
and editable vLLM 0.23.0 from the recorded revision. These observations must
not be combined with non-A100 or cross-node data.

## Correctness evidence

- snapshot at 128 source output tokens;
- 147 computed tokens and exactly one pending known token;
- 48 BF16 KV layers, ten source blocks, block size 16;
- 31,457,280 raw tensor bytes, split equally into four 7,864,320-byte ranks;
- 31,520,588 serialized wire bytes in total, eight frames per rank;
- all four sender receipts reached `READY`;
- all four receiver receipts reached `READY`;
- every sender/receiver payload byte count and SHA256 matched;
- all four ranks passed exact GPU readback into scheduler-owned blocks 1..10;
- all 32 TP4 continuation token IDs exactly matched the live TP1 continuation.

## Debug timing observations

```text
D2H snapshot:              33.918 ms
snapshot preparation:     179.249 ms
receiver rank totals:     307.255, 303.415, 286.251, 225.546 ms
receiver critical path:   307.255 ms
sender send-only times:   23.962, 25.226, 36.668, 23.971 ms
sender thread totals:     703.541, 696.801, 681.510, 620.243 ms
```

Receiver critical path is the maximum rank time, not the sum. Sender totals
include time waiting for the target connection and acknowledgement. These are
single-run debug measurements, not optimized migration performance statistics.

## Evidence boundary

This run proves a live TP1 iteration-boundary snapshot, four real local TCP
streams, authenticated/integrity-checked payloads, TP4 restore/readback, and
greedy shadow continuity. The source completed normally and remained owner.
It does not prove source cancellation, target-gated commit, rollback, arbitrary
sampling/RNG migration, or crash-consistent distributed ownership transfer.
