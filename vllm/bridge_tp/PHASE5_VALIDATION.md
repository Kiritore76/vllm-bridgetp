# BridgeTP D3 Phase 5 validation record

## Result

Phase 5 passed on 2026-08-21. A real TP4 target allocated scheduler-owned KV
blocks, all four workers restored their authenticated rank shards, every worker
passed an exact device readback, and the restored TP4 request produced the same
32 greedy continuation token IDs as a clean TP1 control.

```text
vLLM source branch: bridgetp/d3-phase5-tp4-file-restore
vLLM source commit: 25e4151a7237eb4ca90a9c500a03feda42a893ae
model: Qwen2.5-14B-Instruct
source request: cmpl-bff96b5ee84f93ec-0-bf80dd23
target request: cmpl-b1e5ce73880bc9e9-0-83e5a0ae
source topology: TP1
target topology: TP4
computed tokens restored: 137
pending tokens computed after restore: 1
target logical block table: [1, 2, 3, 4, 5, 6, 7, 8, 9]
all ranks exact readback: true
```

The target block IDs happened to equal the source dump's IDs in this run. They
were nevertheless allocated by the target scheduler and were not imported as
source ownership metadata. All TP ranks correctly received the same logical
block table and applied it to their own GPU-local KV pools.

## Restore receipts

```text
TP rank 0: 236.2813614308834 ms
TP rank 1: 265.9468911588192 ms
TP rank 2: 265.74227306991816 ms
TP rank 3: 276.9781555980444 ms
```

The ranks execute in parallel, so the observed restore critical path is bounded
by the slowest receipt, approximately 277 ms, rather than the sum. These values
include synchronous file loading, CPU-to-GPU copy, synchronization, and exact
GPU readback. They are one debug run, not optimized online migration latency or
a formal performance distribution.

## Exact continuation check

The TP1 control and restored TP4 request both returned:

```text
[78026, 389, 2155, 7611, 382, 18, 13, 3070,
 65411, 95518, 8704, 279, 82599, 525, 4237, 11,
 1052, 3880, 311, 387, 264, 1616, 369, 279,
 7611, 311, 9289, 1995, 320, 72, 1734, 2572]
```

```text
exact token continuity: true
```

This proves the narrow greedy continuation case for the captured request and
validated Qwen2.5-14B layout.

## Evidence boundary

Phase 5 proves:

- real TP1 KV export followed by exact TP1-to-TP4 KV-head resharding;
- target-scheduler block allocation rather than source block ownership reuse;
- synchronous placement into four real TP4 worker KV pools;
- exact per-rank device readback before continuation;
- exact 32-token greedy continuation equality against TP1.

Phase 5 does not prove:

- live TP1 and TP4 overlap or source-to-target network transfer;
- optimized, paced, or background KV movement;
- preservation of arbitrary stochastic sampling/RNG state;
- transfer of the original request object's output/stop/accounting state;
- source cancellation, rollback, or iteration-boundary atomic takeover;
- handoff stall or steady-state online migration performance.

## Raw artifacts to archive

The server-side validation record is not complete until these files are copied
into the project result archive:

```text
phase5_receipts/<target-request-id>/tp_rank_0.json
phase5_receipts/<target-request-id>/tp_rank_1.json
phase5_receipts/<target-request-id>/tp_rank_2.json
phase5_receipts/<target-request-id>/tp_rank_3.json
/tmp/bridgetp_tp4_continuation.json
/tmp/bridgetp_tp1_control.json
/tmp/bridgetp_restore_request.json
phase5_tp4_server.log
git/environment/GPU identity snapshot
```
