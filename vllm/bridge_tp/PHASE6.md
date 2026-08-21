# BridgeTP D3 Phase 6: streaming KV transfer

## Objective

Phase 6 replaces the Phase 5 consumer's shared-file tensor load with an
authenticated, chunked, rate-shaped byte stream while preserving the already
validated target scheduler allocation, per-rank KV placement, exact readback,
and continuation checks.

Phase 6 does not transfer request ownership. The source remains authoritative
and is never cancelled in this phase.

## Hardware boundary

The archived Phase 5 node has four NVIDIA A100-PCIE-40GB GPUs. All four are
needed by the TP4 target, so that node can run Phase 6A but cannot host a live
TP1 source at the same time.

- Phase 6A needs the current 4-GPU node. A CPU producer streams the validated
  rank shards to the live TP4 target. This validates the transport substrate,
  not live TP1 export.
- Phase 6B needs at least five same-node GPUs (one TP1 source plus four TP4
  target GPUs), or separate 1-GPU and 4-GPU A100 nodes. Cross-node timing is a
  separate network platform and must not be merged with same-node PCIe results.

## Phase 6A data path

```text
authenticated Phase 4 shards
              |
              v
CPU producer -> framed TCP stream -> pinned CPU receive buffer
                                         |
                                         v
                              TP4 rank-local GPU KV blocks
                                         |
                                         v
                                 exact GPU readback
```

The target request continues to use the Phase 5 KV Connector lifecycle:

1. the target request explicitly names a migration session;
2. the TP4 scheduler allocates its own nine logical blocks;
3. each worker receives only its rank shard;
4. each worker verifies metadata and payload SHA256 before H2D;
5. each worker writes the scheduler-owned blocks and requires exact readback;
6. the one pending token is computed and greedy continuation is compared with
   the TP1 control.

No target worker may open `kv_shard.pt` directly in Phase 6A.

## Wire protocol

Every connection begins with a length-prefixed JSON handshake:

```text
protocol_version
migration_id
source_request_id
target_request_id
target_tp_size = 4
target_tp_rank
num_computed_tokens
pending_known_tokens
block_size
block_axis
num_layers
rank_tensor_shape
raw_rank_bytes
rank_payload_sha256
```

Payload frames use a fixed binary header followed by bytes:

```text
frame_type
sequence_number
payload_length
payload_sha256
payload
```

The first implementation uses at most 16 MiB per frame and a token-bucket rate
limiter. Correctness smoke points are unlimited, 0.7 GiB/s, and 0.4 GiB/s. The
two shaped rates come from prior A100 PCIe interference evidence; Phase 6 does
not rerun or reinterpret the P2 experiments.

The receiver rejects duplicate, missing, reordered, oversized, or
hash-mismatched frames. A transfer is not ready until all four ranks report the
same migration ID and exact GPU readback.

## Phase 6B live-source replacement

After Phase 6A passes, replace only the producer input:

```text
Phase 6A: Phase 4 shard files -> framed sender
Phase 6B: TP1 iteration-boundary pinned snapshot -> framed sender
```

The existing Phase 1-3 model-runner hook already has the required source facts:
request token history, computed/pending boundary, source block table, layer
ordering, block axis, and real KV tensors. Phase 6B copies a synchronized
snapshot into per-rank pinned CPU buffers and hands those immutable buffers to
the background sender. The source can continue decoding after the snapshot;
Phase 6 still performs no ownership change.

## Run isolation and evidence

Every invocation uses a caller-supplied UUID `migration_id` and writes only to:

```text
results/phase6/<migration_id>/
  producer_manifest.json
  sender_metrics.json
  target_request.json
  target_response.json
  tp1_control.json
  tp_rank_0/receiver_receipt.json
  tp_rank_1/receiver_receipt.json
  tp_rank_2/receiver_receipt.json
  tp_rank_3/receiver_receipt.json
  environment.txt
  gpu_identity.txt
  SHA256SUMS
```

Tools must receive the exact run directory. They must not discover a run with
`find | head`, because that mixed Phase 5 invocations during archival.

Per-rank metrics include separate receive, H2D, readback, and total times. The
critical path is the maximum rank total, never the sum. Phase 6A and Phase 6B
timings are reported separately.

## Phase 6A completion criteria

All conditions are mandatory:

1. target workers do not read shard files;
2. transmitted raw bytes equal 28,311,552 across four ranks;
3. every frame and full-rank payload passes SHA256;
4. every rank writes nine scheduler-owned blocks;
5. all 48 layers on every rank pass exact GPU readback;
6. the restored TP4 request produces the same 32 greedy token IDs as TP1;
7. a corrupted-frame test is rejected without running model forward;
8. all artifacts belong to one explicit migration ID.

## Phase 6B completion criteria

Phase 6B additionally requires:

1. TP1 and TP4 are concurrently resident;
2. the payload originates from a live TP1 iteration-boundary snapshot;
3. the source request remains correct while background transfer runs;
4. target readback and continuation checks pass;
5. transfer throughput and source/target interference are recorded, with the
   exact GPU and network topology attached;
6. no source cancellation or takeover claim is made.

Atomic target-ready acknowledgement, source cancellation, rollback, and request
ownership commit belong to Phase 7.
