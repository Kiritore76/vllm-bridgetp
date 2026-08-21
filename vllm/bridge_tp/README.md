# BridgeTP D3 Phase 1-3

This package contains a narrow, opt-in diagnostic for exporting the real KV
blocks of one TP1 request. It does not implement TP1-to-TP4 transfer, restore,
ownership takeover, or migration control.

The hook is disabled by default. For the first server run, use metadata-only
mode:

```bash
export BRIDGETP_DUMP_ENABLED=1
export BRIDGETP_DUMP_AFTER_OUTPUT_TOKENS=128
export BRIDGETP_DUMP_DIR=/root/autodl-tmp/bridgetp/bridge_dumps
export BRIDGETP_DUMP_TENSORS=0
export BRIDGETP_DUMP_STRICT=1
```

After validating the manifest, enable the request-scoped tensor copy:

```bash
export BRIDGETP_DUMP_TENSORS=1
export BRIDGETP_DUMP_MAX_BYTES=2147483648
```

Run exactly one request unless `BRIDGETP_DUMP_REQUEST_ID` is set. The first
eligible request is dumped once per worker process into:

```text
<dump-dir>/<request-id>/tp_rank_0/
├── manifest.json
├── generated_tokens.json
└── kv_blocks.pt
```

`num_computed_tokens` is the number of tokens represented in KV. The token
history can contain a newly sampled token that has not been used as model input
yet; it is recorded under `known_not_computed_token_ids` and counted by
`pending_known_tokens`.

Start vLLM with `--no-async-scheduling` for Phase 1-3. vLLM 0.23.0 enables
async scheduling automatically when compatible, but its worker stores a
placeholder for the newest sampled token before the asynchronous CPU copy
finishes. That placeholder cannot support the token/KV boundary evidence this
diagnostic is designed to collect.

Phase 1-3 intentionally rejects TP greater than one, speculative decoding,
async scheduling, hybrid/Mamba caches, and non-uniform KV-cache groups. These
limitations prevent unsupported layouts from being mistaken for validated
BridgeTP evidence.

Validate a completed dump with the Bridge development environment:

```bash
/root/autodl-tmp/bridgetp/.venv_bridge/bin/python \
  tools/bridge_tp/inspect_dump.py \
  /root/autodl-tmp/bridgetp/bridge_dumps/<request-id>/tp_rank_0
```

Later online-prototype stages are documented separately:

- `PHASE6.md`: live four-stream TP1 snapshot transfer;
- `PHASE7.md`: application-level atomic takeover and rollback;
- `PHASE7_VALIDATION.md`: returned commit/rollback evidence;
- `PHASE8.md`: background old-KV staging, new-KV mirroring, and cleanup.
- `PHASE8_VALIDATION.md`: returned commit/cancellation evidence and the
  provenance-preserving offline verdict reconstruction.
