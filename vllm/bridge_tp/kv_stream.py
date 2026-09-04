# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Live TP1 iteration-boundary publisher for BridgeTP Phases 6-8."""

from __future__ import annotations

import json
import math
import os
import secrets
import socket
import threading
import time
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from vllm.bridge_tp.controller.anchor_selector import select_source_request_id
from vllm.bridge_tp.kv_export import (
    _copy_request_blocks,
    _estimate_dump_bytes,
    _get_block_axis,
    _get_request_block_ids,
    _ordered_layer_names,
)
from vllm.bridge_tp.kv_reshard import iter_tp_rank_shards
from vllm.bridge_tp.runtime_control import (
    ControlCache,
    RuntimeControl,
    mark_control_honored,
)
from vllm.bridge_tp.stream_protocol import (
    PROTOCOL_VERSION,
    recv_json,
    send_json,
    send_payload_frames,
    serialize_rank_payload,
    sha256_bytes,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUE = {"1", "true", "yes", "on"}
_published_request_ids: set[str] = set()
_disabled_after_error = False
_publishers: list[_RankPublisher] = []


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} is not a boolean: {value!r}")


@dataclass(frozen=True)
class BridgeTPStreamConfig:
    enabled: bool
    takeover_enabled: bool
    phase8_enabled: bool
    migration_id: str
    run_dir: Path
    host: str
    base_port: int
    target_tp_size: int
    head_axis: int
    expected_kv_heads: int
    after_output_tokens: int
    phase8_cutover_output_tokens: int
    phase8_delta_host: str
    phase8_delta_base_port: int
    chunk_bytes: int
    aggregate_rate_gib_s: float
    socket_timeout_s: float
    pin_memory: bool
    strict: bool
    # Optional external request-ID prefix used by the capacity pilot to pick
    # one anchor from a real multi-request scheduler batch.  Empty preserves
    # the Phase 6-8 single-request-only behaviour exactly.
    source_request_id_prefix: str = ""
    # False only when a Phase 9 controller has published a control block that
    # has not armed this migration yet.  Absent a control block this stays True,
    # so Phase 6/7/8 runs behave exactly as before.
    armed: bool = True

    @classmethod
    def from_env(cls) -> BridgeTPStreamConfig:
        migration_id = os.getenv("BRIDGETP_STREAM_MIGRATION_ID", "").strip()
        run_dir = os.getenv("BRIDGETP_STREAM_RUN_DIR", "").strip()
        enabled = _env_bool("BRIDGETP_STREAM_ENABLED", False)
        if enabled and (not migration_id or not run_dir):
            raise ValueError(
                "Live streaming requires BRIDGETP_STREAM_MIGRATION_ID and "
                "BRIDGETP_STREAM_RUN_DIR"
            )
        config = cls(
            enabled=enabled,
            takeover_enabled=_env_bool("BRIDGETP_TAKEOVER_ENABLED", False),
            phase8_enabled=_env_bool("BRIDGETP_PHASE8_ENABLED", False),
            migration_id=migration_id,
            run_dir=Path(run_dir or "/tmp/bridgetp_phase6_disabled").expanduser(),
            host=os.getenv("BRIDGETP_STREAM_HOST", "127.0.0.1"),
            base_port=int(os.getenv("BRIDGETP_STREAM_BASE_PORT", "29600")),
            target_tp_size=int(os.getenv("BRIDGETP_STREAM_TARGET_TP", "4")),
            head_axis=int(os.getenv("BRIDGETP_STREAM_HEAD_AXIS", "3")),
            expected_kv_heads=int(
                os.getenv("BRIDGETP_STREAM_EXPECTED_KV_HEADS", "8")
            ),
            after_output_tokens=int(
                os.getenv("BRIDGETP_STREAM_AFTER_OUTPUT_TOKENS", "128")
            ),
            phase8_cutover_output_tokens=int(
                os.getenv("BRIDGETP_PHASE8_CUTOVER_OUTPUT_TOKENS", "160")
            ),
            phase8_delta_host=os.getenv(
                "BRIDGETP_PHASE8_DELTA_HOST", "127.0.0.1"
            ),
            phase8_delta_base_port=int(
                os.getenv("BRIDGETP_PHASE8_DELTA_BASE_PORT", "29900")
            ),
            chunk_bytes=int(
                os.getenv("BRIDGETP_STREAM_CHUNK_BYTES", str(1024 * 1024))
            ),
            aggregate_rate_gib_s=float(
                os.getenv("BRIDGETP_STREAM_RATE_GIB_S", "0")
            ),
            socket_timeout_s=float(
                os.getenv("BRIDGETP_STREAM_SOCKET_TIMEOUT_S", "600")
            ),
            pin_memory=_env_bool("BRIDGETP_STREAM_PIN_MEMORY", True),
            strict=_env_bool("BRIDGETP_STREAM_STRICT", True),
            source_request_id_prefix=os.getenv(
                "BRIDGETP_STREAM_SOURCE_REQUEST_ID_PREFIX", ""
            ).strip(),
        )
        if config.target_tp_size != 4:
            raise ValueError("BridgeTP Phase 6 currently requires target TP=4")
        if not 0 < config.chunk_bytes <= 16 * 1024 * 1024:
            raise ValueError("BRIDGETP_STREAM_CHUNK_BYTES must be in [1, 16 MiB]")
        config.validate_boundaries()
        return config

    def validate_boundaries(self) -> None:
        """Checks that must hold for env values AND for controller overrides.

        Phase 9 rewrites the rate and the two boundaries at runtime, so these
        cannot live inside ``from_env`` alone or a bad control block would slip
        through unvalidated.
        """
        if self.aggregate_rate_gib_s < 0:
            raise ValueError("BRIDGETP_STREAM_RATE_GIB_S cannot be negative")
        if self.after_output_tokens <= 0:
            raise ValueError("Snapshot output-token boundary must be positive")
        if self.phase8_enabled:
            if not self.takeover_enabled:
                raise ValueError("BridgeTP Phase 8 requires takeover to be enabled")
            if self.phase8_cutover_output_tokens <= self.after_output_tokens:
                raise ValueError(
                    "Phase 8 cutover boundary must follow the old-KV boundary"
                )


def _phase_name(config: BridgeTPStreamConfig) -> str:
    if config.phase8_enabled:
        return "BridgeTP D3 Phase 8"
    if config.takeover_enabled:
        return "BridgeTP D3 Phase 7"
    return "BridgeTP D3 Phase 6"


@lru_cache(maxsize=1)
def get_bridge_tp_stream_env_config() -> BridgeTPStreamConfig:
    """Environment baseline.  Immutable for the lifetime of the process."""
    return BridgeTPStreamConfig.from_env()


@lru_cache(maxsize=1)
def _control_cache(run_dir: str) -> ControlCache:
    return ControlCache(run_dir)


def get_bridge_tp_stream_config() -> BridgeTPStreamConfig:
    """Environment baseline overlaid with the live Phase 9 control block.

    Phase 6/7/8 froze the trigger boundary, the cutover boundary and the
    migration rate at process start, which is correct when a human fixes them
    before launching a run.  A Phase 9 controller has to change them while the
    request is decoding, so this reads a control block from the run directory
    on every call.  The cost is one ``stat`` per decode iteration; the JSON is
    re-parsed only when the controller has actually published a new block.

    When no control block exists the environment wins outright and ``armed`` is
    True, so every existing Phase 6/7/8 run script keeps working unchanged.
    """
    base = get_bridge_tp_stream_env_config()
    if not base.enabled:
        return base
    control = _control_cache(str(base.run_dir)).get()
    if control is None:
        return base
    overlaid = replace(
        base,
        armed=bool(control.armed),
        after_output_tokens=(
            control.trigger_output_tokens
            if control.trigger_output_tokens is not None
            else base.after_output_tokens
        ),
        phase8_cutover_output_tokens=(
            control.cutover_output_tokens
            if control.cutover_output_tokens is not None
            else base.phase8_cutover_output_tokens
        ),
        aggregate_rate_gib_s=(
            control.rate_gib_s
            if control.rate_gib_s is not None
            else base.aggregate_rate_gib_s
        ),
    )
    overlaid.validate_boundaries()
    mark_control_honored(base.run_dir, control.generation)
    return overlaid


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


class _RankPublisher:
    def __init__(
        self,
        *,
        config: BridgeTPStreamConfig,
        session_token: str,
        rank: int,
        payload: bytes,
        header: dict[str, Any],
    ) -> None:
        self.config = config
        self.session_token = session_token
        self.rank = rank
        self.payload = payload
        self.header = header
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.settimeout(config.socket_timeout_s)
        self.listener.bind((config.host, config.base_port + rank))
        self.listener.listen(1)
        self.thread = threading.Thread(
            target=self._serve,
            name=f"bridgetp-stream-rank-{rank}",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def _serve(self) -> None:
        started = time.perf_counter()
        receipt: dict[str, Any] = {
            "format_version": 1,
            "phase": _phase_name(self.config),
            "migration_id": self.config.migration_id,
            "target_tp_rank": self.rank,
            "status": "ERROR",
        }
        try:
            connection, peer = self.listener.accept()
            with connection:
                connection.settimeout(self.config.socket_timeout_s)
                hello = recv_json(connection)
                expected = {
                    "protocol_version": PROTOCOL_VERSION,
                    "migration_id": self.config.migration_id,
                    "session_token": self.session_token,
                    "target_tp_rank": self.rank,
                }
                for key, value in expected.items():
                    if hello.get(key) != value:
                        raise ValueError(
                            f"BridgeTP HELLO field {key} differs: "
                            f"{hello.get(key)!r} != {value!r}"
                        )
                send_json(connection, self.header)
                per_rank_rate = self._current_per_rank_rate_bytes_s()
                rate_provider = None
                if RuntimeControl.path(self.config.run_dir).exists():
                    rate_provider = self._current_per_rank_rate_bytes_s
                transfer = send_payload_frames(
                    connection,
                    self.payload,
                    chunk_bytes=self.config.chunk_bytes,
                    rate_bytes_per_second=per_rank_rate,
                    rate_provider=rate_provider,
                )
                acknowledgement = recv_json(connection)
                if acknowledgement.get("status") != "READY":
                    raise RuntimeError(
                        f"TP rank {self.rank} did not acknowledge READY: "
                        f"{acknowledgement}"
                    )
                receipt.update(
                    {
                        "status": "READY",
                        "peer": list(peer),
                        "target_request_id": acknowledgement.get(
                            "target_request_id"
                        ),
                        "exact_readback": acknowledgement.get("exact_readback"),
                        "rate_limit_aggregate_gib_s": (
                            self.config.aggregate_rate_gib_s
                        ),
                        **transfer,
                    }
                )
        except Exception as error:
            receipt["error"] = f"{type(error).__name__}: {error}"
            logger.exception("BridgeTP sender for TP rank %d failed", self.rank)
        finally:
            receipt["total_seconds"] = time.perf_counter() - started
            receipt["completed_unix_s"] = time.time()
            _atomic_json_dump(
                receipt,
                self.config.run_dir
                / "sender_receipts"
                / f"tp_rank_{self.rank}.json",
            )
            self.listener.close()
            # The Phase 8 old-KV path must release its serialized source copy
            # after the CPU stager acknowledges it.
            self.payload = b""

    def _current_per_rank_rate_bytes_s(self) -> float:
        config = get_bridge_tp_stream_config()
        if not config.aggregate_rate_gib_s:
            return 0.0
        return (
            config.aggregate_rate_gib_s
            * 1024**3
            / config.target_tp_size
        )


def _publish_request(
    *,
    config: BridgeTPStreamConfig,
    request_id: str,
    kv_caches: list[torch.Tensor],
    requests: dict[str, Any],
    input_batch: Any,
    scheduler_output: Any,
    kv_cache_config: Any,
    attn_groups: list[list[Any]],
    cache_dtype: str,
    model_name: str,
    tp_rank: int,
    tp_world_size: int,
    async_scheduling: bool,
) -> None:
    if tp_world_size != 1 or tp_rank != 0:
        raise ValueError("BridgeTP Phase 6 source must be TP1 rank 0")
    if async_scheduling:
        raise ValueError("BridgeTP Phase 6 requires --no-async-scheduling")
    if scheduler_output.scheduled_spec_decode_tokens:
        raise ValueError("BridgeTP Phase 6 does not support speculative decoding")
    if not kv_caches:
        raise RuntimeError("The source has no initialized KV-cache tensors")

    request_index = input_batch.req_id_to_index[request_id]
    request = requests[request_id]
    num_output_tokens = len(request.output_token_ids)
    if num_output_tokens < config.after_output_tokens:
        return
    num_scheduled = int(scheduler_output.num_scheduled_tokens.get(request_id, 0))
    num_computed_before = int(input_batch.num_computed_tokens_cpu[request_index])
    num_computed = num_computed_before + num_scheduled
    num_known = int(request.num_tokens)
    pending = num_known - num_computed
    if pending != 1:
        raise ValueError(
            "BridgeTP Phase 6 currently requires exactly one pending known token; "
            f"observed {pending}"
        )

    block_ids, block_size = _get_request_block_ids(
        input_batch, request_index, num_computed
    )
    block_axis = _get_block_axis(attn_groups, cache_dtype, block_size)
    layer_names = _ordered_layer_names(kv_cache_config)
    raw_source_bytes = _estimate_dump_bytes(
        kv_caches, block_axis, len(block_ids)
    )
    snapshot_started = time.perf_counter()
    source_layers, layer_records, d2h_ms = _copy_request_blocks(
        kv_caches, layer_names, block_ids, block_axis
    )

    session_token = secrets.token_hex(32)
    config.run_dir.mkdir(parents=True, exist_ok=True)
    if (config.run_dir / "session_manifest.json").exists():
        raise FileExistsError(
            f"Phase 6 run directory was already used: {config.run_dir}"
        )
    rank_records: list[dict[str, Any]] = []
    publishers: list[_RankPublisher] = []
    total_raw_rank_bytes = 0
    for rank, rank_layers_unpinned in iter_tp_rank_shards(
        source_layers,
        head_axis=config.head_axis,
        target_tp_size=config.target_tp_size,
        expected_source_kv_heads=config.expected_kv_heads,
    ):
        rank_layers = rank_layers_unpinned
        pinned = False
        if config.pin_memory and torch.cuda.is_available():
            rank_layers = {
                name: tensor.pin_memory() for name, tensor in rank_layers.items()
            }
            pinned = True
        raw_rank_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in rank_layers.values()
        )
        total_raw_rank_bytes += raw_rank_bytes
        payload = serialize_rank_payload(
            {
                "format_version": 1,
                "migration_id": config.migration_id,
                "source_request_id": request_id,
                "target_tp_size": config.target_tp_size,
                "target_tp_rank": rank,
                "block_axis": block_axis,
                "block_size": block_size,
                "num_computed_tokens": num_computed,
                "layers": rank_layers,
            }
        )
        payload_hash = sha256_bytes(payload)
        num_frames = math.ceil(len(payload) / config.chunk_bytes)
        header = {
            "protocol_version": PROTOCOL_VERSION,
            "migration_id": config.migration_id,
            "source_request_id": request_id,
            "target_tp_size": config.target_tp_size,
            "target_tp_rank": rank,
            "num_computed_tokens": num_computed,
            "pending_known_tokens": pending,
            "block_size": block_size,
            "block_axis": block_axis,
            "num_layers": len(rank_layers),
            "raw_tensor_bytes": raw_rank_bytes,
            "payload_bytes": len(payload),
            "payload_sha256": payload_hash,
            "num_frames": num_frames,
            "chunk_bytes": config.chunk_bytes,
        }
        publisher = _RankPublisher(
            config=config,
            session_token=session_token,
            rank=rank,
            payload=payload,
            header=header,
        )
        publishers.append(publisher)
        rank_records.append(
            {
                "target_tp_rank": rank,
                "host": config.host,
                "port": config.base_port + rank,
                "raw_tensor_bytes": raw_rank_bytes,
                "payload_bytes": len(payload),
                "payload_sha256": payload_hash,
                "num_frames": num_frames,
                "pinned_cpu": pinned,
            }
        )

    for publisher in publishers:
        publisher.start()
    _publishers.extend(publishers)

    computed_token_ids = [request.get_token_id(i) for i in range(num_computed)]
    pending_token_ids = [
        request.get_token_id(i) for i in range(num_computed, num_known)
    ]
    all_known_token_ids = computed_token_ids + pending_token_ids
    phase = _phase_name(config)
    manifest = {
        "format_version": 1,
        "phase": phase,
        "scope": (
            "old-KV background snapshot prepared for delta-staged takeover"
            if config.phase8_enabled
            else "live TP1 snapshot prepared for atomic TP4 takeover"
            if config.takeover_enabled
            else "live TP1 snapshot to TP4 shadow continuation; no ownership takeover"
        ),
        "protocol_version": PROTOCOL_VERSION,
        "migration_id": config.migration_id,
        "session_token": session_token,
        "model": model_name,
        "source_request_id": request_id,
        "source_tp_size": 1,
        "target_tp_size": config.target_tp_size,
        "cache_dtype": cache_dtype,
        "block_size": block_size,
        "block_axis": block_axis,
        "head_axis": config.head_axis,
        "expected_source_kv_heads": config.expected_kv_heads,
        "physical_block_ids": block_ids,
        "num_blocks": len(block_ids),
        "num_layers": len(source_layers),
        "num_prompt_tokens": int(request.num_prompt_tokens),
        "snapshot_num_output_tokens": num_output_tokens,
        "num_computed_tokens": num_computed,
        "pending_known_tokens": pending,
        "computed_token_ids": computed_token_ids,
        "pending_token_ids": pending_token_ids,
        "all_known_token_ids": all_known_token_ids,
        "raw_source_tensor_bytes": raw_source_bytes,
        "raw_rank_tensor_bytes_total": total_raw_rank_bytes,
        "d2h_snapshot_ms": d2h_ms,
        "snapshot_prepare_ms": (time.perf_counter() - snapshot_started) * 1000,
        "chunk_bytes": config.chunk_bytes,
        "aggregate_rate_limit_gib_s": config.aggregate_rate_gib_s,
        "layers": layer_records,
        "ranks": rank_records,
    }
    _atomic_json_dump(manifest, config.run_dir / "session_manifest.json")
    if config.takeover_enabled:
        _atomic_json_dump(
            {
                "format_version": 1,
                "phase": phase,
                "scope": (
                    "application-level atomic handoff; "
                    "no crash-consensus claim"
                ),
                "migration_id": config.migration_id,
                "source_request_id": request_id,
                "snapshot_num_output_tokens": num_output_tokens,
                "state": "PREPARING",
                "source_abort_dispatched": False,
                "updated_unix_s": time.time(),
            },
            config.run_dir / "takeover_state.json",
        )
    _published_request_ids.add(request_id)
    if config.phase8_enabled:
        from vllm.bridge_tp.phase8_source import start_phase8_source

        start_phase8_source(
            config=config,
            request_id=request_id,
            session_token=session_token,
            initial_num_computed_tokens=num_computed,
            block_size=block_size,
            block_axis=block_axis,
            layer_names=layer_names,
        )
    logger.warning(
        "%s published live request %s at output=%d, "
        "computed=%d, pending=%d, run=%s",
        phase,
        request_id,
        num_output_tokens,
        num_computed,
        pending,
        config.run_dir,
    )


def _publish_source_progress(
    *,
    config: BridgeTPStreamConfig,
    request_id: str,
    requests: dict[str, Any],
    input_batch: Any,
    scheduler_output: Any,
) -> None:
    """Publish the scheduler boundary observed by the Phase 9 controller."""
    request_index = input_batch.req_id_to_index[request_id]
    request = requests[request_id]
    num_scheduled = int(scheduler_output.num_scheduled_tokens.get(request_id, 0))
    num_computed = (
        int(input_batch.num_computed_tokens_cpu[request_index]) + num_scheduled
    )
    num_known = int(request.num_tokens)
    now = time.time()
    arrival = getattr(request, "arrival_time", None)
    if not isinstance(arrival, (int, float)):
        arrival = now
    _atomic_json_dump(
        {
            "format_version": 1,
            "phase": "BridgeTP D3 Phase 9",
            "source_request_id": request_id,
            "num_prompt_tokens": int(request.num_prompt_tokens),
            "num_output_tokens": len(request.output_token_ids),
            "num_computed_tokens": num_computed,
            "num_pending_tokens": num_known - num_computed,
            "arrival_unix_s": float(arrival),
            "updated_unix_s": now,
        },
        config.run_dir / "source_progress.json",
    )


def maybe_publish_kv_stream(
    *,
    kv_caches: list[torch.Tensor],
    requests: dict[str, Any],
    input_batch: Any,
    scheduler_output: Any,
    kv_cache_config: Any,
    attn_groups: list[list[Any]],
    cache_dtype: str,
    model_name: str,
    tp_rank: int,
    tp_world_size: int,
    async_scheduling: bool,
) -> None:
    """Publish one live TP1 request once it reaches the configured boundary."""
    global _disabled_after_error
    config = get_bridge_tp_stream_config()
    if not config.enabled or _disabled_after_error:
        return
    try:
        request_ids = input_batch.req_ids
        request_id = select_source_request_id(
            request_ids,
            config.source_request_id_prefix,
        )
        if request_id is None:
            return
        if _published_request_ids and request_id not in _published_request_ids:
            # A dedicated validation server owns one migration session. Later
            # clean-control requests must not overwrite its progress evidence
            # or reuse its ports and run directory.
            return
        if RuntimeControl.path(config.run_dir).exists():
            _publish_source_progress(
                config=config,
                request_id=request_id,
                requests=requests,
                input_batch=input_batch,
                scheduler_output=scheduler_output,
            )
        if not config.armed:
            return
        if request_id in _published_request_ids:
            if config.phase8_enabled:
                from vllm.bridge_tp.phase8_source import maybe_publish_phase8_delta

                maybe_publish_phase8_delta(
                    config=config,
                    request_id=request_id,
                    kv_caches=kv_caches,
                    requests=requests,
                    input_batch=input_batch,
                    scheduler_output=scheduler_output,
                    cache_dtype=cache_dtype,
                    attn_groups=attn_groups,
                )
            return
        _publish_request(
            config=config,
            request_id=request_id,
            kv_caches=kv_caches,
            requests=requests,
            input_batch=input_batch,
            scheduler_output=scheduler_output,
            kv_cache_config=kv_cache_config,
            attn_groups=attn_groups,
            cache_dtype=cache_dtype,
            model_name=model_name,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            async_scheduling=async_scheduling,
        )
    except Exception:
        if config.strict:
            raise
        _disabled_after_error = True
        logger.exception("BridgeTP live publisher failed and was disabled")
