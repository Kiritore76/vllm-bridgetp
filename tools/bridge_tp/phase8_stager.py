#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""CPU old-KV/delta stager and TP4 delivery service for Phase 8."""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import torch

from vllm.bridge_tp.runtime_control import RuntimeControl
from vllm.bridge_tp.stream_protocol import (
    PROTOCOL_VERSION,
    deserialize_rank_payload,
    recv_json,
    recv_payload_frames,
    send_json,
    send_payload_frames,
    serialize_rank_payload,
    sha256_bytes,
)


def _atomic_json_dump(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _wait_for_path(path: Path, cleanup: Path, deadline: float) -> bool:
    while not path.exists():
        if cleanup.exists():
            return False
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {path}")
        time.sleep(0.02)
    return True


class _DeltaReceivers:
    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        run_dir: Path,
        host: str,
        base_port: int,
        timeout_s: float,
    ) -> None:
        self.manifest = manifest
        self.run_dir = run_dir
        self.host = host
        self.base_port = base_port
        self.timeout_s = timeout_s
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.by_rank: list[dict[int, dict[str, Any]]] = [
            {} for _ in range(4)
        ]
        self.errors: list[str] = []
        self.listeners: list[socket.socket] = []
        self.threads: list[threading.Thread] = []

    def start(self) -> None:
        for rank in range(4):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((self.host, self.base_port + rank))
            listener.listen(16)
            listener.settimeout(0.1)
            thread = threading.Thread(
                target=self._run_rank,
                args=(rank, listener),
                name=f"bridgetp-phase8-stager-delta-{rank}",
                daemon=True,
            )
            self.listeners.append(listener)
            self.threads.append(thread)
            thread.start()

    def _run_rank(self, rank: int, listener: socket.socket) -> None:
        while not self.stop.is_set():
            try:
                connection, _ = listener.accept()
            except TimeoutError:
                continue
            try:
                with connection:
                    connection.settimeout(self.timeout_s)
                    header = recv_json(connection)
                    expected = {
                        "protocol_version": PROTOCOL_VERSION,
                        "phase": "BridgeTP D3 Phase 8",
                        "migration_id": self.manifest["migration_id"],
                        "session_token": self.manifest["session_token"],
                        "source_request_id": self.manifest["source_request_id"],
                        "target_tp_rank": rank,
                    }
                    for key, value in expected.items():
                        if header.get(key) != value:
                            raise ValueError(
                                f"delta rank {rank} field {key} differs"
                            )
                    payload_bytes, transfer = recv_payload_frames(
                        connection,
                        payload_bytes=int(header["payload_bytes"]),
                        num_frames=int(header["num_frames"]),
                        payload_sha256=str(header["payload_sha256"]),
                        max_frame_bytes=int(header["chunk_bytes"]),
                    )
                    payload = deserialize_rank_payload(payload_bytes)
                    start = int(header["start_token"])
                    end = int(header["end_token"])
                    if not start < end:
                        raise ValueError("empty or reversed Phase 8 delta")
                    if int(payload.get("start_token", -1)) != start or int(
                        payload.get("end_token", -1)
                    ) != end:
                        raise ValueError("Phase 8 delta header/payload differs")
                    with self.lock:
                        if start in self.by_rank[rank]:
                            raise ValueError(
                                f"duplicate Phase 8 delta start {start}"
                            )
                        self.by_rank[rank][start] = payload
                    receipt = {
                        "format_version": 1,
                        "phase": "BridgeTP D3 Phase 8",
                        "status": "STAGED",
                        "migration_id": self.manifest["migration_id"],
                        "target_tp_rank": rank,
                        "start_token": start,
                        "end_token": end,
                        **transfer,
                    }
                    _atomic_json_dump(
                        receipt,
                        self.run_dir
                        / "delta_stage_receipts"
                        / f"tp_rank_{rank}_{start}_{end}.json",
                    )
                    send_json(connection, {"status": "STAGED"})
            except Exception as error:
                self.errors.append(
                    f"rank={rank}: {type(error).__name__}: {error}"
                )

    def close(self) -> None:
        self.stop.set()
        for thread in self.threads:
            thread.join(timeout=2)
        for listener in self.listeners:
            listener.close()
        if self.errors:
            raise RuntimeError("; ".join(self.errors))


def _receive_initial_rank(
    manifest: dict[str, Any], run_dir: Path, rank: int, timeout_s: float
) -> dict[str, Any]:
    record = manifest["ranks"][rank]
    started = time.perf_counter()
    with socket.create_connection(
        (str(record["host"]), int(record["port"])), timeout=timeout_s
    ) as connection:
        connection.settimeout(timeout_s)
        send_json(
            connection,
            {
                "protocol_version": PROTOCOL_VERSION,
                "migration_id": manifest["migration_id"],
                "session_token": manifest["session_token"],
                "target_tp_rank": rank,
                "target_request_id": "phase8-cpu-stager",
            },
        )
        header = recv_json(connection)
        payload_bytes, transfer = recv_payload_frames(
            connection,
            payload_bytes=int(header["payload_bytes"]),
            num_frames=int(header["num_frames"]),
            payload_sha256=str(header["payload_sha256"]),
            max_frame_bytes=int(header["chunk_bytes"]),
        )
        payload = deserialize_rank_payload(payload_bytes)
        if int(payload.get("target_tp_rank", -1)) != rank:
            raise ValueError(f"initial payload rank differs for rank {rank}")
        send_json(
            connection,
            {
                "status": "READY",
                "target_request_id": "phase8-cpu-stager",
                "exact_readback": True,
            },
        )
    _atomic_json_dump(
        {
            "format_version": 1,
            "phase": "BridgeTP D3 Phase 8",
            "status": "STAGED",
            "migration_id": manifest["migration_id"],
            "target_tp_rank": rank,
            "num_computed_tokens": manifest["num_computed_tokens"],
            "stage_ms": (time.perf_counter() - started) * 1000,
            "completed_unix_s": time.time(),
            **transfer,
        },
        run_dir / "initial_stage_receipts" / f"tp_rank_{rank}.json",
    )
    return payload


def _token_axis(tensor: torch.Tensor, block_axis: int, block_size: int) -> int:
    candidates = [
        axis
        for axis, size in enumerate(tensor.shape)
        if axis != block_axis and int(size) == block_size
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"cannot infer staged token axis from {tuple(tensor.shape)}"
        )
    return candidates[0]


def _assemble_rank(
    *,
    initial: dict[str, Any],
    deltas: dict[int, dict[str, Any]],
    initial_end: int,
    final_end: int,
    block_axis: int,
    block_size: int,
) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
    expected = initial_end
    coverage: list[list[int]] = []
    layers = initial.get("layers")
    if not isinstance(layers, dict) or not layers:
        raise ValueError("initial staged payload has no layers")
    final_blocks = math.ceil(final_end / block_size)
    assembled: dict[str, torch.Tensor] = {}
    for name, tensor in layers.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"initial layer {name} is not a tensor")
        missing = final_blocks - int(tensor.shape[block_axis])
        if missing < 0:
            raise ValueError("initial staged tensor exceeds final block count")
        if missing:
            shape = list(tensor.shape)
            shape[block_axis] = missing
            tensor = torch.cat(
                [tensor, torch.zeros(shape, dtype=tensor.dtype)], dim=block_axis
            )
        assembled[name] = tensor
    for start in sorted(deltas):
        delta = deltas[start]
        end = int(delta["end_token"])
        if start != expected or end > final_end:
            raise ValueError(
                f"Phase 8 delta coverage gap/overlap: {start} != {expected}"
            )
        delta_layers = delta.get("layers")
        if not isinstance(delta_layers, dict) or set(delta_layers) != set(assembled):
            raise ValueError("Phase 8 delta layer set differs")
        for name, destination in assembled.items():
            source = delta_layers[name]
            if int(source.shape[0]) != end - start:
                raise ValueError("Phase 8 delta tensor length differs")
            token_axis = _token_axis(destination, block_axis, block_size)
            for offset, token_index in enumerate(range(start, end)):
                index: list[int | slice] = [slice(None)] * destination.ndim
                index[block_axis] = token_index // block_size
                index[token_axis] = token_index % block_size
                destination[tuple(index)].copy_(source[offset])
        coverage.append([start, end])
        expected = end
    if expected != final_end:
        raise ValueError(
            f"Phase 8 delta coverage stopped at {expected}, expected {final_end}"
        )
    return assembled, coverage


class _DeliveryPublisher:
    def __init__(
        self,
        *,
        manifest: dict[str, Any],
        run_dir: Path,
        rank: int,
        host: str,
        port: int,
        payload: bytes,
        header: dict[str, Any],
        timeout_s: float,
    ) -> None:
        self.manifest = manifest
        self.run_dir = run_dir
        self.rank = rank
        self.payload = payload
        self.header = header
        self.timeout_s = timeout_s
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((host, port))
        self.listener.listen(1)
        self.listener.settimeout(timeout_s)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.error: str | None = None

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        started = time.perf_counter()
        receipt: dict[str, Any] = {
            "format_version": 1,
            "phase": "BridgeTP D3 Phase 8",
            "status": "ERROR",
            "migration_id": self.manifest["migration_id"],
            "target_tp_rank": self.rank,
        }
        try:
            connection, _ = self.listener.accept()
            with connection:
                connection.settimeout(self.timeout_s)
                hello = recv_json(connection)
                for key, value in {
                    "protocol_version": PROTOCOL_VERSION,
                    "migration_id": self.manifest["migration_id"],
                    "session_token": self.manifest["session_token"],
                    "target_tp_rank": self.rank,
                }.items():
                    if hello.get(key) != value:
                        raise ValueError(f"delivery HELLO field {key} differs")
                send_json(connection, self.header)
                rate_provider = None
                if RuntimeControl.path(self.run_dir).exists():
                    rate_provider = self._current_per_rank_rate_bytes_s
                transfer = send_payload_frames(
                    connection,
                    self.payload,
                    chunk_bytes=int(self.header["chunk_bytes"]),
                    rate_provider=rate_provider,
                )
                acknowledgement = recv_json(connection)
                if acknowledgement.get("status") != "READY":
                    raise RuntimeError("target did not acknowledge staged payload")
                receipt.update(
                    {
                        "status": "READY",
                        "target_request_id": acknowledgement.get(
                            "target_request_id"
                        ),
                        "exact_readback": acknowledgement.get("exact_readback"),
                        **transfer,
                    }
                )
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            receipt["error"] = self.error
        finally:
            receipt["total_ms"] = (time.perf_counter() - started) * 1000
            _atomic_json_dump(
                receipt,
                self.run_dir
                / "stage_delivery_receipts"
                / f"tp_rank_{self.rank}.json",
            )
            self.payload = b""
            self.listener.close()

    def _current_per_rank_rate_bytes_s(self) -> float:
        control = RuntimeControl.load(self.run_dir)
        aggregate_rate = float(
            self.manifest.get("aggregate_rate_limit_gib_s", 0.0)
        )
        if control is not None and control.rate_gib_s is not None:
            aggregate_rate = control.rate_gib_s
        if not aggregate_rate:
            return 0.0
        return aggregate_rate * 1024**3 / 4


def _cleanup(run_dir: Path, reason: str, staged_ranks: int, deltas: int) -> None:
    _atomic_json_dump(
        {
            "format_version": 1,
            "phase": "BridgeTP D3 Phase 8",
            "status": "CLEANED",
            "component": "cpu_stager",
            "reason": reason,
            "released_rank_buffers": staged_ranks,
            "released_delta_batches": deltas,
            "updated_unix_s": time.time(),
        },
        run_dir / "stager_cleanup_receipt.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--delta-host", default="127.0.0.1")
    parser.add_argument("--delta-base-port", type=int, default=29900)
    parser.add_argument("--delivery-host", default="127.0.0.1")
    parser.add_argument("--delivery-base-port", type=int, default=30000)
    parser.add_argument("--timeout-s", type=float, default=600)
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.timeout_s
    cleanup_path = args.run_dir / "cleanup_request.json"
    manifest_path = args.run_dir / "session_manifest.json"
    if not _wait_for_path(manifest_path, cleanup_path, deadline):
        request = _load_json(cleanup_path)
        _cleanup(args.run_dir, str(request.get("reason")), 0, 0)
        return
    manifest = _load_json(manifest_path)
    if manifest.get("phase") != "BridgeTP D3 Phase 8":
        raise ValueError("CPU stager requires a Phase 8 source manifest")

    delta_receivers = _DeltaReceivers(
        manifest=manifest,
        run_dir=args.run_dir,
        host=args.delta_host,
        base_port=args.delta_base_port,
        timeout_s=args.timeout_s,
    )
    delta_receivers.start()
    with ThreadPoolExecutor(max_workers=4) as executor:
        initial = list(
            executor.map(
                lambda rank: _receive_initial_rank(
                    manifest, args.run_dir, rank, args.timeout_s
                ),
                range(4),
            )
        )

    cutover_path = args.run_dir / "cutover_manifest.json"
    if not _wait_for_path(cutover_path, cleanup_path, deadline):
        delta_receivers.close()
        request = _load_json(cleanup_path)
        delta_count = sum(len(value) for value in delta_receivers.by_rank)
        initial.clear()
        for value in delta_receivers.by_rank:
            value.clear()
        _cleanup(args.run_dir, str(request.get("reason")), 4, delta_count)
        return
    cutover = _load_json(cutover_path)
    delta_receivers.close()

    assembled_by_rank: list[dict[str, torch.Tensor]] = []
    coverage_by_rank: list[list[list[int]]] = []
    for rank in range(4):
        assembled, coverage = _assemble_rank(
            initial=initial[rank],
            deltas=delta_receivers.by_rank[rank],
            initial_end=int(manifest["num_computed_tokens"]),
            final_end=int(cutover["num_computed_tokens"]),
            block_axis=int(manifest["block_axis"]),
            block_size=int(manifest["block_size"]),
        )
        assembled_by_rank.append(assembled)
        coverage_by_rank.append(coverage)

    ranks: list[dict[str, Any]] = []
    publishers: list[_DeliveryPublisher] = []
    for rank, layers in enumerate(assembled_by_rank):
        raw_tensor_bytes = sum(
            value.numel() * value.element_size() for value in layers.values()
        )
        payload = serialize_rank_payload(
            {
                "format_version": 1,
                "migration_id": manifest["migration_id"],
                "source_request_id": manifest["source_request_id"],
                "target_tp_size": 4,
                "target_tp_rank": rank,
                "block_axis": manifest["block_axis"],
                "block_size": manifest["block_size"],
                "num_computed_tokens": cutover["num_computed_tokens"],
                "layers": layers,
            }
        )
        record = {
            "target_tp_rank": rank,
            "host": args.delivery_host,
            "port": args.delivery_base_port + rank,
            "raw_tensor_bytes": raw_tensor_bytes,
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
            "num_frames": math.ceil(len(payload) / int(manifest["chunk_bytes"])),
            "delta_coverage": coverage_by_rank[rank],
        }
        header = {
            "protocol_version": PROTOCOL_VERSION,
            "migration_id": manifest["migration_id"],
            "source_request_id": manifest["source_request_id"],
            "target_tp_size": 4,
            "target_tp_rank": rank,
            "num_computed_tokens": cutover["num_computed_tokens"],
            "pending_known_tokens": 1,
            "block_size": manifest["block_size"],
            "block_axis": manifest["block_axis"],
            "num_layers": manifest["num_layers"],
            "raw_tensor_bytes": raw_tensor_bytes,
            "payload_bytes": len(payload),
            "payload_sha256": record["payload_sha256"],
            "num_frames": record["num_frames"],
            "chunk_bytes": manifest["chunk_bytes"],
        }
        publisher = _DeliveryPublisher(
            manifest=manifest,
            run_dir=args.run_dir,
            rank=rank,
            host=args.delivery_host,
            port=args.delivery_base_port + rank,
            payload=payload,
            header=header,
            timeout_s=args.timeout_s,
        )
        publishers.append(publisher)
        ranks.append(record)
    for publisher in publishers:
        publisher.start()

    staging_manifest = {
        **manifest,
        "phase": "BridgeTP D3 Phase 8",
        "scope": "CPU-staged old-KV plus contiguous new-KV deltas",
        "snapshot_num_output_tokens": cutover["cutover_num_output_tokens"],
        "num_computed_tokens": cutover["num_computed_tokens"],
        "pending_known_tokens": cutover["pending_known_tokens"],
        "computed_token_ids": cutover["computed_token_ids"],
        "pending_token_ids": cutover["pending_token_ids"],
        "all_known_token_ids": cutover["all_known_token_ids"],
        "num_blocks": cutover["num_blocks"],
        "old_kv_num_computed_tokens": manifest["num_computed_tokens"],
        "new_kv_delta_tokens": cutover["delta_tokens"],
        "new_kv_delta_batches": cutover["delta_batches"],
        "ranks": ranks,
    }
    _atomic_json_dump(staging_manifest, args.run_dir / "staging_manifest.json")
    for publisher in publishers:
        publisher.thread.join()
    errors = [publisher.error for publisher in publishers if publisher.error]
    if errors:
        raise RuntimeError("; ".join(errors))
    print(json.dumps({
        "status": "READY",
        "phase": "BridgeTP D3 Phase 8",
        "migration_id": manifest["migration_id"],
        "old_kv_computed_tokens": manifest["num_computed_tokens"],
        "new_kv_delta_tokens": cutover["delta_tokens"],
        "final_computed_tokens": cutover["num_computed_tokens"],
    }, indent=2))


if __name__ == "__main__":
    main()
