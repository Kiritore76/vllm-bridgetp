# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in raw/processed-logit evidence capture for Phase 9 D-0.

The diagnostic is deliberately request- and index-scoped. Disabled runs add
no tensor copies. Enabled runs persist the actual vLLM tensor values before
and after greedy-relevant processors, together with the discrete prefix hash
and sampled token. It does not use an HF re-computation as a substitute for
the real serving path.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
_disabled_after_error = False


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"{name} must be a boolean, got {value!r}")


def _read_csv_ints(name: str) -> tuple[int, ...]:
    raw = os.getenv(name, "").strip()
    if not raw:
        return ()
    values = tuple(int(item.strip()) for item in raw.split(",") if item.strip())
    if any(value < 0 for value in values):
        raise ValueError(f"{name} must contain nonnegative integers")
    return values


def _safe_request_id(request_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", request_id)
    return safe[:160] or "request"


def _resolve_target_request_id(
    configured: str,
    request_ids: list[str],
) -> str | None:
    """Resolve a public completion id to vLLM's internal request id."""
    bases = [configured]
    if not configured.startswith("cmpl-"):
        bases.append(f"cmpl-{configured}-0")

    def matches_filter(request_id: str) -> bool:
        for base in bases:
            if request_id == base:
                return True
            randomized = re.fullmatch(
                rf"{re.escape(base)}-[0-9A-Fa-f]{{8}}",
                request_id,
            )
            if randomized is not None:
                return True
        return False

    matches = [request_id for request_id in request_ids if matches_filter(request_id)]
    if len(matches) > 1:
        raise RuntimeError(
            "BridgeTP logit capture request filter is ambiguous: "
            f"configured={configured!r}, matches={matches!r}"
        )
    return matches[0] if matches else None


def token_ids_sha256(token_ids: list[int]) -> str:
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _atomic_json_dump(data: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


@dataclass(frozen=True)
class LogitCaptureConfig:
    enabled: bool
    output_dir: Path
    target_request_id: str | None
    global_indices: tuple[int, ...]
    global_index_offset: int
    candidate_token_ids: tuple[int, ...]
    topk: int
    include_tensors: bool
    capture_tp_rank: int
    strict: bool

    @classmethod
    def from_env(cls) -> LogitCaptureConfig:
        request_id = os.getenv("BRIDGETP_LOGIT_CAPTURE_REQUEST_ID")
        if request_id is not None:
            request_id = request_id.strip() or None
        topk = int(os.getenv("BRIDGETP_LOGIT_CAPTURE_TOPK", "20"))
        offset = int(os.getenv("BRIDGETP_LOGIT_CAPTURE_GLOBAL_OFFSET", "0"))
        rank = int(os.getenv("BRIDGETP_LOGIT_CAPTURE_TP_RANK", "0"))
        if topk < 2:
            raise ValueError("BRIDGETP_LOGIT_CAPTURE_TOPK must be at least 2")
        if offset < 0 or rank < 0:
            raise ValueError("logit capture offset and TP rank must be nonnegative")
        return cls(
            enabled=_read_bool("BRIDGETP_LOGIT_CAPTURE_ENABLED", False),
            output_dir=Path(
                os.getenv("BRIDGETP_LOGIT_CAPTURE_DIR", "/tmp/bridgetp_logits")
            ).expanduser(),
            target_request_id=request_id,
            global_indices=_read_csv_ints("BRIDGETP_LOGIT_CAPTURE_INDICES"),
            global_index_offset=offset,
            candidate_token_ids=_read_csv_ints(
                "BRIDGETP_LOGIT_CAPTURE_CANDIDATE_TOKEN_IDS"
            ),
            topk=topk,
            include_tensors=_read_bool("BRIDGETP_LOGIT_CAPTURE_TENSORS", True),
            capture_tp_rank=rank,
            strict=_read_bool("BRIDGETP_LOGIT_CAPTURE_STRICT", True),
        )


@lru_cache(maxsize=1)
def get_logit_capture_config() -> LogitCaptureConfig:
    return LogitCaptureConfig.from_env()


class LogitObserver:
    """Capture one request's next-token evidence across sampler stages."""

    def __init__(
        self,
        *,
        config: LogitCaptureConfig,
        request_id: str,
        request_index: int,
        prefix_token_ids: list[int],
        local_output_index: int,
        global_output_index: int,
    ) -> None:
        self.config = config
        self.request_id = request_id
        self.request_index = request_index
        self.prefix_token_ids = prefix_token_ids
        self.local_output_index = local_output_index
        self.global_output_index = global_output_index
        self.output_dir = (
            config.output_dir
            / _safe_request_id(request_id)
            / f"global_{global_output_index:06d}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stages: dict[str, dict[str, Any]] = {}

    def capture(self, stage: str, logits: Any) -> None:
        import torch

        if stage not in {"raw", "processed"}:
            raise ValueError(f"unknown logit capture stage {stage!r}")
        if logits.ndim != 2 or self.request_index >= logits.shape[0]:
            raise RuntimeError(
                "logit rows do not align with the request batch: "
                f"shape={tuple(logits.shape)}, index={self.request_index}"
            )
        row = logits[self.request_index].detach().contiguous().cpu()
        raw_bytes = row.view(torch.uint8).numpy().tobytes()
        values, token_ids = torch.topk(row.float(), min(self.config.topk, row.numel()))
        candidates = {
            str(token_id): float(row[token_id].float().item())
            for token_id in self.config.candidate_token_ids
            if token_id < row.numel()
        }
        tensor_file = None
        if self.config.include_tensors:
            tensor_file = f"{stage}_logits.pt"
            destination = self.output_dir / tensor_file
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(row, temporary)
            os.replace(temporary, destination)
        self.stages[stage] = {
            "dtype": str(row.dtype).replace("torch.", ""),
            "shape": list(row.shape),
            "numel": int(row.numel()),
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "tensor_file": tensor_file,
            "top_token_ids": [int(value) for value in token_ids.tolist()],
            "top_values": [float(value) for value in values.tolist()],
            "candidate_values": candidates,
            "min": float(row.float().min().item()),
            "max": float(row.float().max().item()),
        }

    def finalize(self, sampled_token_ids: Any) -> None:
        sampled = sampled_token_ids.reshape(-1)
        if self.request_index >= sampled.numel():
            raise RuntimeError("sampled-token rows do not align with captured logits")
        payload = {
            "format_version": 1,
            "phase": "BridgeTP D3 Phase 9 D-0",
            "request_id": self.request_id,
            "local_output_index": self.local_output_index,
            "global_output_index": self.global_output_index,
            "prefix_token_count": len(self.prefix_token_ids),
            "prefix_token_ids_sha256": token_ids_sha256(self.prefix_token_ids),
            "candidate_token_ids": list(self.config.candidate_token_ids),
            "sampled_token_id": int(sampled[self.request_index].item()),
            "stages": self.stages,
            "evidence_boundary": (
                "Values are the actual vLLM tensors at this request/index. ULP "
                "counts describe tensor representability; they do not identify "
                "a CUDA kernel or prove that all divergence is rounding."
            ),
        }
        if set(self.stages) != {"raw", "processed"}:
            raise RuntimeError("both raw and processed logits must be captured")
        _atomic_json_dump(payload, self.output_dir / "capture.json")


def maybe_make_logit_observer(
    *,
    req_ids: list[str],
    requests: dict[str, Any],
    tp_rank: int,
) -> LogitObserver | None:
    """Return an observer only for an explicitly selected request/index."""
    global _disabled_after_error

    config = get_logit_capture_config()
    if not config.enabled or _disabled_after_error or tp_rank != config.capture_tp_rank:
        return None
    try:
        if config.target_request_id is not None:
            resolved_request_id = _resolve_target_request_id(
                config.target_request_id,
                req_ids,
            )
            if resolved_request_id is None:
                return None
            request_id = resolved_request_id
        else:
            if len(req_ids) != 1:
                return None
            request_id = req_ids[0]
        request_index = req_ids.index(request_id)
        request = requests[request_id]
        local_index = len(request.output_token_ids)
        global_index = config.global_index_offset + local_index
        if config.global_indices and global_index not in config.global_indices:
            return None
        prefix = [*request.prompt_token_ids, *request.output_token_ids]
        return LogitObserver(
            config=config,
            request_id=request_id,
            request_index=request_index,
            prefix_token_ids=[int(value) for value in prefix],
            local_output_index=local_index,
            global_output_index=global_index,
        )
    except Exception:
        if config.strict:
            raise
        _disabled_after_error = True
        logger.exception("BridgeTP logit capture failed and was disabled")
        return None
