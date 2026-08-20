# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _read_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(
        f"{name} must be one of {sorted(_TRUE_VALUES | _FALSE_VALUES)}, "
        f"but got {value!r}"
    )


def _read_nonnegative_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, but got {value}")
    return value


@dataclass(frozen=True)
class BridgeTPDumpConfig:
    """Configuration for the Phase 1-3 TP1 KV-cache snapshot."""

    enabled: bool
    output_dir: Path
    dump_after_output_tokens: int
    target_request_id: str | None
    include_tensors: bool
    strict: bool
    max_bytes: int

    @classmethod
    def from_env(cls) -> "BridgeTPDumpConfig":
        """Build a validated configuration from environment variables."""
        request_id = os.getenv("BRIDGETP_DUMP_REQUEST_ID")
        if request_id is not None:
            request_id = request_id.strip() or None

        return cls(
            enabled=_read_bool("BRIDGETP_DUMP_ENABLED", False),
            output_dir=Path(
                os.getenv("BRIDGETP_DUMP_DIR", "/tmp/bridgetp_dumps")
            ).expanduser(),
            dump_after_output_tokens=_read_nonnegative_int(
                "BRIDGETP_DUMP_AFTER_OUTPUT_TOKENS", 128
            ),
            target_request_id=request_id,
            include_tensors=_read_bool("BRIDGETP_DUMP_TENSORS", True),
            strict=_read_bool("BRIDGETP_DUMP_STRICT", True),
            max_bytes=_read_nonnegative_int("BRIDGETP_DUMP_MAX_BYTES", 2 * 1024**3),
        )


@lru_cache(maxsize=1)
def get_bridge_tp_dump_config() -> BridgeTPDumpConfig:
    """Return the process-wide immutable dump configuration."""
    return BridgeTPDumpConfig.from_env()
