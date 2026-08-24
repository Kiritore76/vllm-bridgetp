# SPDX-License-Identifier: Apache-2.0
"""Phase 9 controller configuration.

One file, loaded once, echoed verbatim into the audit log so that a run can be
reproduced from its artifacts alone. YAML is used when PyYAML is available and
JSON otherwise, so the controller has no hard third-party dependency.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .policy import InterferenceModel, PolicyConfig, TpotModel
from .rate_controller import RateConfig
from .response_proxy import ProxyMode

GIB = 1024.0**3


@dataclass
class SlowConfig:
    """Slow persistent-risk controller (paper Section 3.8).

    Defaults come from the E4 replay. On a five-GPU testbed the constraint
    n1 + 4*n4 = 5 admits only (5,0) and (1,1), so this degenerates to a
    hysteretic binary switch; the parameters below are what gate that switch.
    """

    ewma_alpha: float = 0.08
    high_threshold: float = 0.65
    low_threshold: float = 0.45
    persistent_windows: int = 150
    recovery_windows: int = 60
    minimum_hold_s: float = 120.0
    window_s: float = 1.0
    # Set False for the fast-controller MVP that only keeps a warm TP4 pool.
    physical_reconfiguration: bool = False


@dataclass
class ControllerConfig:
    # endpoints
    source_url: str = "http://127.0.0.1:8001"
    target_url: str = "http://127.0.0.1:8200"
    run_dir: str = "./run"

    # pool geometry, needed to turn KV usage fraction into free tokens
    block_size: int = 16
    tp1_total_kv_blocks: int = 0
    tp4_total_kv_blocks: int = 0

    # control loop
    tick_s: float = 0.2
    max_ticks: int = 100_000
    handoff_output_tokens: int = 32

    # components
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    rate: RateConfig = field(default_factory=RateConfig)
    slow: SlowConfig = field(default_factory=SlowConfig)
    tpot_tp1: TpotModel = field(default_factory=lambda: TpotModel(0.030, 0.0015))
    tpot_tp4: TpotModel = field(default_factory=lambda: TpotModel(0.021, 0.0011))
    interference: InterferenceModel = field(
        default_factory=lambda: InterferenceModel(s_per_gib_at_ref=0.35)
    )

    # data plane
    proxy_mode: str = ProxyMode.HOLD_BACK.value
    survival_table_path: str = "./calibration/survival_table.json"

    # provenance
    platform_note: str = "A100-PCIe-40GB x5, single node, CPU-staged TCP"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path) -> ControllerConfig:
        text = Path(path).read_text(encoding="utf-8")
        raw: dict[str, Any]
        if str(path).endswith((".yaml", ".yml")):
            try:
                import yaml  # type: ignore
            except ImportError as error:  # pragma: no cover
                raise RuntimeError(
                    "PyYAML is required to read a .yaml config; use .json instead"
                ) from error
            raw = yaml.safe_load(text) or {}
        else:
            raw = json.loads(text)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ControllerConfig:
        nested = {
            "policy": PolicyConfig,
            "rate": RateConfig,
            "slow": SlowConfig,
            "tpot_tp1": TpotModel,
            "tpot_tp4": TpotModel,
            "interference": InterferenceModel,
        }
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in nested and isinstance(value, dict):
                kwargs[key] = nested[key](**value)
            elif key in cls.__dataclass_fields__:
                kwargs[key] = value
            else:
                raise ValueError(f"unknown controller config key: {key!r}")
        config = cls(**kwargs)
        config.validate()
        return config

    def validate(self) -> None:
        if self.tp1_total_kv_blocks <= 0 or self.tp4_total_kv_blocks <= 0:
            raise ValueError(
                "tp1_total_kv_blocks and tp4_total_kv_blocks must be measured from "
                "the running servers; the KV horizon H(t) is meaningless without them"
            )
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.handoff_output_tokens <= 0:
            raise ValueError("handoff_output_tokens must be positive")
        if self.proxy_mode not in {m.value for m in ProxyMode}:
            raise ValueError(f"unknown proxy_mode {self.proxy_mode!r}")
        if self.rate.b_min_bytes_s > self.rate.b_max_bytes_s:
            raise ValueError("rate b_min exceeds b_max")
        if self.slow.low_threshold >= self.slow.high_threshold:
            raise ValueError("slow low_threshold must be below high_threshold")
        if self.policy.theta_min > self.policy.theta_0:
            raise ValueError("theta_min must not exceed theta_0")

        # The three cost/benefit inputs set the break-even remaining length N*.
        # Shipping plausible-looking defaults into a measurement run is how a
        # policy ends up with a formula nobody calibrated, so refuse to start
        # until each one names where its numbers came from.
        uncalibrated = [
            name
            for name, model in (
                ("tpot_tp1", self.tpot_tp1),
                ("tpot_tp4", self.tpot_tp4),
                ("interference", self.interference),
            )
            if (
                not model.calibration_source.strip()
                or model.calibration_source.strip().lower().startswith("fill in")
            )
        ]
        if uncalibrated:
            raise ValueError(
                "refusing to run with uncalibrated cost models: "
                + ", ".join(uncalibrated)
                + ". Set calibration_source on each (see P9-1 in "
                "PHASE9_EXPERIMENT_PLAN.md), then confirm the migrate / "
                "do-not-migrate boundary with tools/bridge_tp/replay_policy.py."
            )
