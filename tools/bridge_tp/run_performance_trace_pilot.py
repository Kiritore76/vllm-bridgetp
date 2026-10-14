# SPDX-License-Identifier: Apache-2.0
"""Run only idle/high-prefill START/STAY pairs to qualify batch telemetry."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import run_benefit_v4_collection as collection  # noqa: E402


def design():
    rows = []
    for index, target in enumerate(("idle", "prefill")):
        base = next(
            r
            for r in collection.formal_expansion_design()
            if r["role"] == "train"
            and r["source_state"] == "slack"
            and r["target_state"] == target
        )
        seed = 20262400 + index
        rows.append(
            {
                **base,
                "pair_id": f"performance-pilot-{target}-{seed}",
                "seed": seed,
                "role": "validation",
                "replicate": 0,
                "source_count": 3,
                "source_prompt_tokens": 4096,
                "target_count": 0 if target == "idle" else 8,
                "target_prompt_tokens": 2048 if target == "idle" else 6144,
                "target_max_tokens": 4096 if target == "idle" else 256,
                "target_period_s": 40 if target == "idle" else 30,
                "target_start_after_anchor": target == "prefill",
                "target_burst_arrivals": target == "prefill",
                "target_burst_offset_s": 1.0,
                "anchor_sections": 24,
                "threshold": 0.0,
                "minimum_u": 0.0,
                "maximum_u": 0.5,
                "diagnostic_use_only": True,
                "source_family_group": f"constructed-performance-pilot-{seed}",
                "purpose": "batch telemetry qualification; not independent final test",
            }
        )
    return rows


def main():
    if os.environ.get("BRIDGETP_PERFORMANCE_TRACE") != "1":
        raise ValueError("pilot requires BRIDGETP_PERFORMANCE_TRACE=1")
    if os.environ.get("BRIDGETP_PERFORMANCE_TRACE_DIR"):
        raise ValueError("trace directories must be assigned separately per arm")
    if "--stage" in sys.argv or "--plan-only" in sys.argv:
        raise ValueError("use --describe for the fixed two-pair design")
    if "--describe" in sys.argv:
        import json

        print(json.dumps(design(), indent=2))
        return
    collection.dispatch_stall_diagnostic_design = design
    sys.argv += ["--stage", "dispatch-stall-diagnostic"]
    collection.main()


if __name__ == "__main__":
    main()
