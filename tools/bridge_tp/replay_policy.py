#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Offline policy replay: where is the migrate / do-not-migrate boundary?

Runs the fast policy over a grid of (decode progress, target load) without any
GPU, and emits the break-even remaining length N* and the escalation
probability p_worth at each point. This is the cheapest way to check that the
policy is behaving sensibly before spending cluster time, and it produces the
Section 3.5 figure showing that N* rises with target load -- the behaviour a
fixed token threshold cannot express.

    python tools/bridge_tp/replay_policy.py \
        --survival-table calibration/survival_table.json \
        --out results/policy_boundary.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.events import (  # noqa: E402
    MigrationState,
    PoolTelemetry,
    SourceRequestView,
)
from vllm.bridge_tp.controller.policy import (  # noqa: E402
    FastPolicy,
    InterferenceModel,
    PolicyConfig,
    TpotModel,
)
from vllm.bridge_tp.controller.predictor import SurvivalTable  # noqa: E402

GIB = 1024.0**3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--survival-table", type=Path, required=True)
    parser.add_argument(
        "--progress", type=int, nargs="+", default=[64, 128, 256, 512, 768, 1024]
    )
    parser.add_argument(
        "--target-load", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.85]
    )
    parser.add_argument("--rate-gib-s", type=float, default=0.5)
    parser.add_argument("--tpot-tp1-s", type=float, default=0.030)
    parser.add_argument("--tpot-tp4-s", type=float, default=0.020)
    parser.add_argument(
        "--interference-s-per-gib",
        type=float,
        default=InterferenceModel.s_per_gib_at_ref,
        help="use the value you calibrated from P2-D, not the anchor default",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table = SurvivalTable.load(args.survival_table)
    policy = FastPolicy(
        config=PolicyConfig(),
        table=table,
        tpot_tp1=TpotModel(args.tpot_tp1_s, 0.0),
        tpot_tp4=TpotModel(args.tpot_tp4_s, 0.0),
        interference=InterferenceModel(
            s_per_gib_at_ref=args.interference_s_per_gib,
            calibration_source="replay",
        ),
    )
    print(
        f"# survival table: {table.source or args.survival_table} "
        f"(calibrated to {table.max_observed_length} tokens)\n"
        f"# tau1={args.tpot_tp1_s * 1e3:.1f}ms tau4={args.tpot_tp4_s * 1e3:.1f}ms "
        f"interference={args.interference_s_per_gib:.2f} s/GiB "
        f"rate={args.rate_gib_s:.2f} GiB/s"
    )

    rows = []
    for produced in args.progress:
        for load in args.target_load:
            req = SourceRequestView(
                request_id="replay",
                prompt_tokens=512,
                output_tokens=produced,
                computed_tokens=produced,
                pending_tokens=1,
                arrival_unix_s=0.0,
                last_token_unix_s=0.0,
            )
            tp1 = PoolTelemetry(2, 0, 0.35, 0, 0.05, 0.03, 2000, 16)
            tp4 = PoolTelemetry(
                max(1, int(load * 10)),
                0,
                load,
                0,
                0.05,
                0.03,
                int(2000 * (1 - load)),
                16,
            )
            decision = policy.evaluate(
                req,
                MigrationState.LOCAL,
                tp1,
                tp4,
                0.2,
                args.rate_gib_s * GIB,
            )
            rows.append(
                {
                    "produced_tokens": produced,
                    "target_load": load,
                    "in_support": table.in_support(produced),
                    "break_even_tokens": round(decision.break_even_tokens, 1),
                    "p_worth": round(decision.p_worth, 4),
                    "theta_esc": round(decision.theta_esc, 4),
                    "expected_remaining": round(decision.expected_remaining_tokens, 1),
                    "cost_s": round(decision.cost_s, 4),
                    "action": decision.action.value,
                    "reason": decision.reason,
                }
            )

    header = list(rows[0].keys())
    print(f"{'produced':>9} {'load':>6} {'N*':>9} {'p_worth':>8} {'theta':>7}  action")
    for row in rows:
        print(
            f"{row['produced_tokens']:>9} {row['target_load']:>6.2f} "
            f"{row['break_even_tokens']:>9.0f} {row['p_worth']:>8.3f} "
            f"{row['theta_esc']:>7.3f}  {row['action']}"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
