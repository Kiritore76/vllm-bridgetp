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
import json
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
    parser.add_argument("--rate-gib-s", type=float, nargs="+", default=[0.4, 0.7, 1.2])
    parser.add_argument("--tpot-model", type=Path)
    parser.add_argument(
        "--tpot-tp1-base-s",
        "--tpot-tp1-s",
        dest="tpot_tp1_base_s",
        type=float,
        default=0.030,
    )
    parser.add_argument("--tpot-tp1-per-running-s", type=float, default=0.0)
    parser.add_argument(
        "--tpot-tp4-base-s",
        "--tpot-tp4-s",
        dest="tpot_tp4_base_s",
        type=float,
        default=0.020,
    )
    parser.add_argument("--tpot-tp4-per-running-s", type=float, default=0.0)
    parser.add_argument("--interference-model", type=Path)
    parser.add_argument(
        "--interference-s-per-gib",
        type=float,
        default=InterferenceModel.s_per_gib_at_ref,
        help="use the value you calibrated from P2-D, not the anchor default",
    )
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def load_tpot_models(args: argparse.Namespace) -> tuple[TpotModel, TpotModel]:
    if args.tpot_model is None:
        return (
            TpotModel(args.tpot_tp1_base_s, args.tpot_tp1_per_running_s),
            TpotModel(args.tpot_tp4_base_s, args.tpot_tp4_per_running_s),
        )
    payload = json.loads(args.tpot_model.read_text(encoding="utf-8"))

    def build(name: str) -> TpotModel:
        raw = payload[name]
        return TpotModel(
            base_s=float(raw["base_s"]),
            per_running_s=float(raw["per_running_s"]),
            calibration_source=(
                f"{args.tpot_model}: {payload.get('status', 'candidate')}"
            ),
        )

    return build("tpot_tp1"), build("tpot_tp4")


def load_interference_model(args: argparse.Namespace) -> InterferenceModel:
    if args.interference_model is None:
        return InterferenceModel(
            s_per_gib_at_ref=args.interference_s_per_gib,
            calibration_source="legacy replay argument",
        )
    payload = json.loads(args.interference_model.read_text(encoding="utf-8"))
    return InterferenceModel(**payload["controller_model"])


def main() -> None:
    args = parse_args()
    table = SurvivalTable.load(args.survival_table)
    tpot_tp1, tpot_tp4 = load_tpot_models(args)
    interference = load_interference_model(args)
    policy = FastPolicy(
        config=PolicyConfig(),
        table=table,
        tpot_tp1=tpot_tp1,
        tpot_tp4=tpot_tp4,
        interference=interference,
    )
    print(
        f"# survival table: {table.source or args.survival_table} "
        f"(calibrated to {table.max_observed_length} tokens)\n"
        f"# TP1=({tpot_tp1.base_s:.6f} + {tpot_tp1.per_running_s:.6f}*running)s "
        f"TP4=({tpot_tp4.base_s:.6f} + {tpot_tp4.per_running_s:.6f}*running)s\n"
        f"# interference={interference.model_kind} "
        f"rates={','.join(str(rate) for rate in args.rate_gib_s)} GiB/s"
    )

    rows = []
    for produced in args.progress:
        for load in args.target_load:
            for rate in args.rate_gib_s:
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
                    rate * GIB,
                )
                rows.append(
                    {
                        "produced_tokens": produced,
                        "target_load": load,
                        "rate_gib_s": rate,
                        "survival_in_support": table.in_support(produced),
                        "interference_in_support": interference.in_support(load, rate),
                        "break_even_tokens": round(decision.break_even_tokens, 1),
                        "p_worth": round(decision.p_worth, 4),
                        "theta_esc": round(decision.theta_esc, 4),
                        "expected_remaining": round(
                            decision.expected_remaining_tokens, 1
                        ),
                        "cost_s": round(decision.cost_s, 4),
                        "action": decision.action.value,
                        "reason": decision.reason,
                    }
                )

    header = list(rows[0].keys())
    print(
        f"{'produced':>9} {'load':>6} {'rate':>6} {'N*':>9} "
        f"{'p_worth':>8} {'theta':>7}  action"
    )
    for row in rows:
        print(
            f"{row['produced_tokens']:>9} {row['target_load']:>6.2f} "
            f"{row['rate_gib_s']:>6.2f} "
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
