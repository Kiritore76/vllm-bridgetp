#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check one Phase 9 run against the pass conditions, from artifacts alone.

Mirrors the gates in PHASE9_EXPERIMENT_PLAN.md. Reads only immutable run
artifacts -- the audit log, the receipts, and the takeover state -- so a run
can be re-checked later without the controller process.

    python tools/bridge_tp/inspect_phase9_run.py --run-dir runs/e2_seed0

Exit status is 0 only when every applicable gate passes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.audit import read_audit  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--control-tokens",
        type=Path,
        default=None,
        help="JSON list of the clean-TP1 control token IDs",
    )
    parser.add_argument(
        "--expect",
        choices=("commit", "rollback", "cancel", "completed"),
        default="commit",
        help="expected terminal path; commit is the strict E-1 default",
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable")
    return parser.parse_args()


class Report:
    def __init__(self) -> None:
        self.checks: list[tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(ok for _n, ok, _d in self.checks)

    def render(self) -> str:
        lines = []
        for name, ok, detail in self.checks:
            mark = "PASS" if ok else "FAIL"
            lines.append(f"[{mark}] {name}" + (f" -- {detail}" if detail else ""))
        lines.append("")
        lines.append("RESULT: " + ("PASS" if self.passed else "FAIL"))
        return "\n".join(lines)

    def to_json(self) -> dict:
        return {
            "passed": self.passed,
            "checks": [
                {"name": n, "status": ("pass" if o else "fail"), "detail": d}
                for n, o, d in self.checks
            ],
        }


def main() -> None:
    args = parse_args()
    run = args.run_dir
    report = Report()

    audit_path = run / "phase9_audit.jsonl"
    if not audit_path.exists():
        report.add("audit log present", False, f"{audit_path} missing")
        print(report.render())
        raise SystemExit(1)
    records = list(read_audit(audit_path))
    report.add("audit log present", True, f"{len(records)} records")

    kinds = [r.get("kind") for r in records]
    decisions = [r for r in records if r.get("kind") == "decision"]
    transitions = [r for r in records if r.get("kind") == "transition"]
    end = next((r for r in reversed(records) if r.get("kind") == "run_end"), None)

    # --- gate 3: every decision is auditable -------------------------------
    required = {
        "p_worth",
        "p_oom",
        "theta_esc",
        "benefit_s",
        "cost_s",
        "break_even_tokens",
        "cost_breakdown",
        "action",
        "reason",
    }
    incomplete = [d for d in decisions if not required.issubset(d)]
    report.add(
        "policy gate: every decision records inputs, benefit, cost, action",
        not incomplete and bool(decisions),
        f"{len(decisions)} decisions, {len(incomplete)} incomplete",
    )

    # --- gate 2: safety ----------------------------------------------------
    state_path = run / "takeover_state.json"
    state = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_takeover = {
        "commit": "COMMITTED",
        "rollback": "ROLLED_BACK",
        "cancel": "CANCELLED",
        "completed": None,
    }[args.expect]
    if expected_takeover is None:
        report.add(
            "expected no committed takeover",
            not state or state.get("state") not in {"COMMITTING", "COMMITTED"},
            f"state={state.get('state') if state else None}",
        )
    else:
        report.add(
            f"expected takeover state {expected_takeover}",
            bool(state and state.get("state") == expected_takeover),
            f"state={state.get('state') if state else None}",
        )
    committed = args.expect == "commit" and bool(
        state and state.get("state") == "COMMITTED"
    )

    commit_transitions = [t for t in transitions if t.get("to") == "TAKEOVER"]
    bad_commit = [t for t in commit_transitions if len(t.get("ranks_ready", [])) < 4]
    report.add(
        "safety gate: no commit without four ready ranks",
        not bad_commit,
        f"{len(commit_transitions)} commit transitions",
    )

    if committed:
        report.add(
            "safety gate: source abort dispatched exactly on commit",
            bool(state.get("source_abort_dispatched")),
            f"state={state.get('state')}",
        )
    elif state and state.get("state") in ("ROLLED_BACK", "CANCELLED"):
        expected_abort = state.get("state") == "CANCELLED"
        report.add(
            f"safety gate: {state['state']} cleanup semantics",
            bool(state.get("source_abort_dispatched")) == expected_abort,
            f"source_abort_dispatched={state.get('source_abort_dispatched')}",
        )
    elif args.expect != "completed":
        report.add("safety gate: terminal takeover state", False, "no takeover state")

    # --- receipts ----------------------------------------------------------
    receiver_root = run / "receiver_receipts"
    if receiver_root.is_dir():
        target_dirs = [p for p in receiver_root.iterdir() if p.is_dir()]
        ok = len(target_dirs) == 1
        detail = f"{len(target_dirs)} target request directories"
        if ok:
            readbacks = []
            for rank in range(4):
                path = target_dirs[0] / f"tp_rank_{rank}.json"
                if path.exists():
                    readbacks.append(
                        json.loads(path.read_text(encoding="utf-8")).get(
                            "exact_readback"
                        )
                    )
            ok = len(readbacks) == 4 and all(readbacks)
            detail = f"exact_readback={readbacks}"
        report.add("four-rank exact readback", ok, detail)
    elif args.expect == "commit":
        report.add("four-rank exact readback", False, "no receiver receipts")

    # --- gate 1: client stream --------------------------------------------
    stream_path = run / "response_proxy_stats.json"
    if stream_path.exists():
        stats = json.loads(stream_path.read_text(encoding="utf-8"))
        token_ids = stats.get("token_ids")
        report.add(
            "correctness gate: unified stream has no gap or duplicate",
            stats.get("emitted_tokens", 0) > 0
            and isinstance(token_ids, list)
            and len(token_ids) == int(stats.get("emitted_tokens", -1)),
            f"{stats.get('emitted_tokens')} tokens, mode={stats.get('mode')}, "
            f"stall={stats.get('handoff_stall_s')}",
        )
        unified_path = run / "unified_response.jsonl"
        if unified_path.exists():
            emitted_rows = [
                json.loads(line)
                for line in unified_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            report.add(
                "correctness gate: emitted JSONL matches proxy state",
                [row.get("index") for row in emitted_rows]
                == list(range(len(emitted_rows)))
                and [row.get("token_id") for row in emitted_rows]
                == list(token_ids or []),
                f"{len(emitted_rows)} externally emitted token records",
            )
        else:
            report.add(
                "correctness gate: emitted JSONL present",
                False,
                "unified_response.jsonl missing",
            )
        if args.expect == "commit":
            report.add(
                "correctness gate: source and target both contributed",
                int(stats.get("source_origin_tokens", 0)) > 0
                and int(stats.get("target_origin_tokens", 0)) > 0,
                f"source={stats.get('source_origin_tokens')} "
                f"target={stats.get('target_origin_tokens')}",
            )
        if args.control_tokens and args.control_tokens.exists():
            control = json.loads(args.control_tokens.read_text(encoding="utf-8"))
            emitted = token_ids if isinstance(token_ids, list) else []
            match = list(emitted) == list(control)
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(emitted, control)) if a != b),
                None,
            )
            report.add(
                "correctness gate: token-identical to greedy control",
                match,
                "exact" if match else f"first divergence at index {first_diff}",
            )
        else:
            report.add(
                "correctness gate: clean TP1 control tokens supplied",
                False,
                "--control-tokens is missing or unreadable",
            )
    else:
        report.add("correctness gate: unified stream", False, "no proxy stats saved")

    # --- run completion ----------------------------------------------------
    if end is not None:
        expected_final = {
            "commit": "TAKEOVER",
            "rollback": "ROLLED_BACK",
            "cancel": "CANCELLED",
            "completed": "COMPLETED_ON_TP1",
        }[args.expect]
        report.add(
            "run reached a terminal state",
            end.get("final_state") == expected_final,
            f"final_state={end.get('final_state')} ticks={end.get('ticks')}",
        )
        if end.get("t_cutover") and end.get("t_committed"):
            report.add(
                "handoff duration recorded",
                True,
                f"cutover->commit = {end['t_committed'] - end['t_cutover']:.3f} s",
            )
    else:
        report.add("run reached a terminal state", False, "no run_end record")

    if "invariant_violation" in kinds:
        report.add("no invariant violations logged", False, "see audit log")
    else:
        report.add("no invariant violations logged", True)

    if args.json:
        print(json.dumps(report.to_json(), indent=2))
    else:
        print(report.render())
    raise SystemExit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
