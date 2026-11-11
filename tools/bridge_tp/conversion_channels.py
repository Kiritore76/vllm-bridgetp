# SPDX-License-Identifier: Apache-2.0
"""Target-scoped channel close protocol for dynamic unified conversions.

This extends the four-card plan/receipt protocol without aggregating identically
named rank receipts from different receivers. No engine or session is stopped
here: the executor must first obtain the controller's retired/drained evidence.
"""

import math
from pathlib import Path

from tools.bridge_tp import pressure_conversion as pc


class TargetChannelGraph:
    def __init__(self, run_dir, targets, source_files, *, clock, event):
        self.root = Path(run_dir).resolve()
        self.targets = {}
        self.seen = set()
        self.source_files = {s: Path(p) for s, p in source_files.items()}
        if any(
            not p.resolve().is_relative_to(self.root)
            for p in self.source_files.values()
        ):
            raise ValueError("source target file escaped owned run")
        self.clock, self.event = clock, event
        self.closing = None
        for target, description in targets.items():
            self.add_target(target, **description)

    def add_target(self, target, *, mechanism_root, channel_plan, tp=2, wake=False):
        """``wake``: a receiver whose engine slept comes back under its id.

        Its old edges were closed before it slept; the new ones carry fresh
        receiver-scoped generations, so receipts never mix incarnations.
        """
        if self.closing is not None and not self.closing["done"]:
            raise ValueError("cannot add a receiver during channel retirement")
        root, path = Path(mechanism_root).resolve(), Path(channel_plan).resolve()
        if target in self.targets or (target in self.seen and not wake):
            raise ValueError("target incarnation cannot be reused")
        if wake and target not in self.seen:
            raise ValueError("only a known receiver wakes")
        if root != self.root / "migration" / target or not path.is_relative_to(root):
            raise ValueError("target channel plan escaped owned receiver")
        if tp != 2:
            raise ValueError("six-card channel target must be TP2")
        self.targets[target] = dict(root=root, path=path, tp=tp)
        self.seen.add(target)

    def rows(self, target):
        rows = pc.read_json(self.targets[target]["path"])
        if not isinstance(rows, list):
            raise ValueError("missing or invalid receiver channel plan")
        if len({r["generation"] for r in rows}) != len(rows):
            raise ValueError("duplicate channel generation within receiver")
        return rows

    def receipts(self, target):
        """Latest receipt for each exact rank, confined to this receiver."""
        result = {}
        for name, row in pc.channel_receipts(self.targets[target]["root"]):
            if name not in {
                f"tp_rank_{r}.json" for r in range(self.targets[target]["tp"])
            }:
                continue
            stamp = row.get("completed_unix_s")
            if (
                type(stamp) not in (int, float)
                or not math.isfinite(stamp)
                or not 0 <= stamp <= self.clock()
            ):
                continue
            if row.get("completed_unix_s", 0) >= result.get(name, {}).get(
                "completed_unix_s", 0
            ):
                result[name] = row
        return result

    def closed_generations(self, target, written_at):
        """Keep close proof across rotations, scoped by receiver and exact rank."""
        result = {}
        for name, row in pc.channel_receipts(self.targets[target]["root"]):
            if name not in {
                f"tp_rank_{r}.json" for r in range(self.targets[target]["tp"])
            }:
                continue
            stamp = row.get("completed_unix_s")
            if (
                type(stamp) in (int, float)
                and math.isfinite(stamp)
                and written_at <= stamp <= self.clock()
            ):
                result.setdefault(name, set()).update(row.get("closed", ()))
        return result

    def acknowledged(self):
        for target, description in self.targets.items():
            planned = {r["generation"] for r in self.rows(target)}
            if not planned:
                continue
            receipts = self.receipts(target)
            for rank in range(description["tp"]):
                row = receipts.get(f"tp_rank_{rank}.json", {})
                if not planned <= set(row.get("planned", ())):
                    return False
        return True

    def retire(self, plan_id, retiring):
        """Remove only incident edges; await every affected receiver's ranks.

        Source files are changed first to prevent selection of a closing edge.
        Generation numbers are receiver-scoped, never instance generations.
        The original selection persists across polls and partial file writes.
        """
        retiring = frozenset(retiring)
        if self.closing is not None and self.closing["plan_id"] != plan_id:
            if not self.closing.get("done"):
                raise ValueError("previous channel close has not completed")
            self.closing = None
        if self.closing is None:
            selected, remaining = {}, {}
            for target in self.targets:
                rows = self.rows(target)
                selected[target] = [
                    r for r in rows if target in retiring or r["source"] in retiring
                ]
                remaining[target] = [r for r in rows if r not in selected[target]]
            self.closing = dict(
                plan_id=plan_id,
                retiring=retiring,
                selected=selected,
                remaining=remaining,
                published=False,
                done=False,
            )
        closing = self.closing
        if closing["retiring"] != retiring:
            raise ValueError("channel close plan changed while pending")
        if closing["done"]:
            return True
        if not closing["published"]:
            removed = {
                (r["source"], target, r["generation"])
                for target, rows in closing["selected"].items()
                for r in rows
            }
            if {source for source, _, _ in removed} - self.source_files.keys():
                raise ValueError("retiring channel has no owned source target file")
            for source, path in self.source_files.items():
                rows = pc.read_json(path)
                if not isinstance(rows, list):
                    raise ValueError("missing or invalid source target file")
                kept = [
                    r
                    for r in rows
                    if (source, r["target"], r["generation"]) not in removed
                ]
                if kept != rows:
                    pc.atomic_json(path, kept)
            # Set the boundary before writing: an immediate receiver ACK counts.
            closing.setdefault("written_at", self.clock())
            for target, selected in closing["selected"].items():
                if selected:
                    pc.atomic_json(
                        self.targets[target]["path"], closing["remaining"][target]
                    )
            closing["published"] = True
            self.event(
                "CHANNEL_GRAPH_CLOSE",
                plan_id=plan_id,
                closing={
                    t: [r["generation"] for r in rows]
                    for t, rows in closing["selected"].items()
                },
            )
        missing = []
        for target, rows in closing["selected"].items():
            if not rows:
                continue
            generations = {r["generation"] for r in rows}
            proofs = closing.setdefault("acks", {}).setdefault(target, {})
            for name, seen in self.closed_generations(
                target, closing["written_at"]
            ).items():
                proofs.setdefault(name, set()).update(seen)
            for rank in range(self.targets[target]["tp"]):
                if not generations <= proofs.get(f"tp_rank_{rank}.json", set()):
                    missing.append((target, rank))
        if missing:
            if self.clock() - closing["written_at"] > 60:
                raise RuntimeError(f"target-scoped channel close timed out: {missing}")
            return False
        closing.setdefault("closed_at", self.clock())
        teardown = {}
        for target, rows in closing["selected"].items():
            receipts = pc.teardown_receipts(self.targets[target]["root"])
            for row in rows:
                for rank in range(self.targets[target]["tp"]):
                    teardown[f"{target}/{rank}/{row['generation']}"] = receipts.get(
                        (f"tp_rank_{rank}", row["generation"])
                    )
        complete = all(s in {"ABORTED", "FAILED"} for s in teardown.values())
        if not complete and self.clock() - closing["closed_at"] < 10:
            return False
        closing["done"] = True
        self.event(
            "CHANNEL_GRAPH_RETIRED",
            plan_id=plan_id,
            teardown=teardown,
            teardown_complete=complete,
        )
        return True

    def remove_target(self, target):
        if (
            self.closing is None
            or not self.closing["done"]
            or target not in self.closing["retiring"]
        ):
            raise ValueError("receiver channels have not completed retirement")
        if self.rows(target):
            raise ValueError("receiver still has channel edges")
        del self.targets[target]
