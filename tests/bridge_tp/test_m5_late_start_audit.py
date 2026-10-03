# SPDX-License-Identifier: Apache-2.0
"""The late-start gate must verify actual source work and KV release."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.bridge_tp.audit_m5_late_start import audit


class TestM5LateStartAudit(unittest.TestCase):
    def make_run(self, root: Path, *, source_start: float = 101.0,
                 output: int = 96, minimum_free: int = 9000,
                 pending_prefill: int = 0) -> Path:
        result = root / "online" / "r01_shadow_only"
        controller = result / "controller"
        background = result / "background"
        provenance = result / "provenance"
        for path in (controller, background, provenance):
            path.mkdir(parents=True)
        rows = [
            {"kind": "telemetry", "unix_s": 100.0, "output_tokens": 1},
            {"kind": "manager_m1_start_decision", "unix_s": 102.0,
             "snapshot": {"generated_tokens": output,
                          "source_guard_free_kv_tokens": 8448},
             "decision": {"action": "START_SHADOW"}},
            {"kind": "telemetry", "unix_s": 102.5,
             "capacity_signal": {
                 "free_kv_tokens": minimum_free,
                 "prefill_pending_kv_tokens": pending_prefill,
             }, "tp1": {"preemptions_total": 0}},
        ]
        events = [
            {"kind": "job_start", "pool": "source", "job_id": "peer",
             "unix_s": source_start},
            {"kind": "job_end", "pool": "source", "job_id": "peer",
             "unix_s": 110.0},
        ]
        (controller / "phase9_audit.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8")
        (background / "background_events.jsonl").write_text(
            "\n".join(json.dumps(row) for row in events), encoding="utf-8")
        (background / "background_manifest.json").write_text(json.dumps({
            "jobs": [{"pool": "source",
                      "start_after_event": "ANCHOR_FIRST_OUTPUT"}],
        }), encoding="utf-8")
        (provenance / "shadow_online_acceptance.json").write_text(
            json.dumps({"status": "PASS", "source_kv_released_unix_s": 104.0}),
            encoding="utf-8")
        return root

    def test_late_start_passes_only_with_peer_and_guard_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_run(Path(directory))
            self.assertEqual(audit(root, 96)["status"], "PASS")

    def test_early_start_or_guard_crossing_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_run(Path(directory), output=95,
                                 minimum_free=8448)
            result = audit(root, 96)
            self.assertEqual(result["status"], "FAIL")
            self.assertEqual(len(result["errors"]), 2)

    def test_peer_arrival_after_start_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_run(Path(directory), source_start=103.0)
            self.assertIn("no source peer was active before M1 started",
                          audit(root, 96)["errors"])

    def test_pending_prefill_consumes_guard_headroom(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_run(Path(directory), pending_prefill=600)
            self.assertIn(
                "source KV headroom reached the guard before TP1 KV release",
                audit(root, 96)["errors"],
            )


if __name__ == "__main__":
    unittest.main()
