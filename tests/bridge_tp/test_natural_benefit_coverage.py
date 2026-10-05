"""Read-only coverage checks for naturally occurring migration decisions."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.bridge_tp.audit_natural_benefit_coverage import (
    bin_name,
    extract_run,
    summarize,
    validate_cuts,
)


def decision(tick: int, source_growth: float, target_running: int) -> list[dict]:
    return [
        {
            "kind": "telemetry", "tick": tick, "state": "LOCAL",
            "output_tokens": 40,
            "capacity_signal": {"active": False, "time_to_guard_s": 60.0},
        },
        {
            "kind": "manager_m5_predictor_shadow",
            "output_tokens": 40, "status": "AVAILABLE",
            "prediction_output_tokens": 40,
            "p_remaining_gt_headroom_runtime_bounds": [0.2, 0.3],
        },
        {
            "kind": "manager_m1_start_decision", "tick": tick,
            "snapshot": {
                "state": "LOCAL", "generated_tokens": 40,
                "request_id": "anchor", "unix_s": 100.0,
                "source_free_kv_tokens": 1000,
                "source_guard_free_kv_tokens": 100,
                "source_prefill_pending_kv_tokens": 200,
                "source_decode_growth_tokens_s": source_growth,
                "target_running": target_running, "target_waiting": 1,
                "target_kv_usage_frac": 0.5,
            },
            "decision": {
                "action": "STAY", "reason": "long-request probability is low",
                "source_time_to_guard_s": 35.0,
                "estimated_preparation_s": 5.0,
            },
        },
    ]


class TestNaturalBenefitCoverage(unittest.TestCase):
    def test_extracts_only_pre_action_evidence_and_one_episode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "controller" / "phase9_audit.jsonl"
            path.parent.mkdir()
            events = decision(1, 20.0, 2)
            events.append({"kind": "paired_stay_intervention", "tick": 1})
            events += decision(2, 22.0, 3)
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            background = root / "background" / "background_events.jsonl"
            background.parent.mkdir()
            background.write_text("".join(json.dumps(row) + "\n" for row in [
                {"kind": "job_start", "pool": "source", "unix_s": 99.0},
                {"kind": "job_start", "pool": "target", "unix_s": 98.0},
            ]), encoding="utf-8")
            (path.parent / "source_response.json").write_text(json.dumps({
                "request_started_unix_s": 95.0,
            }), encoding="utf-8")
            observations, provenance = extract_run(root)
        self.assertEqual(len(observations), 2)
        self.assertEqual(provenance["local_decisions"], 2)
        self.assertEqual(observations[0]["source_headroom_tokens"], 700)
        self.assertEqual(observations[0]["m5_status"], "AVAILABLE")
        self.assertAlmostEqual(observations[0]["source_arrival_rate_rps"], 0.2)
        self.assertTrue(observations[0]["paired_stay_intervention"])
        self.assertEqual(observations[1]["target_busy_count"], 4)
        summary = summarize([observations], 32, {
            "source_arrival_rate_rps": [0.1, 0.3],
            "target_busy_count": [2.0, 5.0],
        })
        self.assertEqual(summary["local_decision_ticks"], 2)
        self.assertEqual(summary["runs_with_candidate"], 1)
        self.assertEqual(summary["cells"], {"MEDIUM/MEDIUM": 1})
        self.assertFalse(summary["benefit_is_estimated"])

    def test_missing_metrics_remain_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "phase9_audit.jsonl"
            row = decision(1, 20.0, 2)[-1]
            row["snapshot"].pop("source_decode_growth_tokens_s")
            row["snapshot"].pop("target_running")
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            observations, _ = extract_run(root)
        summary = summarize([observations], 32, {
            "source_arrival_rate_rps": [0.1, 0.3],
            "target_busy_count": [2.0, 5.0],
        })
        self.assertEqual(summary["cells"], {"UNKNOWN/UNKNOWN": 1})
        self.assertEqual(summary["unknown_source_metric"], 1)
        self.assertEqual(summary["unknown_target_metric"], 1)

    def test_missing_anchor_start_does_not_become_zero_arrivals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "phase9_audit.jsonl").write_text(
                "".join(json.dumps(row) + "\n"
                        for row in decision(1, 20.0, 2)),
                encoding="utf-8",
            )
            background = root / "background" / "background_events.jsonl"
            background.parent.mkdir()
            background.write_text(json.dumps({
                "kind": "job_start", "pool": "target", "unix_s": 98.0,
            }) + "\n", encoding="utf-8")
            rows, _ = extract_run(root)
        self.assertIsNone(rows[0]["source_arrival_rate_rps"])
        self.assertEqual(rows[0]["target_arrival_rate_rps"], 0.1)

    def test_rejects_invalid_cut_points(self) -> None:
        with self.assertRaises(ValueError):
            validate_cuts({
                "source_arrival_rate_rps": [10, 10],
                "target_busy_count": [1, 2],
            })
        self.assertEqual(bin_name(1.0, [1.0, 2.0]), "MEDIUM")

    def test_fails_on_telemetry_snapshot_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = decision(1, 20.0, 2)
            rows[0]["output_tokens"] = 41
            (root / "phase9_audit.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "output mismatch"):
                extract_run(root)


if __name__ == "__main__":
    unittest.main()
