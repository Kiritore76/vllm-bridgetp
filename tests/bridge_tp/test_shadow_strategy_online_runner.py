# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from tools.bridge_tp.build_shadow_strategy_online_manifest import build_manifest
from tools.bridge_tp.run_phase9_capacity_background import percentile
from tools.bridge_tp.run_phase9_cap0_calibration import make_source_request
from tools.bridge_tp.run_phase9_cap0_noop import wait_for_background_first_tokens
from tools.bridge_tp.run_shadow_rate_load_matrix import (
    parse_load_profiles,
    rate_label,
    resolve_design,
)
from tools.bridge_tp.run_shadow_strategy_online_validation import (
    accept_m1_stay,
    accept_paired_stay,
    build_controller_config_overrides,
    controller_completion_errors,
    emitted_boundary_gap_ms,
    has_measured_source_high,
    summarize_emitted_intervals,
    summarize_slo,
    write_measurements,
)
from vllm.bridge_tp.online_shadow_strategy_protocol import (
    summarize_background_windows,
    validate_strategy_timing,
)


class TestOnlineShadowManifest(unittest.TestCase):
    def test_builds_target_only_exact_token_jobs(self) -> None:
        manifest = build_manifest(
            target_jobs=2,
            prompt_tokens=4,
            output_tokens=3,
            max_model_len=8,
        )
        self.assertEqual(len(manifest["jobs"]), 2)
        self.assertTrue(all(job["pool"] == "target" for job in manifest["jobs"]))
        self.assertEqual(manifest["jobs"][0]["request"]["prompt"], [100] * 4)

    def test_rejects_context_overflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds"):
            build_manifest(prompt_tokens=6, output_tokens=3, max_model_len=8)


class TestOnlineStrategyTiming(unittest.TestCase):
    def test_natural_anchor_keeps_exact_prompt_and_eos(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            pinned = root / "pinned.json"
            pinned.write_text(json.dumps({
                "model": "bridgetp-model", "prompt": [11, 12, 13],
                "max_tokens": 4096, "ignore_eos": False,
            }), encoding="utf-8")
            args = Namespace(anchor_request_file=pinned,
                             anchor_prompt_tokens=3, anchor_max_tokens=4096)
            saved = json.loads(make_source_request(args, root).read_text())
            self.assertEqual(saved["prompt"], [11, 12, 13])
            self.assertIs(saved["ignore_eos"], False)
            args.anchor_prompt_tokens = 4
            with self.assertRaisesRegex(ValueError, "pinned anchor"):
                make_source_request(args, root)

    def test_cross_context_smoke_allows_forced_length_only_when_explicit(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            pinned = root / "pinned.json"
            pinned.write_text(json.dumps({
                "model": "bridgetp-model", "prompt": [11, 12, 13],
                "max_tokens": 4, "ignore_eos": True,
            }), encoding="utf-8")
            args = Namespace(anchor_request_file=pinned,
                             anchor_prompt_tokens=3, anchor_max_tokens=4)
            with self.assertRaisesRegex(ValueError, "EOS rule"):
                make_source_request(args, root)
            args.cross_context_smoke = True
            saved = json.loads(make_source_request(args, root).read_text())
            self.assertIs(saved["ignore_eos"], True)
            self.assertEqual(saved["prompt"], [11, 12, 13])

    def test_paired_stay_requires_full_tp1_output_and_suppressed_start(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            controller, background = root / "controller", root / "background"
            controller.mkdir()
            background.mkdir()
            (controller / "source_response.json").write_text(json.dumps({
                "token_ids": [10, 11, 12], "finish_reason": "length",
            }), encoding="utf-8")
            (controller / "response_proxy_stats.json").write_text(json.dumps({
                "emitted_tokens": 3, "source_origin_tokens": 3,
                "target_origin_tokens": 0, "committed": False,
            }), encoding="utf-8")
            (background / "background_summary.json").write_text(json.dumps({
                "jobs": 1, "completed": 1, "failed": 0,
            }), encoding="utf-8")
            rows = [
                {"kind": "manager_m1_start_decision", "decision": {
                    "action": "START_SHADOW", "reason": "source risk",
                }},
                {"kind": "paired_stay_intervention", "action": "STAY"},
                {"kind": "transition", "to": "COMPLETED_ON_TP1"},
                {"kind": "run_end", "final_state": "COMPLETED_ON_TP1"},
            ]
            audit = controller / "phase9_audit.jsonl"
            audit.write_text("\n".join(map(json.dumps, rows)), encoding="utf-8")
            self.assertEqual(
                accept_paired_stay(controller, background, 1, 3)["status"],
                "PASS",
            )
            audit.write_text(
                "\n".join(map(json.dumps, rows[:1] + rows[2:])),
                encoding="utf-8",
            )
            self.assertEqual(
                accept_paired_stay(controller, background, 1, 3)["status"],
                "FAIL",
            )

    def test_paired_stay_accepts_natural_eos_before_cap(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            controller, background = root / "controller", root / "background"
            controller.mkdir()
            background.mkdir()
            (controller / "source_response.json").write_text(json.dumps({
                "token_ids": [10, 11, 12], "finish_reason": "stop",
            }), encoding="utf-8")
            (controller / "response_proxy_stats.json").write_text(json.dumps({
                "emitted_tokens": 3, "source_origin_tokens": 3,
                "target_origin_tokens": 0, "committed": False,
                "finished_reason": "",
            }), encoding="utf-8")
            (background / "background_summary.json").write_text(json.dumps({
                "jobs": 1, "completed": 1, "failed": 0,
            }), encoding="utf-8")
            rows = [
                {"kind": "manager_m1_start_decision", "decision": {
                    "action": "START_SHADOW", "reason": "source risk",
                }},
                {"kind": "paired_stay_intervention", "action": "STAY"},
                {"kind": "transition", "to": "COMPLETED_ON_TP1"},
                {"kind": "run_end", "final_state": "COMPLETED_ON_TP1"},
            ]
            (controller / "phase9_audit.jsonl").write_text(
                "\n".join(map(json.dumps, rows)), encoding="utf-8",
            )
            self.assertEqual(
                accept_paired_stay(
                    controller, background, 1, 4096,
                    natural_eos_anchor=True,
                )["status"], "PASS",
            )
            self.assertEqual(
                accept_paired_stay(controller, background, 1, 4096)["status"],
                "FAIL",
            )

    def test_source_readiness_requires_active_source_tokens(self) -> None:
        with TemporaryDirectory() as temp:
            event_path = Path(temp) / "events.jsonl"
            process = MagicMock()
            process.process.poll.return_value = None
            process.log_path = Path(temp) / "background.log"
            rows = [
                {"kind": "job_first_token", "job_id": "target-1", "pool": "target"},
                {"kind": "job_first_token", "job_id": "source-1", "pool": "source"},
                {"kind": "job_end", "job_id": "source-1", "pool": "source"},
            ]
            event_path.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(TimeoutError, "remain active"):
                wait_for_background_first_tokens(
                    event_path, process, 1, 0.01,
                    pool="source", require_active=True,
                )
            rows.append({
                "kind": "job_first_token", "job_id": "source-2", "pool": "source",
            })
            event_path.write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            wait_for_background_first_tokens(
                event_path, process, 1, 0.01,
                pool="source", require_active=True,
            )

    def test_measured_source_high_rejects_forced_or_post_guard_high(self) -> None:
        decision = {
            "action": "SET_RATE", "profile": "HIGH",
            "reason": "source guard horizon is short",
            "source_time_to_guard_s": 8.0,
        }
        snapshot = {
            "source_free_kv_tokens": 9000,
            "source_guard_free_kv_tokens": 8448,
            "source_running": 3,
        }
        audit = [{
            "kind": "manager_m2_initial_rate",
            "unix_s": 100.0,
            "decision": decision,
            "snapshot": snapshot,
        }]
        peers = [
            {"request_started_unix_s": 90.0, "request_ended_unix_s": 110.0},
            {"request_started_unix_s": 91.0, "request_ended_unix_s": 111.0},
        ]
        self.assertTrue(has_measured_source_high(audit, peers))
        decision["reason"] = "diagnostic HIGH transfer smoke"
        self.assertFalse(has_measured_source_high(audit, peers))
        decision["reason"] = "source guard horizon is short"
        snapshot["source_free_kv_tokens"] = 8000
        self.assertFalse(has_measured_source_high(audit, peers))
        snapshot["source_free_kv_tokens"] = 9000
        peers[0]["request_ended_unix_s"] = 99.0
        peers[1]["request_ended_unix_s"] = 99.0
        self.assertFalse(has_measured_source_high(audit, peers))

    def test_measured_source_high_accepts_known_prefill_reservation(self) -> None:
        audit = [{
            "kind": "rate", "unix_s": 100.0,
            "manager_m2_decision": {
                "action": "SET_RATE", "profile": "HIGH",
                "reason": "source guard horizon is short",
                "source_time_to_guard_s": 0.0,
                "source_capacity_model": (
                    "prefill_reservation_plus_decode_growth"
                ),
            },
            "manager_m2_snapshot": {
                "source_free_kv_tokens": 12000,
                "source_guard_free_kv_tokens": 8448,
                "source_prefill_pending_kv_tokens": 4000,
                "source_running": 3,
            },
        }]
        peers = [
            {"request_started_unix_s": 90.0, "request_ended_unix_s": 110.0},
            {"request_started_unix_s": 91.0, "request_ended_unix_s": 111.0},
        ]
        self.assertTrue(has_measured_source_high(audit, peers))

    def test_m1_short_request_stay_acceptance(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            controller = root / "controller"
            background = root / "background"
            controller.mkdir()
            background.mkdir()
            (controller / "source_response.json").write_text(
                json.dumps({"token_ids": list(range(96)), "finish_reason": "length"}),
                encoding="utf-8",
            )
            (background / "background_summary.json").write_text(
                json.dumps({"jobs": 2, "completed": 2, "failed": 0}),
                encoding="utf-8",
            )
            rows = [
                {"kind": "manager_m1_start_decision", "decision": {
                    "action": "STAY", "reason": "insufficient target output budget"}},
                {"kind": "transition", "to": "COMPLETED_ON_TP1"},
                {"kind": "run_end", "final_state": "COMPLETED_ON_TP1",
                 "trigger_path": None},
            ]
            (controller / "phase9_audit.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            result = accept_m1_stay(controller, background, 2, 96)
            self.assertEqual(result["status"], "PASS")
            rows[0]["decision"]["action"] = "START_SHADOW"
            (controller / "phase9_audit.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            self.assertEqual(
                accept_m1_stay(controller, background, 2, 96)["status"], "FAIL"
            )
            rows[0]["decision"] = {
                "action": "STAY", "reason": "target load exceeds admission guard"
            }
            rows[0]["snapshot"] = {
                "target_waiting": 8,
                "source_free_kv_tokens": 29000,
                "source_guard_free_kv_tokens": 8448,
            }
            (controller / "phase9_audit.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            result = accept_m1_stay(
                controller, background, 2, 96, "target-load"
            )
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["peak_target_waiting"], 8)
            rows[0]["snapshot"]["target_waiting"] = 2
            (controller / "phase9_audit.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
            )
            self.assertEqual(
                accept_m1_stay(
                    controller, background, 2, 96, "target-load"
                )["status"],
                "FAIL",
            )

    def test_controller_completion_accepts_m1_only_when_selected(self) -> None:
        m1_end = [{"final_state": "TAKEOVER", "trigger_path": "MANAGER_M1_START"}]
        fixed_end = [
            {"final_state": "TAKEOVER", "trigger_path": "DIAGNOSTIC_FIXED_BOUNDARY"}
        ]
        self.assertEqual(controller_completion_errors(m1_end, True), [])
        self.assertEqual(controller_completion_errors(fixed_end, False), [])
        self.assertTrue(controller_completion_errors(m1_end, False))
        self.assertTrue(controller_completion_errors(fixed_end, True))

    def test_controller_window_tracks_cli_boundaries(self) -> None:
        overrides = build_controller_config_overrides(
            trigger_output_tokens=64,
            cutover_output_tokens=160,
            fixed_rate_gib_s=0.4,
        )
        self.assertEqual(overrides["handoff_output_tokens"], 96)
        expected_rate = 0.4 * 1024**3
        self.assertEqual(overrides["rate"]["b_min_bytes_s"], expected_rate)
        self.assertEqual(overrides["rate"]["b_max_bytes_s"], expected_rate)

    def test_s_new_requires_history_at_bridge(self) -> None:
        self.assertFalse(
            validate_strategy_timing(
                "S_NEW",
                shadow_start_unix_s=10.0,
                bridge_start_unix_s=12.0,
                history_start_unix_s=12.0,
            )
        )
        self.assertTrue(
            validate_strategy_timing(
                "S_NEW",
                shadow_start_unix_s=10.0,
                bridge_start_unix_s=12.0,
                history_start_unix_s=10.5,
            )
        )

    def test_s_new_old_requires_history_at_shadow(self) -> None:
        self.assertFalse(
            validate_strategy_timing(
                "S_NEW_OLD",
                shadow_start_unix_s=10.0,
                bridge_start_unix_s=12.0,
                history_start_unix_s=10.0,
            )
        )


class TestOnlineWindows(unittest.TestCase):
    def test_natural_eos_before_migration_has_no_shadow_measurements(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_measurements(root, [{
                "repetition": 1,
                "strategy": "S_NEW_OLD",
                "acceptance": {
                    "status": "PASS",
                    "outcome": "NATURAL_EOS_BEFORE_MIGRATION",
                    "source_origin_tokens": 943,
                    "target_origin_tokens": 0,
                },
            }])
            self.assertFalse((root / "measurements.csv").exists())

    def test_visible_interval_summary_preserves_maximum_stall(self) -> None:
        emitted = [
            {"origin": "source", "unix_s": 1.0},
            {"origin": "source", "unix_s": 1.01},
            {"origin": "source", "unix_s": 1.51},
            {"origin": "target", "unix_s": 1.61},
        ]
        source = summarize_emitted_intervals(emitted, origin="source")
        self.assertEqual(source["samples"], 2)
        self.assertAlmostEqual(source["max_ms"], 500.0)
        self.assertAlmostEqual(
            emitted_boundary_gap_ms(
                emitted,
                origin="source",
                output_tokens=3,
            ),
            500.0,
        )
        self.assertIsNone(
            emitted_boundary_gap_ms(
                emitted,
                origin="source",
                output_tokens=4,
            )
        )

    def test_partitions_target_tpot(self) -> None:
        results = [
            {
                "job_id": "target_000",
                "pool": "target",
                "status": "COMPLETED",
                "token_times_unix_s": [9.0, 9.5, 10.5, 11.0, 11.5,
                                        12.5, 12.8, 13.5, 14.0],
            }
        ]
        windows = summarize_background_windows(
            results,
            shadow_start_unix_s=10.0,
            bridge_start_unix_s=12.0,
            committed_unix_s=13.0,
        )
        self.assertEqual(windows["PRE_SHADOW"]["samples"], 1)
        self.assertEqual(windows["SHADOW"]["samples"], 2)
        self.assertEqual(windows["BRIDGE"]["samples"], 1)
        self.assertEqual(windows["POST_COMMIT"]["samples"], 1)
        self.assertEqual(percentile([1.0, 3.0], 0.5), 2.0)

    def test_writes_shadow_only_architecture_pair(self) -> None:
        windows = {
            name: {
                "jobs": 1,
                "samples": 1,
                "tpot_p50_ms": 1.0,
                "tpot_p95_ms": 1.0,
                "tpot_p99_ms": 1.0,
            }
            for name in ("PRE_SHADOW", "SHADOW", "BRIDGE", "POST_COMMIT")
        }
        runs = []
        for architecture, strategy, stall in (
            ("BRIDGE", "S_NEW", 20.0),
            ("SHADOW_ONLY", "S_NEW_OLD", 10.0),
        ):
            architecture_windows = dict(windows)
            if architecture == "SHADOW_ONLY":
                architecture_windows["FINAL_SYNC"] = dict(windows["BRIDGE"])
            runs.append(
                {
                    "repetition": 1,
                    "architecture": architecture,
                    "strategy": strategy,
                    "acceptance": {
                        "status": "PASS",
                        "shadow_duration_ms": 100.0,
                        "bridge_to_commit_ms": stall,
                        "handoff_stall_ms": stall,
                        "source_origin_tokens": 10,
                        "target_origin_tokens": 20,
                        "target_tpot_windows": architecture_windows,
                    },
                }
            )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_measurements(root, runs)
            paired = (root / "paired_comparisons.csv").read_text(
                encoding="utf-8"
            )
            measurements = (root / "measurements.csv").read_text(
                encoding="utf-8"
            )
        self.assertIn("final_sync_ms_saved_by_shadow_only", paired)
        self.assertIn("10.0", paired)
        self.assertIn("final_sync_tpot_p99_ms", measurements)

    def test_slo_summary_counts_token_and_request_violations(self) -> None:
        summary = summarize_slo(
            [
                {
                    "status": "COMPLETED",
                    "token_times_unix_s": [1.0, 1.01, 1.08],
                    "tpot_p99_ms": 70.0,
                    "ttft_ms": 1200.0,
                    "e2e_ms": 2000.0,
                }
            ],
            tpot_ms=50.0,
            ttft_ms=1000.0,
            e2e_ms=3000.0,
        )
        self.assertEqual(summary["tpot_interval_violations"], 1)
        self.assertEqual(summary["request_p99_tpot_violations"], 1)
        self.assertEqual(summary["ttft_violations"], 1)
        self.assertEqual(summary["e2e_violations"], 0)

    def test_one_token_warmup_has_no_tpot_violation(self) -> None:
        summary = summarize_slo(
            [
                {
                    "status": "COMPLETED",
                    "token_times_unix_s": [1.0],
                    "tpot_p99_ms": None,
                    "ttft_ms": 10.0,
                    "e2e_ms": 12.0,
                }
            ],
            tpot_ms=50.0,
            ttft_ms=1000.0,
            e2e_ms=3000.0,
        )
        self.assertEqual(summary["token_intervals"], 0)
        self.assertEqual(summary["request_p99_tpot_violations"], 0)


class TestShadowRateLoadMatrix(unittest.TestCase):
    def test_default_formal_design_covers_three_loads_and_five_rates(self) -> None:
        args = Namespace(
            phase="formal",
            load_profile=None,
            rates_gib_s=None,
            repetitions=None,
            minimum_window_samples=None,
        )
        loads, rates, repetitions, minimum_samples = resolve_design(args)
        self.assertEqual(loads, [("low", 2), ("medium", 8), ("high", 24)])
        self.assertEqual(rates, [0.2, 0.4, 0.8, 1.2, 0.0])
        self.assertEqual(repetitions, 4)
        self.assertEqual(minimum_samples, 128)

    def test_load_profiles_and_rate_labels_are_unambiguous(self) -> None:
        self.assertEqual(
            parse_load_profiles(["quiet:3", "busy:20"], "smoke"),
            [("quiet", 3), ("busy", 20)],
        )
        self.assertEqual(rate_label(0.4), "0p4gibs")
        self.assertEqual(rate_label(0.0), "unlimited")

    def test_rejects_duplicate_load_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_load_profiles(["busy:4", "busy:8"], "formal")


if __name__ == "__main__":
    unittest.main()
