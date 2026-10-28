# SPDX-License-Identifier: Apache-2.0
"""Finite62 online PMF, survival alignment, and backward compatibility."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from tools.bridge_tp.pressure_cpu import load
from tools.bridge_tp.qwen3_online_validation import probe_requests

b = load("bounded_length")
m5 = load("manager_m5")


class FiniteOnlineTests(unittest.TestCase):
    def test_weighted_midpoints_without_subtracting_generated_again(self):
        x = b.bounded_length(
            (0.2, 0.3, 0.5),
            (0, 8, 16),
            generated=100,
            predicted_at=100,
            max_output_tokens=16384,
            overflow_category=False,
        )
        self.assertAlmostEqual(x.mean, 0.3 * 4.5 + 0.5 * 12.5)
        self.assertEqual(x.probability_gt_bounds(16), (0, 0))

    def test_lag_condition_shift_and_stop_crossing(self):
        x = b.bounded_length(
            (0, 1),
            (0, 8),
            generated=6,
            predicted_at=3,
            max_output_tokens=9,
            overflow_category=False,
        )
        # R=3..8 survives uniformly, shifts to0..5, then clips to3.
        self.assertAlmostEqual(x.mean, (0 + 1 + 2 + 3 + 3 + 3) / 6)
        for h in range(4):
            lo, hi = x.probability_gt_bounds(h)
            self.assertLessEqual(lo, x.probability_gt(h) + 1e-12)
            self.assertLessEqual(x.probability_gt(h), hi + 1e-12)

    def test_finite_geometry_keeps_frozen_total_output_stop_contract(self):
        x = b.bounded_length(
            (0, 0, 1),
            (0, 4096, 8192),
            generated=8000,
            predicted_at=8000,
            max_output_tokens=16000,
            overflow_category=False,
        )
        self.assertEqual(x.mean, 192)
        self.assertEqual(x.cap, 192)
        with self.assertRaises(ValueError):
            b.bounded_length(
                (0, 0, 1),
                (0, 8),
                generated=0,
                predicted_at=0,
                max_output_tokens=20,
                overflow_category=False,
            )

    def test_reader_finite_header_and_staleness(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            now = time.time_ns()
            sha = "a" * 64
            header = dict(
                kind="predictor_header",
                format_version=1,
                checkpoint_sha256=sha,
                feature_layer="decoder:31",
                interval=20,
                category_upper_edges=[0, 8, 16],
                overflow_category=False,
                support_scope="natural_eos_within_finite_support",
            )
            event = dict(
                kind="predictor_prediction",
                request_id="r",
                generated_tokens=20,
                checkpoint_sha256=sha,
                captured_unix_ns=now,
                probabilities=[0.2, 0.3, 0.5],
            )
            path.write_text(json.dumps(header) + "\n" + json.dumps(event) + "\n")
            reader = m5.PredictorEventReader(path, sha)
            row = reader.advisory("r", 20, 32, max_output_tokens=100, now_ns=now)
            self.assertEqual(row["status"], "AVAILABLE")
            self.assertFalse(row["overflow_category"])
            self.assertEqual(row["p_remaining_gt_headroom_bounds"], (0, 0))
            self.assertAlmostEqual(b.from_advisory(row, 100).mean, 7.6)
            self.assertEqual(
                reader.advisory("r", 61, 32, now_ns=now)["status"], "STALE"
            )
            self.assertEqual(
                reader.advisory("r", 20, 32, now_ns=now + 31_000_000_000)["status"],
                "STALE",
            )

    def test_finite_header_requires_conditional_scope(self):
        reader = m5.PredictorEventReader(Path("unused"), "a" * 64)
        with self.assertRaises(ValueError):
            reader._read_header(
                dict(
                    kind="predictor_header",
                    format_version=1,
                    checkpoint_sha256="a" * 64,
                    feature_layer="decoder:31",
                    interval=20,
                    category_upper_edges=[0, 8],
                    overflow_category=False,
                )
            )

    def test_probe_uses_natural_eos_and_non_thinking_template(self):
        class Tokenizer:
            def apply_chat_template(self, *args, **kwargs):
                assert kwargs["enable_thinking"] is False
                return [1] * 100

        jobs = probe_requests(Tokenizer())
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all(not j["payload"]["ignore_eos"] for j in jobs))
        self.assertEqual([j["payload"]["max_tokens"] for j in jobs], [512, 2048, 6144])


class LoadedValidationTests(unittest.TestCase):
    def test_stage_arguments_gate_probe_and_keep_both_loaded(self):
        from types import SimpleNamespace

        from tools.bridge_tp import run_qwen3_online_loaded_validation as runner

        args = SimpleNamespace(
            model=Path("model"),
            inputs=Path("inputs"),
            output=Path("out"),
            checkpoint=Path("checkpoint"),
            expected_revision="rev",
        )
        first, second = runner.stages()
        a = runner.child_arguments(args, first)
        c = runner.child_arguments(args, second)
        self.assertIn("--natural-probe", a)
        self.assertNotIn("--natural-probe", c)
        self.assertEqual(a[a.index("--target-background") + 1], "2")
        self.assertEqual(c[c.index("--target-background") + 1], "4")
        self.assertIn("--predictor-checkpoint", a)
        self.assertIn("--predictor-checkpoint", c)

    def test_failed_live_stage_blocks_second(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        from tools.bridge_tp import run_qwen3_online_loaded_validation as runner

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                model=Path("model"),
                inputs=Path("inputs"),
                output=Path(tmp),
                checkpoint=Path("checkpoint"),
                expected_revision="rev",
            )
            ledger = []
            with (
                patch.object(runner, "gpu_idle", return_value=True),
                patch.object(runner, "migration_main", return_value=1) as child,
                self.assertRaises(RuntimeError),
            ):
                runner.execute(args, ledger)
            self.assertEqual(child.call_count, 1)
            self.assertEqual(ledger[0]["status"], "FAILED")


class SummaryNameTests(unittest.TestCase):
    def test_nd_prefix_scenes_produce_reference_windows(self):
        from unittest.mock import patch

        from tools.bridge_tp import summarize_tp2_calibration as summary

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "selection.json").write_text(json.dumps(dict(target_tp=2)))
            for suffix in (1, 2):
                d = root / f"nd8-r{suffix}"
                c = d / "controller"
                c.mkdir(parents=True)
                (d / "recipe.json").write_text("{}")
                (d / "measurement_window.json").write_text(
                    json.dumps(dict(started_unix_s=0, ended_unix_s=40))
                )
                requests = []
                for role in ("source", "target"):
                    requests.append(dict(id=role, pool=role, tokens=800))
                    (c / (role + ".tokens.jsonl")).write_text(
                        "".join(
                            json.dumps(dict(index=i, unix_s=i * 0.05)) + "\n"
                            for i in range(800)
                        )
                    )
                (d / "request_results.json").write_text(
                    json.dumps(dict(status="COMPLETED", requests=requests))
                )
            batches = [
                dict(
                    kind="engine_batch",
                    scheduled_unix_s=i * 0.1,
                    recompute_tokens=0,
                    speculative=False,
                    decode_requests=8,
                    decode_context_sum=8 * 3072,
                    prefill_tokens=0,
                    completion_latency_ms=10,
                )
                for i in range(400)
            ]
            with (
                patch.object(summary, "verify_pool"),
                patch.object(summary, "read_journal", return_value=batches),
            ):
                result = summary.summarize(root)
            self.assertEqual(len(result["cells"]), 2)
            self.assertTrue(
                all(c["health_status"] == "REFERENCE" for c in result["cells"])
            )


class BackgroundJournalTests(unittest.TestCase):
    def test_background_keeps_client_token_timing(self):
        from tools.bridge_tp.qwen3_online_validation import background_completion

        class IO:
            @staticmethod
            def post_streaming_completion(
                url, payload, timeout, token, *args, **kwargs
            ):
                token(0, 7, 100.1)
                token(1, 8, 100.2)
                return dict(token_ids=[7, 8], finish_reason="length")

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            result = background_completion(IO, "url", {}, 10, d, 0)
            events = [
                json.loads(s)
                for s in (d / "background-0.tokens.jsonl").read_text().splitlines()
            ]
            self.assertEqual(result["token_ids"], [7, 8])
            self.assertEqual([e["index"] for e in events], [0, 1])
            self.assertAlmostEqual(events[1]["unix_s"] - events[0]["unix_s"], 0.1)
