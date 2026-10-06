"""CPU contract tests for F1/F2, independent of CUDA and predictor training."""

import itertools
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bridge_tp.experiment_probability_gate import ProbabilityGate
from tools.bridge_tp.horizon_goodoutput import score_horizon
from tools.bridge_tp.risk_urgency import build_snapshot, remaining_ge_bounds
from tools.bridge_tp.run_randomized_goodoutput_pilot_a100 import (
    build_setup,
    configure_action,
)


def prediction(position=0, generated=0, cap=100):
    return {
        "status": "AVAILABLE",
        "probabilities": [0.1, 0.3, 0.6],
        "category_upper_edges": [0, 4],
        "output_tokens": generated,
        "prediction_output_tokens": position,
        "max_remaining_output_tokens": cap,
        "ignore_eos": False,
    }


def snapshot(growth=10, pending=0):
    return {
        "state": "LOCAL",
        "unix_s": 10,
        "source_sampled_unix_s": 10,
        "target_sampled_unix_s": 10,
        "source_free_kv_tokens": 1100,
        "source_guard_free_kv_tokens": 100,
        "source_prefill_pending_kv_tokens": pending,
        "source_decode_growth_tokens_s": growth,
        "generated_tokens": 0,
        "current_context_tokens": 20,
        "target_free_kv_tokens": 10000,
        "target_prefill_pending_kv_tokens": 0,
        "target_kv_usage_frac": 0.1,
        "channel_available": True,
    }


def risk(growth=10, speed=10, pending=0, pred=None):
    return build_snapshot(
        snapshot=snapshot(growth, pending),
        prediction=pred or prediction(cap=2000),
        candidate_rate=speed,
        initial_rate={"rate_bytes_s": 1000000, "profile": "LOW"},
        kv_bytes_per_token=100,
        release_tail_s=5,
        block_size=1,
    ).to_json()


class TestProbabilityCollection(unittest.TestCase):
    def test_partial_service_error_and_input_error_have_separate_receipts(self):
        import urllib.error
        from io import BytesIO

        from vllm.bridge_tp.controller.online_io import post_streaming_completion

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.json"
            truncated = BytesIO(b'data: {"choices":[{"token_ids":[42]}]}\n')
            with (
                patch("urllib.request.urlopen", return_value=truncated),
                self.assertRaisesRegex(RuntimeError, "ended before"),
            ):
                post_streaming_completion(
                    "http://example", {}, 1, lambda *args: None, failure_path=path
                )
            receipt = json.loads(path.read_text())
            self.assertEqual(receipt["token_ids"], [42])
            self.assertEqual(receipt["failure_kind"], "SERVICE_REQUEST_FAILURE")
            self.assertGreaterEqual(
                receipt["completed_unix_s"], receipt["request_started_unix_s"]
            )
            error = urllib.error.HTTPError(
                "http://example", 400, "bad input", {}, BytesIO(b"invalid budget")
            )
            with (
                patch("urllib.request.urlopen", side_effect=error),
                self.assertRaisesRegex(RuntimeError, "HTTP 400"),
            ):
                post_streaming_completion(
                    "http://example", {}, 1, lambda *args: None, failure_path=path
                )
            self.assertEqual(
                json.loads(path.read_text())["failure_kind"], "TECHNICAL_INPUT_FAILURE"
            )

    def test_one_effect_sample_per_episode_and_safety_not_randomized(self):
        from tools.bridge_tp.run_randomized_goodoutput_pilot_a100 import (
            probability_results,
        )

        candidate = {"snapshot": {"generated_tokens": 9, "H_tokens": 1000}}
        observed = {
            "probability_candidate": candidate,
            "actual_probability_executions": [{"actual_action": "START_SHADOW"}],
        }
        start = {"fixed_horizon_goodoutput_tokens_s": 10, "observed_action": observed}
        stay = {
            "fixed_horizon_goodoutput_tokens_s": 8,
            "observed_action": {"probability_candidate": candidate},
        }
        summary = {
            "seed": 1,
            "probability_thresholds": [0.1],
            "cases": {"load": {"prob_0.1_START": start, "prob_0.1_STAY": stay}},
        }
        result = probability_results(summary)
        self.assertEqual(len(result["samples"]), 1)
        self.assertEqual(result["samples"][0]["descriptive_delta_tokens_s"], 2)
        observed["safety_override_ticks"] = [12]
        self.assertFalse(
            probability_results(summary)["samples"][0]["effect_sample_eligible"]
        )

    def test_bucket_bounds_contain_all_discrete_support_allocations(self):
        # Enumerate actual distributions: the middle bucket [1,4] and
        # tail [5,infinity] each put mass at possible points. No reference
        # calculation copies the production interval algorithm.
        for lag, n in itertools.product(range(5), range(1, 9)):
            p = prediction(generated=lag)
            bounds = remaining_ge_bounds(p, n)
            for middle, tail in itertools.product(range(1, 5), [5, 6, 8, 1000]):
                points = [(0, 0.1), (middle, 0.3), (tail, 0.6)]
                survived = sum(m for r, m in points if r >= lag)
                exact = (
                    sum(m for r, m in points if r >= lag and r - lag >= n) / survived
                )
                self.assertLessEqual(bounds[0], exact + 1e-12)
                self.assertGreaterEqual(bounds[1], exact - 1e-12)

    def test_tail_integer_boundary_stop_cap_and_survival(self):
        p = prediction(generated=4)
        self.assertAlmostEqual(remaining_ge_bounds(p, 1)[0], 2 / 3)
        self.assertEqual(remaining_ge_bounds(p, 1)[1], 1)
        self.assertEqual(remaining_ge_bounds(p, 100.1), [0, 0])
        p["ignore_eos"] = True
        self.assertIsNone(remaining_ge_bounds(p, 1))
        p["ignore_eos"] = False
        p["status"] = "STALE"
        self.assertIsNone(remaining_ge_bounds(p, 1))

    def test_single_request_reduction_and_pool_conversion(self):
        single = risk()
        self.assertEqual(single["H_tokens"], 1000)
        self.assertEqual(single["r_guard_tokens"], 1000)
        self.assertEqual(single["p_guard_est_bounds"], single["p_capacity_bounds"])
        parallel = risk(growth=100)
        self.assertEqual(parallel["r_guard_tokens"], 100)
        self.assertEqual(parallel["T_guard_s"], 10)
        self.assertGreater(parallel["U"], single["U"])
        self.assertEqual(risk(pending=128)["H_tokens"], 872)

    def test_no_growth_protection_stale_and_delta_catchup(self):
        for growth in (0, -1, None, math.nan):
            row = risk(growth=growth)
            self.assertIsNone(row["T_guard_s"])
            self.assertIsNone(row["p_guard_est_bounds"])
            self.assertFalse(row["physical_feasible"])
            json.dumps(row, allow_nan=False)
        self.assertEqual(risk(pending=1000)["status"], "PROTECTION_BAND")
        self.assertIn("delta_cannot_catch_up", risk(speed=10000)["physical_rejections"])
        row = snapshot()
        row["target_sampled_unix_s"] = 1
        result = build_snapshot(
            snapshot=row,
            prediction=prediction(),
            candidate_rate=10,
            initial_rate={"rate_bytes_s": 1e6},
            kv_bytes_per_token=100,
            release_tail_s=5,
        ).to_json()
        self.assertFalse(result["physical_feasible"])

    def test_first_crossing_common_feasibility_stay_and_simultaneous(self):
        gate = ProbabilityGate(
            0.5, "seed", assigned_action="STAY", thresholds=(0, 0.2, 0.5)
        )
        s = {
            "status": "VALID",
            "p_guard_est_bounds": [0.6, 0.8],
            "physical_feasible": False,
        }
        first = gate.observe(s, 1)
        self.assertTrue(first["simultaneous_crossing"])
        self.assertFalse(first["first_feasible_candidate"])
        s["physical_feasible"] = True
        second = gate.observe(s, 2)
        self.assertTrue(second["first_feasible_candidate"])
        self.assertEqual(second["requested_action"], "STAY")
        self.assertFalse(gate.observe(s, 3)["first_feasible_candidate"])
        self.assertEqual(gate.candidate["tick"], 2)

    def test_horizon_zeros_late_and_service_failed_without_excluding_window(self):
        rows = [
            dict(request_id=x, pool="source", status=status, slo_success=True)
            for x, status in [
                ("on_time", "COMPLETED"),
                ("late", "COMPLETED"),
                ("failed", "FAILED"),
            ]
        ]
        rows.append(
            dict(
                request_id="anchor", pool="anchor", status="COMPLETED", slo_success=True
            )
        )
        background = {
            "results": [
                dict(
                    job_id=x,
                    request_started_unix_s=0,
                    request_ended_unix_s=end,
                    token_times_unix_s=[1, 2],
                )
                for x, end in [("on_time", 2), ("late", 11), ("failed", 3)]
            ]
        }
        source = {
            "request_started_unix_s": 0,
            "completed_unix_s": 5,
            "finish_reason": "stop",
        }
        proxy = {"emitted": [{"unix_s": 1}], "external_request_id": "anchor"}
        result = score_horizon(
            {"computable": True, "request_rows": rows},
            background,
            source,
            {},
            proxy,
            10,
        )
        self.assertTrue(result["eligible"])
        self.assertEqual(result["good_output_tokens"], 3)
        self.assertEqual(result["unfinished_or_failed_at_H"], 2)
        del background["results"][0]["token_times_unix_s"]
        self.assertFalse(
            score_horizon(
                {"computable": True, "request_rows": rows},
                background,
                source,
                {},
                proxy,
                10,
            )["eligible"]
        )

    def test_probability_command_has_no_fixed_token_trigger_gate(self):
        command = [
            "--manager-m2-force-initial-high",
            "--manager-m2-expected-profile",
            "HIGH",
            "--manager-m2-min-history-byte-frac",
            ".9",
            "--m1-min-output-tokens",
            "128",
        ]
        configure_action(command, "prob_0.01_STAY")
        self.assertNotIn("--experiment-m1-action", command)
        self.assertEqual(command[command.index("--m1-min-output-tokens") + 1], "0")
        self.assertIn("--probability-threshold", command)

    def test_rotated_unique_prompts_actual_context_and_idle_target(self):
        tokenizer = SimpleNamespace(encode=lambda s: [1] * len(s))
        fake = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "input.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": str(i),
                            "source_tree_id": str(i),
                            "split": "test",
                            "workload_group": "long_form" if i % 4 == 0 else "natural",
                            "prompt": "hi" + str(i),
                        }
                    )
                    + "\n"
                    for i in range(350)
                )
            )
            args = SimpleNamespace(
                input=path,
                model=root,
                probability_pilot=True,
                cases=["p00_source1_target2"],
                max_model_len=16384,
                tp4_max_model_len=32768,
                anchor_context_limit=True,
                background_context_limit=True,
                arrival_window_s=30,
                arrival_wave_period_s=10,
            )
            with patch.dict(sys.modules, {"transformers": fake}):
                setup = build_setup(args, root)
            manifest = json.loads(
                Path(setup["manifests"][args.cases[0]]["path"]).read_text()
            )
            jobs = manifest["jobs"]
            self.assertEqual(len(jobs), len({j["input_id"] for j in jobs}))
            self.assertEqual(manifest["target_count"], 0)
            self.assertTrue(all(j["pool"] == "source" for j in jobs))
            self.assertTrue(all(j["request"]["max_tokens"] > 8192 for j in jobs))


if __name__ == "__main__":
    unittest.main()
