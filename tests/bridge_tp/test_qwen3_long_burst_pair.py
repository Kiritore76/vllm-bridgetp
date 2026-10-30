# SPDX-License-Identifier: Apache-2.0
"""Long-I/O paired workload and real native-boundary hook, without GPUs."""

import contextlib
import copy
import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.bridge_tp import qwen3_long_burst_pair as burst
from tools.bridge_tp import qwen3_stay_migrate as matched
from tools.bridge_tp import run_atomic_takeover as takeover
from tools.bridge_tp import run_qwen3_stay_migrate as runner


class LongBurstPairTests(unittest.TestCase):
    def setUp(self):
        self.inputs = dict(
            requests={
                str(n): dict(model="bridgetp-model", prompt=list(range(n)))
                for n in (2048, 6144)
            }
        )

    def test_four_runs_frozen_long_work_no_forced_route_and_short_background(self):
        original = copy.deepcopy(self.inputs)
        cfg = burst.config(self.inputs)
        self.assertEqual(self.inputs, original)
        self.assertEqual(cfg, burst.config(self.inputs))
        self.assertEqual(cfg["arms"], ["stay", "migrate"])
        self.assertEqual(len(cfg["recipes"]) * len(cfg["arms"]), 4)
        self.assertEqual([len(r["jobs"]) for r in cfg["recipes"]], [158, 156])
        self.assertEqual(
            [r["planned_output_tokens"] for r in cfg["recipes"]], [60160, 68352]
        )
        for r in cfg["recipes"]:
            self.assertEqual(r["input_sha256"], matched.digest(r["jobs"]))
            self.assertEqual(len({j["id"] for j in r["jobs"]}), len(r["jobs"]))
            self.assertEqual(
                sum(j["workload_role"] == "short_probe" for j in r["jobs"]), 150
            )
            for j in r["jobs"]:
                self.assertIsNone(j["route"])
                self.assertLess(j["at_s"], 300)
                self.assertLessEqual(
                    len(j["request"]["prompt"]) + j["request"]["max_tokens"], 32768
                )
                self.assertLessEqual(j["request"]["max_tokens"], 8192)
                self.assertTrue(j["request"]["ignore_eos"])

    def test_token_expansion_preserves_boundaries_and_rejects_overlimit(self):
        r = burst.request(self.inputs, 32767, 1)
        self.assertEqual(len(r["prompt"]), 32767)
        self.assertEqual(r["prompt"][:32], list(range(32)))
        self.assertEqual(r["prompt"][-32:], list(range(6144 - 32, 6144)))
        with self.assertRaises(ValueError):
            burst.request(self.inputs, 32768, 1)
        with self.assertRaises(ValueError):
            burst.request(self.inputs, 8192, 8193)

    def test_actual_cli_is_paired_and_automatically_uses_both_32k(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = runner.main(
                [
                    "--long-burst-pair",
                    "--current-machine",
                    "--repetitions",
                    "1",
                    "--model",
                    "unused",
                    "--output",
                    "unused",
                    "--expected-revision",
                    "CPU",
                    "--validate-only",
                ]
            )
        record = json.loads(out.getvalue())
        self.assertEqual(code, 0)
        self.assertEqual(record["runs"], 4)
        self.assertEqual(record["arms"], ["stay", "migrate"])
        self.assertEqual(record["service_context_limits"], dict(tp1=32768, tp2=32768))
        self.assertTrue(record["policy_overrides"]["kv_routing_priority"])

    def test_boundary_hook_executes_both_pools_then_rechecks_empty_inventory(self):
        plan = dict(
            run_dir=Path("unused/controller"),
            config=dict(source_url="source", target_url="target"),
        )
        result = dict(usage=dict(prompt_tokens=32767, completion_tokens=1))
        common = Mock()
        with (
            patch.object(runner, "prepare_native") as native,
            patch.object(takeover, "_post_json", return_value=result) as post,
        ):
            burst.prepare_native(plan, common, self.inputs)
        self.assertEqual(native.call_count, 2)
        self.assertEqual(post.call_count, 2)
        for c in post.call_args_list:
            self.assertEqual(c.args[1], "/v1/completions")
            self.assertEqual(len(c.args[2]["prompt"]), 32767)
            self.assertFalse(c.args[2]["stream"])
            self.assertEqual(c.args[2]["max_tokens"], 1)
        self.assertEqual(common.write_json.call_args.args[1]["status"], "PASSED")
        with (
            patch.object(runner, "prepare_native"),
            patch.object(
                takeover,
                "_post_json",
                return_value=dict(usage=dict(prompt_tokens=10, completion_tokens=1)),
            ),
            self.assertRaises(ValueError),
        ):
            burst.prepare_native(plan, Mock(), self.inputs)
