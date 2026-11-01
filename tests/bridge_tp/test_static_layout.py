# SPDX-License-Identifier: Apache-2.0
"""Four-card static layout screen: runner switch and plans (CPU)."""

import contextlib
import io
import json
import unittest
from pathlib import Path

from tools.bridge_tp import run_qwen3_stay_migrate as runner

INPUTS = "experiments/platforms/qwen3_pressure_observation/inputs.json"
COMMON = ["--current-machine", "--repetitions", "1", "--model", "m"]
COMMON += ["--checkpoint", "c", "--inputs", INPUTS, "--output", "o"]
COMMON += ["--expected-revision", "CPU", "--validate-only"]


def plan(*args):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = runner.main([*args, *COMMON])
    return code, json.loads(out.getvalue())


class StaticLayoutTests(unittest.TestCase):
    def test_one_arm_per_layout_on_both_workload_families(self):
        for layout, arm in (("L0", "stay"), ("L1A", "migrate"), ("L2", "stay")):
            code, value = plan(
                "--routine-four-mode", "--expanded-context", "--static-layout", layout
            )
            self.assertEqual((code, value["arms"], value["runs"]), (0, [arm], 3))
            self.assertEqual(
                [r["scenario"] for r in value["recipes"]],
                ["routine_mixed", "moderate_decode", "transient_burst"],
            )
            self.assertEqual(
                value["service_context_limits"], dict(tp1=32768, tp2=32768)
            )
            code, value = plan("--long-burst-pair", "--static-layout", layout)
            self.assertEqual((code, value["arms"], value["runs"]), (0, [arm], 1))
            self.assertEqual(
                [r["scenario"] for r in value["recipes"]], ["deep_long_burst"]
            )

    def test_l0_pair_source_starts_as_a_routing_only_engine(self):
        """Static screen (11-01): the L0 source still preconnected to a TP2
        receiver that L0 does not have and failed to start."""
        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        env = {
            "PATH": "/bin",
            "CUDA_VISIBLE_DEVICES": "2",
            "BRIDGETP_PRESSURE_POOL": "tp1",
            "BRIDGETP_PRESSURE_INVENTORY_PATH": "/run/source_inventory.jsonl",
            "BRIDGETP_PERFORMANCE_TRACE_DIR": "/run/trace/source",
            "BRIDGETP_PERFORMANCE_FINALIZE_TOKEN": "t",
            "BRIDGETP_GPU_DIRECT_PRECONNECT": "1",
            "BRIDGETP_STREAM_ENABLED": "1",
            "BRIDGETP_TAKEOVER_ENABLED": "1",
            "BRIDGETP_PREDICTOR_LIVE_EVENTS": "/run/predictor_events.jsonl",
        }
        argv = ["python", "-m", "serve", "--port", "8001", "--kv-transfer-config", "{}"]
        plan = dict(
            source_env=env,
            source_argv=argv,
            instances=dict(legacy_pool_aliases={"tp1": "A-g1-tp1"}),
        )
        out_argv, out_env = launch.source_launch(plan)
        self.assertNotIn("--kv-transfer-config", out_argv)
        self.assertEqual(
            sorted(out_env),
            sorted(
                [
                    "PATH",
                    "CUDA_VISIBLE_DEVICES",
                    "BRIDGETP_PRESSURE_POOL",
                    "BRIDGETP_PRESSURE_INVENTORY_PATH",
                    "BRIDGETP_PERFORMANCE_TRACE_DIR",
                    "BRIDGETP_PERFORMANCE_FINALIZE_TOKEN",
                ]
            ),
        )
        self.assertEqual(plan["source_env"], env)  # the plan itself is untouched
        # With a target (L1A, L2's absent source aside) nothing changes.
        plan["instances"] = dict(legacy_pool_aliases={"tp1": "a", "tp4": "b"})
        self.assertEqual(launch.source_launch(plan), (argv, env))

    def test_rate_sweep_runs_four_steady_short_rates_per_layout(self):
        from tools.bridge_tp import qwen3_rate_sweep as sweep

        code, value = plan(
            "--rate-sweep", "--expanded-context", "--static-layout", "L0"
        )
        self.assertEqual((code, value["runs"], value["arms"]), (0, 4, ["stay"]))
        self.assertEqual([r["rate_req_s"] for r in value["recipes"]], [4, 8, 12, 16])
        cfg = sweep.config(json.loads(Path(INPUTS).read_text(encoding="utf-8")))
        for recipe in cfg["recipes"]:
            jobs = recipe["jobs"]
            rate = recipe["rate_req_s"]
            self.assertEqual(len(jobs), rate * 60)
            self.assertAlmostEqual(jobs[1]["at_s"] - jobs[0]["at_s"], 1 / rate, 5)
            self.assertLess(jobs[-1]["at_s"], 60)
            self.assertEqual(
                {
                    (len(j["request"]["prompt"]), j["request"]["max_tokens"])
                    for j in jobs
                },
                {(512, 128)},
            )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(["--rate-sweep", "--expanded-context", *COMMON])

    def test_rate_sweep_accepts_higher_rates(self):
        code, value = plan(
            "--rate-sweep",
            "--expanded-context",
            "--static-layout",
            "L2",
            "--sweep-rates",
            "12,16,20,24",
        )
        self.assertEqual(code, 0)
        self.assertEqual([r["rate_req_s"] for r in value["recipes"]], [12, 16, 20, 24])
        for bad in (["--sweep-rates", "8,8"], ["--sweep-rates", "0"]):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                plan(
                    "--rate-sweep", "--expanded-context", "--static-layout", "L0", *bad
                )
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            plan("--static-layout", "L0", "--sweep-rates", "12")

    def test_conflicting_options_are_rejected(self):
        for args in (
            ["--long-burst-memory-verify", "--static-layout", "L0"],
            ["--long-burst-pair", "--static-layout", "L2", "--layout-l1a-channels"],
            ["--static-layout", "L1A"],
        ):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                runner.main([*args, *COMMON])


if __name__ == "__main__":
    unittest.main()
