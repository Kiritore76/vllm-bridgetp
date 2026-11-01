# SPDX-License-Identifier: Apache-2.0
"""Four-card static layout screen: runner switch and plans (CPU)."""

import contextlib
import io
import json
import unittest

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
