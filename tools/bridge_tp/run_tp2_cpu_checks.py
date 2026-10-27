# SPDX-License-Identifier: Apache-2.0
"""Run pure BridgeTP contracts without executing vLLM's CUDA package startup.

Only package discovery is substituted. The tested modules/functions are actual
repository files. This does not validate torch/CUDA/model execution.
"""

import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

TESTS = [
    "4090d_qwen3",
    "qwen3_recipe_pilot",
    "predictor_capture",
    "predictor_layer_probe",
    "predictor_formal_corpus",
    "predictor_collection_training",
    "predictor_risk76",
    "predictor_mlp4_scratch",
    "tp2_reconfiguration",
    "pressure_session",
    "pressure_sessions",
    "pressure_policy",
    "pressure_workload",
    "pressure_preflight",
    "pressure_launcher",
    "pressure_calibration",
    "pressure_relief",
    "pressure_cancel",
    "pressure_dispatch",
    "pressure_inventory_process",
    "pressure_health_service",
    "pressure_health_reference",
    "pressure_sustained",
    "phase9_state_and_proxy",
    "rolling_cutover",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    temporary_root = args.output / "temp"
    temporary_root.mkdir()
    tempfile.tempdir = str(temporary_root.resolve())
    namespace = types.ModuleType("vllm")
    namespace.__path__ = [str(ROOT / "vllm")]
    sys.modules["vllm"] = namespace
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        ["tests.bridge_tp.test_" + name for name in TESTS]
    )
    with (args.output / "unittest.log").open("w", encoding="utf-8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    receipt = dict(
        tests=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(result.skipped),
        passed=result.wasSuccessful(),
        scope="real CPU modules; vllm package startup bypassed; no GPU acceptance",
        modules=TESTS,
    )
    (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
