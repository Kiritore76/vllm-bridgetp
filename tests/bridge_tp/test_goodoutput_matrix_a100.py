"""Checks for the exploratory A100 paired matrix driver."""

import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools.bridge_tp.run_goodoutput_matrix_a100 import (
    expected_gpu_uuids,
    guard_contract,
    observed_pressure,
    online_command,
    prepare,
    reference_contract,
    verify,
)


class TestObservedPressure(unittest.TestCase):
    def test_visible_diagnostic_reference_matches_linux_checkout(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "experiments/phase9/slo"
            / "slo_v6_slow2pct_visible_diagnostic_a100_tp4_reference_20261008.json"
        )
        contents = path.read_bytes().replace(b"\r\n", b"\n")
        args = SimpleNamespace(
            slo_profile="slow2pct_visible_diagnostic", probability_pilot=True
        )
        self.assertEqual(hashlib.sha256(contents).hexdigest(), reference_contract(args))
        self.assertEqual(json.loads(contents)["max_visible_interval_policy"],
                         "DIAGNOSTIC_ONLY")
        args.probability_pilot = False
        with self.assertRaisesRegex(ValueError, "explicit probability"):
            reference_contract(args)

    def test_new_slo_changes_only_fraction_and_version_metadata(self):
        directory = Path(__file__).resolve().parents[2] / "experiments/phase9/slo"
        old = json.loads(
            (directory / "slo_v6_a100_tp4_reference_20261004.json").read_text()
        )
        path = directory / "slo_v6_slow2pct_a100_tp4_reference_20261006.json"
        contents = path.read_bytes().replace(b"\r\n", b"\n")
        new = json.loads(contents)
        self.assertEqual(new["primary_max_slow_interval_rate"], .02)
        self.assertEqual(new["slo_policy_id"], "v6_slow2pct_20261006")
        for key in old.keys() - {"primary_max_slow_interval_rate", "note"}:
            self.assertEqual(old[key], new[key])
        args = SimpleNamespace(slo_profile="slow2pct", probability_pilot=True)
        self.assertEqual(hashlib.sha256(contents).hexdigest(), reference_contract(args))
        args.probability_pilot = False
        with self.assertRaisesRegex(ValueError, "explicit probability"):
            reference_contract(args)

    def test_reduced_guard_file_matches_frozen_contract(self):
        args = SimpleNamespace(guard_profile="reduced2000", probability_pilot=True)
        value, digest = guard_contract(args)
        path = (
            Path(__file__).resolve().parents[2]
            / "experiments/phase9/guard_free_kv_tokens_2000.txt"
        )
        contents = path.read_bytes().replace(b"\r\n", b"\n")
        self.assertEqual(contents, b"2000\n")
        self.assertEqual(value % 16, 0)
        self.assertEqual(hashlib.sha256(contents).hexdigest(), digest)

    def test_near_guard_requires_fresh_anchor_telemetry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audit = root / "controller" / "phase9_audit.jsonl"
            audit.parent.mkdir()
            rows = [
                {"kind": "telemetry", "unix_s": 10.0, "output_tokens": 0,
                 "tp1": {"free_kv_blocks": 900, "block_size": 16,
                         "kv_usage_frac": 0.5}},
                {"kind": "telemetry", "unix_s": 11.0, "output_tokens": 1,
                 "tp1": {"free_kv_blocks": 600, "block_size": 16,
                         "kv_usage_frac": 0.7}},
            ]
            audit.write_text("".join(json.dumps(row) + "\n" for row in rows))
            result = observed_pressure(root)
            self.assertTrue(result["near_guard_verified"])
            self.assertEqual(result["free_kv_tokens_at_anchor_first_output"],
                             9600)

            rows[1]["tp1"]["free_kv_blocks"] = 500
            audit.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertFalse(observed_pressure(root)["near_guard_verified"])


class TestCommands(unittest.TestCase):
    def test_portable_hardware_keeps_model_and_idle_checks(self) -> None:
        args = SimpleNamespace(expected_revision="rev", expected_host=None,
                               portable_hardware=True,
                               input=Path("input"), model=Path("model"),
                               base=Path("base"), survival=Path("survival"),
                               guard=Path("guard"),
                               checkpoint=Path("checkpoint"),
                               reference=Path("reference"))
        values = iter(["rev", "", "GPU-a\nGPU-b\nGPU-c\nGPU-d\nGPU-e",
                       "\n".join(["NVIDIA A100-PCIE-40GB"] * 5),
                       "GPU-c, 123"])
        with mock.patch(
            "tools.bridge_tp.run_goodoutput_matrix_a100.subprocess.check_output",
            side_effect=lambda *_a, **_k: next(values),
        ):
            with self.assertRaisesRegex(ValueError, "compute process"):
                verify(args)

    def test_gpu_inventory_override_is_explicit_and_unique(self) -> None:
        value = ",".join(f"GPU-{index}" for index in range(5))
        self.assertEqual(expected_gpu_uuids(SimpleNamespace(
            expected_gpu_uuids=value)), value.split(","))
        with self.assertRaisesRegex(ValueError, "five distinct"):
            expected_gpu_uuids(SimpleNamespace(
                expected_gpu_uuids="GPU-a,GPU-a,GPU-b,GPU-c,GPU-d"))

    def test_prepare_builds_all_four_isolated_cells(self) -> None:
        class Tokenizer:
            @staticmethod
            def encode(_value: str, **_kwargs: object) -> list[int]:
                return [1, 2, 3]

        rows = [
            {"id": f"natural-{index}", "workload_group": "natural",
             "prompt": "hello"}
            for index in range(48)
        ] + [
            {"id": f"long-{index}", "workload_group": "long_form",
             "prompt": "hello"}
            for index in range(11)
        ] + [{
            "id": "oasst1:0ffd5b9c-d93a-4c60-b66a-f8786fbea2a0",
            "workload_group": "long_form", "prompt": "hello",
        }]
        transformers = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(
                from_pretrained=lambda *_args, **_kwargs: Tokenizer()))
        selector = types.SimpleNamespace(select_requests=lambda *_args: rows)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(input=Path("input.jsonl"),
                                   model=Path("model"))
            with mock.patch.dict(sys.modules, {
                "transformers": transformers,
                "tools.bridge_tp.run_natural_eos_gap_probe_a100": selector,
            }):
                setup = prepare(args, Path(directory))
            manifests = setup["manifests"]
            self.assertEqual(set(manifests), {
                "A_safe_light", "B_safe_busy", "C_guard_light",
                "D_guard_busy"})
            self.assertEqual([manifests[key]["jobs"] for key in manifests],
                             [14, 59, 8, 53])
            pressure = json.loads(
                Path(manifests["C_guard_light"]["path"]).read_text())
            source = [job for job in pressure["jobs"]
                      if job["pool"] == "source"]
            self.assertEqual(len(source), 6)
            self.assertTrue(all(len(job["request"]["prompt"]) == 3584
                                for job in source))
            self.assertTrue(all("start_after_event" not in job
                                for job in source))

    def test_stay_and_pressure_readiness_are_isolated(self) -> None:
        args = SimpleNamespace(
            model=Path("model"), survival=Path("survival"),
            guard=Path("guard"), checkpoint=Path("checkpoint"),
            expected_revision="abc",
        )
        setup = {
            "anchor_path": "anchor.json", "anchor_sha256": "sha",
            "anchor_prompt_tokens": 155,
            "manifests": {
                "A_safe_light": {"path": "a.json", "sha256": "a"},
                "C_guard_light": {"path": "c.json", "sha256": "c"},
            },
        }
        stay = online_command(args, setup, "A_safe_light", "stay",
                              Path("stay"))
        pressure = online_command(args, setup, "C_guard_light", "migrate",
                                  Path("migrate"))
        self.assertIn("--paired-stay", stay)
        self.assertNotIn("--paired-stay", pressure)
        index = pressure.index("--minimum-ready-source-jobs")
        self.assertEqual(pressure[index + 1], "3")
        index = stay.index("--minimum-ready-source-jobs")
        self.assertEqual(stay[index + 1], "0")
        args.guard_profile = "reduced2000"
        args.probability_pilot = True
        reduced = online_command(args, setup, "A_safe_light", "stay", Path("new"))
        self.assertEqual(reduced[reduced.index("--expected-guard") + 1], "2000")
        self.assertEqual(
            reduced[reduced.index("--expected-guard-sha256") + 1],
            "1d8fa3c8ab49d50b30fccbbd901735d5896a5d7959a5ad7ccecb79c1c849cc66",
        )
        args.probability_pilot = False
        with self.assertRaisesRegex(ValueError, "explicit probability"):
            online_command(args, setup, "A_safe_light", "stay", Path("invalid"))


if __name__ == "__main__":
    unittest.main()
