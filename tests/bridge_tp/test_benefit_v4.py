"""Risk76 runtime, phase observations, evidence integrity and fit boundaries."""

import hashlib
import importlib.util
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from tools.bridge_tp import fit_benefit_v4 as fit
from tools.bridge_tp import run_benefit_v4_collection as collect
from tools.bridge_tp.target_observation import TargetObserver, state_eligibility
from tools.bridge_tp.verify_predictor_feature_journal import verify

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


journal = load("benefit_journal_test", "vllm/bridge_tp/async_journal.py")
runtime = load(
    "benefit_runtime_test", "vllm/bridge_tp/controller/distribution_predictor.py"
)
network = load("benefit_network_test", "tools/bridge_tp/predictor_network.py")
capture = load("benefit_capture_test", "vllm/bridge_tp/predictor_capture.py")


class BenefitV4Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_four_layer_online_matches_training_network(self):
        torch.manual_seed(4)
        model = network.build_mlp(3, 4, 4, 0.1, 4).eval()
        metadata = {
            "format_version": 2,
            "model_type": "remaining_length_categorical_mlp4",
            "network_depth": 4,
            "hidden_widths": [4, 8, 4],
            "hidden_width": 4,
            "input_width": 3,
            "activation": "GELU",
            "dropout": 0.1,
            "feature_mean": torch.tensor([0.1, 0.2]),
            "feature_std": torch.tensor([2.0, 3.0]),
            "category_upper_edges": torch.tensor([0, 8, 8192]),
            "overflow_category": True,
            "position_log_scale": 9.7,
            "temperature": 1.2,
            "feature_layer": "decoder:31",
            "model_config_sha256": "config",
            "capture_input_sha256": "input",
            "state_dict": model.state_dict(),
        }
        path = self.root / "checkpoint.pt"
        torch.save(metadata, path)
        predictor = runtime.DistributionPredictor(
            path,
            checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            model_config_sha256="config",
        )
        hidden = torch.tensor([[0.2, 0.4], [1.1, 1.3]])
        position = torch.tensor([0, 9000])
        features = torch.cat(
            (
                (hidden.half().float() - metadata["feature_mean"])
                / metadata["feature_std"],
                (torch.log1p(position.float()) / 9.7)[:, None],
            ),
            1,
        )
        torch.testing.assert_close(
            predictor.probabilities(hidden, position),
            torch.softmax(model(features) / 1.2, -1),
        )
        metadata["hidden_widths"] = [4, 4, 4]
        torch.save(metadata, path)
        with self.assertRaisesRegex(ValueError, "architecture"):
            runtime.DistributionPredictor(
                path,
                checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                model_config_sha256="config",
            )

    def observe(self, rows, pool=None):
        path = self.root / "events.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        if pool is not None:
            pool = {**pool, "target_observation_unix_s": pool["sampled_unix_s"]}
        return TargetObserver(path).snapshot(
            10,
            pool
            or {
                "sampled_unix_s": 10,
                "target_observation_unix_s": 10,
                "num_running": 4,
                "num_waiting": 0,
                "prefill_pending_tokens": 0,
            },
        )

    def test_only_actual_past_arrivals_and_native_intervals(self):
        rows = [
            {"kind": "native_arrival", "unix_s": 5, "prompt_tokens": 1000},
            {"kind": "native_arrival", "unix_s": 11, "prompt_tokens": 9000},
        ]
        rows += [
            {
                "kind": "native_itl",
                "unix_s": 8 + i / 20,
                "itl_s": 0.05,
                "tpot_reference_s": 0.1,
            }
            for i in range(20)
        ]
        observation = self.observe(rows)
        self.assertEqual(observation["prompt_arrival_tokens_s"], 100)
        self.assertEqual(observation["decode_pressure_p95"], 0.5)
        self.assertEqual(state_eligibility(observation, "decode", 0.25), ())
        self.assertTrue(state_eligibility(observation, "prefill"))

    def test_missing_and_stale_never_mean_idle(self):
        for pending in (None, 1):
            observation = self.observe(
                [],
                {
                    "sampled_unix_s": 10,
                    "num_running": 0,
                    "num_waiting": 0,
                    "prefill_pending_tokens": pending,
                },
            )
            self.assertEqual(observation["decode_status"], "MISSING")
        observation = self.observe(
            [],
            {
                "sampled_unix_s": 7,
                "num_running": 0,
                "num_waiting": 0,
                "prefill_pending_tokens": 0,
            },
        )
        self.assertTrue(state_eligibility(observation, "idle"))
        idle = self.observe(
            [],
            {
                "sampled_unix_s": 10,
                "num_running": 0,
                "num_waiting": 0,
                "prefill_pending_tokens": 0,
            },
        )
        self.assertEqual(idle["decode_status"], "IDLE")
        self.assertEqual(idle["decode_pressure_p95"], 1)

    def test_integrity_footer_rejects_missing_events(self):
        observation = self.observe([{"kind": "evidence_footer", "dropped": 1}])
        self.assertEqual(observation["decode_status"], "MISSING")
        self.assertIsNone(observation["prompt_arrival_tokens_s"])

    def test_idle_report_expires_when_new_work_arrives(self):
        pool = {
            "sampled_unix_s": 10,
            "target_observation_unix_s": 5,
            "num_running": 0,
            "num_waiting": 0,
            "prefill_pending_tokens": 0,
        }
        path = self.root / "idle.jsonl"
        path.write_text("")
        observation = TargetObserver(path).snapshot(10, pool)
        self.assertEqual(observation["decode_status"], "IDLE")
        self.assertEqual(observation["prefill_pending_tokens"], 0)
        path.write_text(
            json.dumps({"kind": "native_arrival", "unix_s": 9, "prompt_tokens": 1000})
            + "\n"
        )
        observation = TargetObserver(path).snapshot(10, pool)
        self.assertEqual(observation["decode_status"], "MISSING")

    def test_fresh_scrape_does_not_refresh_busy_engine_evidence(self):
        path = self.root / "stale.jsonl"
        path.write_text("")
        observation = TargetObserver(path).snapshot(
            10,
            {
                "sampled_unix_s": 10,
                "target_observation_unix_s": 4,
                "num_running": 4,
                "num_waiting": 3,
                "prefill_pending_tokens": 10000,
            },
        )
        self.assertIsNone(observation["prefill_pending_tokens"])
        self.assertTrue(state_eligibility(observation, "prefill"))

    def test_partial_tail_line_is_not_consumed_early(self):
        path = self.root / "partial.jsonl"
        path.write_text('{"kind":"native_arrival","unix_s":5,')
        observer = TargetObserver(path)
        observer.poll()
        self.assertEqual(len(observer.events), 0)
        with path.open("a") as handle:
            handle.write('"prompt_tokens":50}\n')
        observer.poll()
        self.assertEqual(len(observer.events), 1)

    def test_async_feature_receipt_and_hashes(self):
        writer = journal.FeatureJournal(self.root / "features")
        writer.submit("sample", {"hidden": torch.ones(2, 2)})
        writer.close()
        receipt = json.loads((self.root / "features/receipt.json").read_text())
        self.assertEqual(receipt["saved_batches"], 1)
        digest = hashlib.sha256(
            (self.root / "features/sample.pt").read_bytes()
        ).hexdigest()
        self.assertEqual(receipt["sha256"]["sample.pt"], digest)
        writer = journal.AsyncJournal(self.root / "async.jsonl")
        writer.submit({"unix_s": 10})
        writer.close()
        self.assertEqual(
            json.loads((self.root / "async.jsonl").read_text().splitlines()[-1])[
                "dropped"
            ],
            0,
        )

    def test_live_observer_journal_replays_and_associates_candidate(self):
        config_path = self.root / "config.json"
        config_path.write_text("{}")
        config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
        model = network.build_mlp(3, 4, 4, 0.1, 4)
        path = self.root / "checkpoint.pt"
        torch.save(
            {
                "format_version": 2,
                "model_type": "remaining_length_categorical_mlp4",
                "network_depth": 4,
                "hidden_widths": [4, 8, 4],
                "hidden_width": 4,
                "input_width": 3,
                "activation": "GELU",
                "dropout": 0.1,
                "feature_mean": torch.zeros(2),
                "feature_std": torch.ones(2),
                "category_upper_edges": torch.tensor([0, 8, 8192]),
                "overflow_category": True,
                "position_log_scale": math.log1p(8192),
                "temperature": 1.2,
                "feature_layer": "decoder:31",
                "model_config_sha256": config_sha,
                "capture_input_sha256": "input",
                "state_dict": model.state_dict(),
            },
            path,
        )
        events = self.root / "predictor_events.jsonl"
        features = self.root / "saved_features"
        with (
            patch.dict(
                os.environ, {"BRIDGETP_PREDICTOR_FEATURE_JOURNAL": str(features)}
            ),
            patch.dict(
                sys.modules,
                {
                    "vllm.bridge_tp.controller.distribution_predictor": runtime,
                    "vllm.bridge_tp.async_journal": journal,
                },
            ),
        ):
            observer = capture.PredictorLiveObserver(
                path,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                events,
                self.root,
                torch.device("cpu"),
            )
            observer.capture(
                torch.ones(2, 2), ["a", "b"], [0, 20], [15, 16], [1, 1], [16, 16]
            )
            observer.close()
        prediction = json.loads(events.read_text().splitlines()[1])
        audit = self.root / "audit.jsonl"
        audit.write_text(
            json.dumps(
                {
                    "kind": "experiment_probability_gate",
                    "first_feasible_candidate": True,
                    "snapshot": {
                        "prediction": {
                            "feature_id": prediction["feature_id"],
                            "request_id": "a",
                            "prediction_output_tokens": 0,
                        }
                    },
                }
            )
            + "\n"
        )
        result = verify(features, path, config_sha, audit)
        self.assertEqual(result["candidate_references"], 1)
        self.assertEqual(result["saved_states"], 2)
        batch = torch.load(next(features.glob("*.pt")), weights_only=True)
        self.assertEqual(batch["sampling_boundaries"][0]["computed_prompt_tokens"], 15)

    def test_preregistered_roles_and_zero_probability_intervention(self):
        rows = collect.design()
        self.assertEqual(len(rows), 36)
        self.assertEqual(sum(row["role"] == "train" for row in rows), 24)
        self.assertEqual(len({row["seed"] for row in rows}), 36)
        input_path = self.root / "input.jsonl"
        input_path.write_text("{}\n")
        args = SimpleNamespace(
            **{
                key: Path(key)
                for key in (
                    "model",
                    "base",
                    "survival",
                    "guard",
                    "checkpoint",
                    "reference",
                )
            },
            expected_revision="head",
        )
        cmd = collect.command(args, rows[0], input_path, self.root)
        self.assertNotIn("--timing-pilot", cmd)
        self.assertNotIn("--cutover-output-tokens", cmd)
        self.assertEqual(cmd[cmd.index("--probability-thresholds") + 1], "0.0")
        self.assertIn("--save-predictor-features", cmd)

    def test_fit_unknown_has_opportunity_loss_and_keeps_negative_labels(self):
        rows = []
        for index in range(8):
            snapshot = {
                "p_guard_est_bounds": [index / 10, 1],
                "U": 0.2 + index / 10,
                "source": {"source_running": 4, "target_running": index % 3},
                "target_observation": {
                    "version": "TARGET_OBSERVATION_V1",
                    "prompt_arrival_tokens_s": index * 100,
                    "prefill_pending_tokens": 0,
                    "decode_status": "VALID",
                    "decode_pressure_p95": 0.5,
                },
            }
            rows.append(
                {
                    "eligible": True,
                    "snapshot": snapshot,
                    "delta": 10 - index * 4,
                    "stay_q": 100,
                    "pair_id": str(index),
                    "target_state": "decode",
                }
            )
        model = fit.train(rows, 10, 1)
        self.assertTrue(np.isfinite(model["coefficients"]).all())
        row = {**rows[0], "snapshot": {}}
        result = fit.evaluate(model, [row], 0)
        self.assertEqual(result["unknown"], 1)
        self.assertEqual(result["total_regret"], 10)
        self.assertEqual(result["missed_positives"], 1)

    def test_fit_selection_does_not_open_final_test_labels(self):
        plan = collect.design()
        (self.root / "preregistration.json").write_text(
            json.dumps(
                {
                    "protocol": collect.PROTOCOL,
                    "predictor_sha256": fit.PROFILE_SHA,
                    "all_pairs": plan,
                    "selected_pairs": [row["pair_id"] for row in plan],
                }
            )
        )
        for index, row in enumerate(plan):
            if row["role"] == "test":
                continue  # Test paths intentionally do not exist.
            directory = self.root / row["pair_id"]
            directory.mkdir()
            for action in ("START", "STAY"):
                feature_dir = (
                    directory
                    / f"case/prob_0_{action}/r01_shadow_only/controller"
                    / "predictor_features"
                )
                feature_dir.mkdir(parents=True)
                (feature_dir / "replay_verification.json").write_text(
                    json.dumps(
                        {
                            "checkpoint_sha256": fit.PROFILE_SHA,
                            "saved_states": 1,
                            "max_probability_absolute_error": 0.0,
                        }
                    )
                )
            snapshot = {
                "p_guard_est_bounds": [0.1 + index / 100, 1],
                "U": 0.2 + index / 50,
                "source": {
                    "source_running": row["source_count"] + 1,
                    "target_running": row["target_count"],
                },
                "prediction": {"checkpoint_sha256": fit.PROFILE_SHA},
                "target_observation": {
                    "version": "TARGET_OBSERVATION_V1",
                    "prompt_arrival_tokens_s": index * 100,
                    "prefill_pending_tokens": 1024,
                    "decode_status": "VALID",
                    "decode_pressure_p95": 0.5,
                },
            }
            (directory / "paired_outcomes.json").write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "threshold": row["threshold"],
                                "effect_sample_eligible": True,
                                "start_candidate": {"snapshot": snapshot},
                                "stay_candidate": {"snapshot": snapshot},
                                "descriptive_delta_tokens_s": 20 - index,
                                "stay_quality_output_tokens_s": 100,
                            }
                        ]
                    }
                )
            )
        out = self.root / "fit"
        with patch.object(
            sys, "argv", ["fit", "--collections", str(self.root), "--out-dir", str(out)]
        ):
            fit.main()
        result = json.loads((out / "result.json").read_text())
        self.assertEqual(set(result["models"]), {"F8", "F10"})
        self.assertFalse(result["engineering_gate"]["automatic_deployment"])


if __name__ == "__main__":
    unittest.main()
