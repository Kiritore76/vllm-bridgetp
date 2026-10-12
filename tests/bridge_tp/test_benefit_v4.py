"""Risk76 runtime, phase observations, evidence integrity and fit boundaries."""

import ast
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
                    "vllm.bridge_tp.predictor_capture": capture,
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
            self.addCleanup(observer.close)
            observer.capture(
                torch.ones(2, 2), ["a", "b"], [0, 20], [15, 16], [1, 1], [16, 16]
            )
            tree = ast.parse((ROOT / "vllm/v1/worker/gpu_worker.py").read_text())
            worker = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "Worker"
            )
            method = next(
                node
                for node in worker.body
                if isinstance(node, ast.FunctionDef)
                and node.name == "bridge_tp_finalize_predictor_features"
            )
            namespace = {}
            exec(
                compile(ast.Module(body=[method], type_ignores=[]), "rpc", "exec"),
                namespace,
            )
            source_worker = SimpleNamespace(
                model_runner=SimpleNamespace(predictor_feature_capture=observer)
            )
            finalized = namespace[method.name](source_worker)
            self.assertEqual(finalized["status"], "FINALIZED")
            self.assertGreater(finalized["saved_batches"], 0)
            self.assertTrue((features / "receipt.json").is_file())
            self.assertEqual(namespace[method.name](source_worker), finalized)
            # Model teardown follows explicit finalization. It must not hang or
            # rewrite the receipt when it closes the observer for a second time.
            observer.close()
            with self.assertRaisesRegex(RuntimeError, "finalized"):
                observer.begin_forward()
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

    def test_repair_design_is_four_arms_without_using_selection_test_seeds(self):
        rows = collect.repair_design()
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["diagnostic_use_only"] for row in rows))
        self.assertTrue(all(row["threshold"] == 0 for row in rows))
        self.assertTrue(all(row["source_state"] == "slack" for row in rows))
        self.assertEqual([row["target_count"] for row in rows], [12, 4])
        self.assertEqual(rows[1]["target_period_s"], 30)
        self.assertEqual(rows[1]["target_max_tokens"], 256)
        self.assertTrue(rows[1]["target_burst_arrivals"])
        self.assertFalse({row["seed"] for row in rows} &
                         {row["seed"] for row in collect.design()})

    def test_diagnostic_repair_pilot_cannot_enter_model_selection(self):
        (self.root / "preregistration.json").write_text(json.dumps({
            "diagnostic_only": True,
        }))
        with self.assertRaisesRegex(ValueError, "excluded from model selection"):
            fit.load_pairs([self.root], {"train", "validation"})

    def test_dispatch_pair_reuses_decode_recipe_without_expansion(self):
        rows = collect.dispatch_design()
        self.assertEqual(len(rows), 1)
        expected = collect.repair_design()[0]
        self.assertEqual(rows[0]["seed"], 20261251)
        self.assertTrue(rows[0]["diagnostic_use_only"])
        self.assertEqual(
            {key: value for key, value in rows[0].items() if key != "pair_id"},
            {key: value for key, value in expected.items() if key != "pair_id"},
        )

    def test_prefill_repeat_changes_records_not_pressure_recipe(self):
        row = collect.prefill_repeat_design()[0]
        self.assertEqual(row["seed"], 20261253)
        self.assertEqual(row["target_state"], "prefill")
        self.assertTrue(row["target_burst_arrivals"])
        self.assertTrue(row["diagnostic_use_only"])
        self.assertEqual(row["target_count"], 4)
        self.assertEqual(row["target_period_s"], 30)

    def test_urgent_phase_changes_source_not_verified_target_recipes(self):
        rows = collect.urgent_phase_design()
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["target_count"] for row in rows], [0, 12, 4])
        self.assertEqual([row["seed"] for row in rows], [20261261, 20261262, 20261263])
        for row in rows:
            self.assertEqual(row["source_count"], 5)
            self.assertEqual(row["source_prompt_tokens"], 4992)
            self.assertEqual(row["minimum_u"], 0.8)
            self.assertIsNone(row["maximum_u"])
            self.assertEqual(row["threshold"], 0)
            self.assertTrue(row["diagnostic_use_only"])
        self.assertTrue(rows[-1]["target_burst_arrivals"])
        self.assertEqual(rows[-1]["target_max_tokens"], 256)
        self.assertFalse({row["seed"] for row in rows} &
                         {row["seed"] for row in collect.design()})

    def test_urgent_followup_reuses_inputs_and_moves_only_prefill_epoch(self):
        rows = collect.urgent_followup_design()
        self.assertEqual([row["seed"] for row in rows], [20261262, 20261263])
        self.assertEqual([row["target_state"] for row in rows], ["decode", "prefill"])
        self.assertFalse(rows[0]["target_start_after_anchor"])
        self.assertTrue(rows[1]["target_start_after_anchor"])
        self.assertEqual(rows[1]["target_burst_offset_s"], 1.0)
        self.assertTrue(all(row["minimum_u"] == 0.8 and row["threshold"] == 0
                            for row in rows))

    def test_prefill_recovery_does_not_repeat_decode(self):
        rows = collect.urgent_prefill_recovery_design()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0], collect.urgent_followup_design()[1])

    def test_formal_expansion_is_fresh_and_role_isolated(self):
        rows = collect.formal_expansion_design()
        self.assertEqual(len(rows), 24)
        self.assertEqual([sum(r["role"] == role for r in rows)
                          for role in ("train", "validation", "test")], [12, 6, 6])
        self.assertEqual(len({r["seed"] for r in rows}), 24)
        old = collect.design() + collect.repair_design() + collect.urgent_phase_design()
        self.assertFalse({r["seed"] for r in rows} & {r["seed"] for r in old})
        for r in rows:
            self.assertEqual(r["threshold"], 0)
            self.assertNotIn("diagnostic_use_only", r)
            if r["target_state"] == "prefill":
                self.assertTrue(r["target_start_after_anchor"])
                self.assertEqual((r["target_count"], r["target_period_s"],
                                  r["target_max_tokens"]), (4, 30, 256))

    def test_f11_retains_fresh_prefill_and_marks_missing_pressure(self):
        snapshot = {
            "p_guard_est_bounds": [0.6, 0.7], "U": 0.9,
            "source": {"source_running": 6, "target_running": 1},
            "target_observation": {"version": "TARGET_OBSERVATION_V1",
                "prompt_arrival_tokens_s": 2457.6,
                "prefill_pending_tokens": 18432, "decode_status": "MISSING",
                "decode_pressure_p95": None, "telemetry_age_s": 0.5,
                "integrity_errors": []},
        }
        self.assertIsNone(fit.features(snapshot, 10))
        self.assertEqual(fit.features(snapshot, 11)[-3:], [18432, 0.0, 0.0])
        obs = snapshot["target_observation"]
        for change in ({"telemetry_age_s": 3}, {"prefill_pending_tokens": None},
                       {"integrity_errors": ["lost records"]},
                       {"decode_status": "STALE"}, {"prefill_pending_tokens": 0}):
            with self.subTest(change=change):
                altered = {**snapshot, "target_observation": {**obs, **change}}
                self.assertIsNone(fit.features(altered, 11))
        observed = {**snapshot, "target_observation": {
            **obs, "decode_status": "VALID", "decode_pressure_p95": 0.3}}
        self.assertEqual(fit.features(observed, 11)[-2:], [0.3, 1.0])
        rows = [{"eligible": True, "delta": float(i), "snapshot": state}
                for i, state in enumerate([snapshot, observed, snapshot, observed])]
        model = fit.train(rows, 11, 1.0)
        self.assertEqual(model["training_pairs"], 4)
        self.assertIsNotNone(fit.predict(model, rows[0]))
        model["features"] = model["features"][:-1]
        with self.assertRaisesRegex(ValueError, "contract"):
            fit.predict(model, rows[0])

    def test_formal_first_batch_plan_selects_only_six_training_pairs(self):
        args = ["collect", "--stage", "formal-train1", "--plan-only",
                "--out-dir", str(self.root / "formal"),
                "--expected-revision", "local-check"]
        for key in ("model", "base", "survival", "guard", "checkpoint", "reference"):
            args += [f"--{key}", str(self.root / key)]
        with patch.object(sys, "argv", args):
            collect.main()
        commands = json.loads((self.root / "formal/commands.json").read_text())
        self.assertEqual(len(commands), 6)
        self.assertEqual({r["pair"]["seed"] for r in commands},
                         set(range(20261500, 20261506)))
        plan = json.loads((self.root / "formal/preregistration.json").read_text())
        self.assertFalse(plan["diagnostic_only"])
        self.assertTrue(all(r["pair"]["role"] == "train" for r in commands))

    def test_prefix_diagnostic_ignores_post_action_divergence(self):
        arms = {}
        for action, boundary in (("prob_0_STAY", 20), ("prob_0_START", 25)):
            path = self.root / "case" / action / "r01_shadow_only/controller"
            path.mkdir(parents=True)
            tokens = list(range(40))
            if action.endswith("START"):
                tokens[30] = 99
            (path / "source_response.json").write_text(json.dumps({
                "token_ids": tokens,
            }))
            arms[action] = {"observed_action": {"probability_candidate": {
                "snapshot": {"generated_tokens": boundary},
            }}}
        (self.root / "pilot_summary.json").write_text(json.dumps({
            "cases": {"case": arms},
        }))
        collect.prefix_diagnostic(self.root)
        report = json.loads((self.root / "pre_action_prefix_diagnostic.json")
                            .read_text())
        self.assertTrue(report["identical_pre_action_prefix"])
        self.assertEqual(report["comparison_tokens"], 20)
        self.assertFalse(report["selection_on_goodoutput"])
        arms["prob_0_START"]["observed_action"] = {}
        (self.root / "pilot_summary.json").write_text(json.dumps({
            "cases": {"case": arms},
        }))
        collect.prefix_diagnostic(self.root)
        report = json.loads((self.root / "pre_action_prefix_diagnostic.json")
                            .read_text())
        self.assertEqual(report["status"], "CANDIDATE_NOT_OBSERVED")
        self.assertIsNone(report["identical_pre_action_prefix"])

    def test_fit_rejects_legacy_file_and_renamed_wrong_contract(self):
        row = collect.design()[0]
        (self.root / "preregistration.json").write_text(json.dumps({
            "protocol": collect.PROTOCOL, "predictor_sha256": fit.PROFILE_SHA,
            "all_pairs": [row], "selected_pairs": [row["pair_id"]],
        }))
        directory = self.root / row["pair_id"]
        directory.mkdir()
        old = {"status": "PAIRED_ONLY", "samples": []}
        (directory / "paired_outcomes.json").write_text(json.dumps(old))
        with self.assertRaisesRegex(ValueError, "missing probability outcome"):
            fit.load_pairs([self.root], {"train"})
        (directory / "probability_outcomes.json").write_text(json.dumps(old))
        with self.assertRaisesRegex(
            ValueError, "probability collector outcome contract"
        ):
            fit.load_pairs([self.root], {"train"})

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
            (directory / "probability_outcomes.json").write_text(
                json.dumps(
                    {
                        "status": "COLLECTOR_PILOT_NOT_FITTED_NET_BENEFIT",
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
        self.assertEqual(set(result["models"]), {"F8", "F11"})
        self.assertFalse(result["engineering_gate"]["automatic_deployment"])


if __name__ == "__main__":
    unittest.main()
