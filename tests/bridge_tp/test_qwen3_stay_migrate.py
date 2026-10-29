# SPDX-License-Identifier: Apache-2.0
"""Execution gating, causal predictor pairing and matched STAY/MIGRATE metrics."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.bridge_tp import qwen3_stay_migrate as matched
from tools.bridge_tp import run_pressure_balanced_pilot as controller
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp.pressure_cpu import load

m5 = load("manager_m5")
SHA = "a" * 64


def event(position, captured, published=None):
    return dict(
        kind="predictor_prediction",
        checkpoint_sha256=SHA,
        request_id="r",
        generated_tokens=position,
        captured_unix_ns=captured,
        published_unix_ns=published if published is not None else captured + 1,
        probabilities=[0.0, 0.0, 1.0],
    )


def reader(path, **kwargs):
    result = m5.PredictorEventReader(path, SHA, **kwargs)
    result._header = dict(
        category_upper_edges=[0, 8, 16],
        overflow_category=False,
        feature_layer="decoder:31",
        interval=20,
        support_scope="natural_eos_within_finite_support",
    )
    return result


class CausalTests(unittest.TestCase):
    def test_inventory_behind_latest_selects_prior_causal_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = reader(Path(tmp) / "absent", align_history=True)
            value._read_event(event(20, 1_000_000_000))
            value._read_event(event(40, 1_500_000_000))
            result = value.advisory(
                "r", 30, 100, now_ns=2_000_000_000, max_output_tokens=100
            )
            self.assertEqual(result["status"], "AVAILABLE")
            self.assertEqual(result["prediction_output_tokens"], 20)
            self.assertEqual(result["prediction_lag_tokens"], 10)

    def test_future_publication_is_not_consumed_and_legacy_default_is_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            for align in (False, True):
                value = reader(Path(tmp) / "absent", align_history=align)
                value._read_event(event(20, 1_000_000_000))
                value._read_event(event(40, 1_500_000_000, 3_000_000_000))
                result = value.advisory(
                    "r", 40, 100, now_ns=2_000_000_000, max_output_tokens=100
                )
                self.assertEqual(
                    result["prediction_output_tokens"], 20 if align else 40
                )

    def test_history_bounded_and_unavailable_clears_previous_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            value = reader(Path(tmp) / "absent", align_history=True, history_size=2)
            for position in (20, 40, 60):
                value._read_event(event(position, 1_000_000_000 + position))
            self.assertEqual(len(value._history["r"]), 2)
            self.assertEqual(
                value.advisory("r", 30, 100, now_ns=2_000_000_000)["status"],
                "NO_CAUSAL_HISTORY",
            )
            value._read_event(
                dict(
                    kind="predictor_unavailable",
                    checkpoint_sha256=SHA,
                    request_id="r",
                    generated_tokens=80,
                    reason="capture failed",
                )
            )
            self.assertNotIn("r", value._history)
            self.assertEqual(value.advisory("r", 80, 100)["status"], "UNAVAILABLE")


class PairTests(unittest.TestCase):
    def test_trace_finalization_uses_two_target_ranks(self):
        from tools.bridge_tp import run_atomic_takeover as atomic
        from tools.bridge_tp import run_pressure_comparison as comparison
        from tools.bridge_tp import verify_performance_trace as trace

        plan = dict(
            run_dir=Path("r/controller"),
            config=dict(source_url="s", target_url="t", target_tp_size=2),
            source_env=dict(BRIDGETP_PERFORMANCE_FINALIZE_TOKEN="s"),
            target_env=dict(BRIDGETP_PERFORMANCE_FINALIZE_TOKEN="t"),
        )
        with (
            patch.object(comparison.calibration, "wait_empty", return_value=[]),
            patch.object(comparison.pilot, "announce"),
            patch.object(atomic, "_post_json", return_value=dict(status="FINALIZED")),
            patch.object(trace, "verify_pool", return_value={}) as verify,
        ):
            comparison.finalize(plan, Mock())
        self.assertEqual([c.args[1] for c in verify.call_args_list], [1, 2])

    def test_closed_loop_requires_all_target_ranks_and_source_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            session = d / "sessions/migration-0001"
            session.mkdir(parents=True)
            final = dict(
                kind="pressure_final",
                outcome="COMPLETED",
                sessions=[
                    dict(
                        started=True,
                        run_dir=str(session),
                        migration_id="m",
                        settled=True,
                    )
                ],
            )
            (d / "pressure_audit.jsonl").write_text(
                json.dumps(dict(kind="pressure_actual_start"))
                + "\n"
                + json.dumps(final)
                + "\n"
            )
            state = dict(
                state="COMMITTED",
                migration_id="m",
                ready_sender_ranks=[0, 1],
                ready_receiver_ranks=[0, 1],
                source_abort_dispatched=True,
                source_request_id="r",
            )
            (session / "takeover_state.json").write_text(json.dumps(state))
            (session / "source_kv_release_receipt.json").write_text(
                json.dumps(dict(status="SOURCE_KV_RELEASED", request_id="r"))
            )
            with patch(
                "tools.bridge_tp.pressure_session.channel_idle", return_value=True
            ):
                self.assertEqual(
                    runner.closed_loop_proof(dict(run_dir=d), "migrate")[
                        "actual_starts"
                    ],
                    1,
                )
                with self.assertRaises(ValueError):
                    runner.closed_loop_proof(dict(run_dir=d), "stay")
                state["ready_receiver_ranks"] = [0]
                (session / "takeover_state.json").write_text(json.dumps(state))
                with self.assertRaises(ValueError):
                    runner.closed_loop_proof(dict(run_dir=d), "migrate")

    def test_migrate_arm_is_explicitly_enabled_and_stay_disabled(self):
        for arm in ("stay", "migrate"):
            plan = dict(
                source_env={},
                target_env={},
                run_dir=Path("r"),
                controller_argv=["python", "entry"]
                + (["--no-migration"] if arm == "stay" else []),
            )
            with (
                patch.object(runner, "adjust_migration_plan"),
                patch.object(runner, "configure_predictor"),
            ):
                runner.adapt_plan(plan, arm, Path("c"), Path("m"))
            self.assertEqual(
                "--qwen3-execute-migrations" in plan["controller_argv"],
                arm == "migrate",
            )
            self.assertEqual("--no-migration" in plan["controller_argv"], arm == "stay")

    def test_qwen3_cli_explicit_execution_requires_consistent_flags(self):
        from types import SimpleNamespace

        from tools.bridge_tp.run_qwen3_pressure_observation import child_arguments

        args = child_arguments(
            SimpleNamespace(), dict(run_dir=Path("d")), Path("d"), Path("jobs")
        )
        args.remove("--no-migration")
        args.append("--qwen3-execute-migrations")
        self.assertFalse(controller.parse_args(args).no_migration)
        args.append("--no-migration")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            controller.parse_args(args)

    def test_lengths_frozen_offline_before_either_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "result.json").write_text(
                json.dumps(
                    dict(status="COMPLETED_OBSERVATION_ONLY", total_profile_attempts=31)
                )
            )
            for name in (
                "kv_pressure",
                "short_burst_recovery",
                "natural_long_pressure",
                "target_busy",
            ):
                (d / name).mkdir()
                value = [
                    dict(
                        id="r",
                        at_s=1.0,
                        route=None,
                        request=dict(prompt=[1, 2], max_tokens=100, ignore_eos=False),
                    )
                ]
                (d / name / "arrivals.json").write_text(json.dumps(value))
                (d / name / "observation_summary.json").write_text(
                    json.dumps(dict(request_quality=[dict(job_id="r", tokens=30)]))
                )
            values = matched.recipes(d, 2, include_natural=False)
            self.assertEqual(len(values), 8)
            for recipe in values:
                self.assertEqual(recipe["jobs"][0]["request"]["max_tokens"], 30)
                self.assertTrue(recipe["jobs"][0]["request"]["ignore_eos"])
                self.assertEqual(
                    matched.digest(copy.deepcopy(recipe["jobs"])),
                    recipe["input_sha256"],
                )
            self.assertEqual(matched.config(values)["maximum_migrations"], 0)
            supplemented = matched.recipes(d, 2)
            self.assertEqual(len(supplemented), 9)
            self.assertFalse(supplemented[-1]["fixed_output"])
            self.assertFalse(supplemented[-1]["jobs"][0]["request"]["ignore_eos"])
            self.assertEqual(supplemented[-1]["jobs"][0]["request"]["max_tokens"], 100)

    def test_metrics_keep_handoff_stall_and_use_common_window_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            request = dict(prompt=[1], max_tokens=3)
            jobs = [dict(id="r", at_s=0.0, route=None, request=request)]
            row = dict(
                job_id="r",
                initial_pool="tp1",
                source_error=None,
                response=dict(
                    emitted=[
                        dict(
                            index=i,
                            unix_s=t,
                            token_id=i,
                            origin="source" if i == 0 else "target",
                        )
                        for i, t in enumerate([100.1, 102.1, 102.2])
                    ],
                    committed=True,
                    handoff_stall_s=2.0,
                ),
                source_transport=dict(request_started_unix_s=100.0),
            )
            (d / "pressure_request_results.json").write_text(json.dumps([row]))
            audit = [
                dict(kind="pressure_clock_origin", unix_s=100.0),
                dict(kind="pressure_actual_start"),
                dict(kind="pressure_source_release_confirmed"),
                dict(kind="pressure_final", outcome="COMPLETED"),
            ]
            (d / "pressure_audit.jsonl").write_text(
                "\n".join(json.dumps(r) for r in audit) + "\n"
            )
            result, values, _ = matched.metrics(d, jobs, 300.0)
            self.assertFalse(values[0]["slo_pass"])
            self.assertEqual(result["window_raw_tokens_s"], 3 / 300)
            self.assertEqual(result["maximum_handoff_s"], 2.0)
            self.assertFalse(result["load_coverage_pass"])
            attempted = copy.deepcopy(result)
            attempted.update(
                migration_starts=1, source_releases=0, committed_requests=0
            )
            paired = matched.compare(result, attempted)
            self.assertTrue(paired["actual_start_observed"])
            self.assertFalse(paired["actual_migration_observed"])
