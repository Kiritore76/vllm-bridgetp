# SPDX-License-Identifier: Apache-2.0
"""Real six-card assessor and arrival validation, with synthetic CPU evidence."""

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from tests.bridge_tp.test_runtime_instance_ids import canonical_plan
from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp import six_conversion as six
from tools.bridge_tp import six_split_probe as probe


def fixture(root):
    (root / "controller").mkdir()
    step = dict(
        plan_id="split",
        triggered_unix_s=110,
        completed_unix_s=251,
        plan=dict(state="COMPLETED"),
        layout=dict(A=1, B=2, C=1),
        layout_revision=6,
        controller_acknowledged=True,
        active_instances={name: dict(tp_degree=1) for name in probe.NEW_A},
    )
    events = [
        dict(
            kind=kind,
            plan_id="split",
            unix_s=at,
            changes={probe.INITIAL_A: "QUIESCING"},
        )
        for kind, at in [
            ("LAYOUT_CONTROL", 110),
            ("FENCED", 210),
            ("RESOURCES_RELEASED", 212),
            ("ROUTE_PUBLISHED", 250),
        ]
    ]
    rows = []
    for i, (source, first, last) in enumerate(
        [
            (probe.INITIAL_A, 31, 200),
            (probe.NEW_A[0], 331, 550),
            (probe.NEW_A[1], 451, 670),
        ]
    ):
        rows.append(
            dict(
                job_id=f"job-{i}",
                server_id=f"request-{i}",
                initial_pool=source,
                source_done=True,
                source_error=None,
                request=dict(prompt=[1] * 16384, max_tokens=1024),
                dispatch_timing=dict(
                    planned_unix_s=first - 1, http_thread_started_unix_s=first - 1
                ),
                response=dict(
                    committed=i > 0,
                    emitted=[
                        dict(index=n, unix_s=first + (last - first) * n / 1023)
                        for n in range(1024)
                    ],
                ),
            )
        )
    audit = [dict(kind="pressure_layout_applied", revision=6, unix_s=251)]
    sessions = []
    for i, at in ((1, 370), (2, 490)):
        audit.append(
            dict(
                kind="pressure_actual_start",
                migration_id=f"migration-{i}",
                request_id=f"request-{i}",
                target_epoch=probe.TARGET,
                unix_s=at,
            )
        )
        sessions.append(
            dict(
                migration_id=f"migration-{i}",
                state="COMMITTED",
                settled=True,
                target_instance=probe.TARGET,
                source_release=dict(
                    status="SOURCE_KV_RELEASED", request_id=f"request-{i}"
                ),
            )
        )
    return dict(step=step, events=events, rows=rows, audit=audit, sessions=sessions)


def assess(root, data, required=True):
    def write(path, value):
        (root / path).write_text(json.dumps(value), encoding="utf-8")

    write(
        "conversion_result.json",
        dict(state="COMPLETED", error=None, steps=[data["step"]]),
    )
    write(
        "conversion_sequence_result.json",
        dict(state="COMPLETED", error=None, steps=[data["step"]]),
    )
    write("closed_loop_acceptance.json", dict(sessions=data["sessions"]))
    write("controller/pressure_request_results.json", data["rows"])
    for path, key in [
        ("conversion_events.jsonl", "events"),
        ("controller/pressure_audit.jsonl", "audit"),
    ]:
        (root / path).write_text(
            "\n".join(json.dumps(v) for v in data[key]), encoding="utf-8"
        )
    return six.assess(root, 3, split_probe=required)


class CoverageTests(unittest.TestCase):
    def check(self, change=lambda data: None, required=True):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = fixture(root)
            change(data)
            return assess(root, data, required)

    def test_real_assessor_requires_loaded_drain_and_both_joined_sources(self):
        result = self.check()
        self.assertTrue(result["mechanism_passed"])
        self.assertTrue(result["passed"])
        self.assertEqual(result["loaded_drain_requests"], ["job-0"])
        self.assertEqual(
            result["joined_source_coverage"], dict.fromkeys(probe.NEW_A, True)
        )

    def test_no_migration_is_false_even_when_optional_mechanism_passes(self):
        def clear(data):
            data["audit"] = data["audit"][:1]
            data["sessions"] = []

        result = self.check(clear, required=False)
        self.assertTrue(result["passed"])
        self.assertFalse(result["migration_coverage"])
        self.assertFalse(result["migration_coverage_required"])
        required = self.check(clear)
        self.assertTrue(required["mechanism_passed"])
        self.assertFalse(required["passed"])

    def test_only_one_joined_source_is_a_coverage_gap(self):
        result = self.check(lambda data: data["sessions"].pop())
        self.assertTrue(result["migration_coverage"])
        self.assertFalse(result["coverage_passed"])
        verdict, code = runner.six_conversion_result(
            NS(conversion="L2C-L1AC", conversion_decider=None),
            [dict(status="COMPLETED", conversion_sequence=result)],
            1,
            None,
        )
        self.assertEqual((verdict["status"], code), ("COMPLETED_WITH_COVERAGE_GAPS", 2))

    def test_start_only_cancelled_unsettled_and_wrong_release_never_cover(self):
        for patch in (
            dict(state="CANCELLED"),
            dict(settled=False),
            dict(source_release={}),
            dict(source_release=dict(status="SOURCE_KV_RELEASED", request_id="other")),
        ):
            with self.subTest(patch=patch):
                result = self.check(
                    lambda data, patch=patch: data["sessions"][0].update(patch)
                )
                self.assertFalse(result["joined_source_coverage"][probe.NEW_A[0]])

    def test_wrong_target_or_pre_join_transfer_does_not_cover_joined_source(self):
        for patch in (dict(target_epoch="A-g3-tp2"), dict(unix_s=240)):
            with self.subTest(patch=patch):
                result = self.check(
                    lambda data, patch=patch: data["audit"][1].update(patch)
                )
                self.assertFalse(result["coverage_passed"])

    def test_empty_or_migrated_drain_is_not_natural_loaded_drain(self):
        for change in (
            lambda r: r["response"]["emitted"][-1].update(unix_s=109),
            lambda r: r["response"].update(committed=True),
        ):
            with self.subTest(change=change):
                result = self.check(lambda data, change=change: change(data["rows"][0]))
                self.assertTrue(result["mechanism_passed"])
                self.assertFalse(result["loaded_drain_coverage"])
                self.assertFalse(result["passed"])

    def test_non_completed_conversion_still_fails_mechanism(self):
        result = self.check(
            lambda data: data["step"]["plan"].update(state="RECOVERY_REQUIRED")
        )
        self.assertFalse(result["mechanism_passed"])
        self.assertFalse(result["passed"])

    def test_saved_profile_cannot_be_bypassed_by_offline_default(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = fixture(root)
            data["sessions"] = []
            (root / "instances.json").write_text(
                json.dumps(dict(diagnostic_arrival_profile=probe.PROFILE))
            )
            result = assess(root, data, required=False)
            self.assertTrue(result["migration_coverage_required"])
            self.assertFalse(result["passed"])


class WorkloadTests(unittest.TestCase):
    def inputs(self):
        return dict(requests={"6144": dict(model="cpu", prompt=list(range(128)))})

    def test_recipe_is_frozen_sized_and_keeps_background_autonomous(self):
        first = probe.recipe(self.inputs())
        self.assertEqual(first, probe.recipe(self.inputs()))
        self.assertEqual(
            (len(first["jobs"]), first["planned_output_tokens"]), (366, 95232)
        )
        pinned = [r for r in first["jobs"] if r["route"] is not None]
        self.assertEqual(
            [(r["at_s"], r["route"]) for r in pinned],
            [
                (30, probe.INITIAL_A),
                (32, probe.INITIAL_A),
                (330, probe.NEW_A[0]),
                (332, probe.NEW_A[0]),
                (450, probe.NEW_A[1]),
                (452, probe.NEW_A[1]),
            ],
        )
        self.assertTrue(
            all(
                len(r["request"]["prompt"]) == 16384
                and r["request"]["max_tokens"] == 8192
                for r in pinned
            )
        )

    def test_declared_future_arrivals_do_not_register_future_instances(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = canonical_plan(Path(folder), "L2C")
            six.configure(plan)
            before = copy.deepcopy(plan["instances"]["instances"])
            probe.configure(plan)
            self.assertEqual(plan["instances"]["instances"], before)
            args = NS(
                source_context_limit=32768,
                target_context_limit=32768,
                scheduler="unified",
            )
            pilot.validate_arrivals(
                probe.recipe(self.inputs())["jobs"],
                args,
                NS(dynamic_conversion=True),
                plan["instances"],
            )
            with self.assertRaises(ValueError):
                pilot.validate_arrivals(
                    probe.recipe(self.inputs())["jobs"],
                    args,
                    NS(dynamic_conversion=False),
                    plan["instances"],
                )
            plan["instances"].pop("diagnostic_arrival_profile")
            with self.assertRaises(ValueError):
                pilot.validate_arrivals(
                    probe.recipe(self.inputs())["jobs"],
                    args,
                    NS(dynamic_conversion=True),
                    plan["instances"],
                )

    def test_unknown_future_id_and_context_overflow_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = canonical_plan(Path(folder), "L2C")
            manifest = plan["instances"]
            manifest["diagnostic_arrival_profile"] = probe.PROFILE
            args = NS(
                source_context_limit=32768,
                target_context_limit=32768,
                scheduler="unified",
            )
            for route, prompt in [("A-g9-tp1", 16384), (probe.NEW_A[0], 32768)]:
                with (
                    self.subTest(route=route, prompt=prompt),
                    self.assertRaises(ValueError),
                ):
                    pilot.validate_arrivals(
                        [
                            dict(
                                id="x",
                                at_s=330,
                                route=route,
                                request=dict(prompt=[1] * prompt, max_tokens=8192),
                            )
                        ],
                        args,
                        NS(dynamic_conversion=True),
                        manifest,
                    )

    def test_cli_probe_is_scoped_to_the_declared_split_and_trigger(self):
        base = [
            "--model",
            "m",
            "--output",
            "unused",
            "--expected-revision",
            "CPU",
            "--cards",
            "6",
            "--static-layout",
            "L2C",
            "--conversion",
            "L2C-L1AC",
            "--long-burst-pair",
            "--current-machine",
            "--expanded-context",
            "--scheduler",
            "unified",
            "--hold-queue",
            "--kv-short-exempt",
            "--overload-migrate",
            "--migrate-current-length",
            "--six-split-probe",
            "--conversion-trigger-s",
            "110",
            "--validate-only",
        ]
        base += ["--repetitions", "1"]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(base), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["runs"], 1)
        self.assertEqual(value["recipes"][0]["scenario"], probe.PROFILE)
        self.assertEqual(value["recipes"][0]["planned_output_tokens"], 95232)
        for option in ("60",):
            invalid = list(base)
            invalid[invalid.index("110")] = option
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                runner.main(invalid)
