# SPDX-License-Identifier: Apache-2.0
"""Engine-reuse smoke: scoped CLI, workload and the evidence check."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp import six_sleep_smoke as smoke
from tools.bridge_tp.conversion_sequence import validate

BASE = [
    "--model",
    "m",
    "--output",
    "unused",
    "--expected-revision",
    "CPU",
    "--cards",
    "6",
    "--static-layout",
    "L1AC",
    "--conversion",
    "L1AC-L2C",
    "--long-burst-pair",
    "--current-machine",
    "--expanded-context",
    "--scheduler",
    "unified",
    "--hold-queue",
    "--kv-short-exempt",
    "--overload-migrate",
    "--migrate-current-length",
    "--conversion-trigger-s",
    "110",
    "--conversion-prestart",
    "--conversion-recompute",
    "--conversion-engine-reuse",
    "sleep",
    "--six-sleep-smoke",
    "--repetitions",
    "1",
    "--validate-only",
]

TP1 = ["A-g1-tp1", "A-g2-tp1"]
TP2 = "A-g7-tp2"


class CliTests(unittest.TestCase):
    def test_cli_runs_the_smoke_workload(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(BASE), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["recipes"][0]["scenario"], smoke.PROFILE)
        self.assertEqual(
            value["recipes"][0]["planned_output_tokens"],
            300 * 128 + 6 * 8192 + 6 * 4096,
        )

    def test_cli_requires_reuse_trigger_and_layout(self):
        for drop in ("--conversion-engine-reuse", "--long-burst-pair"):
            argv = list(BASE)
            index = argv.index(drop)
            del argv[index : index + (2 if drop == "--conversion-engine-reuse" else 1)]
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                runner.main(argv)
        argv = list(BASE)
        argv[argv.index("110")] = "120"
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(argv)

    def test_sequence_is_a_valid_six_card_request(self):
        steps = smoke.sequence()
        self.assertEqual([s["target_mode"] for s in steps], ["TP2", "TP1", "TP2"])
        validate(
            dict(
                profile="six_card",
                base_port=8400,
                engine_reuse="sleep",
                recompute=True,
                prestart=True,
                sequence=steps,
            )
        )

    def test_long_bursts_precede_both_merges(self):
        jobs = smoke.recipe(
            dict(requests={"6144": dict(prompt=list(range(200)), max_tokens=1)})
        )["jobs"]
        longs = sorted(j["at_s"] for j in jobs if j["workload_role"] != "short_probe")
        self.assertEqual(len(longs), 12)
        merges = [at for mode, at in smoke.SEQUENCE if mode == "TP2"]
        self.assertLess(max(longs[:6]), merges[0])
        self.assertLess(max(longs[6:]), merges[1])
        self.assertGreater(min(longs[6:]), smoke.SEQUENCE[1][1])


def write_scene(root, *, split_woken=TP1, remerge_cold=(), fallback=False, work=True):
    scene = root / "trial" / "scene"
    (scene / "controller").mkdir(parents=True)
    origin = 1000.0
    steps, events = [], []

    def step(pid, mode, begin, end, slept, cold, woken):
        steps.append(
            dict(
                plan_id=pid,
                triggered_unix_s=origin + begin,
                completed_unix_s=origin + end,
                plan=dict(state="COMPLETED", target_bundle_mode=mode),
                layout={},
            )
        )
        events.append(dict(kind="FENCED", plan_id=pid, unix_s=origin + begin + 5))
        events.append(dict(kind="SLEPT", instances=slept, sleep_s={}, wait_s=1.4,
                           used_mib={"2": 900}, unix_s=origin + begin + 6))
        events.append(dict(kind="RESOURCES_RELEASED", plan_id=pid,
                           unix_s=origin + begin + 6))
        for name in cold:
            events.append(dict(kind="LOAD_START", instance=name,
                               unix_s=origin + begin + 7))
        for name in woken:
            events.append(dict(kind="WAKE_START", instance=name,
                               unix_s=origin + begin + 7))
            events.append(dict(kind="WOKEN", instance=name, wake_s=1.8,
                               unix_s=origin + begin + 9))
        if fallback:
            events.append(dict(kind="SLEEP_FALLBACK", unix_s=origin + begin + 6))
        events.append(dict(kind="ROUTE_PUBLISHED", plan_id=pid,
                           unix_s=origin + end - 1))

    step("s1", "TP2", 110, 200, TP1, [TP2], [])
    step("s2", "TP1", 240, 300, [TP2], [], list(split_woken))
    step("s3", "TP2", 440, 520, TP1, list(remerge_cold), [] if remerge_cold else [TP2])
    (scene / "conversion_sequence_result.json").write_text(json.dumps(dict(steps=steps)))
    (scene / "conversion_events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )
    (scene / "controller" / "layout_status.json").write_text(
        json.dumps(dict(workload_origin_unix_s=origin))
    )
    (scene / "conversion_sequence_acceptance.json").write_text(
        json.dumps(dict(mechanism_passed=True))
    )
    rows = [
        dict(
            job_id="j1",
            request=dict(max_tokens=2),
            response=dict(emitted=[{}, {}]),
            source_error=None,
            recompute=dict(state="COMPLETED", source_pool=TP1[0], target_pool=TP2)
            if work
            else None,
        )
    ]
    (scene / "controller" / "pressure_request_results.json").write_text(
        json.dumps(rows)
    )
    (scene / "controller" / "pressure_audit.jsonl").write_text("")
    return root / "trial"


class CheckTests(unittest.TestCase):
    def test_full_reuse_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            value, code = smoke.check(write_scene(Path(folder)))
            self.assertEqual(code, 0, value)
            self.assertTrue(value["reuse_exercised"])
            self.assertEqual(value["steps"][1]["woken"], TP1)

    def test_no_work_on_woken_engines_is_coverage_missing(self):
        with tempfile.TemporaryDirectory() as folder:
            _, code = smoke.check(write_scene(Path(folder), work=False))
            self.assertEqual(code, 2)

    def test_cold_start_or_fallback_or_wrong_wake_fails(self):
        for kwargs in (
            dict(remerge_cold=["A-g9-tp2"]),
            dict(fallback=True),
            dict(split_woken=["A-g1-tp1"]),
        ):
            with tempfile.TemporaryDirectory() as folder:
                value, code = smoke.check(write_scene(Path(folder), **kwargs))
                self.assertEqual(code, 1, (kwargs, value))


if __name__ == "__main__":
    unittest.main()
