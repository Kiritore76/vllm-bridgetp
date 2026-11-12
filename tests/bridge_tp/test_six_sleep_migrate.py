# SPDX-License-Identifier: Apache-2.0
"""Engine-reuse migrate profile: C pressure migrates into the woken A TP2."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from tests.bridge_tp.test_six_sleep_reuse import reuse_plan
from tests.bridge_tp.test_six_sleep_smoke import BASE, TP2, write_scene
from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp import six_sleep_smoke as smoke

INPUTS = dict(requests={"6144": dict(prompt=list(range(200)), max_tokens=1)})


def migrate_argv(*extra):
    argv = list(BASE)
    argv[argv.index("sleep")] = "standby"
    return argv + ["--six-sleep-profile", "migrate", "--history-direct-to-kv", *extra]


class CliTests(unittest.TestCase):
    def test_cli_runs_the_migrate_workload(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(migrate_argv()), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["recipes"][0]["scenario"], smoke.MIGRATE_PROFILE)

    def test_cli_requires_standby_and_the_direct_path(self):
        sleep = migrate_argv()
        sleep[sleep.index("standby")] = "sleep"
        plain = [a for a in migrate_argv() if a != "--history-direct-to-kv"]
        for argv in (sleep, plain):
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                runner.main(argv)


class WorkloadTests(unittest.TestCase):
    def test_heavy_plus_two_c_longs_after_every_merge(self):
        heavy = smoke.recipe(INPUTS, "heavy")["jobs"]
        jobs = smoke.recipe(INPUTS, "migrate")["jobs"]
        self.assertEqual(len(jobs), len(heavy) + 8)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
        self.assertTrue(all(j["id"].startswith(smoke.MIGRATE_PROFILE) for j in jobs))
        merges = [at for mode, at in smoke.HEAVY_SEQUENCE if mode == "TP2"]
        for k, merge in enumerate(merges):
            mine = [j for j in jobs if j["workload_phase"] == f"migrate_{k}"]
            self.assertEqual(sorted(j["at_s"] for j in mine), [merge - 30, merge - 28])
            # Before the short burst, so C-g4 still has room for both prompts.
            self.assertTrue(all(j["at_s"] < merge - 5 for j in mine))
            self.assertTrue(all(j["route"] == smoke.MIGRATE_SOURCE_POOL for j in mine))
            # Together beyond one TP1's KV (37,488 tokens): C must migrate.
            self.assertGreater(
                sum(len(j["request"]["prompt"]) + j["request"]["max_tokens"]
                    for j in mine),
                37488,
            )
        self.assertEqual(smoke.sequence("migrate"), smoke.sequence("heavy"))


class WiringTests(unittest.TestCase):
    def test_configure_forces_a_standby_tp2_and_marks_the_routes(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = reuse_plan(Path(folder), "L1AC", "standby")
            target = smoke.configure_migrate(plan)
            standby = {e["instance_id"]: e for e in plan["standby_engines"]}
            self.assertEqual((standby[target]["bundle"], standby[target]["tp"]),
                             ("A", 2))
            argv = plan["controller_argv"]
            self.assertEqual(argv[argv.index("--force-migration-target") + 1], target)
            self.assertEqual(
                plan["instances"]["diagnostic_arrival_profile"], smoke.MIGRATE_PROFILE
            )
            with self.assertRaises(ValueError):
                smoke.configure_migrate(plan)  # forced twice
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                smoke.configure_migrate(reuse_plan(Path(folder), "L1AC", "sleep"))

    def test_pinned_c_routes_pass_only_for_dynamic_unified(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = reuse_plan(Path(folder), "L1AC", "standby")
            smoke.configure_migrate(plan)
            jobs = smoke.recipe(INPUTS, "migrate")["jobs"]
            args = NS(source_context_limit=32768, target_context_limit=32768,
                      scheduler="unified")
            pilot.validate_arrivals(jobs, args, NS(dynamic_conversion=True),
                                    plan["instances"])
            with self.assertRaises(ValueError):
                pilot.validate_arrivals(jobs, args, NS(dynamic_conversion=False),
                                        plan["instances"])
            plan["instances"].pop("diagnostic_arrival_profile")
            with self.assertRaises(ValueError):
                pilot.validate_arrivals(jobs, args, NS(dynamic_conversion=True),
                                        plan["instances"])

    def test_dynamic_controller_accepts_a_forced_runtime_receiver(self):
        source = Path(pilot.__file__).read_text(encoding="utf-8")
        check = source[source.index('force = getattr(args, "force_migration_target"'):]
        check = check[: check.index("raise ValueError")]
        self.assertIn('not getattr(controller, "dynamic_conversion", False)', check)


def add_migration(trial, *, state="COMMITTED", path="DIRECT_TO_KV", memory=True):
    scene = trial / "scene"
    rows_path = scene / "controller" / "pressure_request_results.json"
    rows = json.loads(rows_path.read_text())
    for row in rows:
        row["job_id"] = f"{smoke.MIGRATE_PROFILE}-{row['job_id']}"
    rows.append(dict(job_id=f"{smoke.MIGRATE_PROFILE}-c0-80",
                     request=dict(max_tokens=1),
                     response=dict(emitted=[{}], handoff_stall_s=0.25),
                     source_error=None, recompute=None))
    rows_path.write_text(json.dumps(rows))
    session = scene / "controller" / "migration" / TP2 / "sessions" / "migration-0001"
    (session / "gpu_initial_receipts").mkdir(parents=True)
    (session / "takeover_state.json").write_text(
        json.dumps(dict(state=state, updated_unix_s=1180.0))
    )
    for rank in (0, 1):
        (session / "gpu_initial_receipts" / f"tp_rank_{rank}.json").write_text(
            json.dumps(dict(history_path=path, direct_copy=dict(chunks=3)))
        )
    if memory:
        (scene / "controller" / f"gpu_memory_{TP2}_tp0.jsonl").write_text(
            "\n".join(json.dumps(dict(max_reserved_bytes=b, device_free_bytes=f))
                      for b, f in ((2**30, 2**31), (2 * 2**30, 2**30))) + "\n"
        )
    return trial


class CheckTests(unittest.TestCase):
    def test_committed_direct_migration_into_the_woken_tp2_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            trial = add_migration(write_scene(Path(folder), standby=True))
            value, code = smoke.check(trial)
            self.assertEqual(code, 0, value)
            migration = value["migration"]
            self.assertEqual(migration["target"], TP2)
            self.assertEqual(migration["committed_into_woken_tp2"], 1)
            self.assertEqual(migration["paths"], ["DIRECT_TO_KV"])
            self.assertEqual(migration["handoff_stall_s"], dict(n=1, max=0.25))
            memory = migration["gpu_memory"][f"gpu_memory_{TP2}_tp0"]
            self.assertEqual(memory["max_reserved_mib"], 2048.0)
            self.assertEqual(memory["min_device_free_mib"], 1024.0)

    def test_staged_or_uncommitted_migration_fails(self):
        for kwargs in (dict(path="STAGED_INJECT"), dict(state="CANCELLED")):
            with tempfile.TemporaryDirectory() as folder:
                trial = add_migration(write_scene(Path(folder), standby=True), **kwargs)
                value, code = smoke.check(trial)
                self.assertEqual(code, 1, (kwargs, value))

    def test_other_profiles_need_no_migration(self):
        with tempfile.TemporaryDirectory() as folder:
            value, code = smoke.check(write_scene(Path(folder), standby=True))
            self.assertEqual(code, 0, value)
            self.assertNotIn("migration", value)


if __name__ == "__main__":
    unittest.main()
