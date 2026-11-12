# SPDX-License-Identifier: Apache-2.0
"""Hang diagnostics (11-12) and the loaded engine-reuse stress scene."""

import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.bridge_tp.test_six_sleep_smoke import BASE, write_scene
from tools.bridge_tp import hang_watch
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp import six_sleep_smoke as smoke

ROOT = Path(__file__).resolve().parents[2]
DUMP = ROOT / "tools/bridge_tp/stack_dump"


def fake_proc(root, pid, *, environ, cmdline, stack="[<0>] nccl wait\n"):
    entry = root / str(pid)
    entry.mkdir(parents=True)
    (entry / "environ").write_bytes(b"\0".join(environ) + b"\0")
    (entry / "cmdline").write_bytes(b"\0".join(cmdline) + b"\0")
    (entry / "status").write_text("State:\tS (sleeping)\n")
    (entry / "wchan").write_text("futex_wait_queue")
    (entry / "stack").write_text(stack)


class HangWatchTests(unittest.TestCase):
    def test_scan_reports_only_new_matching_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "scene" / "A-g6-tp2.log"
            log.parent.mkdir()
            log.write_text("ok\n")
            offsets = {}
            self.assertEqual(hang_watch.scan(folder, offsets), [])
            with log.open("a") as stream:
                stream.write(f"x {hang_watch.PATTERN} in 60 seconds\n")
            found = hang_watch.scan(folder, offsets)
            self.assertEqual([name for name, _ in found], ["A-g6-tp2.log"])
            self.assertEqual(hang_watch.scan(folder, offsets), [])

    def test_capture_signals_only_registered_python_processes(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = Path(folder) / "proc"
            fake_proc(
                proc,
                101,
                environ=[b"PY_STACK_DUMP_ON_USR1=1", b"BRIDGETP_PRESSURE_POOL=A-g6-tp2"],
                cmdline=[b"/venv/bin/python", b"-m", b"vllm"],
            )
            # Same variable but not Python (the sampling shell): never signalled.
            fake_proc(proc, 102, environ=[b"PY_STACK_DUMP_ON_USR1=1"], cmdline=[b"bash"])
            fake_proc(proc, 103, environ=[b"OTHER=1"], cmdline=[b"python3"])
            sent = []
            out = hang_watch.capture(
                Path(folder) / "out",
                1,
                [("A-g6-tp2.log", "line")],
                kill=lambda pid, sig: sent.append((pid, sig)),
                proc=proc,
            )
            usr1 = getattr(signal, "SIGUSR1", None)
            self.assertEqual(sent, [(101, usr1)] if usr1 is not None else [])
            value = json.loads((out / "processes.json").read_text())
            self.assertEqual([p["pid"] for p in value["processes"]], [101])
            self.assertEqual(value["processes"][0]["pool"], "A-g6-tp2")
            self.assertIn("nccl wait", value["processes"][0]["kernel_stack"])
            self.assertTrue((out / "nvidia_smi.txt").exists())


@unittest.skipUnless(hasattr(signal, "SIGUSR1"), "SIGUSR1 is POSIX only")
class StackDumpTests(unittest.TestCase):
    def test_sigusr1_prints_all_thread_stacks_and_chains(self):
        with tempfile.TemporaryDirectory() as folder:
            other = Path(folder)
            (other / "sitecustomize.py").write_text("import os\nos.environ['CHAINED'] = '1'\n")
            code = (
                "import os, signal, threading, time\n"
                "assert os.environ.get('CHAINED') == '1'\n"
                "threading.Thread(target=time.sleep, args=(5,), daemon=True).start()\n"
                "os.kill(os.getpid(), signal.SIGUSR1)\n"
                "time.sleep(0.5)\n"
            )
            env = dict(
                os.environ,
                PYTHONPATH=os.pathsep.join([str(DUMP), str(other)]),
                PY_STACK_DUMP_ON_USR1="1",
            )
            done = subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True, text=True
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("Thread", done.stderr)
            self.assertGreaterEqual(done.stderr.count("File"), 2)

    def test_registered_only_with_the_variable(self):
        code = (
            "import faulthandler, signal\n"
            "print(faulthandler.unregister(signal.SIGUSR1))\n"
        )
        for flag, expected in (("1", "True"), (None, "False")):
            env = dict(os.environ, PYTHONPATH=str(DUMP))
            env.pop("PY_STACK_DUMP_ON_USR1", None)
            if flag:
                env["PY_STACK_DUMP_ON_USR1"] = flag
            done = subprocess.run(
                [sys.executable, "-c", code], env=env, capture_output=True, text=True
            )
            self.assertEqual(done.stdout.strip(), expected, done.stderr)


class StressSceneTests(unittest.TestCase):
    def test_stress_profile_has_six_loaded_rounds(self):
        steps = smoke.sequence("stress")
        self.assertEqual(len(steps), 12)
        self.assertEqual([s["target_mode"] for s in steps], ["TP2", "TP1"] * 6)
        triggers = [s["trigger_s"] for s in steps]
        self.assertEqual(triggers, sorted(triggers))
        jobs = smoke.recipe(
            dict(requests={"6144": dict(prompt=list(range(200)), max_tokens=1)}),
            "stress",
        )["jobs"]
        longs = sorted(j["at_s"] for j in jobs if j["workload_role"] != "short_probe")
        self.assertEqual(len(longs), 36)
        for k, merge in enumerate(triggers[0::2]):
            mine = longs[6 * k : 6 * k + 6]
            self.assertTrue(all(merge - 20 <= t < merge for t in mine))
        shorts = [j for j in jobs if j["workload_role"] == "short_probe"]
        self.assertGreater(len(shorts), 1300)
        self.assertLess(max(j["at_s"] for j in jobs), smoke.STRESS_HORIZON_S)

    def test_cli_runs_the_stress_workload_only_with_the_smoke(self):
        argv = list(BASE)
        argv[argv.index("sleep")] = "standby"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(argv + ["--six-sleep-profile", "stress"]), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["recipes"][0]["scenario"], smoke.STRESS_PROFILE)
        plain = [a for a in argv if a != "--six-sleep-smoke"]
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(plain + ["--six-sleep-profile", "stress"])

    def test_heavy_profile_pins_recomputable_longs_and_builds_a_backlog(self):
        from types import SimpleNamespace as NS

        from tools.bridge_tp import run_pressure_balanced_pilot as pilot

        steps = smoke.sequence("heavy")
        self.assertEqual([s["target_mode"] for s in steps], ["TP2", "TP1"] * 4)
        jobs = smoke.recipe(
            dict(requests={"6144": dict(prompt=list(range(200)), max_tokens=1)}),
            "heavy",
        )["jobs"]
        merges = [s["trigger_s"] for s in steps if s["target_mode"] == "TP2"]
        longs = [j for j in jobs if j["workload_role"] != "short_probe"]
        self.assertEqual(len(longs), 8)
        for k, merge in enumerate(merges):
            mine = [j for j in longs if j["workload_phase"] == f"recompute_{k}"]
            self.assertEqual(len(mine), 2)
            # Pinned, still running at the 45 s check, both resident in a TP1.
            self.assertTrue(all(j["route"] == smoke.HEAVY_PINNED_POOL for j in mine))
            self.assertTrue(all(merge - 20 <= j["at_s"] < merge for j in mine))
            self.assertLessEqual(
                sum(len(j["request"]["prompt"]) + 1500 for j in mine), 37488
            )
            window = [
                j
                for j in jobs
                if j["workload_role"] == "short_probe"
                and merge <= j["at_s"] < merge + 60
            ]
            self.assertEqual(len(window), 60 * smoke.HEAVY_BURST_RATE)
        manifest = dict(
            legacy_pool_aliases={"tp1": "A-g1-tp1", "tp4": "B-g2-tp2"},
            runtime_pool_ids={"tp1": "A-g1-tp1", "tp4": "B-g2-tp2"},
            instances=[],
        )
        args = NS(source_context_limit=32768, target_context_limit=32768)
        pilot.validate_arrivals(jobs, args, NS(dynamic_conversion=True), manifest)

    def test_cli_runs_the_heavy_workload(self):
        argv = list(BASE)
        argv[argv.index("sleep")] = "standby"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(runner.main(argv + ["--six-sleep-profile", "heavy"]), 0)
        value = json.loads(output.getvalue())
        self.assertEqual(value["recipes"][0]["scenario"], smoke.HEAVY_PROFILE)

    def test_check_accepts_three_steps_and_rejects_partial_rounds(self):
        with tempfile.TemporaryDirectory() as folder:
            value, code = smoke.check(write_scene(Path(folder), standby=True))
            self.assertEqual(code, 0, value)
            self.assertEqual(len(value["fence_to_route_s"]), 3)


if __name__ == "__main__":
    unittest.main()
