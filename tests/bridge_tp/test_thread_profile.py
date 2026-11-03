# SPDX-License-Identifier: Apache-2.0
"""Optional controller thread profile: off the serving path, groups and stacks."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from tools.bridge_tp.pressure_cpu import load

profile = load("thread_profile")


def busy(stop):
    while not stop.is_set():
        sum(range(1000))


class ThreadProfileTests(unittest.TestCase):
    def test_pool_members_share_a_group(self):
        self.assertEqual(
            profile.group_name("ThreadPoolExecutor-0_17"), "ThreadPoolExecutor"
        )
        self.assertEqual(profile.group_name("MainThread"), "MainThread")
        self.assertEqual(profile.group_name("12"), "unnamed")

    def test_reports_busy_and_idle_threads_by_group(self):
        stop = threading.Event()
        workers = [
            threading.Thread(target=busy, args=(stop,), name="spin-1", daemon=True),
            threading.Thread(target=stop.wait, name="idle-1", daemon=True),
        ]
        for worker in workers:
            worker.start()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.jsonl"
            with profile.ThreadProfile(path, sample_s=0.01, report_s=0.1):
                time.sleep(0.25)
            stop.set()
            rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertGreaterEqual(len(rows), 2)
        self.assertTrue(all("thread_cpu_s" in row for row in rows))
        busy_groups = {s["group"] for row in rows for s in row["busy_stacks"]}
        idle_groups = {s["group"] for row in rows for s in row["idle_stacks"]}
        self.assertIn("spin", busy_groups)
        self.assertIn("idle", idle_groups)
        self.assertNotIn("thread-profile", busy_groups)

    def test_thread_cpu_reads_proc_style_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "17").mkdir()
            fields = ["0"] * 20
            fields[11], fields[12] = "150", "50"
            (root / "17" / "stat").write_text("17 (py thon) " + " ".join(fields))
            self.assertEqual(
                profile.thread_cpu_seconds(root), {17: 200 / profile.TICKS}
            )
        self.assertEqual(profile.thread_cpu_seconds(Path(directory) / "gone"), {})


if __name__ == "__main__":
    unittest.main()
