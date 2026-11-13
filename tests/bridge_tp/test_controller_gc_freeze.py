# SPDX-License-Identifier: Apache-2.0
"""Controller GC freeze (11-13): full collections must not pause admission.

guard40 R4/split5: every admission that waited >0.5 s overlapped a
generation-2 collection of the controller (up to 1.1 s late in the run).
"""

import gc
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.pressure_cpu import load

g = load("gc_freeze")
PILOT = Path(__file__).resolve().parents[2] / "tools/bridge_tp/run_pressure_balanced_pilot.py"


class FakeGc:
    def __init__(self):
        self.calls = []
        self.frozen = 0

    def collect(self, generation):
        self.calls.append(("collect", generation))
        return 3

    def freeze(self):
        self.calls.append(("freeze",))
        self.frozen += 100

    def get_freeze_count(self):
        self.calls.append(("get_freeze_count",))
        return self.frozen

    def get_count(self):
        return (5, 2, 1)

    def unfreeze(self):
        self.calls.append(("unfreeze",))
        self.frozen = 0


class FreezerTests(unittest.TestCase):
    def setUp(self):
        self.now = [0.0]
        self.rows = []
        self.gc = FakeGc()

    def freezer(self, interval):
        return g.GcFreezer(
            interval, clock=lambda: self.now[0], record=self.rows.append, collector=self.gc
        )

    def test_start_collects_fully_then_freezes(self):
        f = self.freezer(30)
        f.tick()  # before start: nothing
        self.assertEqual(self.gc.calls, [])
        f.start()
        self.assertEqual(
            self.gc.calls, [("collect", 2), ("freeze",), ("get_freeze_count",)]
        )
        row = self.rows[0]
        self.assertEqual((row["kind"], row["reason"]), ("controller_gc_freeze", "start"))
        self.assertEqual((row["collected"], row["frozen_total"]), (3, 100))
        self.assertEqual(row["young_counts"], [5, 2, 1])
        self.assertGreaterEqual(row["duration_s"], 0)

    def test_periodic_freeze_uses_a_young_collection_once_per_interval(self):
        f = self.freezer(30)
        f.start()
        self.gc.calls.clear()
        self.now[0] = 29.9
        f.tick()
        self.assertEqual(self.gc.calls, [])
        self.now[0] = 30.0
        f.tick()
        self.assertEqual(self.gc.calls, [("collect", 1), ("freeze",)])
        self.now[0] = 45.0
        f.tick()  # interval counts from the last freeze
        self.assertEqual(len(self.gc.calls), 2)
        self.assertEqual([r["reason"] for r in self.rows], ["start", "periodic"])
        self.assertEqual(f.freezes, 2)

    def test_periodic_freezes_never_walk_the_permanent_generation(self):
        """R5: get_freeze_count() walks every frozen object (0.2-0.4 s per
        periodic freeze on the server at 1M objects)."""
        f = self.freezer(30)
        f.start()
        self.gc.calls.clear()
        for second in range(30, 3000, 30):
            self.now[0] = second
            f.tick()
        self.assertNotIn(("get_freeze_count",), self.gc.calls)
        self.assertTrue(all("frozen_total" not in r for r in self.rows[1:]))

    def test_close_unfreezes_only_after_a_freeze(self):
        f = self.freezer(30)
        f.close()
        self.assertEqual(self.gc.calls, [])
        f.start()
        f.close()
        self.assertEqual(self.gc.calls[-2:], [("get_freeze_count",), ("unfreeze",)])
        self.assertEqual(self.rows[-1]["reason"], "close")
        self.now[0] = 100
        f.tick()  # closed: no further freezes
        self.assertEqual(self.gc.calls[-1], ("unfreeze",))

    def test_zero_disables_and_negative_is_rejected(self):
        f = self.freezer(0)
        f.start()
        self.now[0] = 1e6
        f.tick()
        self.assertEqual(self.gc.calls, [])
        self.assertFalse(f.enabled)
        with self.assertRaises(ValueError):
            self.freezer(-1)

    def test_environment_sets_the_interval(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(g.GcFreezer.from_env().interval_s, g.DEFAULT_INTERVAL_S)
        with patch.dict(os.environ, {g.ENV: "0"}):
            self.assertFalse(g.GcFreezer.from_env().enabled)

    def test_real_freeze_takes_objects_out_of_full_collections(self):
        before = gc.get_freeze_count()
        keep = [[i] for i in range(20000)]  # lists are always GC-tracked
        try:
            f = g.GcFreezer(30, record=self.rows.append)
            f.start()
            self.assertGreaterEqual(gc.get_freeze_count() - before, len(keep))
        finally:
            gc.unfreeze()  # leave the test process as it was
            del keep


class WiringTests(unittest.TestCase):
    def test_controller_freezes_before_arrivals_and_ticks_in_its_loop(self):
        text = PILOT.read_text(encoding="utf-8")
        start = text.index("gc_freezer.start()")
        self.assertLess(text.index('load("gc_freeze").GcFreezer.from_env('), start)
        # Before the workload clock: the full collection delays no arrival.
        self.assertLess(start, text.index("    started = time.monotonic()"))
        self.assertLess(start, text.index("dispatcher.start()"))
        loop = text[text.index("while time.monotonic() - started < args.max_seconds:"):]
        self.assertLess(loop.index("gc_freezer.tick()"), loop.index("apply_layout_control()"))
        final = text[text.index("        if gc_freezer is not None:"):]
        self.assertIn("gc_freezer.close()", final.splitlines()[2])


if __name__ == "__main__":
    unittest.main()
