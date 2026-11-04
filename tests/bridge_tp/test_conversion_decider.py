# SPDX-License-Identifier: Apache-2.0
"""Rule-based conversion decider (record-only first) on CPU."""

import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from tools.bridge_tp import conversion_decider as d
from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp.pressure_cpu import load

instance_pool = load("instance_pool")

# L1A on A={2,3}, B={0,1}: two TP1s (A) and one TP2 (B).
BUNDLES = {"A": 1, "B": 2}
MEMBERS = {"A": ["A-g1-tp1", "A-g3-tp1"], "B": ["B-g2-tp2"]}
EDGES = {"L1A": [["A", "TP2"], ["B", "TP1"]]}


def status(
    short=0.0,
    long_arrivals=0,
    capacity=0.2,
    tp1_capacity=None,
    waiting=0,
    deep=None,
    instances=None,
):
    """``short`` req/s; ``deep``: {instance: deep long requests in flight}."""

    def row(instance, tp):
        cap = tp1_capacity if tp == 1 and tp1_capacity is not None else capacity
        requests = [
            dict(
                request_id=f"{instance}-{n}",
                prompt_tokens=16384,
                max_output_tokens=8192,
                generated=10,
            )
            for n in range((deep or {}).get(instance, 0))
        ]
        return dict(
            lifecycle="ACTIVE",
            pressure=dict(capacity=cap, zone="SPARE"),
            tp_degree=tp,
            inventory=dict(waiting=waiting if tp > 1 else 0),
            requests=requests,
        )

    return dict(
        unix_s=time.time(),
        instances={
            name: row(name, tp)
            for name, tp in (
                instances or {"A-g1-tp1": 1, "A-g3-tp1": 1, "B-g2-tp2": 2}
            ).items()
        },
        workload=dict(
            window_s=60,
            arrivals=int(short * 60) + long_arrivals,
            long_arrivals=long_arrivals,
            migration_starts=[],
        ),
    )


L0 = {"A": 1, "B": 1}
L0_MEMBERS = {"A": ["A1", "A2"], "B": ["B1", "B2"]}
L0_INSTANCES = dict(A1=1, A2=1, B1=1, B2=1)
L2 = {"A": 2, "B": 2}
L2_MEMBERS = {"A": ["A-tp2"], "B": ["B-tp2"]}
L2_INSTANCES = {"A-tp2": 2, "B-tp2": 2}
# Seven deep long requests on L1A: above the 9817 blocks of L1A -> L2.
MANY_LONG = {"A-g1-tp1": 2, "A-g3-tp1": 2, "B-g2-tp2": 3}


def run(
    decider, value, seconds, step=1.0, start=1000.0, bundles=BUNDLES, members=MEMBERS
):
    records, now = [], start
    while now <= start + seconds:
        records.append(decider.observe(value, bundles, members, now))
        now += step
    return records


def edge(record):
    proposal = record["proposal"]
    return proposal["bundle"], proposal["target_mode"]


class CurrentLengthTests(unittest.TestCase):
    def decider(self):
        return d.ConversionDecider(d.EDGES, dict(current_length=True))

    def signals(self, decider, value, now=1000.0):
        return decider.signals(value, BUNDLES, MEMBERS, now)

    def test_a_short_request_with_a_large_cap_is_not_long_yet(self):
        value = status()
        value["instances"]["B-g2-tp2"]["requests"] = [
            dict(request_id="a", prompt_tokens=512, max_output_tokens=8192, generated=9)
        ]
        s = self.signals(self.decider(), value)
        self.assertEqual((s["long_blocks"], s["long_arrivals"]), (0, 0))
        self.assertGreater(
            self.signals(d.ConversionDecider(d.EDGES), value)["long_blocks"], 0
        )

    def test_it_counts_the_current_length_once_a_request_turns_long(self):
        value = status()
        value["instances"]["B-g2-tp2"]["requests"] = [
            dict(
                request_id="a",
                prompt_tokens=512,
                max_output_tokens=8192,
                generated=7680,
            )
        ]
        s = self.signals(self.decider(), value)
        self.assertEqual((s["long_blocks"], s["long_arrivals"]), (512, 1))

    def test_a_long_request_is_new_only_for_one_window(self):
        decider = self.decider()
        value = status(deep={"B-g2-tp2": 1}, long_arrivals=5)
        self.assertEqual(self.signals(decider, value, 1000.0)["long_arrivals"], 1)
        self.assertEqual(self.signals(decider, value, 1059.0)["long_arrivals"], 1)
        self.assertEqual(self.signals(decider, value, 1061.0)["long_arrivals"], 0)
        # Still in flight and first seen long-ago: it does not count again.
        self.assertEqual(self.signals(decider, value, 1062.0)["long_arrivals"], 1)

    def test_held_long_prompts_count_as_demand_and_as_new(self):
        value = status()
        value["workload"]["held_long_prompt_tokens"] = [16384, 16384]
        s = self.signals(self.decider(), value)
        self.assertEqual((s["long_blocks"], s["long_arrivals"]), (2048, 2))

    def test_the_default_still_reads_output_caps(self):
        value = status(deep={"B-g2-tp2": 1})
        self.assertEqual(
            self.signals(d.ConversionDecider(d.EDGES), value)["long_blocks"], 1536
        )


class DeciderRuleTests(unittest.TestCase):
    def test_many_short_few_long_on_l1a_goes_to_l0_after_the_longer_hold(self):
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, status(short=12, deep={"A-g1-tp1": 3}), 130)
        (first,) = [r for r in records if r["proposal"]]
        self.assertEqual(edge(first), ("B", "TP1"))
        self.assertEqual(first["evaluated_unix_s"] - 1000, 120)

    def test_entering_l0_is_strict(self):
        for value, reason in (
            (status(short=10), "short rate within what L1A carries"),
            (
                status(short=12, deep={"A-g1-tp1": 2, "A-g3-tp1": 2}),
                "too many long-request blocks for L0",
            ),
        ):
            decider = d.ConversionDecider(d.EDGES)
            records = run(decider, value, 300)
            self.assertFalse(any(r["proposal"] for r in records), reason)
            self.assertIn(reason, records[-1]["candidates"]["B->TP1"]["failed"])

    def test_a_split_waits_without_limit_for_its_long_requests(self):
        decider = d.ConversionDecider(d.EDGES)
        value = status(short=12, deep={"B-g2-tp2": 1})
        records = run(decider, value, 2000)
        self.assertFalse(any(r["proposal"] for r in records))
        row = records[-1]["candidates"]["B->TP1"]
        self.assertEqual(row["failed"], [])
        self.assertEqual(row["waiting"], ["bundle serves a long request"])
        # Once it is free the split goes ahead at once.
        record = decider.observe(status(short=12), BUNDLES, MEMBERS, 3001)
        self.assertEqual(edge(record), ("B", "TP1"))

    def test_sustained_long_requests_merge_but_a_burst_is_left_to_migration(self):
        decider = d.ConversionDecider(d.EDGES)
        value = status(short=1, long_arrivals=2, deep=MANY_LONG)
        records = run(decider, value, 80)
        (first,) = [r for r in records if r["proposal"]]
        self.assertEqual(edge(first), ("A", "TP2"))
        self.assertEqual(first["evaluated_unix_s"] - 1000, 75)
        # The same blocks with no new long request: a burst.
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, status(short=1, deep=MANY_LONG), 300)
        self.assertFalse(any(r["proposal"] for r in records))
        self.assertIn(
            "no new long request: a burst is left to migration",
            records[-1]["candidates"]["A->TP2"]["failed"],
        )

    def test_merge_to_l2_needs_few_short_and_many_blocks(self):
        for value, reason in (
            (
                status(short=7, long_arrivals=2, deep=MANY_LONG),
                "short rate too high for L2",
            ),
            (
                status(short=1, long_arrivals=2, deep={"B-g2-tp2": 6}),
                "too few long-request blocks",
            ),
        ):
            decider = d.ConversionDecider(d.EDGES)
            records = run(decider, value, 300)
            self.assertFalse(any(r["proposal"] for r in records), reason)
            self.assertIn(reason, records[-1]["candidates"]["A->TP2"]["failed"])

    def test_kv_at_guard_merges_urgently(self):
        decider = d.ConversionDecider(d.EDGES)
        value = status(short=1, long_arrivals=2, deep=MANY_LONG, tp1_capacity=1.05)
        records = run(decider, value, 40)
        first = next(r for r in records if r["proposal"])
        self.assertEqual(first["evaluated_unix_s"] - 1000, 30)
        self.assertTrue(first["candidates"]["A->TP2"]["urgent"])

    def test_short_rate_leaves_l2_urgently_when_it_queues(self):
        for waiting, at in ((0, 75), (3, 30)):
            decider = d.ConversionDecider(d.EDGES)
            value = status(short=8, waiting=waiting, instances=L2_INSTANCES)
            records = run(decider, value, 80, bundles=L2, members=L2_MEMBERS)
            first = next(r for r in records if r["proposal"])
            self.assertEqual(edge(first), ("A", "TP1"))
            self.assertEqual(first["evaluated_unix_s"] - 1000, at)
        # Between the lines nothing happens; a long request delays the split.
        decider = d.ConversionDecider(d.EDGES)
        value = status(short=6.5, instances=L2_INSTANCES)
        records = run(decider, value, 300, bundles=L2, members=L2_MEMBERS)
        self.assertFalse(any(r["proposal"] for r in records))
        decider = d.ConversionDecider(d.EDGES)
        value = status(short=8, deep={"A-tp2": 1}, instances=L2_INSTANCES)
        records = run(decider, value, 1000, bundles=L2, members=L2_MEMBERS)
        self.assertFalse(any(r["proposal"] for r in records))

    def test_long_requests_leave_l0_quickly_even_through_a_drain(self):
        decider = d.ConversionDecider(d.EDGES)
        value = status(
            short=12, deep=dict(A1=1, A2=1, B1=1, B2=1), instances=L0_INSTANCES
        )
        records = run(decider, value, 40, bundles=L0, members=L0_MEMBERS)
        first = next(r for r in records if r["proposal"])
        self.assertEqual(edge(first), ("B", "TP2"))
        self.assertEqual(first["evaluated_unix_s"] - 1000, 30)

    def test_l0_gives_the_tp2_back_when_l1a_carries_the_short_rate(self):
        decider = d.ConversionDecider(d.EDGES)
        value = status(short=9, instances=L0_INSTANCES)
        records = run(decider, value, 80, bundles=L0, members=L0_MEMBERS)
        first = next(r for r in records if r["proposal"])
        self.assertEqual(first["evaluated_unix_s"] - 1000, 75)
        # That calm direction does not drain a long request.
        decider = d.ConversionDecider(d.EDGES)
        value = status(short=9, deep=dict(B1=1), instances=L0_INSTANCES)
        records = run(decider, value, 1000, bundles=L0, members=L0_MEMBERS)
        self.assertFalse(any(r["proposal"] for r in records))
        # Short requests above the line keep L0.
        decider = d.ConversionDecider(d.EDGES)
        value = status(short=10, instances=L0_INSTANCES)
        records = run(decider, value, 300, bundles=L0, members=L0_MEMBERS)
        self.assertFalse(any(r["proposal"] for r in records))

    def test_a_migrating_long_request_counts_once(self):
        decider = d.ConversionDecider(d.EDGES)
        value = status(deep={"A-g1-tp1": 1})
        value["instances"]["B-g2-tp2"]["requests"] = value["instances"]["A-g1-tp1"][
            "requests"
        ]
        record = decider.observe(value, BUNDLES, MEMBERS, 1000)
        self.assertEqual(record["signals"]["long_blocks"], 1536)
        self.assertEqual(record["signals"]["bundle_long_blocks"], dict(A=1536, B=1536))

    def test_a_broken_condition_restarts_the_hold(self):
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, status(short=12), 100)
        records += run(decider, status(short=10), 5, start=1101)
        records += run(decider, status(short=12), 200, start=1107)
        first = next(r for r in records if r["proposal"])
        self.assertEqual(first["evaluated_unix_s"], 1107 + 120)

    def test_record_only_never_repeats_an_edge_within_the_cooldown(self):
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, status(short=12), 400)
        times = [r["evaluated_unix_s"] - 1000 for r in records if r["proposal"]]
        self.assertEqual(times, [120, 241, 362])
        self.assertEqual(records[0]["mode"], "RECORD_ONLY")

    def test_execute_cools_down_every_edge_after_a_conversion(self):
        decider = d.ConversionDecider(d.EDGES, execute=True)
        decider.converted(1000)
        records = run(decider, status(short=12), 200, start=1001)
        times = [r["evaluated_unix_s"] - 1000 for r in records if r["proposal"]]
        # The condition held through the cooldown: proposed as it ends.
        self.assertEqual(times, [121])
        self.assertTrue(records[0]["cooling"])

    def test_layout_names_and_registry_layout(self):
        self.assertEqual(d.layout_name({"A": 1, "B": 1}), "L0")
        self.assertEqual(d.layout_name({"A": 2, "B": 2}), "L2")
        self.assertEqual(d.layout_name(BUNDLES), "L1A")
        registry, _ = instance_pool.fixed_pair_registry(
            source_gpu="2",
            target_gpus=["0", "1"],
            source_url="s",
            target_url="t",
            identity="qwen",
            source_context_limit=32768,
            target_context_limit=32768,
            extras=[
                dict(bundle="A", tp=1, gpus=["3"], endpoint="e", context_limit=32768)
            ],
        )
        self.assertEqual(d.registry_layout(registry), ({}, {}))  # PREPARING
        for row in registry.instances.values():
            registry.transition(row.ref, "ACTIVE")
        bundles, members = d.registry_layout(registry)
        self.assertEqual(bundles, BUNDLES)
        self.assertEqual(members, MEMBERS)


class PrepareStageTests(unittest.TestCase):
    def test_tilt_starts_shortly_before_the_proposal(self):
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, status(short=12), 120)
        tilted = [r["evaluated_unix_s"] - 1000 for r in records if r["tilt"]]
        self.assertEqual((tilted[0], tilted[-1]), (60, 120))
        self.assertEqual(records[60]["tilt"], ["B"])
        self.assertEqual(records[60]["prepared"]["target_mode"], "TP1")
        self.assertIsNotNone(records[120]["proposal"])

    def test_tilt_needs_the_other_bundle_quiet_but_the_proposal_does_not(self):
        decider = d.ConversionDecider(d.EDGES)
        value = status(short=1, long_arrivals=2, deep=MANY_LONG)
        value["instances"]["B-g2-tp2"]["pressure"]["zone"] = "CRITICAL"
        records = run(decider, value, 75)
        self.assertFalse(any(r["tilt"] for r in records))
        self.assertIsNone(records[-1]["prepared"])
        self.assertIsNotNone(records[-1]["proposal"])  # liveness

    def test_a_failed_candidate_ends_the_tilt_at_once(self):
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, status(short=12), 70)
        self.assertEqual(records[-1]["tilt"], ["B"])
        record = decider.observe(status(short=1), BUNDLES, MEMBERS, 1071)
        self.assertEqual(record["tilt"], [])
        self.assertIsNone(record["prepared"])

    def test_a_waiting_split_keeps_long_requests_away_until_its_limit(self):
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, status(short=12, deep={"B-g2-tp2": 1}), 600)
        tilted = [r["evaluated_unix_s"] - 1000 for r in records if r["tilt"]]
        self.assertEqual((tilted[0], tilted[-1]), (60, 420))
        self.assertEqual(len(tilted), 361)
        # The limit only ends the tilt: the split still waits, without draining.
        self.assertFalse(any(r["proposal"] for r in records))

    def test_a_waiting_split_needs_kv_room_elsewhere_not_quiet(self):
        busy = status(short=8, deep={"A-tp2": 1}, instances=L2_INSTANCES)
        busy["instances"]["B-tp2"]["pressure"]["zone"] = "CRITICAL"
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, busy, 100, bundles=L2, members=L2_MEMBERS)
        self.assertEqual(records[-1]["tilt"], ["A"])
        # The same bundle without long requests would not tilt under this load.
        free = status(short=8, instances=L2_INSTANCES)
        free["instances"]["B-tp2"]["pressure"]["zone"] = "CRITICAL"
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, free, 70, bundles=L2, members=L2_MEMBERS)
        self.assertFalse(any(r["tilt"] for r in records))
        # KV near full on the other bundle: no tilt to enter.
        full = status(short=8, deep={"A-tp2": 1}, instances=L2_INSTANCES)
        full["instances"]["B-tp2"]["pressure"]["capacity"] = 0.9
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, full, 100, bundles=L2, members=L2_MEMBERS)
        self.assertFalse(any(r["tilt"] for r in records))

    def test_a_long_tilt_pauses_while_the_other_bundle_lacks_kv_room(self):
        value = status(short=12, deep={"B-g2-tp2": 1})
        decider = d.ConversionDecider(d.EDGES)
        records = run(decider, value, 100)
        self.assertEqual(records[-1]["tilt"], ["B"])
        full = status(short=12, deep={"B-g2-tp2": 1}, tp1_capacity=0.9)
        record = decider.observe(full, BUNDLES, MEMBERS, 1101)
        self.assertEqual(record["tilt"], [])
        # Room again, after the capacity smoothing.
        records = run(decider, value, 20, start=1102)
        tilted = [r["evaluated_unix_s"] - 1000 for r in records if r["tilt"]]
        self.assertEqual(tilted[0], 117)

    def test_signals_are_smoothed(self):
        decider = d.ConversionDecider(d.EDGES)
        decider.observe(status(short=2, tp1_capacity=1.4), BUNDLES, MEMBERS, 1000)
        record = decider.observe(status(short=2), BUNDLES, MEMBERS, 1001)
        self.assertEqual(record["signals"]["tp1_capacity_max"], 1.4)
        record = decider.observe(status(short=2), BUNDLES, MEMBERS, 1016)
        self.assertEqual(record["signals"]["tp1_capacity_max"], 0.2)
        # One waiting sample in ten seconds is not a queue; most of them are.
        decider = d.ConversionDecider(d.EDGES)
        for t in range(10):
            value = status(short=2, waiting=3 if t == 5 else 0)
            record = decider.observe(value, BUNDLES, MEMBERS, 1000 + t)
            self.assertFalse(record["signals"]["tpk_waiting"])
        decider = d.ConversionDecider(d.EDGES)
        for t in range(10):
            value = status(short=2, waiting=0 if t == 5 else 3)
            record = decider.observe(value, BUNDLES, MEMBERS, 1000 + t)
        self.assertTrue(record["signals"]["tpk_waiting"])


class PrepareDriverTests(unittest.TestCase):
    def driver(self, root, mode, **request):
        driver, events = DeciderDriverTests.driver(DeciderDriverTests(), root, mode)
        driver.request = dict(driver.request, **request)
        # Prepare at once, proposal at 60s.
        driver.decider.rules.update(hold_s=60.0, l0_hold_s=60.0)
        driver.events = events
        return driver

    @staticmethod
    def tick(driver, root, value):
        value["unix_s"] = time.time()
        (root / "layout_status.json").write_text(json.dumps(value))
        driver.decided_at = -float("inf")
        driver.decide()

    def test_tilt_is_published_only_when_asked(self):
        for pre_tilt in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                driver = self.driver(root, "RECORD_ONLY", pre_tilt=pre_tilt)
                self.tick(driver, root, status(short=12))
                control = root / "layout_control.json"
                if pre_tilt:
                    self.assertEqual(
                        json.loads(control.read_text())["tilt"], ["B-g2-tp2"]
                    )
                    self.assertIn("PRE_TILT", driver.events)
                else:
                    self.assertFalse(control.exists())
                    self.assertNotIn("PRE_TILT", driver.events)

    def test_execute_prestarts_while_prepared_and_drops_when_it_ends(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = self.driver(root, "EXECUTE", pre_tilt=True)
            started, stopped = [], []
            ops = driver.backend.ops
            ops.start = lambda extra: started.append(extra["instance_id"])
            ops.stop = stopped.extend
            driver.backend.prestart_enabled = True
            self.tick(driver, root, status(short=12))
            self.assertEqual(started, ["B-g4-tp1", "B-g5-tp1"])
            self.tick(driver, root, status(short=12))
            self.assertEqual(len(started), 2)  # still the same preparation
            self.tick(driver, root, status(short=1))
            self.assertEqual(stopped, ["B-g4-tp1", "B-g5-tp1"])
            self.assertEqual(driver.backend.layout_tilt, [])
            self.assertIsNone(driver.backend.prestarted)

    def test_record_only_never_prestarts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = self.driver(root, "RECORD_ONLY")
            driver.backend.ops.start = lambda extra: self.fail("started an engine")
            driver.backend.prestart_enabled = True
            self.tick(driver, root, status(short=12))

    def test_a_tilted_proposal_skips_only_the_service_part_of_the_gate(self):
        for pre_tilt in (False, True):
            with tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                driver = self.driver(root, "EXECUTE", pre_tilt=pre_tilt, assist=True)
                driver.decider.rules.update(hold_s=0.0, l0_hold_s=0.0)
                value = status(short=1, long_arrivals=2, deep=MANY_LONG)
                self.tick(driver, root, value)
                plan = driver.executor.plan
                self.assertEqual(plan["bundle_id"], "A")
                self.assertEqual(plan["method"], "SELECTIVE_ROLL")
                backend = driver.backend
                self.assertEqual(backend.service_gate_passed, pre_tilt)
                # The tilt lasts until quiesce; the controller is held.
                control = json.loads((root / "layout_control.json").read_text())
                self.assertTrue(control["conversion_pending"])
                self.assertEqual(bool(control["tilt"]), pre_tilt)
                # The TP2 got busy from the tilt: only KV headroom is checked.
                busy = status(short=1, long_arrivals=2, deep=MANY_LONG)
                busy["unix_s"] = time.time()
                busy["instances"]["B-g2-tp2"]["pressure"]["zone"] = "CRITICAL"
                (root / "layout_status.json").write_text(json.dumps(busy))
                self.assertEqual(backend.gate(plan), pre_tilt)
                full = json.loads(json.dumps(busy))
                full["instances"]["B-g2-tp2"]["pressure"]["capacity"] = 0.95
                (root / "layout_status.json").write_text(json.dumps(full))
                self.assertFalse(backend.gate(plan))

    def test_a_decided_split_drains(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = self.driver(root, "EXECUTE", assist=True)
            driver.decider.rules.update(hold_s=0.0, l0_hold_s=0.0)
            self.tick(driver, root, status(short=12))
            self.assertEqual(driver.executor.plan["method"], "DRAIN")

    def test_the_tilt_time_limit_covers_a_waiting_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = self.driver(root, "RECORD_ONLY", pre_tilt=True)
            self.tick(driver, root, status(short=12))
            self.assertTrue(driver.backend.layout_tilt)
            driver.end_expired_tilt()
            self.assertTrue(driver.backend.layout_tilt)
            driver.tilt_since = time.monotonic() - 91
            driver.end_expired_tilt()
            self.assertEqual(driver.backend.layout_tilt, [])

    def test_quiesce_clears_the_tilt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver = self.driver(root, "RECORD_ONLY", pre_tilt=True)
            self.tick(driver, root, status(short=12))
            driver.backend.rows["B-g2-tp2"].setdefault("endpoint", "t")
            driver.backend.quiesce(
                dict(
                    plan_id="p",
                    bundle_id="B",
                    target_bundle_mode="TP1",
                    refs=[dict(instance_id="B-g2-tp2", instance_generation=2)],
                )
            )
            self.assertEqual(driver.backend.layout_tilt, [])


class PreemptedMigrationTests(unittest.TestCase):
    def test_allocation_acknowledgement_survives_a_target_preemption(self):
        from tools.bridge_tp import run_pressure_balanced_pilot as pilot

        policy = load("pressure_policy")
        ledger = policy.ReservationLedger()
        ledger.entries["m"] = policy.Reservation(
            "m", "r", "tp4", 1200, 0.0, 0, "migration", state="COMMITTED"
        )
        entry = ledger.entries["m"]
        for owned, expected in ((500, 500), (1145, 1145), (0, 1145), (510, 1145)):
            value = pilot.acknowledged_blocks(entry, owned)
            self.assertEqual(value, expected)
            ledger.observe("m", allocated_blocks=value, service_observed=True)
        self.assertEqual(entry.state, "COMMITTED")
        # Allocation beyond the promise is capped, and completes the promise.
        value = pilot.acknowledged_blocks(entry, 5000)
        self.assertEqual(value, 1200)
        ledger.observe("m", allocated_blocks=value, service_observed=True)
        self.assertEqual(entry.state, "ACCOUNTED")


class StartGateTests(unittest.TestCase):
    def test_gate_blocks_until_opened_and_times_out(self):
        import os
        import threading
        from unittest.mock import patch

        from vllm.bridge_tp import start_gate

        self.assertIsNone(start_gate.wait_for_start_gate())
        with tempfile.TemporaryDirectory() as folder:
            gate = Path(folder) / "gate"
            with patch.dict(os.environ, {start_gate.ENV: str(gate)}):
                threading.Timer(0.2, gate.touch).start()
                waited = start_gate.wait_for_start_gate(poll_s=0.01)
                self.assertGreaterEqual(waited, 0.15)
            never = {
                start_gate.ENV: str(gate) + "-never",
                start_gate.TIMEOUT_ENV: "0.1",
            }
            with patch.dict(os.environ, never), self.assertRaises(TimeoutError):
                start_gate.wait_for_start_gate(poll_s=0.01)

    def test_worker_waits_at_the_gate_before_any_cuda_use(self):
        import ast

        root = Path(__file__).resolve().parents[2]
        tree = ast.parse((root / "vllm/v1/worker/gpu_worker.py").read_text("utf-8"))
        init = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "init_device"
        )
        calls = [  # the body only (decorators are not part of the call order)
            n.func.id
            for statement in init.body
            for n in ast.walk(statement)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        ]
        self.assertEqual(calls[0], "wait_for_start_gate")
        first_if = next(n for n in init.body if isinstance(n, ast.If))
        gate_line = next(
            n.lineno
            for n in ast.walk(init)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "wait_for_start_gate"
        )
        self.assertLess(gate_line, first_if.lineno)


class DeciderDriverTests(unittest.TestCase):
    def driver(self, root, mode):
        registry, aliases = instance_pool.fixed_pair_registry(
            source_gpu="2",
            target_gpus=["0", "1"],
            source_url="s",
            target_url="t",
            identity="qwen",
            source_context_limit=32768,
            target_context_limit=32768,
            extras=[
                dict(bundle="A", tp=1, gpus=["3"], endpoint="e", context_limit=32768)
            ],
        )
        events = []
        backend = pc.ControllerConversionBackend(
            registry,
            instance_pool.registry_manifest(registry, aliases),
            root,
            NS(event=lambda kind, **values: events.append(kind)),
            base_port=8400,
            context_limits=(32768, 32768),
        )
        request = dict(decider=mode, deadline_s=600, base_port=8400, edges=EDGES)
        return pc.ConversionDriver(backend, request, root / "plan.json"), events

    def test_record_only_logs_and_never_holds_the_controller(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver, events = self.driver(root, "RECORD_ONLY")
            driver.start()
            control = json.loads((root / "layout_control.json").read_text())
            self.assertFalse(control["conversion_pending"])
            value = status(short=12)
            driver.decider.rules.update(hold_s=0.0, l0_hold_s=0.0)
            for _ in range(3):
                value["unix_s"] = time.time()
                (root / "layout_status.json").write_text(json.dumps(value))
                driver.decided_at = -float("inf")
                driver.decide()
            outcome = driver.stop()
            self.assertEqual(outcome["state"], "NOT_TRIGGERED")
            self.assertGreaterEqual(outcome["decisions"], 3)
            self.assertEqual(
                [(p["bundle"], p["target_mode"]) for p in outcome["proposals"]],
                [("B", "TP1")],
            )
            self.assertIn("DECIDER_PROPOSAL", events)
            self.assertNotIn("TRIGGER", events)
            rows = (root / "conversion_decider.jsonl").read_text().splitlines()
            self.assertGreaterEqual(len(rows), 3)
            control = json.loads((root / "layout_control.json").read_text())
            self.assertFalse(control["conversion_pending"])

    def test_a_decider_run_without_conversion_is_a_valid_scene(self):
        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        decided = dict(decider="RECORD_ONLY", deadline_s=600, base_port=8400)
        explicit = dict(bundle="A", target_mode="TP2", trigger_s=30)
        self.assertTrue(launch.conversion_outcome_ok(decided, "NOT_TRIGGERED"))
        self.assertFalse(launch.conversion_outcome_ok(explicit, "NOT_TRIGGERED"))
        for request in (decided, explicit):
            self.assertTrue(launch.conversion_outcome_ok(request, "COMPLETED"))
            self.assertTrue(launch.conversion_outcome_ok(request, "ABORTED"))
            self.assertFalse(launch.conversion_outcome_ok(request, "FENCED"))

    def test_a_failed_result_write_does_not_stop_scene_cleanup(self):
        """Disk full (11-01): the write raised in run_scene's finally, ahead of
        stop_processes, and the engines outlived the scene."""
        import ast

        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        stopped = []
        driver = NS(stop=lambda: stopped.append(1) or dict(state="NOT_TRIGGERED"))
        with tempfile.TemporaryDirectory() as folder:
            missing = Path(folder) / "gone"
            with self.assertRaises(OSError):
                launch.finish_conversion(driver, missing)
            self.assertIsNone(launch.finish_conversion_best_effort(driver, missing))
        self.assertEqual(len(stopped), 2)
        tree = ast.parse(Path(launch.__file__).read_text("utf-8"))
        run_scene = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "run_scene"
        )
        (handler,) = [
            n for n in ast.walk(run_scene) if isinstance(n, ast.Try) and n.finalbody
        ]
        names = [
            n.func.id if isinstance(n.func, ast.Name) else n.func.attr
            for statement in handler.finalbody
            for n in ast.walk(statement)
            if isinstance(n, ast.Call) and isinstance(n.func, (ast.Name, ast.Attribute))
        ]
        self.assertNotIn("finish_conversion", names)
        self.assertIn("finish_conversion_best_effort", names)

    def test_stale_status_is_not_decided_on(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            driver, _ = self.driver(root, "RECORD_ONLY")
            value = status(short=12)
            value["unix_s"] = time.time() - 10
            (root / "layout_status.json").write_text(json.dumps(value))
            driver.decide()
            self.assertEqual(driver.decisions, 0)

    def test_launcher_uses_the_edges_of_every_layout(self):
        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        registry, _ = instance_pool.fixed_pair_registry(
            source_gpu="2",
            target_gpus=["0", "1"],
            source_url="s",
            target_url="t",
            identity="qwen",
            source_context_limit=32768,
            target_context_limit=32768,
            extras=[
                dict(bundle="A", tp=1, gpus=["3"], endpoint="e", context_limit=32768)
            ],
        )
        request = dict(decider="RECORD_ONLY", deadline_s=600, base_port=8400)
        value = launch.conversion_request(
            dict(layout=dict(name="L1A"), conversion=request), registry
        )
        self.assertEqual(value["edges"], d.EDGES)
        self.assertEqual(value["edges"]["L1A"], EDGES["L1A"])
        for bad in (
            dict(request, decider="MAYBE"),
            dict(request, rules=dict(nothing=1)),
            dict(request, bundle="A"),
        ):
            with self.assertRaises(ValueError):
                launch.conversion_request(
                    dict(layout=dict(name="L1A"), conversion=bad), registry
                )


class PhaseShiftTests(unittest.TestCase):
    def test_scenarios_keep_their_long_share(self):
        from tools.bridge_tp import qwen3_phase_shift as phase

        root = Path(__file__).resolve().parents[2]
        inputs = json.loads(
            (
                root / "experiments/platforms/qwen3_pressure_observation/inputs.json"
            ).read_text(encoding="utf-8")
        )

        def share(scenario):
            jobs = [  # what the decider's window sees during the main phase
                j
                for j in phase.jobs(inputs, scenario)
                if j["workload_phase"] in {"long", "peak"}
            ]
            long = [
                j
                for j in jobs
                if len(j["request"]["prompt"]) + j["request"]["max_tokens"]
                >= d.LONG_TOKENS
            ]
            return long, len(long) / len(jobs)

        self.assertGreater(share("merge_heavy")[1], 0.30)
        self.assertGreater(share("merge_moderate")[1], 0.30)
        long, value = share("merge_deep")
        self.assertLess(value, 0.30)
        self.assertEqual(
            (len(long[0]["request"]["prompt"]), long[0]["request"]["max_tokens"]),
            (16384, 8192),
        )
        long, value = share("split_trickle")
        self.assertTrue(long)
        self.assertLess(value, 0.10)

    def test_boundary_grid_steps_the_long_rate_at_fixed_sizes(self):
        from tools.bridge_tp import qwen3_phase_shift as phase

        root = Path(__file__).resolve().parents[2]
        inputs = json.loads(
            (
                root / "experiments/platforms/qwen3_pressure_observation/inputs.json"
            ).read_text(encoding="utf-8")
        )

        def longs(scenario):
            rows = [
                j
                for j in phase.jobs(inputs, scenario)
                if j["workload_role"] == "long_input_output"
            ]
            sizes = {
                (len(j["request"]["prompt"]), j["request"]["max_tokens"]) for j in rows
            }
            return len(rows), sizes

        for scenario, count in (
            ("merge_light", 42),
            ("merge_moderate", 105),
            ("merge_heavy", 168),
        ):
            self.assertEqual(longs(scenario), (count, {(6144, 2048)}))
        for scenario, count in (
            ("deep_light", 8),
            ("deep_mid", 14),
            ("merge_deep", 21),
        ):
            self.assertEqual(longs(scenario), (count, {(16384, 8192)}))
        self.assertEqual(longs("short_mid"), (36, {(6144, 2048)}))
        self.assertEqual(longs("split_trickle"), (36, {(6144, 2048)}))

    def test_phases_give_each_rule_its_window(self):
        from tools.bridge_tp import qwen3_phase_shift as phase

        root = Path(__file__).resolve().parents[2]
        inputs = json.loads(
            (
                root / "experiments/platforms/qwen3_pressure_observation/inputs.json"
            ).read_text(encoding="utf-8")
        )
        jobs = phase.jobs(inputs)
        self.assertEqual(len({j["id"] for j in jobs}), len(jobs))
        by = {}
        for job in jobs:
            by.setdefault(job["workload_phase"], []).append(job)
        p1 = [j for j in by["P1_short"] if j["at_s"] >= 60]
        self.assertEqual(len(p1), 2700)
        long = [j for j in by["P2_long"] if j["workload_role"] == "long_input_output"]
        self.assertEqual(len(long), 144)
        self.assertAlmostEqual(len(long) / len(by["P2_long"]), 144 / 324)
        request = long[0]["request"]
        self.assertEqual((len(request["prompt"]), request["max_tokens"]), (6144, 2048))
        self.assertGreaterEqual(
            len(request["prompt"]) + request["max_tokens"], d.LONG_TOKENS
        )
        self.assertLess(max(j["at_s"] for j in jobs), 840)


if __name__ == "__main__":
    unittest.main()
