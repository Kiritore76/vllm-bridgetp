# SPDX-License-Identifier: Apache-2.0
"""CPU checks for fixed arrivals, KV bounds and actual concurrency coverage."""

import copy
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools.bridge_tp import pressure_efficient_workload as e
from tools.bridge_tp import run_pressure_calibration as c


class EfficientRegionTests(unittest.TestCase):
    def setUp(self):
        base = c.ROOT / "experiments/phase9/pressure_balanced_v2"
        self.inputs = c.pilot.read(base / "calibration.json")
        self.plan = c.pilot.read(base / "efficient_region.json")

    def test_frozen_inputs_and_test_kv_bounds(self):
        e.validate_plan(self.plan, self.inputs)
        for scene in self.plan["scenes"]:
            receipt = c.capacity_check(
                scene,
                [
                    dict(block_size=16, free_blocks=1962),
                    dict(block_size=16, free_blocks=35000),
                ],
            )
            self.assertLessEqual(receipt["source"]["required_blocks"], 1962)
        scene = self.plan["scenes"][2]
        self.assertEqual(
            c.capacity_check(
                scene,
                [
                    dict(block_size=16, free_blocks=1962),
                    dict(block_size=16, free_blocks=35000),
                ],
            )["source"]["required_blocks"],
            1949,
        )
        with self.assertRaises(ValueError):
            c.capacity_check(
                scene,
                [
                    dict(block_size=16, free_blocks=1948),
                    dict(block_size=16, free_blocks=35000),
                ],
            )
        changed = copy.deepcopy(self.plan)
        changed["scenes"][-1]["injection_prompt_tokens"] = [2048]
        with self.assertRaisesRegex(ValueError, "equal total"):
            e.validate_plan(changed, self.inputs)

    def test_last_only_replays_exactly_short_scene_repetition_two(self):
        scenes, repetitions = e.collection_selection(self.plan, last_only=True)
        jobs = [(s["name"], r) for r in repetitions for s in scenes]
        self.assertEqual(jobs, [("region-short-n8-n24", 2)])
        self.assertEqual(scenes[0], self.plan["scenes"][0])
        full_scenes, full_repetitions = e.collection_selection(self.plan)
        self.assertEqual(full_scenes, self.plan["scenes"])
        self.assertEqual(full_repetitions, [1, 2])
        changed = copy.deepcopy(self.plan)
        changed["scenes"][0]["name"] = "unrelated"
        with self.assertRaisesRegex(ValueError, "identity differs"):
            e.collection_selection(changed, last_only=True)

    def test_fixed_arrivals_do_not_wait_for_previous_completion(self):
        clock = [100.0]

        class FakeWork:
            timeout_s = 900
            phase = ""

            def __init__(self, injection_duration=3):
                self.rows, self.futures = [], []
                self.injection_duration = injection_duration

            def submit(self, pool, prompt, output, label):
                end = clock[0] + (
                    70 if "background" in label else self.injection_duration
                )
                row = dict(
                    id=label,
                    pool=pool,
                    prompt_tokens=prompt,
                    first_token_unix_s=clock[0],
                    last_token_unix_s=end,
                    response=dict(request_started_unix_s=clock[0]),
                )
                self.rows.append(row)
                future = SimpleNamespace(done=lambda: clock[0] >= end)
                self.futures.append(future)
                return label, future

            def snapshot(self):
                return [
                    dict(r, done=clock[0] >= r["last_token_unix_s"]) for r in self.rows
                ]

            def wait(self, predicate, deadline):
                while not predicate():
                    clock[0] += 0.25
                    if clock[0] >= deadline:
                        raise TimeoutError("fake workload deadline")

        work = FakeWork()
        with tempfile.TemporaryDirectory(dir=c.ROOT) as directory:
            root = Path(directory)
            with (
                patch.object(e.time, "time", lambda: clock[0]),
                patch.object(e.time, "monotonic", lambda: clock[0]),
            ):
                e.drive(work, self.plan["scenes"][-2], 1, root)
            receipt = c.pilot.read(root / "controlled_phase.json")
            starts = [
                r["response"]["request_started_unix_s"]
                for r in work.rows
                if r["pool"] == "source" and "prefill" in r["id"]
            ]
            self.assertEqual(starts, [105 + 2 * i for i in range(24)])
            self.assertEqual(receipt["ended_unix_s"], 153)
            self.assertEqual(len(receipt["injection_arrivals"]), 48)
            self.assertEqual(
                c.pilot.read(root / "request_results.json")["status"], "COMPLETED"
            )
        # An invalid overload must preserve partial evidence and stop, not queue
        # or stretch the fixed schedule beyond its checked physical envelope.
        clock[0] = 100.0
        work = FakeWork(injection_duration=9)
        with tempfile.TemporaryDirectory(dir=c.ROOT) as directory:
            root = Path(directory)
            with (
                patch.object(e.time, "time", lambda: clock[0]),
                patch.object(e.time, "monotonic", lambda: clock[0]),
            ):
                c.drive_guarded(work, self.plan["scenes"][-2], 1, root)
            failed = c.pilot.read(root / "request_results.json")
            self.assertEqual(failed["status"], "TECHNICAL_FAILURE")
            self.assertIn("KV envelope", failed["error"])
            self.assertEqual(
                len(c.pilot.read(root / "controlled_phase.json")["injection_arrivals"]),
                4,
            )

    def test_38_streams_can_start_without_client_executor_queue(self):
        entered, release = threading.Event(), threading.Event()
        lock, count = threading.Lock(), [0]

        def stream(_url, _request, _timeout, sink, *_args, **_kwargs):
            with lock:
                count[0] += 1
                if count[0] == 38:
                    entered.set()
            release.wait(5)
            sink(0, 42, 100.0)
            return dict(token_ids=[42], finish_reason="length", chunks=[])

        with tempfile.TemporaryDirectory(dir=c.ROOT) as directory:
            work = c.Workload(
                self.inputs,
                Path(directory),
                dict(source="unused", target="unused"),
                5,
                max_workers=64,
            )
            try:
                with patch.object(c.io, "post_streaming_completion", stream):
                    for i in range(38):
                        work.submit("source" if i < 6 else "target", 1024, 1, str(i))
                    self.assertTrue(entered.wait(3), "hidden client queue limits N")
                    release.set()
                    for future in work.futures:
                        future.result(timeout=3)
            finally:
                release.set()
                work.executor.shutdown(wait=True)

    def coverage_fixture(self, root, source_duration=30, target_nd=32):
        rows = []
        for pool, count in (("source", 6), ("target", 32)):
            duration = source_duration if pool == "source" else 30
            rows.extend(
                dict(
                    id=f"{pool}-{i}",
                    pool=pool,
                    first_token_unix_s=100,
                    last_token_unix_s=100 + duration,
                )
                for i in range(count)
            )
            native_nd = 6 if pool == "source" else target_nd
            engine = [dict(kind="trace_header", schema="BRIDGETP_BATCH_PERFORMANCE_V1")]
            engine += [
                dict(
                    kind="engine_batch",
                    scheduled_unix_s=100 + i / 4,
                    completed_unix_s=100 + i / 4 + 0.02,
                    decode_requests=native_nd,
                    decode_context_sum=native_nd * 4500,
                    prefill_tokens=0,
                    recompute_tokens=0,
                )
                for i in range(120)
            ]
            engine.append(
                dict(
                    kind="evidence_footer",
                    saved_records=len(engine),
                    dropped=0,
                    writer_error=None,
                )
            )
            path = root / "controller" / "performance_trace" / pool / "engine.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in engine))
        c.io.atomic_json_dump(dict(requests=rows), root / "request_results.json")
        c.io.atomic_json_dump(
            dict(background_ids=[r["id"] for r in rows], injection_arrivals=[]),
            root / "controlled_phase.json",
        )

    def test_actual_short_lifetime_and_wrong_native_nd_are_coverage_gaps(self):
        for duration, target_nd, status in (
            (30, 32, "COVERED"),
            (5, 32, "COVERAGE_GAP"),
            (30, 16, "COVERAGE_GAP"),
        ):
            with (
                self.subTest(duration=duration, target_nd=target_nd),
                tempfile.TemporaryDirectory(dir=c.ROOT) as directory,
            ):
                root = Path(directory)
                self.coverage_fixture(root, duration, target_nd)
                result = e.assess_coverage(root, self.plan["scenes"][2])
                self.assertEqual(result["status"], status)

    def test_mixed_coverage_requires_real_prefill_and_timely_http_dispatch(self):
        for lag, prefill_tokens, expected in (
            (0.2, 1024, "COVERED"),
            (0.8, 1024, "COVERAGE_GAP"),
            (0.2, 0, "COVERAGE_GAP"),
        ):
            with (
                self.subTest(lag=lag, prefill_tokens=prefill_tokens),
                tempfile.TemporaryDirectory(dir=c.ROOT) as directory,
            ):
                root = Path(directory)
                self.coverage_fixture(root)
                result = c.pilot.read(root / "request_results.json")
                phase = c.pilot.read(root / "controlled_phase.json")
                phase.update(started_unix_s=105, ended_unix_s=125)
                for pool in e.POOLS:
                    label = pool + "-injection"
                    result["requests"].append(
                        dict(id=label, response=dict(request_started_unix_s=105 + lag))
                    )
                    phase["injection_arrivals"].append(
                        dict(id=label, pool=pool, planned_unix_s=105)
                    )
                    path = root / "controller/performance_trace" / pool / "engine.jsonl"
                    engine = [
                        json.loads(line) for line in path.read_text().splitlines()
                    ]
                    engine.insert(
                        -1,
                        dict(
                            kind="engine_batch",
                            scheduled_unix_s=110,
                            completed_unix_s=110.2,
                            decode_requests=0,
                            prefill_tokens=prefill_tokens,
                            recompute_tokens=0,
                        ),
                    )
                    engine[-1]["saved_records"] += 1
                    path.write_text("".join(json.dumps(r) + "\n" for r in engine))
                c.io.atomic_json_dump(result, root / "request_results.json")
                c.io.atomic_json_dump(phase, root / "controlled_phase.json")
                scene = dict(
                    self.plan["scenes"][2],
                    injection_prompt_tokens=[1024],
                    fixed_schedule=dict(count=1, max_dispatch_lag_s=0.5),
                )
                receipt = e.assess_coverage(root, scene)
                self.assertEqual(receipt["status"], expected)


if __name__ == "__main__":
    unittest.main()
