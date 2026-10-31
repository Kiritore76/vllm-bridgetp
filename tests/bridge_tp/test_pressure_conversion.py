# SPDX-License-Identifier: Apache-2.0
"""S3b-3: explicit DRAIN conversion driven against the live control loop (CPU)."""

import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from tests.bridge_tp.test_performance_trace import journal
from tests.bridge_tp.test_pressure_policy import config
from tools.bridge_tp import conversion_planner
from tools.bridge_tp import pressure_conversion as pc
from tools.bridge_tp import run_pressure_balanced_pilot as pilot
from tools.bridge_tp.pressure_cpu import load

i, o, instance_pool = (
    load(name) for name in ("pressure_ingress", "pressure_observation", "instance_pool")
)


class Reader:
    def __init__(self, pool_id):
        self.pool_id, self.generation = pool_id, 0

    def poll(self, now):
        self.generation += 1
        return dict(
            schema="PRESSURE_INVENTORY_V1",
            dropped_before=0,
            writer_error=None,
            pool_id=self.pool_id,
            sampled_at=now,
            generation=self.generation,
            total_blocks=2001,
            usage_denominator_blocks=2000,
            free_blocks=2000,
            block_size=16,
            context_limit=8192,
            pending_prompt_tokens=0,
            waiting=0,
            running=0,
            requests=[],
        )


class Harness:
    """L1A on A={2,3}, B={0,1}: TP1@2 (pair source), TP1@3, TP2(0,1) target."""

    def __init__(
        self,
        root,
        bundle,
        target_mode,
        channel_plan=False,
        slow=(),
        acknowledge=True,
    ):
        self.root, self.slow = root, set(slow)
        self.calls, self.events, self.stopped = [], [], set()
        self.lock = threading.Lock()
        registry, aliases = instance_pool.fixed_pair_registry(
            source_gpu="2",
            target_gpus=["0", "1"],
            source_url="source",
            target_url="target",
            identity="qwen",
            source_context_limit=8192,
            target_context_limit=8192,
            extras=[
                dict(bundle="A", tp=1, gpus=["3"], endpoint="extra", context_limit=8192)
            ],
        )
        self.registry = registry
        manifest = instance_pool.registry_manifest(registry, aliases)
        paths = {"A-g1-tp1": "source", "B-g2-tp2": "target", "A-g3-tp1": "extra"}
        for row in manifest["instances"]:
            row["inventory_path"] = str(root / paths[row["instance_id"]])
        self.manifest = manifest
        self.path_of = {"source": root / "source", "target": root / "target"}
        self.path_of["extra"] = root / "extra"
        self.readers = {
            "source": Reader("tp1"),
            "target": Reader("tp4"),
            "extra": Reader("A-g3-tp1"),
        }
        for name in list(self.readers):
            self.publish(name)
        self.channel_plan = None
        if channel_plan:
            self.channel_plan = root / "gpu_channel_plan.json"
            self.channel_plan.write_text(
                json.dumps(
                    [
                        dict(source="A-g1-tp1", base_port=30400, generation=1),
                        dict(source="A-g3-tp1", base_port=30410, generation=3),
                    ]
                )
            )
            if acknowledge:
                # Each target rank receipts its first accepted plan file load.
                directory = root / "session-1" / "gpu_channel_plan_receipts"
                directory.mkdir(parents=True)
                for rank in (0, 1):
                    (directory / f"tp_rank_{rank}.json").write_text(
                        json.dumps(
                            dict(
                                opened=[],
                                closed=[],
                                deferred=[],
                                planned=[1, 3],
                                completed_unix_s=time.time() - 60,
                            )
                        )
                    )
        ops = NS(
            stop=lambda ids: self.stopped.update(ids),
            alive=lambda instance_id: instance_id not in self.stopped,
            gpus_released=lambda gpus: True,
            scrape=lambda endpoint, native: dict(
                num_running=0,
                num_waiting=0,
                free_kv_blocks=native["free_blocks"],
                sampled_unix_s=time.time(),
            ),
            start=self.start,
            warm=lambda extra: None,
            event=self.event,
        )
        self.backend = pc.ControllerConversionBackend(
            registry,
            manifest,
            root,
            ops,
            base_port=8400,
            context_limits=(8192, 8192),
            channel_plan=self.channel_plan,
            target_tp_ranks=2,
            shadow_planner=conversion_planner.ShadowPlanner(
                conversion_planner.load_profile(), manifest["instances"]
            ),
        )
        self.driver = pc.ConversionDriver(
            self.backend,
            dict(
                bundle=bundle,
                target_mode=target_mode,
                trigger_s=0.3,
                deadline_s=20,
                base_port=8400,
            ),
            root / "conversion_plan.json",
            interval_s=0.02,
        )

    def event(self, kind, **values):
        self.events.append(dict(kind=kind, unix_s=time.time(), **values))
        if kind == "CHANNEL_PLAN_WRITTEN" and values["closing"]:
            # Both target ranks reconcile the rewritten plan file.
            directory = self.root / "session-1" / "gpu_channel_plan_receipts"
            directory.mkdir(parents=True, exist_ok=True)
            for rank in (0, 1):
                (directory / f"tp_rank_{rank}.json").write_text(
                    json.dumps(
                        dict(
                            opened=[],
                            closed=values["closing"],
                            deferred=[],
                            completed_unix_s=time.time() + 0.01,
                        )
                    )
                )
                # Background communicator teardown of each closed channel.
                for generation in values["closing"]:
                    name = f"tp_rank_{rank}-retired-{generation}.json"
                    (directory / name).write_text(json.dumps(dict(status="ABORTED")))

    def publish(self, name, after=-float("inf")):
        while time.time() <= after:
            time.sleep(0.001)
        value = self.readers[name].poll(time.time())
        with self.path_of[name].open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(value) + "\n")

    def start(self, extra):
        with self.lock:
            self.readers[extra["endpoint"]] = Reader(extra["instance_id"])
            self.path_of[extra["endpoint"]] = Path(extra["inventory_path"])
            self.publish(extra["endpoint"])

    def post(self, url, payload, timeout, sink):
        with self.lock:
            self.calls.append((time.time(), url))
        if url in self.stopped_endpoints():
            raise RuntimeError("request routed to a stopped engine")
        sink(0, 42, time.time())
        # In flight across the quiesce: the drain must wait for these.
        time.sleep(0.8 if url in self.slow else 0.25)
        completed = time.time()
        with self.lock:
            self.publish(url, after=completed)
        return dict(request_started_unix_s=completed, completed_unix_s=completed)

    def stopped_endpoints(self):
        rows = {r["instance_id"]: r for r in self.manifest["instances"]}
        return {rows[i]["endpoint"] for i in self.stopped if i in rows}

    def run(self, jobs):
        actual_ingress = i.PressureIngress

        def make_ingress(*args, **kwargs):
            return actual_ingress(*args, **kwargs, post=self.post)

        session = NS(
            sessions=[],
            record=None,
            stream=None,
            target_future=None,
            settled=True,
            migration_id="test",
            ready_for_start=True,
            rotate=lambda: False,
            cleanup_requires_shutdown=False,
            cancel_future=None,
            close=lambda: None,
        )
        args = NS(
            run_dir=self.root,
            source_inventory=self.root / "source",
            target_inventory=self.root / "target",
            source_context_limit=8192,
            target_context_limit=8192,
            predictor_events=self.root / "predictor",
            predictor_sha256="0" * 64,
            migration_id="test",
            tick_s=0.005,
            max_seconds=15,
            no_progress_s=5,
            no_migration=True,
            maximum_migrations=1,
            routing_strategy="pressure",
            routing_weights=[1, 1],
            test_only_cancel_first_shadow=False,
            instances=self.root / "instances.json",
        )
        (self.root / "instances.json").write_text(json.dumps(self.manifest))
        controller = NS(
            source_url="source",
            target_url="target",
            block_size=16,
            tp1_total_kv_blocks=2001,
            tp4_total_kv_blocks=2001,
            target_tp_size=2,
        )
        checked = pilot.instance_manifest(args, controller)
        self.driver.start()
        try:
            with (
                patch.dict(sys.modules, {"vllm.bridge_tp.async_journal": journal}),
                patch(
                    "tools.bridge_tp.pressure_sessions.PressureSessions",
                    return_value=session,
                ),
                patch.object(o, "scrape_inventory_metrics", return_value=None),
                patch.object(i, "PressureIngress", side_effect=make_ingress),
            ):
                pilot._run(
                    args,
                    (
                        config(),
                        controller,
                        NS(release_budget=17),
                        jobs,
                        {"instance_manifest": checked},
                    ),
                )
        finally:
            outcome = self.driver.stop()
        return outcome


def jobs(count, spacing, busy=None):
    """``busy``: pair pool that the first two arrivals are pinned to."""
    return [
        dict(
            id=str(n),
            at_s=n * 0.05 if busy and n < 2 else n * spacing,
            request=dict(model="qwen", prompt=[1], max_tokens=8),
            **(dict(route=busy) if busy and n < 2 else {}),
        )
        for n in range(count)
    ]


class ConversionTests(unittest.TestCase):
    def check_common(self, harness, outcome, retired, added):
        root = harness.root
        self.assertEqual(outcome["state"], "COMPLETED", outcome)
        self.assertIsNone(outcome["error"])
        result = json.loads((root / "pressure_request_results.json").read_text())
        self.assertTrue(all(row["source_done"] for row in result))
        progress = json.loads((root / "pressure_progress.json").read_text())
        self.assertEqual(progress["stage"], "COMPLETED")
        self.assertEqual(harness.stopped, set(retired))
        kinds = [e["kind"] for e in harness.events]
        for kind in (
            "TRIGGER",
            "FENCED",
            "CHANNEL_RETIRE",
            "STOP_RETURN",
            "RESOURCES_RELEASED",
            "ROUTE_PUBLISHED",
        ):
            self.assertIn(kind, kinds)
        published = next(e for e in harness.events if e["kind"] == "ROUTE_PUBLISHED")
        self.assertEqual(published["instances"], added)
        # Routing closed at QUIESCING: nothing reaches the converted bundle's
        # engines after the controller applied it.
        audit = [
            json.loads(line)
            for line in (root / "pressure_audit.jsonl").read_text().splitlines()
        ]
        quiesced = min(
            r["unix_s"]
            for r in audit
            if r["kind"] == "pressure_instance_lifecycle" and r["to"] == "QUIESCING"
        )
        old = {
            r.get("legacy_pool") or r["instance_id"]
            for r in harness.manifest["instances"]
            if r["instance_id"] in retired
        }
        late = [
            r
            for r in audit
            if r["kind"] == "pressure_dispatch"
            and r["pool"] in old
            and r["admission_finished_unix_s"] > quiesced
        ]
        self.assertEqual(late, [])
        # The converted bundle had requests in flight at the trigger; the
        # fence waited until the controller saw them finish.
        checks = [
            json.loads(line)
            for line in (root / "conversion_drain_observations.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertTrue(
            any(row["streams_pending"] for c in checks for row in c["rows"])
        )
        fenced = next(e for e in harness.events if e["kind"] == "FENCED")["unix_s"]
        self.assertGreaterEqual(fenced - quiesced, 0.3)
        # New instances joined routing once their inventory arrived.
        pools = {row["initial_pool"] for row in result}
        self.assertTrue(set(added) & pools, pools)
        status = json.loads((root / "layout_status.json").read_text())
        for instance_id in retired:
            self.assertEqual(status["instances"][instance_id]["lifecycle"], "RETIRED")
        for instance_id in added:
            self.assertEqual(status["instances"][instance_id]["lifecycle"], "ACTIVE")
        for instance_id in added:
            self.assertEqual(
                harness.registry.instances[instance_id].lifecycle, "ACTIVE"
            )
        # S3b-4: the reusable acceptance agrees with the checks above.
        from tools.bridge_tp import conversion_acceptance as acceptance

        events = root / "conversion_events.jsonl"
        events.write_text("".join(json.dumps(e) + "\n" for e in harness.events))
        (root / "conversion_result.json").write_text(json.dumps(outcome))
        value = acceptance.assess(root, run_dir=root)
        self.assertTrue(value["passed"], value["checks"])
        self.assertEqual(value["retired"], retired)
        self.assertEqual(value["new_instances"], added)
        self.assertTrue(set(value["first_request_after_publish_s"]) <= set(added))
        self.assertIsNone(value["service_by_arrival_phase"])  # no runner metrics
        # S4: the SHADOW planner ran while draining and never failed. This
        # harness serves 8192 context, outside the frozen profile: UNKNOWN.
        shadow = value["planner_shadow"]
        self.assertGreaterEqual(shadow["records"], 1)
        self.assertEqual(shadow["errors"], 0)
        first = json.loads(
            (root / "conversion_planner_shadow.jsonl").read_text().splitlines()[0]
        )
        self.assertIn("context", first["profile_mismatch"])
        self.assertEqual(first["drain_only"]["support_status"], "UNKNOWN")
        self.assertGreaterEqual(value["phases_s"]["drain"], 0.3)
        self.assertGreater(sum(value["drain_inflight_peak"].values()), 0)
        self.assertEqual(
            sum(sum(c.values()) for c in value["routes_by_phase"].values()),
            len(result),
        )
        # A single admission to a retired instance after quiesce fails it.
        rows = (root / "pressure_audit.jsonl").read_text().splitlines()
        forged = dict(
            kind="pressure_dispatch",
            job_id="forged",
            pool=next(
                r["legacy_pool"] or r["instance_id"]
                for r in harness.manifest["instances"]
                if r["instance_id"] in retired
            ),
            admission_finished_unix_s=quiesced + 0.5,
        )
        (root / "pressure_audit.jsonl").write_text(
            "\n".join(rows + [json.dumps(forged)]) + "\n"
        )
        value = acceptance.assess(root, run_dir=root)
        self.assertFalse(value["passed"])
        self.assertFalse(value["checks"]["no_admission_to_retired_after_quiesce"])
        self.assertEqual(value["admitted_to_retired_after_quiesce"], ["forged"])
        return audit

    def test_split_retires_pair_target_and_routes_to_new_tp1(self):
        """L1A -> L0: bundle B TP2(0,1) -> TP1@0 + TP1@1 while serving."""
        with tempfile.TemporaryDirectory() as directory:
            harness = Harness(Path(directory), "B", "TP1", slow={"target"})
            outcome = harness.run(jobs(16, 0.3, busy="tp4"))
            self.check_common(harness, outcome, ["B-g2-tp2"], ["B-g4-tp1", "B-g5-tp1"])
            # No channel plan: retiring the target closes nothing on a target.
            retire = next(e for e in harness.events if e["kind"] == "CHANNEL_RETIRE")
            self.assertEqual(retire["closed"], [])
            # Scene finalization drains and finalizes only the surviving side.
            from tools.bridge_tp import launch_pressure_balanced_pilot as launch

            self.assertEqual(
                launch.live_pair(
                    dict(run_dir=harness.root, instances=harness.manifest)
                ),
                ["source"],
            )

    def test_merge_closes_retired_sources_channels_first(self):
        """L1A -> L2: bundle A TP1@2 + TP1@3 -> TP2(2,3); their channels close."""
        with tempfile.TemporaryDirectory() as directory:
            harness = Harness(
                Path(directory), "A", "TP2", channel_plan=True, slow={"source"}
            )
            outcome = harness.run(jobs(16, 0.3, busy="tp1"))
            self.check_common(harness, outcome, ["A-g1-tp1", "A-g3-tp1"], ["A-g4-tp2"])
            self.assertEqual(json.loads(harness.channel_plan.read_text()), [])
            kinds = [e["kind"] for e in harness.events]
            self.assertLess(kinds.index("CHANNEL_RETIRE"), kinds.index("STOP_RETURN"))
            retire = next(e for e in harness.events if e["kind"] == "CHANNEL_RETIRE")
            self.assertEqual(retire["closed"], [1, 3])
            self.assertEqual(retire["ranks"], ["tp_rank_0.json", "tp_rank_1.json"])
            self.assertTrue(retire["teardown_complete"])
            self.assertEqual(
                retire["teardown"],
                {f"tp_rank_{r}/{g}": "ABORTED" for r in (0, 1) for g in (1, 3)},
            )

    def test_scene_runner_binds_owned_engines_and_writes_result(self):
        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness(root, "B", "TP1")
            for row in harness.manifest["instances"]:
                row.pop("inventory_path")
            processes = [NS(name="target TP2"), NS(name="source TP1")]
            processes.append(NS(name="A-g3-tp1"))
            plan = dict(
                run_dir=root / "controller",
                instances=harness.manifest,
                extra_engines=[dict(instance_id="A-g3-tp1")],
                conversion=dict(
                    bundle="B",
                    target_mode="TP1",
                    trigger_s=600,
                    deadline_s=900,
                    base_port=8400,
                ),
                context_limits=(8192, 8192),
                config=dict(target_tp_size=2),
                registry=harness.registry,
                channel_plan=None,
            )
            plan["run_dir"].mkdir()
            driver = launch.start_conversion(
                dict(startup_timeout_s=1, run_timeout_s=1),
                dict(name="s"),
                plan,
                NS(),
                processes,
                lambda: None,
                [],
            )
            rows = driver.backend.rows
            self.assertEqual(
                rows["A-g1-tp1"]["inventory_path"],
                str(root / "controller" / "source_inventory.jsonl"),
            )
            self.assertEqual(
                rows["B-g2-tp2"]["inventory_path"],
                str(root / "controller" / "target_inventory.jsonl"),
            )
            outcome = launch.finish_conversion(driver, root)
            self.assertEqual(outcome["state"], "NOT_TRIGGERED")
            self.assertEqual(
                json.loads((root / "conversion_result.json").read_text())["state"],
                "NOT_TRIGGERED",
            )

    def test_runner_conversion_edges_build_on_the_l1a_layout(self):
        import contextlib
        import io

        from tools.bridge_tp import run_qwen3_stay_migrate as runner

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(
                ["--long-burst-memory-verify", "--conversion", "L1A-L2"]
                + ["--current-machine", "--model", "m", "--output", "o"]
                + ["--expected-revision", "CPU", "--validate-only"]
            )
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
        from tools.bridge_tp import launch_pressure_balanced_pilot as launch

        for edge, value in runner.CONVERSIONS.items():
            request = dict(value, trigger_s=30, deadline_s=420, base_port=8400)
            self.assertEqual(
                launch.conversion_request(
                    dict(layout=dict(name="L1A"), conversion=request), registry
                ),
                request,
                edge,
            )

    def test_nothing_is_quiesced_until_every_target_rank_follows_the_plan(self):
        """S3b-3 GPU (10-31): a target that rejected the plan file stalled the
        conversion after both sources were already retired."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness(root, "A", "TP2", channel_plan=True, acknowledge=False)
            plan = dict(
                plan_id="p",
                refs=[
                    dict(instance_id="A-g1-tp1", instance_generation=1),
                    dict(instance_id="A-g3-tp1", instance_generation=3),
                ],
            )
            backend = harness.backend
            self.assertFalse(backend.precheck(plan))
            self.assertFalse(backend.precheck(plan))
            reported = [
                e
                for e in harness.events
                if e["kind"] == "CHANNEL_PLAN_NOT_ACKNOWLEDGED"
            ]
            self.assertEqual(len(reported), 1)
            self.assertFalse((root / "layout_control.json").exists())
            directory = root / "s" / "gpu_channel_plan_receipts"
            directory.mkdir(parents=True)
            receipt = dict(opened=[], closed=[], deferred=[], completed_unix_s=1)
            (directory / "tp_rank_0.json").write_text(
                json.dumps(dict(receipt, planned=[1, 3]))
            )
            # A receipt without the plan's generations does not count.
            (directory / "tp_rank_1.json").write_text(
                json.dumps(dict(receipt, planned=[1]))
            )
            self.assertFalse(backend.precheck(plan))
            (directory / "tp_rank_1.json").write_text(
                json.dumps(dict(receipt, planned=[1, 3]))
            )
            self.assertTrue(backend.precheck(plan))

    def test_channel_close_that_never_arrives_fails_before_any_engine_stops(self):
        """S3b-3 GPU run 2 (10-31) idled in FENCED until the workload ended."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness(root, "A", "TP2", channel_plan=True)
            backend = harness.backend
            harness.event = lambda kind, **values: None  # target never closes
            backend.ops.event = harness.event
            plan = dict(
                plan_id="p",
                refs=[
                    dict(instance_id="A-g1-tp1", instance_generation=1),
                    dict(instance_id="A-g3-tp1", instance_generation=3),
                ],
            )
            (root / "layout_status.json").write_text(
                json.dumps(
                    dict(
                        unix_s=time.time(),
                        instances={
                            name: dict(lifecycle="RETIRED")
                            for name in ("A-g1-tp1", "A-g3-tp1")
                        },
                    )
                )
            )
            now = [time.time()]
            backend.clock = lambda: now[0]
            self.assertFalse(backend.retire_channels(plan))
            now[0] += 61
            with self.assertRaises(RuntimeError):
                backend.retire_channels(plan)
            self.assertEqual(harness.stopped, set())

    def test_channel_retirement_waits_for_every_target_rank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            harness = Harness(root, "A", "TP2", channel_plan=True)
            backend = harness.backend
            plan = dict(
                plan_id="p",
                refs=[
                    dict(instance_id="A-g1-tp1", instance_generation=1),
                    dict(instance_id="A-g3-tp1", instance_generation=3),
                ],
            )
            # Controller already shows both retired.
            (root / "layout_status.json").write_text(
                json.dumps(
                    dict(
                        unix_s=time.time(),
                        instances={
                            name: dict(lifecycle="RETIRED")
                            for name in ("A-g1-tp1", "A-g3-tp1")
                        },
                    )
                )
            )
            harness.event = lambda kind, **values: None  # no receipts at all
            backend.ops.event = harness.event
            self.assertFalse(backend.retire_channels(plan))
            receipts = root / "s" / "gpu_channel_plan_receipts"
            receipts.mkdir(parents=True)
            stamp = time.time() + 1
            (receipts / "tp_rank_0.json").write_text(
                json.dumps(dict(closed=[1, 3], completed_unix_s=stamp))
            )
            self.assertFalse(backend.retire_channels(plan))
            (receipts / "tp_rank_1.json").write_text(
                json.dumps(dict(closed=[1], deferred=[3], completed_unix_s=stamp))
            )
            self.assertFalse(backend.retire_channels(plan))
            (receipts / "tp_rank_1.json").write_text(
                json.dumps(dict(closed=[3], completed_unix_s=stamp + 1))
            )
            # Closed everywhere; teardown receipts are awaited for up to 10s.
            now = [time.time()]
            backend.clock = lambda: now[0]
            self.assertFalse(backend.retire_channels(plan))
            for rank in (0, 1):
                (receipts / f"tp_rank_{rank}-retired-1.json").write_text(
                    json.dumps(dict(status="ABORTED"))
                )
            (receipts / "tp_rank_0-retired-3.json").write_text(
                json.dumps(dict(status="TEARDOWN_STARTED"))
            )
            self.assertFalse(backend.retire_channels(plan))
            now[0] += 11  # one teardown never returned: proceed and record it
            self.assertTrue(backend.retire_channels(plan))


if __name__ == "__main__":
    unittest.main()
