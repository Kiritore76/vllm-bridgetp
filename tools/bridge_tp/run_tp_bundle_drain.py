# SPDX-License-Identifier: Apache-2.0
"""Controlled four-GPU DRAIN lifecycle test, not a pressure-policy comparison.

Uses real model requests and owned engines. No migration channel is configured.
The retained bundle is never stopped during the conversion. Final test cleanup
stops only this run's process groups. Start with --validate-only on CPU.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.pressure_cpu import load  # noqa: E402

pool = load("instance_pool")
conversion = load("reconfiguration_executor")
EDGES = {
    "L0-L1A": ("L0", "L1A", "B", "TP2"),
    "L1A-L0": ("L1A", "L0", "B", "TP1"),
    "L1A-L2": ("L1A", "L2", "A", "TP2"),
    "L2-L1A": ("L2", "L1A", "A", "TP1"),
}


class NativeDrainBackend:
    """Live backend scoped to a fresh run with NO request migration channels."""

    def __init__(self, args, registry, common, inputs):
        self.args, self.registry, self.common, self.inputs = (
            args,
            registry,
            common,
            inputs,
        )
        self.lock = threading.RLock()
        self.processes = {}
        self.inventories = {}
        self.future_by_request = {}
        self.next_port = args.base_port
        self.sequence = 0
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.events = []
        self.responses = []

    def event(self, kind, **values):
        row = dict(kind=kind, unix_s=time.time(), **values)
        with self.lock:
            with (self.args.out_root / "lifecycle.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(row) + "\n")
            self.events.append(row)
        print(json.dumps(row), flush=True)

    def start(self, bundle, tp, gpus, activate=True):
        with self.lock:
            self.sequence += 1
            name = f"{bundle}-g{self.sequence}-tp{tp}"
            port, self.next_port = self.next_port, self.next_port + 1
            ref = pool.InstanceRef(name, self.sequence)
            row = pool.InstanceRecord(
                ref,
                tp,
                tuple(gpus),
                bundle,
                f"http://127.0.0.1:{port}",
                str(self.args.model_path),
                8192,
            )
            self.registry.register(row)
            path = self.args.out_root / name
            path.mkdir()
            env = {k: v for k, v in os.environ.items() if not k.startswith("BRIDGETP_")}
            env.update(
                CUDA_VISIBLE_DEVICES=",".join(gpus),
                OMP_NUM_THREADS="1",
                BRIDGETP_PRESSURE_POOL="tp1" if tp == 1 else "tp4",
                BRIDGETP_PRESSURE_INVENTORY_PATH=str(path / "inventory.jsonl"),
            )
            settings = SimpleNamespace(
                python_bin=Path(sys.executable),
                model_path=self.args.model_path,
                dtype="bfloat16",
                max_model_len=8192,
                tp4_max_model_len=8192,
                gpu_memory_utilization=0.88,
            )
            argv = self.common.server_command(settings, tp, port)
            self.common.write_json(path / "argv.json", argv)
            self.event("LOAD_START", instance=name, bundle=bundle, tp=tp, gpus=gpus)
            process = self.common.start_process(name, argv, env, path / "server.log")
            self.processes[name] = process
            self.inventories[name] = path / "inventory.jsonl"
        self.common.wait_healthy(row.endpoint, process, self.args.startup_timeout_s)
        from tools.bridge_tp.run_pressure_calibration import payload

        request = payload(self.inputs, 2048, 16, "warm-" + name)
        self.io.post_streaming_completion(
            row.endpoint,
            request,
            120,
            lambda *a: None,
            path / "warm_failure.json",
            retain_chunks=False,
        )
        if activate:
            self.registry.transition(ref, "ACTIVE")
        self.event("READY", instance=name, bundle=bundle, tp=tp)
        return row

    @property
    def io(self):
        return load("online_io")

    def submit(self, request_id, output_tokens):
        from tools.bridge_tp.run_pressure_calibration import payload

        with self.lock:
            active = sorted(
                (
                    r
                    for r in self.registry.instances.values()
                    if r.lifecycle == "ACTIVE"
                ),
                key=lambda r: r.ref,
            )
            if not active:
                raise RuntimeError("no active retained service")
            row = active[len(self.future_by_request) % len(active)]
            self.registry.acquire_dispatch_lease(row.ref, request_id)
            request = payload(self.inputs, 2048, output_tokens, request_id)

            def run():
                events = self.args.out_root / (request_id + ".tokens.jsonl")
                try:
                    with events.open("x", encoding="utf-8") as stream:

                        def sink(index, token, stamp):
                            stream.write(
                                json.dumps(
                                    dict(index=index, token_id=token, unix_s=stamp)
                                )
                                + "\n"
                            )

                        result = self.io.post_streaming_completion(
                            row.endpoint,
                            request,
                            self.args.request_timeout_s,
                            sink,
                            self.args.out_root / (request_id + ".failure.json"),
                            retain_chunks=False,
                        )
                    if len(result["token_ids"]) != output_tokens:
                        raise RuntimeError("incomplete controlled output")
                    result.pop("chunks", None)
                    self.io.atomic_json_dump(
                        result, self.args.out_root / (request_id + ".result.json")
                    )
                    self.registry.reconcile_dispatch(row.ref, request_id, "TERMINAL")
                    return result
                except Exception:
                    # No blind resubmit and no declaration of drain on absence.
                    self.registry.reconcile_dispatch(row.ref, request_id, "UNCERTAIN")
                    raise

            self.future_by_request[request_id] = self.executor.submit(run)
            self.event(
                "DISPATCH",
                request_id=request_id,
                instance=row.ref.instance_id,
                output_tokens=output_tokens,
            )

    def precheck(self, plan):
        return all(
            self.registry.get(pool.InstanceRef(**r)).lifecycle == "ACTIVE"
            for r in plan["refs"]
        )

    def quiesce(self, plan):
        with self.lock:
            for value in plan["refs"]:
                self.registry.transition(pool.InstanceRef(**value), "QUIESCING")
            self.event("QUIESCE", plan_id=plan["plan_id"])

    def kv_emergency(self, plan):
        # This isolated fixture has no fast controller or migration channels.
        # Do not represent it as the frozen-fast DRAIN performance comparison.
        return False

    def drain_evidence(self, plan):
        from tools.bridge_tp.run_pressure_calibration import latest_inventory

        evidence = []
        for value in plan["refs"]:
            ref = pool.InstanceRef(**value)
            native = latest_inventory(self.inventories[ref.instance_id])
            if native is None:
                return []
            leases = [
                x
                for x in self.registry.leases.values()
                if x.ref == ref and x.state != "TERMINAL"
            ]
            # Native empty is event-driven. Revalidate it using fresh metrics;
            # retain the native sample separately rather than rewriting its age.
            telemetry = load("telemetry")
            scraper = telemetry.MetricsScraper(
                self.registry.get(ref).endpoint,
                native["block_size"],
                native["usage_denominator_blocks"],
                timeout_s=1,
            )
            observed = load("pressure_observation").scrape_inventory_metrics(
                scraper, native
            )
            stamp = native["sampled_at"]
            if (
                not native["requests"]
                and native["running"] == native["waiting"] == 0
                and observed.get("num_running") == observed.get("num_waiting") == 0
                and observed.get("free_kv_blocks") == native["free_blocks"]
            ):
                stamp = observed["sampled_unix_s"]
            tails = sum(
                not future.done()
                for request_id, future in self.future_by_request.items()
                if self.registry.leases[request_id].ref == ref
            )
            evidence.append(
                conversion.DrainEvidence(
                    ref.instance_id,
                    ref.instance_generation,
                    stamp,
                    max(native["running"], observed.get("num_running", 1)),
                    max(native["waiting"], observed.get("num_waiting", 1)),
                    max(len(leases), len(native["requests"])),
                    len(leases),
                    0,
                    0,
                    0,
                    0,
                    0,
                    tails,
                    tails,
                    True,
                )
            )
        return evidence

    def fence(self, plan):
        with self.lock:
            if any(
                x.state != "TERMINAL" and as_ref(x.ref) in plan["refs"]
                for x in self.registry.leases.values()
            ):
                return False
            self.event("FENCED", plan_id=plan["plan_id"])
            return True

    def retire_channels(self, plan):
        self.event(
            "CHANNEL_RETIRE",
            plan_id=plan["plan_id"],
            evidence="no transport configured for this lifecycle fixture",
        )
        return True

    def stop_owned_engines(self, plan):
        selected = [self.processes[r["instance_id"]] for r in plan["refs"]]
        self.common.stop_processes(selected)
        self.event("STOP_RETURN", plan_id=plan["plan_id"])

    def resources_released(self, plan):
        if any(
            self.common.process_group_alive(self.processes[r["instance_id"]])
            for r in plan["refs"]
        ):
            return False
        allocated = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid",
                "--format=csv,noheader",
            ],
            text=True,
        )
        target_ids = {
            gpu
            for r in plan["refs"]
            for gpu in self.registry.get(pool.InstanceRef(**r)).physical_gpu_ids
        }
        uuids = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True
        )
        target_uuids = {
            line.split(",")[1].strip()
            for line in uuids.splitlines()
            if line.split(",")[0].strip() in target_ids
        }
        if any(
            line.split(",")[0].strip() in target_uuids
            for line in allocated.splitlines()
        ):
            return False
        if len(target_uuids) != len(target_ids):
            raise RuntimeError("GPU UUID mapping incomplete")
        self.event("RESOURCES_RELEASED", plan_id=plan["plan_id"])
        return True

    def start_replacements(self, plan):
        gpus = sorted(
            gpu
            for r in plan["refs"]
            for gpu in self.registry.get(pool.InstanceRef(**r)).physical_gpu_ids
        )
        tp = int(plan["target_bundle_mode"][2:])
        for devices in [gpus] if tp > 1 else [[g] for g in gpus]:
            self.start(plan["bundle_id"], tp, devices, activate=False)

    def warm_and_verify(self, plan):
        rows = [
            r
            for r in self.registry.instances.values()
            if r.bundle_id == plan["bundle_id"] and r.lifecycle != "RETIRED"
        ]
        return bool(rows) and all(r.lifecycle == "PREPARING" for r in rows)

    def publish_ready(self, plan):
        with self.lock:
            for row in self.registry.instances.values():
                if row.bundle_id == plan["bundle_id"] and row.lifecycle == "PREPARING":
                    self.registry.transition(row.ref, "ACTIVE")
        self.event("ROUTE_PUBLISHED", plan_id=plan["plan_id"])

    def settle_before_abort(self, plan):
        return all(x.state != "UNCERTAIN" for x in self.registry.leases.values())

    def restore_admission(self, plan):
        self.event("ADMISSION_RESTORED", plan_id=plan["plan_id"])

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=False)
        live = [
            p for p in self.processes.values() if self.common.process_group_alive(p)
        ]
        self.common.stop_processes(live)
        receipt = dict(
            all_owned_groups_ended=not any(
                self.common.process_group_alive(p) for p in self.processes.values()
            ),
            instances=[
                dict(instance=name, pid=p.process.pid, returncode=p.process.poll())
                for name, p in self.processes.items()
            ],
        )
        self.common.write_json(
            self.args.out_root / "owned_shutdown_receipt.json", receipt
        )
        if not receipt["all_owned_groups_ended"]:
            raise RuntimeError("owned process group remains; retain evidence")
        return receipt


def as_ref(ref):
    return dict(
        instance_id=ref.instance_id, instance_generation=ref.instance_generation
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edge", choices=EDGES, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--trigger-s", type=float, default=20)
    parser.add_argument("--arrival-duration-s", type=float, default=60)
    parser.add_argument("--arrival-interval-s", type=float, default=4)
    parser.add_argument("--output-tokens", type=int, default=512)
    parser.add_argument("--startup-timeout-s", type=float, default=900)
    parser.add_argument("--request-timeout-s", type=float, default=600)
    parser.add_argument("--plan-timeout-s", type=float, default=900)
    parser.add_argument("--base-port", type=int, default=8500)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--empty",
        action="store_true",
        help="empty lifecycle cycle, no benchmark requests",
    )
    args = parser.parse_args(argv)
    initial, final, bundle, mode = EDGES[args.edge]
    layouts = pool.layouts(2, ["0", "1", "2", "3"])
    if any(
        not math.isfinite(v)
        for v in (
            args.trigger_s,
            args.arrival_duration_s,
            args.arrival_interval_s,
            args.request_timeout_s,
            args.plan_timeout_s,
        )
    ):
        raise ValueError("finite workload/deadlines required")
    if (
        min(
            args.trigger_s,
            args.arrival_duration_s,
            args.arrival_interval_s,
            args.request_timeout_s,
            args.plan_timeout_s,
        )
        <= 0
        or args.output_tokens < 1
        or args.output_tokens > 6144
    ):
        raise ValueError("invalid finite workload/deadline")
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    initial=layouts[initial],
                    final=layouts[final],
                    bundle=bundle,
                    target_mode=mode,
                    purpose="fixed-route lifecycle fixture; no fast policy",
                ),
                indent=2,
            )
        )
        return 0
    from tools.bridge_tp import run_phase9_cap0_calibration as common
    from tools.bridge_tp.run_tp2_calibration import prepare

    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != 4 or any(n.strip() != "NVIDIA A100-PCIE-40GB" for n in names):
        raise RuntimeError("requires four A100 PCIe 40GB")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPUs busy; nothing stopped")
    spec = prepare(args.model_path)
    import hashlib

    if (
        hashlib.sha256((args.model_path / "config.json").read_bytes()).hexdigest()
        != spec["model_config_sha256"]
    ):
        raise ValueError("frozen model identity differs")
    args.out_root.mkdir(parents=True, exist_ok=False)
    inputs = json.loads(
        (ROOT / "experiments/phase9/pressure_balanced_v2/calibration.json").read_text()
    )
    registry = pool.Registry()
    from tools.bridge_tp.freeze_pressure_baseline import cloud_identity, freeze

    freeze(args.out_root / "source_identity")
    common.write_json(
        args.out_root / "runtime_identity.json",
        cloud_identity(args.model_path, spec["checkpoint"]),
    )
    (args.out_root / "gpu_topology.txt").write_bytes(
        subprocess.check_output(["nvidia-smi", "topo", "-m"])
    )
    common.write_json(
        args.out_root / "experiment_config.json",
        {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    )
    backend = NativeDrainBackend(args, registry, common, inputs)
    executor = conversion.ConversionExecutor(
        registry, backend, args.out_root / "plan.json"
    )
    result, code = {}, 0
    arrival_stop, arrival_errors = threading.Event(), []
    arrival_thread = None
    try:
        for row in layouts[initial]:
            backend.start(row["bundle_id"], row["tp_degree"], row["gpu_ids"])
        origin = time.monotonic()
        trigger = origin + args.trigger_s

        def arrivals():
            try:
                submitted = 0
                while submitted * args.arrival_interval_s < args.arrival_duration_s:
                    scheduled = origin + submitted * args.arrival_interval_s
                    if arrival_stop.wait(max(0, scheduled - time.monotonic())):
                        return
                    lag = time.monotonic() - scheduled
                    backend.event("ARRIVAL", index=submitted, dispatch_lag_s=lag)
                    backend.submit(f"drain-{submitted:04d}", args.output_tokens)
                    submitted += 1
            except Exception as error:
                arrival_errors.append(repr(error))

        arrival_thread = threading.Thread(target=arrivals, daemon=True)
        if args.empty:
            args.arrival_duration_s = 0
        arrival_thread.start()
        while True:
            now = time.monotonic()
            elapsed = now - origin
            if arrival_errors:
                raise RuntimeError(arrival_errors[0])
            if now >= trigger and executor.plan is None:
                executor.plan_create(
                    "manual-drain",
                    registry.registry_revision,
                    bundle,
                    mode,
                    "DRAIN",
                    "ACTIVE",
                    time.time() + args.plan_timeout_s,
                    "NO_PROFILE",
                )
            if executor.plan:
                state = executor.plan_execute(time.time())["state"]
                if state in {"RECOVERY_REQUIRED", "ABORTED"}:
                    raise RuntimeError(f"conversion ended in {state}")
            with backend.lock:
                futures = list(backend.future_by_request.values())
            for future in futures:
                if future.done():
                    future.result()
            if (
                elapsed >= args.arrival_duration_s
                and executor.plan
                and executor.plan["state"] == "COMPLETED"
                and not arrival_thread.is_alive()
                and all(f.done() for f in futures)
            ):
                break
            if (
                elapsed
                > args.plan_timeout_s + args.arrival_duration_s + args.request_timeout_s
            ):
                raise TimeoutError("finite lifecycle experiment deadline")
            time.sleep(0.1)
        result = dict(status="COMPLETED", plan=executor.plan_status())
    except Exception as error:
        result, code = (
            dict(
                status="TECHNICAL_FAILURE",
                error=repr(error),
                plan=executor.plan_status(),
            ),
            1,
        )
    finally:
        arrival_stop.set()
        if arrival_thread:
            arrival_thread.join(5)
        try:
            backend.close()
        except Exception as error:
            result.update(status="TECHNICAL_FAILURE", shutdown_error=repr(error))
            code = 1
        result.update(
            runner_code=code,
            purpose="fixed-route lifecycle fixture",
            frozen_fast_policy_validated=False,
            migration_validated=False,
        )
        common.write_json(args.out_root / "result.json", result)
        import tarfile

        with tarfile.open(str(args.out_root) + ".tar.gz", "w:gz") as archive:
            archive.add(args.out_root, arcname=args.out_root.name)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
