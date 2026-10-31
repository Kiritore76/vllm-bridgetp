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
QWEN3_NATIVE_CONTEXT = 32768
# Physical GPU order handed to layouts(): first pair is bundle A, second B.
DEFAULT_BUNDLE_GPUS = "0,1,2,3"
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

    def context_limits(self):
        """(TP1, TP2+) model length. Native 32K needs explicit opt-in."""
        if getattr(self.args, "platform", "a100") != "qwen3":
            return 8192, 8192
        if getattr(self.args, "expanded_context", False):
            return QWEN3_NATIVE_CONTEXT, QWEN3_NATIVE_CONTEXT
        return 8192, 16384

    def launch(self, bundle, tp, gpus):
        """Register and spawn one owned engine; readiness is awaited separately."""
        tp1_limit, merged_limit = self.context_limits()
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
                merged_limit if tp > 1 else tp1_limit,
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
            if getattr(self.args, "platform", "a100") == "qwen3":
                from tools.bridge_tp.qwen3_calibration_contract import (
                    engine_environment,
                )

                engine_environment(env)
            settings = SimpleNamespace(
                python_bin=Path(sys.executable),
                model_path=self.args.model_path,
                dtype="bfloat16",
                max_model_len=tp1_limit,
                tp4_max_model_len=merged_limit,
                gpu_memory_utilization=0.88,
            )
            argv = self.common.server_command(settings, tp, port)
            if getattr(self.args, "platform", "a100") == "qwen3":
                from tools.bridge_tp.qwen3_calibration_contract import engine_arguments

                argv.extend(engine_arguments())
            self.common.write_json(path / "argv.json", argv)
            self.event(
                "LOAD_START",
                instance=name,
                bundle=bundle,
                tp=tp,
                gpus=gpus,
                max_model_len=row.context_limit,
            )
            process = self.common.start_process(name, argv, env, path / "server.log")
            self.processes[name] = process
            self.inventories[name] = path / "inventory.jsonl"
        return row

    def await_ready(self, row, activate=True):
        name, path = row.ref.instance_id, self.args.out_root / row.ref.instance_id
        self.common.wait_healthy(
            row.endpoint, self.processes[name], self.args.startup_timeout_s
        )
        self.event("HEALTHY", instance=name, bundle=row.bundle_id, tp=row.tp_degree)
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
            self.registry.transition(row.ref, "ACTIVE")
        self.event("READY", instance=name, bundle=row.bundle_id, tp=row.tp_degree)
        return row

    def start(self, bundle, tp, gpus, activate=True):
        return self.await_ready(self.launch(bundle, tp, gpus), activate)

    def start_parallel(self, rows, activate):
        """Spawn every engine first, then await them concurrently.

        Serial start would count each extra engine's load time in rebuild cost.
        """
        launched = [
            self.launch(r["bundle_id"], r["tp_degree"], r["gpu_ids"]) for r in rows
        ]
        with ThreadPoolExecutor(max_workers=max(1, len(launched))) as pool_:
            futures = [
                pool_.submit(self.await_ready, row, activate) for row in launched
            ]
            errors = []
            for future in futures:
                try:
                    future.result()
                except Exception as error:  # report after every wait has ended
                    errors.append(error)
        if errors:
            raise errors[0]
        return launched

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
        # Every observation is kept; plan.json only holds the latest check.
        with (self.args.out_root / "drain_observations.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(
                json.dumps(
                    dict(
                        collected_unix_s=time.time(),
                        rows=[vars(row) for row in evidence],
                    )
                )
                + "\n"
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
        self.start_parallel(
            [
                dict(bundle_id=plan["bundle_id"], tp_degree=tp, gpu_ids=devices)
                for devices in ([gpus] if tp > 1 else [[g] for g in gpus])
            ],
            activate=False,
        )

    def warm_and_verify(self, plan):
        """One fresh, non-routed request per replacement before publication."""
        from tools.bridge_tp.run_pressure_calibration import payload

        rows = [
            r
            for r in self.registry.instances.values()
            if r.bundle_id == plan["bundle_id"] and r.lifecycle != "RETIRED"
        ]
        if not rows or any(r.lifecycle != "PREPARING" for r in rows):
            return False
        verified = self.__dict__.setdefault("verified", set())
        for row in rows:
            name = row.ref.instance_id
            if name in verified:
                continue
            result = self.io.post_streaming_completion(
                row.endpoint,
                payload(self.inputs, 2048, 16, "verify-" + name),
                120,
                lambda *a: None,
                self.args.out_root / name / "verify_failure.json",
                retain_chunks=False,
            )
            if len(result["token_ids"]) != 16:
                raise RuntimeError(f"replacement verification incomplete: {name}")
            verified.add(name)
            self.event("VERIFIED", instance=name, bundle=row.bundle_id)
        return True

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
    parser.add_argument("--platform", choices=("a100", "qwen3"), default="a100")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--expected-revision")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument(
        "--empty",
        action="store_true",
        help="empty lifecycle cycle, no benchmark requests",
    )
    parser.add_argument(
        "--expanded-context",
        action="store_true",
        help="Qwen3 native 32768 limit for every TP1 and TP2 engine",
    )
    parser.add_argument(
        "--bundle-gpus",
        default=DEFAULT_BUNDLE_GPUS,
        help="four physical GPU ids; first pair is bundle A, second bundle B",
    )
    args = parser.parse_args(argv)
    initial, final, bundle, mode = EDGES[args.edge]
    if args.expanded_context and args.platform != "qwen3":
        raise ValueError("expanded context is defined only for Qwen3")
    bundle_gpus = [g.strip() for g in args.bundle_gpus.split(",")]
    if any(not g.isdigit() for g in bundle_gpus):
        raise ValueError("bundle GPUs must be physical indices")
    layouts = pool.layouts(2, bundle_gpus)
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
                    bundle_gpus=bundle_gpus,
                    expanded_context=args.expanded_context,
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
    if args.platform == "a100" and (
        len(names) != 4 or any(n.strip() != "NVIDIA A100-PCIE-40GB" for n in names)
    ):
        raise RuntimeError("requires four A100 PCIe 40GB")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPUs busy; nothing stopped")
    if args.platform == "qwen3":
        from tools.bridge_tp.prepare_4090d_qwen3 import PROFILE, read, runtime_inventory
        from tools.bridge_tp.qwen3_calibration_contract import (
            migration_spec,
            require_revision,
            verified_inputs,
        )

        require_revision(args.expected_revision)
        runtime_inventory(read(PROFILE))
        inputs = verified_inputs(args.model_path, args.inputs)
        spec = migration_spec(args.model_path)
        if args.expanded_context:
            positions = json.loads((args.model_path / "config.json").read_text())[
                "max_position_embeddings"
            ]
            if positions < QWEN3_NATIVE_CONTEXT:
                raise ValueError("model does not natively support 32K context")
    else:
        spec = prepare(args.model_path)
        inputs = json.loads(
            (
                ROOT / "experiments/phase9/pressure_balanced_v2/calibration.json"
            ).read_text()
        )
    import hashlib

    if (
        hashlib.sha256((args.model_path / "config.json").read_bytes()).hexdigest()
        != spec["model_config_sha256"]
    ):
        raise ValueError("frozen model identity differs")
    args.out_root.mkdir(parents=True, exist_ok=False)
    registry = pool.Registry()
    from tools.bridge_tp.freeze_pressure_baseline import cloud_identity, freeze

    freeze(args.out_root / "source_identity")
    common.write_json(
        args.out_root / "runtime_identity.json",
        cloud_identity(
            args.model_path, spec["checkpoint"] if args.platform == "a100" else None
        ),
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
        registry, backend, args.out_root / "plan.json", clock=time.time
    )
    # Only an observed COMPLETED plan may end with code 0.
    result, code = dict(status="INCOMPLETE"), 1
    arrival_stop, arrival_errors = threading.Event(), []
    arrival_thread = None
    try:
        backend.start_parallel(layouts[initial], activate=True)
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
        result, code = dict(status="COMPLETED", plan=executor.plan_status()), 0
    except KeyboardInterrupt:
        result, code = (
            dict(status="INTERRUPTED", plan=executor.plan_status()),
            130,
        )
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
        if not args.no_archive:
            import tarfile

            with tarfile.open(str(args.out_root) + ".tar.gz", "w:gz") as archive:
                archive.add(args.out_root, arcname=args.out_root.name)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
