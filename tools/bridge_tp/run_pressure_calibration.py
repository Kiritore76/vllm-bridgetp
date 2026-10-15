# SPDX-License-Identifier: Apache-2.0
"""Collect only missing no-migration pressure/output-length baselines."""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import launch_pressure_balanced_pilot as pilot  # noqa: E402
from tools.bridge_tp.pressure_cpu import load  # noqa: E402
from tools.bridge_tp.pressure_progress import save_progress  # noqa: E402

io = load("online_io")


def payload(config, prompt_tokens, max_tokens, request_id):
    """Reuse tracked token inputs, with explicit context and output bounds."""
    key = "2048" if prompt_tokens <= 2048 else "6144"
    result = copy.deepcopy(config["requests"][key])
    tokens = result["prompt"]
    if len(tokens) != prompt_tokens:
        # Controlled performance inputs, not natural-EOS prediction evaluation.
        tokens = tokens[: prompt_tokens - 32] + tokens[-32:]
    if len(tokens) != prompt_tokens or prompt_tokens + max_tokens > 8192:
        raise ValueError("invalid prompt or output context boundary")
    result.update(
        prompt=tokens,
        max_tokens=max_tokens,
        ignore_eos=True,
        seed=20261015,
        request_id=request_id,
        stream=True,
        return_token_ids=True,
    )
    return result


def validate_config(config):
    if (
        config["schema"] != "PRESSURE_CALIBRATION_V1"
        or config["migration_enabled"] is not False
        or config["repetitions"] != 2
        or config["itl_s"] != 0.1
        or config["slow_fraction"] != 0.01
        or len(config["scenes"]) != 5
    ):
        raise ValueError("calibration scope differs from frozen plan")
    names = set()
    for scene in config["scenes"] + config.get("pressure_screen_scenes", []):
        if scene["name"] in names:
            raise ValueError("duplicate scene")
        names.add(scene["name"])
        if not 1 <= scene["source_count"] <= 6 or not 1 <= scene["target_count"] <= 16:
            raise ValueError("concurrency outside collection scope")
        for pool in ("source", "target"):
            payload(
                config,
                scene["prompt_tokens"],
                scene.get(pool + "_max_tokens", scene["max_tokens"]),
                "validate",
            )
        if set(scene.get("injection_pools", ["source", "target"])) - {
            "source",
            "target",
        }:
            raise ValueError("invalid injection pool")
        for count in scene.get("injection_prompt_tokens", []):
            payload(config, count, 16, "validate")


def latest_inventory(path):
    """Read the last complete native sample; incomplete append stays unread."""
    if not path.exists():
        return None
    for line in reversed(path.read_bytes().splitlines(keepends=True)):
        if not line.endswith(b"\n"):
            continue
        row = json.loads(line)
        if row.get("schema") == "PRESSURE_INVENTORY_V1":
            if any(
                row.get(k)
                for k in (
                    "writer_error",
                    "inventory_error",
                    "dropped",
                    "dropped_before",
                )
            ):
                raise RuntimeError("native inventory integrity error")
            return row
    return None


def wait_empty(run_dir, timeout_s=30, after_generations=None):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        rows = [
            latest_inventory(run_dir / f"{p}_inventory.jsonl")
            for p in ("source", "target")
        ]
        # Native inventory is event driven: a correctly idle pool may not append
        # another row. Require advancement after the measured work, not a fake
        # heartbeat on the previously accepted empty state.
        if all(
            r
            and r["sampled_at"] <= time.time()
            and (after_generations is None or r["generation"] > after_generations[i])
            and not r["requests"]
            and not r["running"]
            and not r["waiting"]
            for i, r in enumerate(rows)
        ):
            return rows
        time.sleep(0.2)
    raise TimeoutError("native inventory did not settle empty")


def capacity_check(scene, inventories):
    """Reject physically infeasible test shapes, without changing online routing."""
    receipt = {}
    for pool, inv in zip(("source", "target"), inventories):
        count = scene[pool + "_count"]
        tokens = scene["prompt_tokens"] + scene.get(
            pool + "_max_tokens", scene["max_tokens"]
        )
        per_request = (tokens + inv["block_size"] - 1) // inv["block_size"]
        injection = max(scene.get("injection_prompt_tokens", [0])) + (
            16 if "injection_prompt_tokens" in scene else 0
        )
        extra = (
            (injection + inv["block_size"] - 1) // inv["block_size"]
            if pool in scene.get("injection_pools", ["source", "target"])
            else 0
        )
        required = count * per_request + extra
        receipt[pool] = dict(required_blocks=required, free_blocks=inv["free_blocks"])
        if required > inv["free_blocks"]:
            raise ValueError(
                f"test shape not physically feasible: {pool} {receipt[pool]}"
            )
    return receipt


class Workload:
    """Bounded SSE clients, per-request durable tokens and concise progress."""

    def __init__(self, config, run_dir, urls, timeout_s):
        self.config, self.run_dir, self.urls = config, run_dir, urls
        self.timeout_s = timeout_s
        self.lock = threading.Lock()
        self.records = {}
        self.stop = threading.Event()
        self.futures = []
        self.executor = ThreadPoolExecutor(max_workers=32)
        self.observer_error = None

    def submit(self, pool, prompt, output, label):
        request = payload(self.config, prompt, output, label)
        row = dict(
            id=label,
            pool=pool,
            prompt_tokens=prompt,
            max_tokens=output,
            ignore_eos=True,
            seed=request["seed"],
            submitted_unix_s=time.time(),
            tokens=0,
            first_token_unix_s=None,
            done=False,
            error=None,
        )
        with self.lock:
            self.records[label] = row
        io.atomic_json_dump(row, self.run_dir / f"{label}.request.json")

        def run():
            events = self.run_dir / f"{label}.tokens.jsonl"
            try:
                with events.open("w", encoding="utf-8") as handle:

                    def sink(index, token, timestamp):
                        handle.write(
                            json.dumps(
                                dict(index=index, token_id=token, unix_s=timestamp)
                            )
                            + "\n"
                        )
                        with self.lock:
                            row["tokens"] = index + 1
                            if row["first_token_unix_s"] is None:
                                row["first_token_unix_s"] = timestamp

                    result = io.post_streaming_completion(
                        self.urls[pool],
                        request,
                        self.timeout_s,
                        sink,
                        self.run_dir / f"{label}.failure.json",
                        retain_chunks=False,
                    )
                result.pop("chunks", None)
                token_ids = result.pop("token_ids")
                if len(token_ids) != output or result["finish_reason"] != "length":
                    raise RuntimeError("forced-output stream did not reach stop cap")
                with self.lock:
                    row["response"] = result
            except Exception as error:
                with self.lock:
                    row["error"] = repr(error)
                raise
            finally:
                with self.lock:
                    row.update(done=True, completed_unix_s=time.time())
                    snapshot = copy.deepcopy(row)
                io.atomic_json_dump(snapshot, self.run_dir / f"{label}.result.json")

        future = self.executor.submit(run)
        self.futures.append(future)
        return label, future

    def check(self):
        if self.observer_error is not None:
            raise RuntimeError(self.observer_error)
        for future in list(self.futures):
            if future.done():
                future.result()

    def wait(self, predicate, deadline):
        while not predicate():
            self.check()
            if self.stop.is_set() or time.monotonic() >= deadline:
                raise TimeoutError("workload deadline or stop reached")
            time.sleep(0.1)
        self.check()

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(list(self.records.values()))


def drive(work, scene, repetition, root):
    deadline = time.monotonic() + work.timeout_s
    background = {}
    for pool in ("source", "target"):
        background[pool] = [
            work.submit(
                pool,
                scene["prompt_tokens"],
                scene.get(pool + "_max_tokens", scene["max_tokens"]),
                f"r{repetition}-{pool}-background-{index}",
            )[0]
            for index in range(scene[pool + "_count"])
        ]
    if "injection_prompt_tokens" in scene:
        # Do not call a scheduled arrival "mixed" unless background streams live.
        def ready():
            rows = {r["id"]: r for r in work.snapshot()}
            return all(
                rows[k]["first_token_unix_s"] is not None
                for keys in background.values()
                for k in keys
            )

        work.wait(ready, deadline)
        if work.stop.wait(scene.get("settle_s", 0)):
            raise RuntimeError("workload stopped before injection")
        for size in scene["injection_prompt_tokens"]:
            for wave in range(scene["injections_per_size"]):
                rows = {r["id"]: r for r in work.snapshot()}
                if any(rows[k]["done"] for keys in background.values() for k in keys):
                    raise RuntimeError("mixed-load background ended before injection")
                injected = [
                    work.submit(
                        pool, size, 16, f"r{repetition}-{pool}-prefill-{size}-{wave}"
                    )[1]
                    for pool in scene.get("injection_pools", ["source", "target"])
                ]
                work.wait(lambda batch=injected: all(f.done() for f in batch), deadline)
                for future in injected:
                    future.result()
                work.stop.wait(scene["recovery_s"])
    work.wait(lambda: all(f.done() for f in work.futures), deadline)
    io.atomic_json_dump(
        {"status": "COMPLETED", "migration_starts": 0, "requests": work.snapshot()},
        root / "request_results.json",
    )


def observe(work, receipt):
    """Persist actual inventory observation times and independently live metrics."""
    observation = load("pressure_observation")
    telemetry = load("telemetry")
    readers = {
        pool: observation.InventoryReader(
            work.run_dir / f"{pool}_inventory.jsonl",
            "tp1" if pool == "source" else "tp4",
        )
        for pool in ("source", "target")
    }
    scrapers = {}
    with (work.run_dir / "observation_events.jsonl").open("w", encoding="utf-8") as out:
        while not work.stop.is_set():
            for pool, reader in readers.items():
                now = time.time()
                try:
                    inventory = reader.poll(now)
                    if reader.error:
                        work.observer_error = reader.error
                        raise RuntimeError(reader.error)
                    if inventory is None:
                        continue
                    if pool not in scrapers:
                        scrapers[pool] = telemetry.MetricsScraper(
                            work.urls[pool],
                            inventory["block_size"],
                            inventory["usage_denominator_blocks"],
                            timeout_s=1,
                        )
                    metrics = observation.scrape_inventory_metrics(
                        scrapers[pool], inventory
                    )
                    event = dict(
                        pool=pool,
                        generation=inventory["generation"],
                        sampled_at=inventory["sampled_at"],
                        observed_at=reader.observed_at,
                        metrics=metrics,
                    )
                    receipt["valid_records"] += 1
                    counts = receipt.setdefault("valid_by_pool", {})
                    counts[pool] = counts.get(pool, 0) + 1
                except Exception as error:
                    event = dict(pool=pool, error=repr(error))
                    receipt["error_records"] += 1
                event["captured_unix_s"] = time.time()
                out.write(json.dumps(event) + "\n")
                out.flush()
            work.stop.wait(0.2)


def run_scene(
    spec, config, scene, repetition, root, common, settings, capture_observations=False
):
    from tools.bridge_tp.run_atomic_takeover import _post_json
    from tools.bridge_tp.run_phase9_cap0_noop import warmup_prompt_shapes
    from tools.bridge_tp.run_phase9_controller import RuntimeControl
    from tools.bridge_tp.verify_performance_trace import verify_pool

    root.mkdir(parents=True, exist_ok=False)

    def stage(label, **details):
        save_progress(
            root.parent / "pilot_progress.json", scene=root.name, stage=label, **details
        )
        pilot.announce(root.name, label, **details)

    plan = pilot.build_scene(
        spec,
        {"name": root.name, "arrivals": spec["scenarios"][0]["arrivals"]},
        root.parent,
        common,
        settings,
    )
    run_dir = plan["run_dir"]
    run_dir.mkdir()
    RuntimeControl(
        armed=False,
        migration_id=root.name,
        source_request_id_prefix="calibration-unarmed-",
    ).write(run_dir)
    # No controller or stager is started. Source hooks remain unarmed throughout.
    urls = {pool: plan["config"][pool + "_url"] for pool in ("source", "target")}
    processes, work, observer = [], None, None
    observation_receipt = dict(valid_records=0, error_records=0)
    warm_requests = [
        payload(config, size, 16, "warm")
        for size in sorted(
            {scene["prompt_tokens"], *scene.get("injection_prompt_tokens", [])}
        )
    ]
    for pool in ("source", "target"):
        plan[pool + "_env"].update(
            BRIDGETP_PERFORMANCE_TRACE_DIR=str(run_dir / "performance_trace" / pool),
            BRIDGETP_PERFORMANCE_SAMPLE_EVERY="8",
            BRIDGETP_PERFORMANCE_FINALIZE_TOKEN=secrets.token_hex(32),
        )
    common.write_json(
        root / "effective_argv.json",
        {p: plan[p + "_argv"] for p in ("source", "target")},
    )
    common.write_json(
        root / "effective_environment.json",
        {
            p: {
                k: v
                for k, v in plan[p + "_env"].items()
                if k.startswith(("BRIDGETP_", "CUDA_", "OMP_", "VLLM_"))
                and not k.endswith("TOKEN")
            }
            for p in ("source", "target")
        },
    )
    common.write_json(root / "recipe.json", scene)
    try:
        for pool in ("target", "source"):
            stage("启动服务", pool=pool)
            process = common.start_process(
                pool, plan[pool + "_argv"], plan[pool + "_env"], root / f"{pool}.log"
            )
            processes.append(process)
            common.wait_healthy(urls[pool], process, spec["startup_timeout_s"])
            stage("暖机", pool=pool)
            warmup_prompt_shapes(
                urls[pool],
                warm_requests,
                root / f"{pool}_warmup.json",
                spec["run_timeout_s"],
            )
        inventories = wait_empty(run_dir)
        initial_generations = [r["generation"] for r in inventories]
        common.write_json(
            root / "capacity_check.json", capacity_check(scene, inventories)
        )
        common.write_json(
            root / "measurement_window.json", {"started_unix_s": time.time()}
        )
        work = Workload(config, run_dir, urls, spec["run_timeout_s"])
        if capture_observations:
            observer = threading.Thread(
                target=observe, args=(work, observation_receipt), daemon=True
            )
            observer.start()
        driver = threading.Thread(
            target=drive_guarded, args=(work, scene, repetition, root), daemon=True
        )
        driver.start()
        while driver.is_alive():
            rows = work.snapshot()
            status = dict(
                scene=root.name,
                stage="采集中",
                submitted=len(rows),
                completed=sum(r["done"] for r in rows),
                visible_tokens=sum(r["tokens"] for r in rows),
            )
            save_progress(root.parent / "pilot_progress.json", **status)
            pilot.announce(
                root.name,
                "采集中",
                **{k: v for k, v in status.items() if k not in {"scene", "stage"}},
            )
            driver.join(10)
        result = pilot.read(root / "request_results.json")
        if result["status"] != "COMPLETED":
            raise RuntimeError(result["error"])
        if capture_observations and not all(
            observation_receipt.get("valid_by_pool", {}).get(pool, 0)
            for pool in ("source", "target")
        ):
            raise RuntimeError("no live observation evidence recorded")
        inventories = wait_empty(run_dir, after_generations=initial_generations)
        common.write_json(root / "native_empty_receipt.json", inventories)
        window = pilot.read(root / "measurement_window.json")
        window["ended_unix_s"] = time.time()
        common.write_json(root / "measurement_window.json", window)
        for pool, ranks in (("source", 1), ("target", 4)):
            stage("收尾性能轨迹", pool=pool)
            result = _post_json(
                urls[pool],
                "/bridge_tp/v1/performance_trace/finalize",
                {"token": plan[pool + "_env"]["BRIDGETP_PERFORMANCE_FINALIZE_TOKEN"]},
                60,
            )
            common.write_json(root / f"{pool}_trace_finalization.json", result)
            if result.get("status") != "FINALIZED":
                raise RuntimeError("trace finalization failed")
            receipt = verify_pool(run_dir / "performance_trace" / pool, ranks)
            common.write_json(root / f"{pool}_trace_acceptance.json", receipt)
        return dict(
            status="COMPLETED",
            coverage="inspect actual Nd/chunks/context",
            requests=len(work.records),
            migration_starts=0,
        )
    finally:
        if work:
            work.stop.set()
        if observer:
            observer.join(5)
            common.write_json(root / "observation_receipt.json", observation_receipt)
        stage("清理本场受管进程")
        common.stop_processes(processes)
        if work:
            work.executor.shutdown(wait=True, cancel_futures=True)
        ended = not any(common.process_group_alive(p) for p in processes)
        common.write_json(
            root / "owned_shutdown_receipt.json",
            dict(
                all_owned_groups_ended=ended,
                processes=[
                    dict(name=p.name, pid=p.process.pid, returncode=p.process.poll())
                    for p in processes
                ],
            ),
        )
        if not ended:
            raise RuntimeError("owned process groups did not end")


def drive_guarded(work, scene, repetition, root):
    try:
        drive(work, scene, repetition, root)
    except Exception as error:
        io.atomic_json_dump(
            {
                "status": "TECHNICAL_FAILURE",
                "error": repr(error),
                "requests": work.snapshot(),
            },
            root / "request_results.json",
        )


def check_gpu():
    names = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines()
    if len(names) != 5 or any(n.strip() != "NVIDIA A100-PCIE-40GB" for n in names):
        raise RuntimeError("five A100-PCIE-40GB GPUs required")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPUs busy; no unrelated processes will be stopped")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--pressure-screen", action="store_true")
    args = parser.parse_args()
    spec = pilot.load_spec(args.spec)
    pilot.verify_spec(spec)
    config_path = args.spec.parent / "calibration.json"
    config = pilot.read(config_path)
    validate_config(config)
    selected = (
        config["pressure_screen_scenes"] if args.pressure_screen else config["scenes"]
    )
    total_scenes = len(selected) * config["repetitions"]
    common, settings = pilot.arguments(spec)
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    for tp, port in ((1, settings.tp1_port), (4, settings.tp4_port)):
        command = common.server_command(settings, tp, port)
        actual = make_arg_parser(FlexibleArgumentParser()).parse_args(command[3:])
        validate_parsed_serve_args(actual)
    if args.validate_only:
        pilot.announce("calibration", "CPU/输入/serving参数检查通过；尚未启动GPU")
        return 0
    check_gpu()
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "unittest",
            "tests.bridge_tp.test_pressure_calibration",
            "tests.bridge_tp.test_prefill_cost_reference",
            "tests.bridge_tp.test_rolling_cutover",
            "tests.bridge_tp.test_pressure_observation",
            "tests.bridge_tp.test_performance_trace",
        ],
        cwd=ROOT,
        check=True,
    )
    args.out_root.mkdir(parents=True, exist_ok=False)
    common.write_json(args.out_root / "frozen_spec.json", spec)
    common.write_json(args.out_root / "calibration_config.json", config)
    common.write_json(
        args.out_root / "selection.json",
        dict(
            mode="pressure-screen" if args.pressure_screen else "calibration",
            scenes=[s["name"] for s in selected],
            repetitions=config["repetitions"],
            migration_enabled=False,
            capture_observations=args.pressure_screen,
        ),
    )
    (args.out_root / "gpu_inventory.txt").write_text(
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=uuid,name,memory.total", "--format=csv"],
            text=True,
        )
    )
    results = []
    for repetition in range(1, config["repetitions"] + 1):
        for scene in selected:
            name = f"{scene['name']}-r{repetition}"
            pilot.announce(
                name, "开始场景", number=f"{len(results) + 1}/{total_scenes}"
            )
            try:
                check_gpu()
                result = run_scene(
                    spec,
                    config,
                    scene,
                    repetition,
                    args.out_root / name,
                    common,
                    settings,
                    capture_observations=args.pressure_screen,
                )
            except Exception as error:
                result = dict(status="TECHNICAL_FAILURE", error=repr(error))
            results.append(dict(scene=name, **result))
            common.write_json(args.out_root / "scene_results.json", results)
            save_progress(
                args.out_root / "pilot_progress.json",
                scene=name,
                stage="场景完成" if result["status"] == "COMPLETED" else "失败",
                completed_scenes=len(results),
                total_scenes=total_scenes,
                **result,
            )
            pilot.announce(name, "场景结束", **result)
            if result["status"] != "COMPLETED":
                return 1  # Preserve the failed batch; do not burn more GPU runs.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
