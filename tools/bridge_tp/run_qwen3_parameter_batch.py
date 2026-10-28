# SPDX-License-Identifier: Apache-2.0
"""One bounded batch of service, empty DRAIN and explicit migration measurements."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.prepare_4090d_qwen3 import (  # noqa: E402
    PROFILE,
    inspect_model,
    read,
    runtime_inventory,
    sha,
)
from tools.bridge_tp.qwen3_calibration_contract import (  # noqa: E402
    INPUT_SHA,
    require_revision,
    runtime_versions,
    verified_inputs,
)
from tools.bridge_tp.result_archive import pack  # noqa: E402
from tools.bridge_tp.run_qwen3_tp2_calibration import (  # noqa: E402
    accepted_scene,
    service_facts,
)
from tools.bridge_tp.run_qwen3_tp2_calibration import (  # noqa: E402
    main as service_main,
)
from tools.bridge_tp.run_tp2_migration_smoke import main as migration_main  # noqa: E402
from tools.bridge_tp.run_tp_bundle_drain import EDGES  # noqa: E402
from tools.bridge_tp.run_tp_bundle_drain import main as drain_main  # noqa: E402
from tools.bridge_tp.summarize_tp2_calibration import (  # noqa: E402
    summarize,
    workload_from_cells,
)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def gpu_idle():
    return not subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        text=True,
    ).strip()


def stages():
    return (
        [dict(name="service", kind="service", units=7)]
        + [
            dict(name="empty-" + edge, kind="empty_drain", edge=edge, units=1)
            for edge in EDGES
        ]
        + [
            dict(name=f"migration-{n}", kind="migration", history=n, units=1)
            for n in (128, 2048)
        ]
    )


def child_arguments(args, stage):
    directory = args.output / stage["name"]
    if stage["kind"] == "service":
        return [
            "--model",
            str(args.model),
            "--output",
            str(directory),
            "--expected-revision",
            args.expected_revision,
            "--phase",
            "remaining",
            "--prior-service-attempts",
            "3",
            "--pilot-batch",
            str(args.pilot_batch),
            "--no-archive",
        ]
    common = [
        "--model-path",
        str(args.model),
        "--out-root",
        str(directory),
        "--platform",
        "qwen3",
        "--inputs",
        str(args.pilot_batch / "inputs.json"),
        "--expected-revision",
        args.expected_revision,
        "--no-archive",
    ]
    if stage["kind"] == "migration":
        return common + ["--history-tokens", str(stage["history"])]
    return common + [
        "--edge",
        stage["edge"],
        "--empty",
        "--trigger-s",
        "1",
        "--arrival-duration-s",
        "1",
        "--base-port",
        "8700",
    ]


def complete(directory, kind):
    result = read(directory / "result.json")
    if result["status"] != "COMPLETED" or result["runner_code"] != 0:
        return False
    if kind == "service":
        attempts = read(directory / "attempt_ledger.json")
        return len(attempts) == 7 and all(r["status"] == "COMPLETED" for r in attempts)
    if kind == "migration":
        return bool(
            result["source_release_confirmed"] and result["all_owned_groups_ended"]
        )
    return bool(
        read(directory / "owned_shutdown_receipt.json")["all_owned_groups_ended"]
    )


def summary(output, pilot):
    """Keep unsupported costs null; completed individual measurements stay explicit."""
    profiles = [read(pilot / "service_profile.json")]
    if (output / "service/result.json").is_file():
        try:
            new_profile = summarize(output / "service")
            new_profile["native_capacity_and_startup"] = service_facts(
                output / "service"
            )
            profiles.append(new_profile)
        except Exception as error:
            write(output / "service_summary_error.json", dict(error=repr(error)))
    cells = [dict(c, batch=i) for i, p in enumerate(profiles) for c in p["cells"]]
    lookup = [
        dict(c, batch=i) for i, p in enumerate(profiles) for c in p["service_lookup"]
    ]
    workload, missing = workload_from_cells(cells, lookup)
    measurements, lifecycle = [], []
    for stage in stages()[1:]:
        directory = output / stage["name"]
        if not (directory / "result.json").is_file():
            continue
        result = read(directory / "result.json")
        if not complete(directory, stage["kind"]):
            continue
        if stage["kind"] == "migration":
            measurements.append(
                dict(
                    history_trigger_tokens=stage["history"],
                    observed_generated_at_start=result["observed_generated_at_start"],
                    start_to_source_release_s=result["start_to_release_s"],
                    start_to_release_and_idle_observed_s=result[
                        "start_to_source_release_and_idle_observed_s"
                    ],
                    visible_handoff_gap_s=result["visible_handoff_gap_s"],
                    native_initial=read(directory / "native_initial_receipt.json"),
                    evidence=str(
                        (directory / "mechanism/controller").relative_to(output)
                    ),
                    scope="single idle-target observation, not a general release bound",
                )
            )
        else:
            events = [
                json.loads(line) for line in (directory / "lifecycle.jsonl").open()
            ]
            plan = read(directory / "plan.json")
            starts = [e for e in events if e["kind"] == "LOAD_START"]
            releases = [e for e in events if e["kind"] == "RESOURCES_RELEASED"]
            published = [e for e in events if e["kind"] == "ROUTE_PUBLISHED"]
            lifecycle.append(
                dict(
                    edge=stage["edge"],
                    scope="empty DRAIN, no retained background traffic",
                    plan_state=plan["state"],
                    engine_starts=starts,
                    resources_released=bool(releases),
                    release_to_route_ready_s=published[-1]["unix_s"]
                    - releases[-1]["unix_s"]
                    if releases and published
                    else None,
                    evidence=str(directory.relative_to(output)),
                )
            )
    return dict(
        status="CALIBRATION_EVIDENCE_ONLY",
        deployment_ready=False,
        workload=workload,
        cells=cells,
        service_lookup=lookup,
        service_native_capacity_and_startup=[
            dict(row, batch=i)
            for i, p in enumerate(profiles)
            for row in p.get("native_capacity_and_startup", [])
        ],
        migration_observations=measurements,
        empty_rebuild_observations=lifecycle,
        capacity_guard_blocks=None,
        capacity_lookahead_s=None,
        general_migration_release_s=None,
        general_rebuild_ready_s=None,
        delta_g=None,
        C_pre=None,
        C_post=None,
        missing=missing
        + [
            "loaded target migration/rebuild support",
            "matched layout utility",
            "live finite-62 predictor and frozen pressure-policy integration",
            "statistically supported guard/release bounds",
        ],
        score_note="90% of measured peak; unsupported cells stay diagnostic",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pilot-batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    stages=stages(),
                    new_profile_units=9,
                    new_functional_empty_units=4,
                    prior_profile_attempts=3,
                    profile_limit=12,
                    dependent_large_migration_requires_small_success=True,
                    predictor_training=False,
                    live_policy=False,
                ),
                indent=2,
            )
        )
        return 0
    require_revision(args.expected_revision)
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        raise ValueError("results must be outside the checkout")
    if not args.resume:
        args.output.mkdir(parents=True, exist_ok=False)
    elif not (args.output / "identity.json").is_file():
        raise ValueError("resume requires this batch's identity")
    archive = Path(
        str(args.output)
        + (f".resume_{time.time_ns()}" if args.resume else "")
        + ".tar.gz"
    )
    code, result = 1, dict(status="TECHNICAL_FAILURE")
    ledger = (
        read(args.output / "stages.json")
        if (args.output / "stages.json").exists()
        else []
    )
    try:
        hardware = runtime_inventory(read(PROFILE))
        if not gpu_idle():
            raise RuntimeError("GPUs busy; no unrelated work is stopped")
        pilot_pre = read(args.pilot_batch / "preflight.json")
        if (
            pilot_pre["revision"] != "f90c48b721dc3ab37ecb7db6b09b62ecf088f95b"
            or pilot_pre["input_sha256"] != INPUT_SHA
            or read(args.pilot_batch / "hardware.json")["gpus"] != hardware["gpus"]
        ):
            raise ValueError("audited pilot/GPU identity differs")
        for rep in (1, 2):
            accepted_scene(args.pilot_batch / f"qwen3-nd2-r{rep}")
        verified_inputs(args.model, args.pilot_batch / "inputs.json")
        identity = inspect_model(args.model, read(PROFILE))
        assets = read(ROOT / "experiments/platforms/qwen3_8b_verified_assets.json")[
            "sha256"
        ]
        if {k: v["sha256"] for k, v in identity["files"].items()} != assets:
            raise ValueError("model weights differ")
        versions = json.loads(json.dumps(runtime_versions()))
        if pilot_pre["versions"] != versions:
            raise ValueError("audited pilot runtime/backend environment differs")
        key = dict(
            revision=args.expected_revision,
            model=str(args.model.resolve()),
            pilot=str(args.pilot_batch.resolve()),
            pilot_preflight_sha256=sha(args.pilot_batch / "preflight.json"),
            gpu=hardware["gpus"],
            versions=versions,
        )
        if args.resume and read(args.output / "identity.json") != key:
            raise ValueError("resume identity differs")
        write(args.output / "identity.json", key)
        write(args.output / "hardware.json", hardware)
        write(args.output / "model_identity.json", identity)
        write(
            args.output / "pilot_profile_snapshot.json",
            read(args.pilot_batch / "service_profile.json"),
        )
        write(
            args.output / "plan.json",
            dict(stages=stages(), profile_prior=3, profile_new=9, functional_new=4),
        )
        small_passed = any(
            r["name"] == "migration-128" and r["status"] == "COMPLETED" for r in ledger
        )
        for stage in stages():
            directory = args.output / stage["name"]
            prior = next((r for r in ledger if r["name"] == stage["name"]), None)
            if prior:
                # Never automatically repeat a failed or uncertain physical action.
                if prior["status"] == "COMPLETED" and not complete(
                    directory, stage["kind"]
                ):
                    raise ValueError("previous completed stage no longer verifies")
                continue
            if stage["name"] == "migration-2048" and not small_passed:
                ledger.append(dict(name=stage["name"], status="SKIPPED_DEPENDENCY"))
                write(args.output / "stages.json", ledger)
                continue
            if not gpu_idle():
                raise RuntimeError(
                    "GPU processes remain; stop the batch without killing them"
                )
            record = dict(
                name=stage["name"],
                kind=stage["kind"],
                status="STARTED",
                started_unix_s=time.time(),
            )
            ledger.append(record)
            write(args.output / "stages.json", ledger)
            print(f"[批次] {stage['name']} 开始", flush=True)
            try:
                function = (
                    service_main
                    if stage["kind"] == "service"
                    else migration_main
                    if stage["kind"] == "migration"
                    else drain_main
                )
                rc = function(child_arguments(args, stage))
                record.update(
                    status="COMPLETED"
                    if rc == 0 and complete(directory, stage["kind"])
                    else "FAILED",
                    exit_code=rc,
                )
            except Exception as error:
                record.update(status="FAILED", error=repr(error))
            record["ended_unix_s"] = time.time()
            write(args.output / "stages.json", ledger)
            print(f"[批次] {stage['name']} {record['status']}", flush=True)
            if stage["name"] == "migration-128":
                small_passed = record["status"] == "COMPLETED"
            if not gpu_idle():
                raise RuntimeError(
                    "post-stage GPU cleanup incomplete; subsequent stages blocked"
                )
        write(
            args.output / "parameter_summary.json",
            summary(args.output, args.pilot_batch),
        )
        code = 0 if all(r["status"] == "COMPLETED" for r in ledger) else 1
        result = dict(
            status="COMPLETED" if code == 0 else "PARTIAL_EVIDENCE",
            stages=ledger,
            deployment_ready=False,
        )
    except KeyboardInterrupt:
        code, result = 130, dict(status="INTERRUPTED", stages=ledger)
    except Exception as error:
        result = dict(status="TECHNICAL_FAILURE", error=repr(error), stages=ledger)
    finally:
        service_ledger = args.output / "service/attempt_ledger.json"
        service_attempts = len(read(service_ledger)) if service_ledger.exists() else 0
        result.update(
            runner_code=code,
            predictor_training=False,
            budget=dict(
                prior_profile_attempts=3,
                service_attempts=service_attempts,
                migration_attempts=sum(r.get("kind") == "migration" for r in ledger),
                functional_empty_attempts=sum(
                    r.get("kind") == "empty_drain" for r in ledger
                ),
                profile_limit=12,
            ),
        )
        write(args.output / "result.json", result)
        receipt = pack(args.output, archive)
        Path(str(archive) + ".sha256").write_text(
            receipt["sha256"] + "  " + archive.name + "\n"
        )
        print(json.dumps(receipt), flush=True)
        print(f"[批次结果] {result['status']}；交回完整包：{archive}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
