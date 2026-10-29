# SPDX-License-Identifier: Apache-2.0
"""Two bounded loaded migrations, gated by the finite62 live source probe."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp.prepare_4090d_qwen3 import (  # noqa: E402
    PROFILE,
    read,
    runtime_inventory,
)
from tools.bridge_tp.pressure_cpu import load  # noqa: E402
from tools.bridge_tp.qwen3_calibration_contract import (  # noqa: E402
    require_revision,
    runtime_versions,
    verified_inputs,
)
from tools.bridge_tp.qwen3_online_validation import CHECKPOINT_SHA  # noqa: E402
from tools.bridge_tp.result_archive import pack  # noqa: E402
from tools.bridge_tp.run_qwen3_parameter_batch import gpu_idle, write  # noqa: E402
from tools.bridge_tp.run_qwen3_scheduler_supplement import (  # noqa: E402
    verify_migration,
)
from tools.bridge_tp.run_tp2_migration_smoke import main as migration_main  # noqa: E402


class PerformanceFailure(RuntimeError):
    """Mechanism completed, but the required output SLO did not pass."""


def gpu_configuration(rows):
    """Compare device capabilities, allowing replacement cards and reindexing."""
    return sorted((r["name"], r["memory_mib"], r["driver"]) for r in rows)


def stages(diagnostic=False, channel_ablation=False, runtime_fix=False):
    if runtime_fix:
        return [
            dict(name="runtime-fix-loaded4", background=4, history=2048, probe=False)
        ]
    if channel_ablation:
        return [
            dict(
                name="channel-ablation-loaded4", background=4, history=2048, probe=False
            )
        ]
    return [
        dict(
            name="diagnostic-loaded2" if diagnostic else "live-natural-and-loaded2",
            background=2,
            history=128,
            probe=not diagnostic,
        ),
        dict(
            name="diagnostic-loaded4" if diagnostic else "loaded4-history2048",
            background=4,
            history=2048,
            probe=False,
        ),
    ]


def child_arguments(args, stage):
    argv = [
        "--platform",
        "qwen3",
        "--model-path",
        str(args.model),
        "--inputs",
        str(args.inputs),
        "--out-root",
        str(args.output / stage["name"]),
        "--expected-revision",
        args.expected_revision,
        "--no-archive",
        "--predictor-checkpoint",
        str(args.checkpoint),
        "--target-background",
        str(stage["background"]),
        "--history-tokens",
        str(stage["history"]),
        "--timeout-s",
        "1200",
    ]
    if getattr(args, "diagnostic_probe", False):
        return (
            argv
            + ["--diagnostic-probe"]
            + (
                ["--channel-ablation"]
                if getattr(args, "channel_ablation", False)
                else []
            )
        )
    return argv + ["--natural-probe"] if stage["probe"] else argv


def execute(args, ledger):
    for stage in stages(
        getattr(args, "diagnostic_probe", False),
        getattr(args, "channel_ablation", False),
        getattr(args, "runtime_fix_probe", False),
    ):
        if not gpu_idle():
            raise RuntimeError("GPUs busy; do not stop unrelated processes")
        row = dict(name=stage["name"], status="STARTED")
        ledger.append(row)
        write(args.output / "attempt_ledger.json", ledger)
        try:
            print(f"[在线带载] {stage['name']} 开始", flush=True)
            if migration_main(child_arguments(args, stage)) != 0:
                raise RuntimeError("child validation failed; subsequent stage blocked")
            directory = args.output / stage["name"]
            result = verify_migration(directory)
            events = directory / "mechanism/controller/predictor_events.jsonl"
            load("manager_m5").PredictorEventReader(events, CHECKPOINT_SHA).poll()
            rows = [json.loads(line) for line in events.read_text().splitlines()]
            if (
                rows[0].get("overflow_category") is not False
                or len(rows[0]["category_upper_edges"]) != 62
                or not any(r.get("kind") == "predictor_prediction" for r in rows)
                or any(r.get("kind") == "predictor_unavailable" for r in rows)
            ):
                raise ValueError("finite62 live prediction coverage failed")
            if (
                not result.get("predictor_loaded")
                or result.get("target_background_requests") != stage["background"]
            ):
                raise ValueError("predictor/background identity differs")
            if stage["probe"] and not getattr(args, "diagnostic_probe", False):
                probe = read(
                    directory / "mechanism/controller/online_predictor_receipt.json"
                )
                if probe["status"] != "COMPLETED" or probe["fresh_advisories"] < 1:
                    raise ValueError("live predictor receipt missing")
            if not gpu_idle():
                raise RuntimeError("GPU cleanup incomplete")
            if getattr(args, "channel_ablation", False):
                proof = read(
                    directory / "mechanism/controller/channel_ablation_receipt.json"
                )
                if proof.get("status") != "COMPLETED" or not proof.get(
                    "retired_kept_model_process"
                ):
                    raise ValueError("three-state ablation proof missing")
                for role in ("source", "target"):
                    retired = read(
                        directory
                        / f"mechanism/controller/{role}_channel_retirement.json"
                    )
                    if retired.get("status") != "RETIRED" or not all(
                        r.get("identity_unchanged") for r in retired["ranks"]
                    ):
                        raise ValueError("channel retirement rank proof missing")
            if getattr(args, "diagnostic_probe", False):
                from tools.bridge_tp.monitor_pressure_host import validate_journal

                write(
                    directory / "monitor_acceptance.json",
                    validate_journal(directory / "host_monitor.jsonl"),
                )
            from tools.bridge_tp.qwen3_migration_diagnostics import performance_report

            performance = performance_report(directory, result)
            if getattr(args, "runtime_fix_probe", False):
                proof = runtime_proof(directory)
                if getattr(args, "prior_runtime", None) is not None:
                    proof["post_cleanup"] = verify_worker_cleanup(directory, proof)
                write(directory / "runtime_fix_acceptance.json", proof)
            row.update(
                status="COMPLETED" if performance["passed"] else "PERFORMANCE_FAILURE",
                measurement=result,
                performance=performance,
            )
            if not performance["passed"] and not getattr(
                args, "diagnostic_probe", False
            ):
                raise PerformanceFailure("handoff or output stream SLO failed")
        except BaseException as error:
            row["status"] = (
                "INTERRUPTED"
                if isinstance(error, KeyboardInterrupt)
                else (
                    "PERFORMANCE_FAILURE"
                    if isinstance(error, PerformanceFailure)
                    else "FAILED"
                )
            )
            raise
        finally:
            write(args.output / "attempt_ledger.json", ledger)


def runtime_proof(directory):
    controller = directory / "mechanism/controller"
    ranks = []
    for rank in range(2):
        receipt = read(controller / f"persistent_payload_cleanup/tp_rank_{rank}.json")
        state = receipt.get("channel_runtime", {})
        polling = state.get("manifest_polling") or {}
        if (
            receipt.get("status") != "PERSISTENT_SESSION_PAYLOAD_RELEASED"
            or receipt.get("tp_rank") != rank
            or not state.get("same_receiver")
            or state.get("lifecycle_state") != "IDLE"
            or state.get("active_session")
            or not state.get("communicator_present")
            or state.get("live_load_threads") != 0
            or not 1 <= polling.get("parses", 0) <= polling.get("polls", 0)
        ):
            raise ValueError(f"rank {rank} worker/channel cleanup proof invalid")
        ranks.append(receipt)
    return dict(
        status="COMPLETED", ranks=ranks, channel_retired=False, target_restarted=False
    )


def verify_worker_cleanup(directory, proof):
    controller = directory / "mechanism/controller"
    request_ids = {r["target_request_id"] for r in proof["ranks"]}
    if len(request_ids) != 1:
        raise ValueError("worker cleanup request identity differs across ranks")
    accepted = []
    for path in sorted(
        (controller / "worker_cleanup_receipts").glob("*/tp_rank_*.json")
    ):
        row = read(path)
        if row.get("target_request_id") not in request_ids:
            continue
        state = row.get("worker_state", {})
        counts = state.get("counts", {})
        expected = {
            "load_threads",
            "copy_done_events",
            "model_wait_streams",
            "model_wait_pending_requests",
            "gpu_restore_quiescence",
            "completed_recvs",
            "reported_recvs",
            "finished_worker_requests",
        }
        if (
            row.get("status") != "WORKER_REQUEST_STATE_CLEANED"
            or set(counts) != expected
            or any(counts.values())
            or state.get("receiver_payload_present") is not False
            or state.get("receiver_payload_capacity_elements") != 0
            or set(state.get("stream_ready", {})) != {"migration_receive", "restore"}
            or any(
                v is not True and v is not None for v in state["stream_ready"].values()
            )
        ):
            raise ValueError("post-cleanup worker state is not empty/idle")
        accepted.append(row)
    if len(accepted) != 2 or {r.get("tp_rank") for r in accepted} != {0, 1}:
        raise ValueError("post-cleanup evidence requires both target ranks")
    return accepted


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "checkpoint", "inputs", "prior-supplement", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--diagnostic-probe", action="store_true")
    parser.add_argument("--prior-online", type=Path)
    parser.add_argument("--channel-ablation", action="store_true")
    parser.add_argument("--prior-ready", type=Path)
    parser.add_argument("--runtime-fix-probe", action="store_true")
    parser.add_argument("--prior-ablation", type=Path)
    parser.add_argument("--prior-runtime", type=Path)
    args = parser.parse_args(argv)
    if args.diagnostic_probe and args.prior_online is None:
        parser.error("diagnostics require the previous online validation directory")
    if args.channel_ablation and (
        not args.diagnostic_probe or args.prior_ready is None
    ):
        parser.error(
            "channel ablation requires diagnostics and the previous ready probe"
        )
    if args.runtime_fix_probe and (
        not args.diagnostic_probe
        or args.channel_ablation
        or args.prior_ready is None
        or args.prior_ablation is None
    ):
        parser.error(
            "runtime fix requires diagnostics, ready/ablation evidence, no retirement"
        )
    if args.prior_runtime is not None and not args.runtime_fix_probe:
        parser.error("previous runtime evidence requires runtime fix probe")
    prior_count = (
        25
        if args.runtime_fix_probe
        else (24 if args.channel_ablation else (22 if args.diagnostic_probe else 20))
    )
    if args.prior_runtime is not None:
        prior_count = 26
    new_limit = 1 if args.channel_ablation or args.runtime_fix_probe else 2
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="STATIC_PLAN_ONLY",
                    stages=stages(
                        args.diagnostic_probe,
                        args.channel_ablation,
                        args.runtime_fix_probe,
                    ),
                    prior_profile_attempts=prior_count,
                    additional_limit=new_limit,
                    total_limit=prior_count + new_limit,
                    checkpoint_sha256=CHECKPOINT_SHA,
                    automatic_policy=False,
                ),
                indent=2,
            )
        )
        return 0
    require_revision(args.expected_revision)
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        raise ValueError("results must be outside checkout")
    args.output.mkdir(parents=True, exist_ok=False)
    ledger, result, code = [], dict(status="TECHNICAL_FAILURE"), 1
    try:
        if args.runtime_fix_probe:
            if args.prior_runtime is not None:
                previous = read(args.prior_runtime / "result.json")
                if (
                    previous.get("total_profile_attempts") != 26
                    or previous.get("status") != "COMPLETED"
                    or previous.get("runner_code") != 0
                    or len(previous.get("stages", [])) != 1
                ):
                    raise ValueError("requires the completed first runtime validation")
                verify_migration(args.prior_runtime / "runtime-fix-loaded4")
            previous = read(args.prior_ablation / "result.json")
            if (
                previous.get("total_profile_attempts") != 25
                or previous.get("attempted_new") != 1
                or previous.get("status") != "TECHNICAL_FAILURE"
            ):
                raise ValueError("requires the previous failed single ablation result")
        if args.channel_ablation or args.runtime_fix_probe:
            ready = read(args.prior_ready / "result.json")
            if (
                ready.get("total_profile_attempts") != 24
                or len(ready.get("stages", [])) != 2
            ):
                raise ValueError("requires the audited ready/slowdown probe")
            for previous in stages(True):
                verify_migration(args.prior_ready / previous["name"])
        if args.diagnostic_probe:
            online = read(args.prior_online / "result.json")
            if (
                online.get("total_profile_attempts") != 22
                or len(online.get("stages", [])) != 2
            ):
                raise ValueError("requires the audited two-stage online result")
            for previous in stages():
                verify_migration(args.prior_online / previous["name"])
        prior = read(args.prior_supplement / "result.json")
        prior_identity = read(args.prior_supplement / "identity.json")
        if (
            prior["status"] != "COMPLETED"
            or prior["runner_code"] != 0
            or prior["total_profile_attempts"] != 20
            or prior_identity["revision"] != "9832589c3e0a89967b6dee44a48dc185cdc37f37"
        ):
            raise ValueError("requires the audited completed parameter supplement")
        gpu = runtime_inventory(read(PROFILE))
        versions = runtime_versions()
        comparison = dict(
            uuid_and_index_identity_required=False,
            gpu_configuration_matches=(
                gpu_configuration(gpu["gpus"])
                == gpu_configuration(prior_identity["gpu"]["gpus"])
            ),
            runtime_matches=(
                json.loads(json.dumps(versions)) == prior_identity["versions"]
            ),
            previous_gpu=prior_identity["gpu"],
            current_gpu=gpu,
            previous_versions=prior_identity["versions"],
            current_versions=versions,
        )
        write(args.output / "runtime_identity_comparison.json", comparison)
        if (
            not comparison["gpu_configuration_matches"]
            or not comparison["runtime_matches"]
        ):
            raise ValueError("GPU/driver/runtime identity differs")
        verified_inputs(args.model, args.inputs)
        from tools.bridge_tp.qwen3_online_validation import configure_predictor

        configure_predictor(
            dict(source_env={}, run_dir=args.output), args.checkpoint, args.model
        )
        write(
            args.output / "identity.json",
            dict(
                revision=args.expected_revision,
                gpu=gpu,
                versions=versions,
                checkpoint_sha256=CHECKPOINT_SHA,
                prior_profile_attempts=prior_count,
                maximum_new_attempts=new_limit,
                total_limit=prior_count + new_limit,
                predictor_training=False,
                diagnostic_probe=args.diagnostic_probe,
                channel_ablation=args.channel_ablation,
                runtime_fix_probe=args.runtime_fix_probe,
                automatic_pressure_policy=False,
            ),
        )
        execute(args, ledger)
        performance_failed = any(r["status"] == "PERFORMANCE_FAILURE" for r in ledger)
        result, code = (
            (dict(status="PERFORMANCE_FAILURE"), 2)
            if performance_failed
            else (dict(status="COMPLETED"), 0)
        )
    except KeyboardInterrupt:
        result, code = dict(status="INTERRUPTED"), 130
    except PerformanceFailure as error:
        result, code = dict(status="PERFORMANCE_FAILURE", error=str(error)), 2
    except Exception as error:
        result["error"] = repr(error)
        print(f"[错误] {error!r}", flush=True)
    finally:
        result.update(
            runner_code=code,
            stages=ledger,
            prior_profile_attempts=prior_count,
            attempted_new=len(ledger),
            total_profile_attempts=prior_count + len(ledger),
            deployment_ready=False,
            general_guard_blocks=None,
        )
        write(args.output / "result.json", result)
        archive = Path(str(args.output) + ".tar.gz")
        receipt = pack(args.output, archive)
        Path(str(archive) + ".sha256").write_text(
            receipt["sha256"] + "  " + archive.name + "\n"
        )
        print(f"[结果] {result['status']}；请拿回 {archive}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
