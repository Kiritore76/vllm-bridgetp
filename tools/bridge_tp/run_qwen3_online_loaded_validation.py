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


def stages():
    return [
        dict(name="live-natural-and-loaded2", background=2, history=128, probe=True),
        dict(name="loaded4-history2048", background=4, history=2048, probe=False),
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
    return argv + ["--natural-probe"] if stage["probe"] else argv


def execute(args, ledger):
    for stage in stages():
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
            if stage["probe"]:
                probe = read(
                    directory / "mechanism/controller/online_predictor_receipt.json"
                )
                if probe["status"] != "COMPLETED" or probe["fresh_advisories"] < 1:
                    raise ValueError("live predictor receipt missing")
            if not gpu_idle():
                raise RuntimeError("GPU cleanup incomplete")
            row.update(status="COMPLETED", measurement=result)
        except BaseException:
            row["status"] = (
                "INTERRUPTED" if sys.exc_info()[0] is KeyboardInterrupt else "FAILED"
            )
            raise
        finally:
            write(args.output / "attempt_ledger.json", ledger)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "checkpoint", "inputs", "prior-supplement", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="STATIC_PLAN_ONLY",
                    stages=stages(),
                    prior_profile_attempts=20,
                    additional_limit=2,
                    total_limit=22,
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
        if (
            gpu["gpus"] != prior_identity["gpu"]["gpus"]
            or json.loads(json.dumps(versions)) != prior_identity["versions"]
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
                prior_profile_attempts=20,
                maximum_new_attempts=2,
                total_limit=22,
                predictor_training=False,
                automatic_pressure_policy=False,
            ),
        )
        execute(args, ledger)
        result, code = dict(status="COMPLETED"), 0
    except KeyboardInterrupt:
        result, code = dict(status="INTERRUPTED"), 130
    except Exception as error:
        result["error"] = repr(error)
        print(f"[错误] {error!r}", flush=True)
    finally:
        result.update(
            runner_code=code,
            stages=ledger,
            prior_profile_attempts=20,
            attempted_new=len(ledger),
            total_profile_attempts=20 + len(ledger),
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
