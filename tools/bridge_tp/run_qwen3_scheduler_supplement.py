# SPDX-License-Identifier: Apache-2.0
"""Eight additional measurements after the audited twelve-attempt batch."""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from tools.bridge_tp import run_pressure_calibration as collect  # noqa: E402
from tools.bridge_tp.prepare_4090d_qwen3 import (  # noqa: E402
    PROFILE,
    inspect_model,
    read,
    runtime_inventory,
    sha,
)
from tools.bridge_tp.pressure_session import channel_idle  # noqa: E402
from tools.bridge_tp.qwen3_calibration_contract import (  # noqa: E402
    require_revision,
    runtime_versions,
    verified_inputs,
)
from tools.bridge_tp.result_archive import pack  # noqa: E402
from tools.bridge_tp.run_qwen3_parameter_batch import gpu_idle, write  # noqa: E402
from tools.bridge_tp.run_qwen3_tp2_calibration import accepted_scene  # noqa: E402
from tools.bridge_tp.run_tp2_migration_smoke import main as migration_main  # noqa: E402
from tools.bridge_tp.summarize_tp2_calibration import summarize  # noqa: E402

PRIOR_RESULT_SHA = "cacab4b7eb7f423e84073df56373e60b9401dffb1e8e26df229f8b2a806ddddc"


def stages():
    migration = [
        dict(name=f"migration-{n}-r{r}", kind="migration", history=n)
        for r in (1, 2)
        for n in (2048, 128)
    ]
    service = [
        dict(
            name=f"nd{n}-r{r}",
            kind="service",
            repetition=r,
            scene=dict(
                name=f"nd{n}",
                prompt_tokens=2048,
                max_tokens=6144,
                source_count=4,
                target_count=n,
                headroom_blocks=125,
            ),
        )
        for r in (1, 2)
        for n in ((8, 16) if r == 1 else (16, 8))
    ]
    return migration + service


def verify_migration(directory):
    result = read(directory / "result.json")
    if not (
        result.get("status") == "COMPLETED"
        and result.get("runner_code") == 0
        and result.get("all_owned_groups_ended")
        and result.get("source_release_confirmed")
        and result.get("source_release_to_channel_idle_checked")
        and result.get("target_tp") == 2
        and result.get("mechanism_only")
        and result.get("source_finish") == "abort"
        and result.get("target_finish") == "length"
    ):
        raise ValueError("migration completion/release/cleanup failed")
    speed = result["full_speed"]
    if any(
        speed.get(k) != v
        for k, v in dict(
            status="COMPLETED",
            transfer_mode="FULL_SPEED",
            actual_starts=1,
            history_rank_receipts=2,
            history_sender_receipts=1,
        ).items()
    ):
        raise ValueError("FULL_SPEED rank evidence differs")
    tokens = read(directory / "mechanism/controller/unified_tokens.json")
    if (
        len(tokens) != 6144
        or [t["index"] for t in tokens] != list(range(6144))
        or tokens[0]["origin"] != "source"
        or tokens[-1]["origin"] != "target"
        or sum(a["origin"] != b["origin"] for a, b in zip(tokens, tokens[1:])) != 1
    ):
        raise ValueError("unified token stream differs")
    controller = directory / "mechanism/controller"
    state = read(controller / "takeover_state.json")
    release = read(controller / "source_kv_release_receipt.json")
    if (
        state.get("state") != "COMMITTED"
        or state.get("ready_sender_ranks") != [0, 1]
        or state.get("ready_receiver_ranks") != [0, 1]
        or release.get("status") != "SOURCE_KV_RELEASED"
        or release.get("request_id") != state.get("source_request_id")
        or not channel_idle(controller, "mechanism", state["source_request_id"])
    ):
        raise ValueError("commit/release/IDLE evidence differs")
    for key in (
        "start_to_release_s",
        "visible_handoff_gap_s",
        "start_to_source_release_and_idle_observed_s",
    ):
        if not math.isfinite(result[key]) or result[key] < 0:
            raise ValueError("invalid migration timing")
    return result


def execute_stage(args, stage, inputs):
    directory = args.output / stage["name"]
    if stage["kind"] == "migration":
        code = migration_main(
            [
                "--platform",
                "qwen3",
                "--model-path",
                str(args.model),
                "--inputs",
                str(args.inputs),
                "--expected-revision",
                args.expected_revision,
                "--history-tokens",
                str(stage["history"]),
                "--timeout-s",
                "900",
                "--out-root",
                str(directory),
                "--no-archive",
            ]
        )
        if code != 0:
            raise RuntimeError(f"migration child exited {code}")
        result = verify_migration(directory)
        if result["history_trigger_tokens"] != stage["history"]:
            raise ValueError("migration history trigger differs")
        return result
    settings = SimpleNamespace(
        python_bin=Path(sys.executable),
        model_path=args.model,
        dtype="bfloat16",
        max_model_len=8192,
        tp4_max_model_len=16384,
        gpu_memory_utilization=0.88,
        tp1_gpu="2",
        tp4_gpus="0,1",
        tp1_port=8610,
        tp4_port=8611,
        target_tp_size=2,
    )
    collect.run_scene(
        dict(service_only=True, startup_timeout_s=900, run_timeout_s=1200),
        inputs,
        stage["scene"],
        stage["repetition"],
        directory,
        common,
        settings,
        True,
    )
    accepted_scene(directory)
    return dict(status="COMPLETED")


def run_stages(args, inputs, ledger):
    for stage in stages():
        if not gpu_idle():
            raise RuntimeError("GPU busy; no unrelated process is stopped")
        row = dict(
            name=stage["name"],
            kind=stage["kind"],
            status="STARTED",
            started_unix_s=time.time(),
        )
        ledger.append(row)
        write(args.output / "attempt_ledger.json", ledger)
        print(f"[补测] {stage['name']} 开始", flush=True)
        try:
            execute_stage(args, stage, inputs)
            if not gpu_idle():
                raise RuntimeError("GPU cleanup incomplete; subsequent stages blocked")
            row["status"] = "COMPLETED"
        except BaseException:
            row["status"] = (
                "INTERRUPTED" if sys.exc_info()[0] is KeyboardInterrupt else "FAILED"
            )
            raise
        finally:
            row["ended_unix_s"] = time.time()
            write(args.output / "attempt_ledger.json", ledger)
            print(f"[补测] {stage['name']} {row['status']}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--prior-smoke", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="STATIC_PLAN_ONLY",
                    prior_attempts=12,
                    additional_limit=8,
                    total_limit=20,
                    stages=stages(),
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
        if sha(args.prior_smoke / "result.json") != PRIOR_RESULT_SHA:
            raise ValueError("requires the audited successful small migration")
        verify_migration(args.prior_smoke)
        inputs = verified_inputs(args.model, args.inputs)
        hardware = runtime_inventory(read(PROFILE))
        if [g["index"] for g in hardware["gpus"]] != ["0", "1", "2", "3"]:
            raise ValueError("four GPUs at physical indices 0,1,2,3 required")
        assets = inspect_model(args.model, read(PROFILE))
        expected = read(ROOT / "experiments/platforms/qwen3_8b_verified_assets.json")[
            "sha256"
        ]
        if {k: v["sha256"] for k, v in assets["files"].items()} != expected:
            raise ValueError("model weight identity differs")
        write(
            args.output / "identity.json",
            dict(
                revision=args.expected_revision,
                gpu=hardware,
                versions=runtime_versions(),
                model=assets,
                input_sha256=sha(args.inputs),
                prior_smoke_result_sha256=PRIOR_RESULT_SHA,
            ),
        )
        write(
            args.output / "selection.json",
            dict(
                target_tp=2,
                stages=stages(),
                prior_profile_attempts=12,
                additional_limit=8,
                total_limit=20,
                predictor_enabled=False,
                policy_enabled=False,
                conversion_enabled=False,
            ),
        )
        run_stages(args, inputs, ledger)
        write(args.output / "service_profile.json", summarize(args.output))
        observations = [
            dict(name=s["name"], **verify_migration(args.output / s["name"]))
            for s in stages()
            if s["kind"] == "migration"
        ]
        write(
            args.output / "migration_observations.json",
            dict(
                observations=observations,
                release_max_observed_s=max(
                    r["start_to_release_s"] for r in observations
                ),
                general_release_bound_s=None,
                capacity_guard_blocks=None,
                scope="repeated idle-target measurements; no loaded-target guarantee",
            ),
        )
        result, code = dict(status="COMPLETED"), 0
    except KeyboardInterrupt:
        result, code = dict(status="INTERRUPTED"), 130
    except Exception as error:
        result["error"] = repr(error)
        print(f"[错误] {error!r}", flush=True)
    finally:
        result.update(
            runner_code=code,
            attempts=ledger,
            prior_profile_attempts=12,
            attempted_new=len(ledger),
            total_profile_attempts=12 + len(ledger),
            deployment_ready=False,
            predictor_training=False,
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
