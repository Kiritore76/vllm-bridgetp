# SPDX-License-Identifier: Apache-2.0
"""Execute matched STAY/MIGRATE pairs through the original full controller."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import launch_pressure_balanced_pilot as pilot  # noqa: E402
from tools.bridge_tp import qwen3_pressure_observation as observation  # noqa: E402
from tools.bridge_tp import qwen3_stay_migrate as matched  # noqa: E402
from tools.bridge_tp import run_pressure_comparison as comparison  # noqa: E402
from tools.bridge_tp.prepare_4090d_qwen3 import (  # noqa: E402
    PROFILE,
    read,
    runtime_inventory,
)
from tools.bridge_tp.qwen3_calibration_contract import (  # noqa: E402
    adjust_migration_plan,
    migration_spec,
    require_revision,
    runtime_versions,
    verified_inputs,
)
from tools.bridge_tp.qwen3_online_validation import (  # noqa: E402
    CHECKPOINT_SHA,
    configure_predictor,
)
from tools.bridge_tp.result_archive import pack  # noqa: E402
from tools.bridge_tp.run_qwen3_online_loaded_validation import (  # noqa: E402
    gpu_configuration,
)
from tools.bridge_tp.run_qwen3_parameter_batch import gpu_idle, write  # noqa: E402


def adapt_plan(plan, arm, checkpoint, model):
    adjust_migration_plan(plan)
    configure_predictor(plan, checkpoint, model)
    plan["controller_argv"] += ["--platform", "qwen3"]
    if arm == "migrate":
        if "--no-migration" in plan["controller_argv"]:
            raise ValueError("MIGRATE arm must execute actions")
        plan["controller_argv"].append("--qwen3-execute-migrations")
    elif arm == "stay":
        if "--no-migration" not in plan["controller_argv"]:
            raise ValueError("STAY arm must disable migration")
    else:
        raise ValueError("unknown comparison arm")


def prepare_native(plan, common):
    from tools.bridge_tp.run_pressure_calibration import wait_empty

    rows = wait_empty(plan["run_dir"])
    plan["config"].update(
        tp1_total_kv_blocks=rows[0]["total_blocks"],
        tp4_total_kv_blocks=rows[1]["total_blocks"],
    )
    plan["config"]["policy"]["kv_bytes_per_token"] = 147456
    common.write_json(plan["config_path"], plan["config"])
    common.write_json(plan["run_dir"].parent / "native_initial_receipt.json", rows)


def serving_preflight(plans):
    from tools.bridge_tp import phase8_stager
    from vllm.entrypoints.openai.cli_args import (
        make_arg_parser,
        validate_parsed_serve_args,
    )
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    for _, _, _, plan in plans:
        plan["config_path"].parent.mkdir(parents=True, exist_ok=True)
        write(plan["config_path"], plan["config"])
        for role in ("source", "target"):
            parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(
                plan[role + "_argv"][3:]
            )
            validate_parsed_serve_args(parsed)
        phase8_stager.parse_args(plan["stager_argv"][2:])
        parsed = pilot.controller.parse_args(
            plan["controller_argv"][2:] + ["--validate-only"]
        )
        pilot.controller.preflight(parsed)


def closed_loop_proof(plan, arm):
    from tools.bridge_tp.pressure_session import channel_idle

    directory = plan["run_dir"]
    audit = [
        json.loads(line)
        for line in (directory / "pressure_audit.jsonl").read_text().splitlines()
    ]
    final = [r for r in audit if r.get("kind") == "pressure_final"][-1]
    starts = [r for r in audit if r.get("kind") == "pressure_actual_start"]
    reserved = [
        r
        for r in audit
        if r.get("kind") == "pressure_decision" and r.get("execution") == "RESERVED"
    ]
    if arm == "stay" and (starts or reserved):
        raise ValueError("STAY arm executed or reserved a migration")
    sessions = []
    for session in final["sessions"]:
        if not session["started"]:
            continue
        run_dir = Path(session["run_dir"])
        state_path = run_dir / "takeover_state.json"
        state = read(state_path) if state_path.exists() else {}
        row = dict(
            migration_id=session["migration_id"],
            state=state.get("state"),
            settled=session["settled"],
        )
        if state.get("state") == "COMMITTED":
            release = read(run_dir / "source_kv_release_receipt.json")
            if (
                state.get("migration_id") != session["migration_id"]
                or not session["settled"]
                or state.get("ready_sender_ranks") != [0, 1]
                or state.get("ready_receiver_ranks") != [0, 1]
                or not state.get("source_abort_dispatched")
                or release.get("status") != "SOURCE_KV_RELEASED"
                or release.get("request_id") != state.get("source_request_id")
                or not channel_idle(
                    run_dir, session["migration_id"], state["source_request_id"]
                )
            ):
                raise ValueError(
                    "committed migration lacks matching rank/release/IDLE proof"
                )
            row["source_release"] = release
        elif state.get("state") is None:
            if not (
                final["outcome"] == "COMPLETED_PENDING_OWNER_CLEANUP"
                and final.get("shutdown_cleanup_required")
            ):
                raise ValueError("started migration has no terminal ownership proof")
        elif state.get("state") != "CANCELLED":
            raise ValueError("migration left an unresolved ownership state")
        sessions.append(row)
    if len(sessions) != len(starts):
        raise ValueError("started session count differs from action journal")
    return dict(
        arm=arm,
        actual_starts=len(starts),
        reserved_decisions=len(reserved),
        sessions=sessions,
        final_controller_outcome=final["outcome"],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "checkpoint", "inputs", "prior-observation", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--repetitions", type=int, choices=(1, 2), default=2)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    values = matched.recipes(args.prior_observation, args.repetitions)
    cfg = matched.config(values)
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="CPU_MATCHED_PLAN_ONLY",
                    runs=len(values) * 2,
                    migration_execution=True,
                    maximum_migrations_per_run=0,
                    routing_policy="same original pressure policy in both arms",
                    recipes=[
                        {k: v for k, v in r.items() if k != "jobs"} for r in values
                    ],
                    output_contract=(
                        "primary pairs have identical fixed output budgets; "
                        "additional natural-EOS pair reported separately"
                    ),
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
    result, code, ledger, pairs = dict(status="TECHNICAL_FAILURE"), 1, [], []
    try:
        hardware, versions = runtime_inventory(read(PROFILE)), runtime_versions()
        old = read(args.prior_observation / "identity.json")
        write(
            args.output / "identity.json",
            dict(
                revision=args.expected_revision,
                gpu=hardware,
                versions=versions,
                checkpoint_sha256=CHECKPOINT_SHA,
            ),
        )
        if (
            gpu_configuration(hardware["gpus"]) != gpu_configuration(old["gpu"]["gpus"])
            or json.loads(json.dumps(versions)) != old["versions"]
        ):
            raise ValueError("GPU capability/driver/runtime differ; UUID not compared")
        inputs = verified_inputs(args.model, args.inputs)
        spec = migration_spec(args.model)
        spec.update(
            checkpoint=str(args.checkpoint),
            checkpoint_sha256=CHECKPOINT_SHA,
            policy_config=str(observation.ASSETS / "policy.json"),
            service_evidence=str(observation.ASSETS / "service_evidence.json"),
            health_reference=str(observation.ASSETS / "health_reference.json"),
            run_timeout_s=600,
            startup_timeout_s=900,
            no_progress_s=90,
            require_stager_process_ready=True,
        )
        common, settings = pilot.arguments(spec)
        scope = read(observation.ASSETS / "scope.json")
        scope.update(
            migration_execution_enabled=True,
            experiment_only=True,
            arms=["stay", "migrate"],
            repetitions=args.repetitions,
            maximum_live_runs=len(values) * 2,
            maximum_migrations_per_run=0,
            fairness=(
                "same routing policy, prompts, fixed output budgets, "
                "arrivals, seeds and hooks"
            ),
            guard_status=(
                "original nominal guard retained for development trial; "
                "NOT a calibrated guarantee"
            ),
        )
        write(args.output / "scope.json", scope)
        write(args.output / "comparison_recipe.json", cfg)
        plans = comparison.build_plans(spec, cfg, inputs, args.output, common, settings)
        for _, arm, _, plan in plans:
            adapt_plan(plan, arm, args.checkpoint, args.model)
        serving_preflight(plans)
        for recipe, arm, scene, plan in plans:
            if not gpu_idle():
                raise RuntimeError("GPUs busy; unrelated processes are not stopped")
            if matched.digest(read(Path(scene["arrivals"]))) != recipe["input_sha256"]:
                raise ValueError("paired workload changed")
            row = dict(
                name=scene["name"],
                pair=recipe["name"],
                arm=arm,
                status="STARTED",
                input_sha256=recipe["input_sha256"],
            )
            ledger.append(row)
            write(args.output / "attempt_ledger.json", ledger)
            pilot.run_scene(
                spec,
                scene,
                plan,
                common,
                settings,
                before_workload=lambda current: prepare_native(current, common),
                before_shutdown=lambda current: comparison.finalize(current, common),
                diagnose_pauses=True,
            )
            if not read(plan["run_dir"].parent / "owned_shutdown_receipt.json")[
                "all_owned_groups_ended"
            ]:
                raise RuntimeError("owned process cleanup failed")
            proof = closed_loop_proof(plan, arm)
            summary, requests, _ = matched.metrics(
                plan["run_dir"], recipe["jobs"], natural_eos=not recipe["fixed_output"]
            )
            write(plan["run_dir"].parent / "closed_loop_acceptance.json", proof)
            write(
                plan["run_dir"].parent / "comparison_metrics.json",
                dict(summary=summary, requests=requests),
            )
            row.update(status="COMPLETED", summary=summary, closed_loop=proof)
            write(args.output / "attempt_ledger.json", ledger)
            arms = {
                r["arm"]: r
                for r in ledger
                if r["pair"] == recipe["name"] and r["status"] == "COMPLETED"
            }
            if set(arms) == {"stay", "migrate"}:
                comparison_row = matched.compare(
                    arms["stay"]["summary"], arms["migrate"]["summary"]
                )
                pairs.append(
                    dict(
                        pair=recipe["name"],
                        scenario=recipe["scenario"],
                        repetition=recipe["repetition"],
                        fixed_output=recipe["fixed_output"],
                        comparison=comparison_row,
                    )
                )
                write(args.output / "paired_comparison.json", pairs)
            if not gpu_idle():
                raise RuntimeError("GPU cleanup incomplete")
        valid = all(
            r["comparison"]["matched_output_work"] and r["comparison"]["arrivals_valid"]
            for r in pairs
            if r["fixed_output"]
        )
        observed = any(
            r.get("summary", {}).get("committed_requests", 0) > 0
            and r.get("summary", {}).get("source_releases", 0) > 0
            for r in ledger
            if r["arm"] == "migrate"
        )
        result, code = (
            (dict(status="COMPLETED_PAIRED_EXPERIMENT"), 0)
            if valid and observed
            else (dict(status="COMPLETED_WITH_COVERAGE_GAPS"), 2)
        )
    except KeyboardInterrupt:
        result, code = dict(status="INTERRUPTED"), 130
    except Exception as error:
        result["error"] = repr(error)
        print(f"[错误] {error!r}", flush=True)
    finally:
        if ledger and ledger[-1]["status"] == "STARTED":
            ledger[-1]["status"] = "FAILED"
        result.update(
            runner_code=code,
            runs=ledger,
            pairs=pairs,
            previous_profile_attempts=31,
            total_profile_attempts=31 + len(ledger),
            maximum_live_runs=len(values) * 2,
            maximum_migrations_per_run=0,
            migration_execution=True,
            deployment_ready=False,
            advantages_preselected=False,
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
