# SPDX-License-Identifier: Apache-2.0
"""Execute matched STAY/MIGRATE pairs through the original full controller."""

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.bridge_tp import launch_pressure_balanced_pilot as pilot  # noqa: E402
from tools.bridge_tp import qwen3_four_mode as four  # noqa: E402
from tools.bridge_tp import qwen3_kv_guard_trial as guard_trial  # noqa: E402
from tools.bridge_tp import qwen3_kv_priority_verify as priority  # noqa: E402
from tools.bridge_tp import qwen3_migration_matrix as matrix  # noqa: E402
from tools.bridge_tp import qwen3_pressure_four_mode as pressure_four  # noqa: E402
from tools.bridge_tp import qwen3_pressure_observation as observation  # noqa: E402
from tools.bridge_tp import qwen3_resident_pressure as resident  # noqa: E402
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
    CHECKPOINT_PATH,
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
    elif arm in {"stay", "weighted", "least"}:
        if "--no-migration" not in plan["controller_argv"]:
            raise ValueError("baseline arms must disable migration")
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
    if arm != "migrate" and (starts or reserved):
        raise ValueError("baseline arm executed or reserved a migration")
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
    for key in ("model", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument(
        "--inputs", type=Path, default=observation.ASSETS / "inputs.json"
    )
    parser.add_argument("--prior-observation", type=Path)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--repetitions", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--four-mode", action="store_true")
    parser.add_argument("--migration-matrix", action="store_true")
    parser.add_argument("--kv-guard-trial", action="store_true")
    parser.add_argument("--resident-pressure", action="store_true")
    parser.add_argument("--pressure-four-mode", action="store_true")
    parser.add_argument("--kv-priority-verify", action="store_true")
    parser.add_argument("--prior-comparison", type=Path)
    parser.add_argument("--complete-four-mode", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    matrix_mode = args.migration_matrix or args.kv_guard_trial or args.resident_pressure
    inherited, evaluations, missing = [], [], None
    baseline_reference = None
    previous_attempts = 49 if args.four_mode else 31
    if args.kv_priority_verify:
        if (
            args.four_mode
            or args.pressure_four_mode
            or matrix_mode
            or args.complete_four_mode
            or args.repetitions != 1
            or args.prior_comparison is None
        ):
            parser.error(
                "KV priority verification requires one repetition "
                "and prior pressure trial"
            )
        prior = read(args.prior_comparison / "result.json")
        if (
            prior.get("experiment_mode") != "pressure-four-mode"
            or len(prior.get("runs", [])) != 12
            or any(r["status"] != "COMPLETED" for r in prior["runs"])
            or prior.get("status")
            not in {
                "COMPLETED_PRESSURE_FOUR_MODE_TRIAL",
                "COMPLETED_WITH_COVERAGE_GAPS",
            }
        ):
            raise ValueError("requires the completed twelve-run pressure trial")
        args.four_mode = True
        previous_attempts = prior["total_profile_attempts"]
        cfg, baseline_reference = priority.config(
            read(args.inputs), args.prior_comparison
        )
        values = cfg["recipes"]
    elif args.pressure_four_mode:
        if (
            args.four_mode
            or matrix_mode
            or args.complete_four_mode
            or args.repetitions != 1
            or args.prior_comparison is None
        ):
            parser.error(
                "pressure four-mode requires one repetition and prior resident trial"
            )
        prior = read(args.prior_comparison / "result.json")
        if (
            prior.get("experiment_mode") != "resident-pressure"
            or prior.get("status")
            not in {"COMPLETED_RESIDENT_PRESSURE_TRIAL", "COMPLETED_WITH_COVERAGE_GAPS"}
            or len(prior.get("runs", [])) != 8
            or any(r["status"] != "COMPLETED" for r in prior["runs"])
        ):
            raise ValueError("requires all eight completed resident comparison runs")
        args.four_mode = True
        previous_attempts = prior["total_profile_attempts"]
        cfg = pressure_four.config(read(args.inputs))
        values = cfg["recipes"]
    elif args.resident_pressure:
        if (
            args.migration_matrix
            or args.kv_guard_trial
            or args.four_mode
            or args.complete_four_mode
            or args.repetitions != 1
            or args.prior_comparison is None
        ):
            parser.error(
                "resident pressure requires one repetition and prior guard trial"
            )
        prior = read(args.prior_comparison / "result.json")
        if (
            prior.get("experiment_mode") != "kv-guard-trial"
            or prior.get("status")
            not in {"COMPLETED_KV_GUARD_TRIAL", "COMPLETED_WITH_COVERAGE_GAPS"}
            or len(prior.get("runs", [])) != 6
            or any(r["status"] != "COMPLETED" for r in prior["runs"])
        ):
            raise ValueError("requires all six completed guard screening runs")
        previous_attempts = prior["total_profile_attempts"]
        cfg = resident.config(read(args.inputs))
        values = cfg["recipes"]
    elif args.kv_guard_trial:
        if (
            args.migration_matrix
            or args.four_mode
            or args.complete_four_mode
            or args.repetitions != 1
            or args.prior_comparison is None
        ):
            parser.error("guard trial requires one repetition and prior matrix only")
        prior = read(args.prior_comparison / "result.json")
        if (
            prior.get("experiment_mode") != "migration-matrix"
            or prior.get("status")
            not in {"COMPLETED_MIGRATION_MATRIX", "COMPLETED_WITH_COVERAGE_GAPS"}
            or len(prior.get("runs", [])) != 18
            or any(r["status"] != "COMPLETED" for r in prior["runs"])
        ):
            raise ValueError("guard trial requires all eighteen completed matrix runs")
        previous_attempts = prior["total_profile_attempts"]
        cfg = guard_trial.config(read(args.inputs))
        values = cfg["recipes"]
    elif args.migration_matrix:
        if (
            args.four_mode
            or args.complete_four_mode
            or args.repetitions != 3
            or args.prior_comparison is None
        ):
            parser.error("matrix requires three repetitions and prior comparison only")
        prior = read(args.prior_comparison / "result.json")
        if prior["status"] not in {
            "COMPLETED_FOUR_MODE_SUPPLEMENT",
            "COMPLETED_FOUR_MODE_EXPERIMENT",
        }:
            raise ValueError("matrix requires completed four-mode evidence")
        previous_attempts = prior["total_profile_attempts"]
        cfg = matrix.config(read(args.inputs))
        values = cfg["recipes"]
    elif args.complete_four_mode:
        if not args.four_mode or args.repetitions != 3:
            parser.error("completion requires four-mode and three repetitions")
        cfg, inherited, evaluations, missing, previous_attempts = four.supplement(
            args.complete_four_mode.resolve()
        )
        values = cfg["recipes"]
    elif args.four_mode:
        if args.repetitions != 3 or args.prior_comparison is None:
            parser.error("four-mode requires three repetitions and prior comparison")
        if args.prior_observation is None:
            parser.error("four-mode requires prior observation")
        cfg = four.config(
            args.prior_observation,
            read(args.inputs),
            read(observation.ASSETS / "health_reference.json"),
        )
        previous = read(args.prior_comparison / "result.json")
        if (
            previous.get("status") != "COMPLETED_PAIRED_EXPERIMENT"
            or previous.get("total_profile_attempts") != 49
        ):
            raise ValueError("requires the completed eighteen-run comparison")
        values = cfg["recipes"]
    else:
        if args.prior_observation is None:
            parser.error("paired mode requires prior observation")
        values = matched.recipes(args.prior_observation, args.repetitions)
        cfg = matched.config(values)
    run_count = len(missing) if missing is not None else len(values) * len(cfg["arms"])
    if args.validate_only:
        print(
            json.dumps(
                dict(
                    status="CPU_MATCHED_PLAN_ONLY",
                    runs=run_count,
                    inherited_runs=len(inherited),
                    missing_cells=sorted(missing) if missing is not None else None,
                    arms=cfg["arms"],
                    policy_overrides=cfg["policy_overrides"],
                    routing_weights=cfg["routing_weights"],
                    migration_execution=True,
                    maximum_migrations_per_run=0,
                    routing_policy="single pressure MIGRATE; urgent KV soft-gate fix"
                    if args.kv_priority_verify
                    else "six initial TP1 residents; later stress arrivals "
                    "use original pressure router; per-recipe guard125/384"
                    if args.resident_pressure
                    else "fixed pools for growth/burst; original router "
                    "for stress arrivals in routed pair; experimental guard384"
                    if args.kv_guard_trial
                    else "fixed initial pools; unchanged migration policy"
                    if args.migration_matrix
                    else "original four allocation strategies"
                    if args.four_mode
                    else "same original pressure policy in both arms",
                    recipes=[
                        {k: v for k, v in r.items() if k != "jobs"} for r in values
                    ],
                    output_contract=(
                        "fixed-output system comparison"
                        if args.four_mode or matrix_mode
                        else "identical fixed budgets in primary pairs; "
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
        prior_identity = (
            args.prior_comparison
            if matrix_mode or args.pressure_four_mode or args.kv_priority_verify
            else args.complete_four_mode or args.prior_observation
        )
        old = read(prior_identity / "identity.json")
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
            run_timeout_s=900 if args.four_mode or matrix_mode else 600,
            startup_timeout_s=900,
            no_progress_s=90,
            require_stager_process_ready=True,
        )
        common, settings = pilot.arguments(spec)
        scope = read(observation.ASSETS / "scope.json")
        scope.update(
            migration_execution_enabled=True,
            experiment_only=True,
            arms=cfg["arms"],
            repetitions=args.repetitions,
            maximum_live_runs=run_count,
            maximum_migrations_per_run=0,
            fairness=(
                "same prompts, fixed output budgets, arrivals, seeds, service "
                "configuration and hooks; allocation differs by declared arm"
            ),
            guard_status=(
                "same-batch guard125/384 screening; per-recipe policy; "
                "neither is a calibrated guarantee"
                if args.resident_pressure
                else "guard384 offline-selected GPU screening candidate; "
                "NOT a calibrated guarantee"
                if args.kv_guard_trial
                else "original nominal guard retained for development trial; "
                "NOT a calibrated guarantee"
            ),
            policy_overrides=cfg["policy_overrides"],
        )
        write(args.output / "scope.json", scope)
        write(args.output / "comparison_recipe.json", cfg)
        if baseline_reference is not None:
            write(args.output / "baseline_reference.json", baseline_reference)
        if args.complete_four_mode:
            write(
                args.output / "completion_provenance.json",
                dict(
                    original_batch=str(args.complete_four_mode.resolve()),
                    original_revision=old["revision"],
                    current_revision=args.expected_revision,
                    inherited_runs=inherited,
                    new_cells=sorted(missing),
                    note="new archive holds supplement only; retain original archive; "
                    "third repetition combines runs from different times",
                ),
            )
        plans = comparison.build_plans(spec, cfg, inputs, args.output, common, settings)
        if missing is not None:
            plans = [p for p in plans if p[2]["name"] in missing]
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
            summary, requests, audit = matched.metrics(
                plan["run_dir"], recipe["jobs"], natural_eos=not recipe["fixed_output"]
            )
            write(plan["run_dir"].parent / "closed_loop_acceptance.json", proof)
            write(
                plan["run_dir"].parent / "comparison_metrics.json",
                dict(summary=summary, requests=requests),
            )
            row.update(status="COMPLETED", summary=summary, closed_loop=proof)
            if args.four_mode or matrix_mode:
                raw = read(plan["run_dir"] / "pressure_request_results.json")
                score = four.full_score(
                    raw,
                    cfg,
                    next(
                        r["unix_s"]
                        for r in audit
                        if r["kind"] == "pressure_clock_origin"
                    ),
                    audit,
                )
                if args.pressure_four_mode or args.kv_priority_verify:
                    score = pressure_four.decorate(score, recipe)
                if args.kv_priority_verify:
                    row["kv_priority"] = priority.assess(raw, audit)
                    write(
                        args.output / "kv_priority_acceptance.json", row["kv_priority"]
                    )
                write(plan["run_dir"].parent / "sustained_goodoutput.json", score)
                evaluations.append(
                    dict(arm=arm, score=score, scenario=recipe["scenario"])
                )
                if matrix_mode:
                    evaluations[-1]["roles"] = {
                        j["id"]: j["comparison_role"] for j in recipe["jobs"]
                    }
                    row["mechanism"] = matrix.assess(recipe, arm, raw, audit, score)
                    matching_count = row["mechanism"]["matching_committed_released"]
                    print(
                        f"[场景验收] {scene['name']} "
                        f"匹配路径提交并释放={matching_count} "
                        f"机制覆盖={row['mechanism']['mechanism_coverage_pass']}；"
                        "性能得分及未触发结果均保留",
                        flush=True,
                    )
                    write(
                        plan["run_dir"].parent / "migration_matrix_acceptance.json",
                        row["mechanism"],
                    )
                    write(
                        args.output / "migration_matrix_summary.json",
                        matrix.aggregate(evaluations),
                    )
                else:
                    write(
                        args.output / "four_mode_summary.json",
                        pressure_four.aggregate(evaluations)
                        if args.pressure_four_mode or args.kv_priority_verify
                        else four.aggregate(evaluations),
                    )
                    write(
                        args.output / "four_mode_comparison.json",
                        four.comparisons(inherited + ledger),
                    )
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
        if args.four_mode and not args.kv_priority_verify:
            paired = four.comparisons(inherited + ledger)
            pairs = paired
            write(args.output / "four_mode_comparison.json", paired)
            valid = len(paired) == 3 and all(
                p["arrivals_valid"]
                and p["matched_output_work"]
                and p["all_load_covered"]
                for p in paired
            )
            result, code = (
                (
                    dict(
                        status="COMPLETED_PRESSURE_FOUR_MODE_TRIAL"
                        if args.pressure_four_mode
                        else "COMPLETED_FOUR_MODE_SUPPLEMENT"
                        if args.complete_four_mode
                        else "COMPLETED_FOUR_MODE_EXPERIMENT"
                    ),
                    0,
                )
                if valid
                else (dict(status="COMPLETED_WITH_COVERAGE_GAPS"), 2)
            )
        if matrix_mode:
            coverage = (
                len(pairs) == len(values)
                and valid
                and all(r["mechanism"]["mechanism_coverage_pass"] for r in ledger)
            )
            result, code = (
                (
                    dict(
                        status="COMPLETED_RESIDENT_PRESSURE_TRIAL"
                        if args.resident_pressure
                        else "COMPLETED_KV_GUARD_TRIAL"
                        if args.kv_guard_trial
                        else "COMPLETED_MIGRATION_MATRIX"
                    ),
                    0,
                )
                if coverage
                else (dict(status="COMPLETED_WITH_COVERAGE_GAPS"), 2)
            )
        if args.kv_priority_verify:
            accepted = (
                len(ledger) == 1
                and ledger[0]["summary"]["fixed_arrivals_valid"]
                and ledger[0]["kv_priority"]["verification_pass"]
            )
            result, code = (
                (dict(status="COMPLETED_KV_PRIORITY_VERIFY"), 0)
                if accepted
                else (dict(status="COMPLETED_WITH_COVERAGE_GAPS"), 2)
            )
    except KeyboardInterrupt:
        result, code = dict(status="INTERRUPTED"), 130
    except Exception as error:
        result["error"] = repr(error)
        (args.output / "failure_traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        print(f"[错误] {error!r}", flush=True)
    finally:
        if ledger and ledger[-1]["status"] == "STARTED":
            ledger[-1]["status"] = "FAILED"
        result.update(
            runner_code=code,
            runs=ledger,
            pairs=pairs,
            previous_profile_attempts=previous_attempts,
            total_profile_attempts=previous_attempts + len(ledger),
            inherited_runs=inherited,
            maximum_live_runs=run_count,
            experiment_mode="kv-priority-verify"
            if args.kv_priority_verify
            else "pressure-four-mode"
            if args.pressure_four_mode
            else "resident-pressure"
            if args.resident_pressure
            else "kv-guard-trial"
            if args.kv_guard_trial
            else "migration-matrix"
            if args.migration_matrix
            else "four-mode"
            if args.four_mode
            else "stay-migrate",
            actual_migration_observed=any(
                r.get("summary", {}).get("committed_requests", 0) for r in ledger
            ),
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
