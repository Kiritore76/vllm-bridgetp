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
from tools.bridge_tp import qwen3_long_burst_pair as long_burst  # noqa: E402
from tools.bridge_tp import qwen3_migration_matrix as matrix  # noqa: E402
from tools.bridge_tp import qwen3_pressure_four_mode as pressure_four  # noqa: E402
from tools.bridge_tp import qwen3_pressure_observation as observation  # noqa: E402
from tools.bridge_tp import qwen3_resident_pressure as resident  # noqa: E402
from tools.bridge_tp import qwen3_routine_four_mode as routine  # noqa: E402
from tools.bridge_tp import qwen3_stay_migrate as matched  # noqa: E402
from tools.bridge_tp import run_pressure_comparison as comparison  # noqa: E402
from tools.bridge_tp.prepare_4090d_qwen3 import (  # noqa: E402
    PROFILE,
    read,
    runtime_inventory,
)
from tools.bridge_tp.qwen3_calibration_contract import (  # noqa: E402
    adjust_migration_plan,
    expanded_context_spec,
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

# S3b-3 GPU check: L1A on A={2,3}, B={0,1}. 30s after the controller starts
# (before the deep burst), one bundle is drained, rebuilt and published.
CONVERSIONS = {
    # A: TP1@2 (pair source, channel 1) + TP1@3 (channel 3) -> TP2(2,3).
    "L1A-L2": dict(bundle="A", target_mode="TP2"),
    # B: TP2(0,1) (pair target) -> TP1@0 + TP1@1.
    "L1A-L0": dict(bundle="B", target_mode="TP1"),
}


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

    # A single-class layout starts one side of the pair; the absent side keeps
    # the template geometry (no inventory, never routed).
    pools = [p for p in ("source", "target") if p not in pilot.absent_pair(plan)]
    rows = wait_empty(plan["run_dir"], pools=pools)
    by_pool = dict(zip(pools, rows))
    limits = dict(zip(("source", "target"), plan.get("expected_context_limits", ())))
    for pool, row in by_pool.items():
        limit = limits.get(pool)
        if limit is not None and (
            row["context_limit"] != limit
            or row["free_blocks"] * row["block_size"] < limit + 125 * row["block_size"]
        ):
            raise ValueError("expanded context or native KV headroom differs")
    for pool, key in (("source", "tp1"), ("target", "tp4")):
        if pool in by_pool:
            plan["config"][key + "_total_kv_blocks"] = by_pool[pool]["total_blocks"]
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
        pilot.write_instances(plan)
        absent = pilot.absent_pair(plan)
        argvs = [
            plan[side + "_argv"] for side in ("source", "target") if side not in absent
        ] + [
            pilot.extra_engine(plan, extra)[0]
            for extra in plan.get("extra_engines", ())
        ]
        for argv in argvs:
            parsed = make_arg_parser(FlexibleArgumentParser()).parse_args(argv[3:])
            validate_parsed_serve_args(parsed)
        if not absent:
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
    parser.add_argument("--updated-pressure-four-mode", action="store_true")
    parser.add_argument("--routine-four-mode", action="store_true")
    parser.add_argument("--long-burst-pair", action="store_true")
    parser.add_argument("--long-burst-memory-verify", action="store_true")
    parser.add_argument("--long-burst-deep-stay", action="store_true")
    parser.add_argument("--expanded-context", action="store_true")
    parser.add_argument("--kv-priority-verify", action="store_true")
    parser.add_argument("--kv-routing-verify", action="store_true")
    parser.add_argument("--current-machine", action="store_true")
    parser.add_argument("--prior-comparison", type=Path)
    parser.add_argument("--complete-four-mode", type=Path)
    parser.add_argument(
        "--layout-l1a-channels",
        action="store_true",
        help="G2: L1A on A={2,3}, B={0,1}; TP1@2 and TP1@3 both migrate to TP2(0,1)",
    )
    parser.add_argument(
        "--conversion",
        choices=sorted(CONVERSIONS),
        help="S3b-3: explicit DRAIN conversion of one L1A bundle while serving",
    )
    parser.add_argument(
        "--conversion-assist",
        action="store_true",
        help="S5a: SELECTIVE_ROLL drain assist (merging edge L1A-L2 only)",
    )
    parser.add_argument(
        "--conversion-trigger-s",
        type=float,
        default=30,
        help="seconds after the controller starts; S4 GPU check: while the "
        "90-100s long requests run on the converted bundle",
    )
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.layout_l1a_channels and not args.long_burst_memory_verify:
        parser.error("layout-l1a-channels runs with long-burst-memory-verify only")
    if args.conversion and not args.layout_l1a_channels:
        parser.error("conversion runs on the L1A channel layout only")
    if args.conversion_assist and args.conversion != "L1A-L2":
        parser.error("drain assist merges TP1s into a TP2 (L1A-L2) only")
    if not 0 <= args.conversion_trigger_s < 300:
        parser.error("conversion trigger must fall inside the 300s workload")
    if args.long_burst_memory_verify and args.long_burst_deep_stay:
        parser.error("deep STAY supplement and memory verification are separate runs")
    if args.long_burst_memory_verify or args.long_burst_deep_stay:
        args.long_burst_pair = True
    if args.long_burst_pair:
        if args.routine_four_mode or args.updated_pressure_four_mode:
            parser.error("long-burst-pair is a separate STAY/MIGRATE experiment")
        args.updated_pressure_four_mode = True
        args.expanded_context = True
    if args.routine_four_mode:
        if args.updated_pressure_four_mode:
            parser.error("routine and updated pressure trials are separate modes")
        args.updated_pressure_four_mode = True
    if args.expanded_context and not (
        args.updated_pressure_four_mode and args.current_machine
    ):
        parser.error("expanded-context requires a current-machine four-mode trial")
    if args.kv_routing_verify and not args.kv_priority_verify:
        parser.error("kv-routing-verify requires kv-priority-verify")
    if args.current_machine and (
        not (args.kv_priority_verify or args.updated_pressure_four_mode)
        or args.prior_comparison
    ):
        parser.error(
            "current-machine requires standalone KV or updated four-mode trial"
        )
    matrix_mode = args.migration_matrix or args.kv_guard_trial or args.resident_pressure
    inherited, evaluations, missing = [], [], None
    baseline_reference = None
    previous_attempts = 49 if args.four_mode else 31
    if args.updated_pressure_four_mode:
        if (
            args.four_mode
            or args.pressure_four_mode
            or matrix_mode
            or args.complete_four_mode
            or args.kv_priority_verify
            or args.repetitions != 1
            or not args.current_machine
            or args.prior_comparison
            or args.prior_observation
        ):
            parser.error(
                "updated four-mode requires current-machine, one repetition, "
                "and no other experiment mode or prior directory"
            )
        args.four_mode = args.pressure_four_mode = True
        previous_attempts = 0
        cfg = (
            long_burst.config(read(args.inputs))
            if args.long_burst_pair
            else routine.config(read(args.inputs))
            if args.routine_four_mode
            else pressure_four.updated_config(read(args.inputs))
        )
        if args.long_burst_memory_verify or args.long_burst_deep_stay:
            cfg["recipes"] = [
                r for r in cfg["recipes"] if r["scenario"] == "deep_long_burst"
            ]
            cfg["arms"] = ["stay" if args.long_burst_deep_stay else "migrate"]
            cfg["recipes"][0]["arm_order_offset"] = 0
            cfg["note"] += (
                "; deep STAY supplement; compare separately with completed MIGRATE"
                if args.long_burst_deep_stay
                else "; one failed-workload memory verification, not a new pair"
            )
        values = cfg["recipes"]
    elif args.kv_priority_verify:
        if (
            args.four_mode
            or args.pressure_four_mode
            or matrix_mode
            or args.complete_four_mode
            or args.repetitions != 1
            or (args.prior_comparison is None and not args.current_machine)
        ):
            parser.error(
                "KV priority verification requires one repetition "
                "and prior pressure trial"
            )
        prior = (
            None
            if args.current_machine
            else read(args.prior_comparison / "result.json")
        )
        if prior is not None and (
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
        previous_attempts = 0 if prior is None else prior["total_profile_attempts"]
        cfg, baseline_reference = priority.config(
            read(args.inputs),
            args.prior_comparison,
            kv_routing=args.kv_routing_verify,
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
    if args.expanded_context:
        cfg["service_context_limits"] = dict(tp1=32768, tp2=32768)
        cfg["context_profile"] = "QWEN3_NATIVE_32K_BOTH_V1"
        cfg["note"] += "; native context limits raised; boundary GPU validation pending"
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
                    service_context_limits=cfg.get(
                        "service_context_limits", dict(tp1=8192, tp2=16384)
                    ),
                    environment_mode="CURRENT_MACHINE_NO_PRIOR_COMPARISON"
                    if args.current_machine
                    else "PRIOR_ENVIRONMENT_COMPARISON",
                    policy_overrides=cfg["policy_overrides"],
                    routing_weights=cfg["routing_weights"],
                    migration_execution=True,
                    maximum_migrations_per_run=0,
                    routing_policy="KV-first ingress during warning/recovery; "
                    "original service routing otherwise"
                    if args.kv_routing_verify or args.updated_pressure_four_mode
                    else "single pressure MIGRATE; urgent KV soft-gate fix"
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
        old = None if args.current_machine else read(prior_identity / "identity.json")
        write(
            args.output / "identity.json",
            dict(
                revision=args.expected_revision,
                gpu=hardware,
                versions=versions,
                checkpoint_sha256=CHECKPOINT_SHA,
                prior_environment_comparison=not args.current_machine,
            ),
        )
        if old is not None and (
            gpu_configuration(hardware["gpus"]) != gpu_configuration(old["gpu"]["gpus"])
            or json.loads(json.dumps(versions)) != old["versions"]
        ):
            raise ValueError("GPU capability/driver/runtime differ; UUID not compared")
        inputs = verified_inputs(args.model, args.inputs)
        spec = migration_spec(args.model)
        if args.expanded_context:
            expanded_context_spec(spec, read(args.model / "config.json"))
        if args.layout_l1a_channels:
            spec["layout"] = dict(
                name="L1A",
                bundle_gpus=["2", "3", "0", "1"],
                base_port=8300,
                migration_sources=True,
            )
        if args.conversion:
            spec["conversion"] = dict(
                CONVERSIONS[args.conversion],
                trigger_s=args.conversion_trigger_s,
                method="SELECTIVE_ROLL" if args.conversion_assist else "DRAIN",
                # A drain behind 8192-token outputs can take minutes.
                deadline_s=600,
                base_port=8400,
            )
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
            current_machine_standalone=args.current_machine,
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
            service_context_limits=cfg.get(
                "service_context_limits", dict(tp1=8192, tp2=16384)
            ),
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
            if args.expanded_context:
                plan["expected_context_limits"] = (32768, 32768)
                plan["source_env"].update(
                    BRIDGETP_HISTORY_PACK_BUDGET_BYTES=str(16 * 1024 * 1024),
                    BRIDGETP_HISTORY_FORWARD_RESERVE_BYTES=str(512 * 1024 * 1024),
                )
            if args.kv_priority_verify or args.updated_pressure_four_mode:
                priority.enable_urgent_target(plan)
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
                before_workload=lambda current: (
                    long_burst.prepare_native(current, common, inputs)
                    if args.long_burst_pair
                    else prepare_native(current, common)
                ),
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
            if args.conversion:
                from tools.bridge_tp.conversion_acceptance import assess

                row["conversion"] = assess(plan["run_dir"].parent)
                write(
                    plan["run_dir"].parent / "conversion_acceptance.json",
                    row["conversion"],
                )
            row.update(status="COMPLETED", summary=summary, closed_loop=proof)
            if args.long_burst_memory_verify:
                row["history_memory"] = long_burst.assess_memory(plan, summary)
                write(
                    args.output / "history_memory_acceptance.json",
                    row["history_memory"],
                )
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
                if args.updated_pressure_four_mode and arm in {"stay", "migrate"}:
                    row["kv_routing"] = priority.assess_routing(audit)
                    write(
                        plan["run_dir"].parent / "kv_routing_acceptance.json",
                        row["kv_routing"],
                    )
                    # Its invariants name the fixed pair; after a conversion
                    # new instances take redirects too, so only record it.
                    if not args.conversion and (
                        not row["kv_routing"]["invariants_pass"]
                        or row["kv_routing"]["checked_arrivals"] != len(raw)
                    ):
                        raise ValueError("updated pressure routing audit incomplete")
                if args.kv_priority_verify:
                    row["kv_priority"] = priority.assess(raw, audit)
                    if args.kv_routing_verify:
                        row["kv_routing"] = priority.assess_routing(audit)
                        write(
                            args.output / "kv_routing_acceptance.json",
                            row["kv_routing"],
                        )
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
                        args.output
                        / (
                            "stay_migrate_summary.json"
                            if args.long_burst_pair
                            else "four_mode_summary.json"
                        ),
                        pressure_four.aggregate(evaluations)
                        if args.pressure_four_mode or args.kv_priority_verify
                        else four.aggregate(evaluations),
                    )
                    if not args.long_burst_pair:
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
        if args.long_burst_pair:
            valid = (
                len(pairs) == len(values)
                and len(ledger) == run_count
                and all(
                    p["comparison"]["matched_output_work"]
                    and p["comparison"]["arrivals_valid"]
                    and p["comparison"]["load_coverage_pass"]
                    for p in pairs
                )
            )
            result, code = (
                (dict(status="COMPLETED_LONG_BURST_PAIRED_TRIAL"), 0)
                if valid
                else (dict(status="COMPLETED_WITH_COVERAGE_GAPS"), 2)
            )
            if args.long_burst_memory_verify:
                accepted = len(ledger) == 1 and ledger[0]["history_memory"]["pass"]
                result, code = (
                    (dict(status="COMPLETED_HISTORY_MEMORY_VERIFY"), 0)
                    if accepted
                    else (dict(status="COMPLETED_WITH_COVERAGE_GAPS"), 2)
                )
            if args.conversion:
                # The scene already failed unless the plan COMPLETED; the memory
                # check needs every session COMMITTED, which retiring a source
                # early does not provide, so it is kept as a record only.
                conversion = read(
                    plans[0][3]["run_dir"].parent / "conversion_result.json"
                )
                accepted = (
                    len(ledger) == 1
                    and ledger[0]["status"] == "COMPLETED"
                    and conversion["state"] == "COMPLETED"
                    and ledger[0]["conversion"]["passed"]
                )
                result, code = (
                    (dict(status="COMPLETED_CONVERSION", edge=args.conversion), 0)
                    if accepted
                    # S5a gate/valve: routing restored, nothing stopped.
                    else (dict(status="CONVERSION_ABORTED_SAFELY"), 2)
                    if conversion["state"] == "ABORTED"
                    else (dict(status="COMPLETED_WITH_COVERAGE_GAPS"), 2)
                )
            if args.long_burst_deep_stay:
                accepted = len(ledger) == 1 and long_burst.assess_stay(
                    ledger[0]["summary"], values[0]
                )
                result, code = (
                    (dict(status="COMPLETED_DEEP_LONG_BURST_STAY"), 0)
                    if accepted
                    else (dict(status="COMPLETED_WITH_COVERAGE_GAPS"), 2)
                )
        if args.four_mode and not args.kv_priority_verify and not args.long_burst_pair:
            paired = four.comparisons(inherited + ledger)
            pairs = paired
            write(args.output / "four_mode_comparison.json", paired)
            valid = len(paired) == len(values) and all(
                p["arrivals_valid"]
                and p["matched_output_work"]
                and p["all_load_covered"]
                for p in paired
            )
            result, code = (
                (
                    dict(
                        status="COMPLETED_ROUTINE_FOUR_MODE_TRIAL"
                        if args.routine_four_mode
                        else "COMPLETED_UPDATED_PRESSURE_FOUR_MODE_TRIAL"
                        if args.updated_pressure_four_mode
                        else "COMPLETED_PRESSURE_FOUR_MODE_TRIAL"
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
            if args.kv_routing_verify:
                moves = ledger[0]["kv_priority"]["migrations"] if ledger else []
                accepted = (
                    len(ledger) == 1
                    and ledger[0]["summary"]["fixed_arrivals_valid"]
                    and ledger[0]["kv_routing"]["verification_pass"]
                    and (
                        not moves
                        or (
                            ledger[0]["kv_priority"]["handoff_slo_pass"] is True
                            and ledger[0]["kv_priority"]["joint_scoring_pass"]
                        )
                    )
                )
                result, code = (
                    (dict(status="COMPLETED_KV_ROUTING_VERIFY"), 0)
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
            experiment_mode="conversion-" + args.conversion
            if args.conversion
            else "long-burst-deep-stay"
            if args.long_burst_deep_stay
            else "long-burst-memory-verify"
            if args.long_burst_memory_verify
            else "long-burst-pair"
            if args.long_burst_pair
            else "routine-four-mode"
            if args.routine_four_mode
            else "updated-pressure-four-mode"
            if args.updated_pressure_four_mode
            else "kv-routing-verify"
            if args.kv_routing_verify
            else "kv-priority-verify"
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
        receipt = pack(args.output, archive, compact=True)
        Path(str(archive) + ".sha256").write_text(
            receipt["sha256"] + "  " + archive.name + "\n"
        )
        print(f"[结果] {result['status']}；请拿回 {archive}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
