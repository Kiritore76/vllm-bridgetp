# SPDX-License-Identifier: Apache-2.0
"""Six-card static platform checks; one existing migration target (bundle B)."""

import time
from pathlib import Path

from tools.bridge_tp import pressure_conversion as pc

GPUS = ("2", "3", "0", "1", "4", "5")
LAYOUTS = ("L0", "L1AC", "L2C", "L3")
MIXED = ("L1AC", "L2C")


def configure(plan):
    """Give every source a separate target file and warmup receipt identity.

    Stage 1 uses only the existing B target. The extra TP2 in L2C serves new
    requests but is not a migration receiver until the multi-target stage.
    """
    plan["static_cards"] = 6
    aliases = plan["instances"]["legacy_pool_aliases"]
    target = aliases.get("tp4")
    extras = {r["instance_id"]: r for r in plan["extra_engines"]}
    channels = {r["source"]: r for r in plan["channel_sources"]}
    files = {}
    if set(aliases) != {"tp1", "tp4"}:
        return
    for row in plan["instances"]["instances"]:
        source = row["instance_id"]
        if row["tp_degree"] != 1:
            continue
        channel = channels[source]
        path = str(plan["run_dir"].parent / f"gpu_targets_{source}.json")
        files[path] = [
            dict(
                target=target,
                host="127.0.0.1",
                tp=2,
                base_port=channel["base_port"],
                generation=channel["generation"],
            )
        ]
        row["targets_file"] = path
        if source == aliases["tp1"]:
            plan["source_env"].pop("BRIDGETP_GPU_DIRECT_TARGETS", None)
            plan["source_env"]["BRIDGETP_GPU_DIRECT_TARGETS_FILE"] = path
        else:
            extras[source]["targets_file"] = path
    plan["runtime_target_files"] = files


def instances(plan):
    """All static engine endpoints, native files and trace identities."""
    aliases = plan["instances"]["legacy_pool_aliases"]
    result = []
    for row in plan["instances"]["instances"]:
        name = row["instance_id"]
        side = next(
            (
                side
                for side, key in (("source", "tp1"), ("target", "tp4"))
                if aliases.get(key) == name
            ),
            None,
        )
        environment = plan[
            (side or ("source" if row.get("migration_channel") else "target")) + "_env"
        ]
        inventory_path = (
            row["inventory_path"]
            if plan.get("canonical_instance_ids") or not side
            else str(plan["run_dir"] / f"{side}_inventory.jsonl")
        )
        result.append(
            dict(
                row,
                inventory_path=inventory_path,
                trace_name=name if plan.get("canonical_instance_ids") else side or name,
                finalize_token=environment["BRIDGETP_PERFORMANCE_FINALIZE_TOKEN"],
            )
        )
    return result


def wait_empty(plan, timeout_s=30):
    rows, until = instances(plan), time.monotonic() + timeout_s
    while True:
        inventories = {
            row["instance_id"]: pc.load("native_journal").latest_inventory(
                Path(row["inventory_path"])
            )
            for row in rows
        }
        if all(
            value
            and value["sampled_at"] <= time.time()
            and not value["requests"]
            and not value["running"]
            and not value["waiting"]
            for value in inventories.values()
        ):
            return inventories
        if time.monotonic() >= until:
            raise TimeoutError("six-card static inventories did not settle empty")
        time.sleep(0.2)


def prepare(plan, common):
    """Check native context/headroom and every configured warm channel."""
    from tools.bridge_tp.run_qwen3_stay_migrate import prepare_native

    prepare_native(plan, common)
    inventories = wait_empty(plan)
    for row in inventories.values():
        if row["context_limit"] != 32768 or (
            row["free_blocks"] * row["block_size"] < 32768 + 125 * row["block_size"]
        ):
            raise ValueError("six-card native context/headroom differs")
    receipts = warm_channels(plan)
    aliases = plan["instances"]["legacy_pool_aliases"]
    common.write_json(
        plan["run_dir"].parent / "six_card_initial_receipt.json",
        dict(
            inventories=inventories,
            channels=receipts,
            migration_target=aliases.get("tp4"),
            multi_target_migration=False,
        ),
    )


def warm_channels(plan, timeout_s=30):
    """The target-file followers warm asynchronously after engine startup."""
    if plan.get("receiver_stage") in (2, 3):
        from tools.bridge_tp.migration_receivers import warm_channels as warm

        return warm(plan, pc.preconnect_receipts, timeout_s)
    aliases = plan["instances"]["legacy_pool_aliases"]
    if set(aliases) != {"tp1", "tp4"}:
        return {}
    until = time.monotonic() + timeout_s
    while True:
        receipts, missing = pc.preconnect_receipts(plan["run_dir"]), []
        for channel in plan["channel_sources"]:
            source = channel["source"]
            pool = source if plan.get("canonical_instance_ids") else (
                "tp1" if source == aliases["tp1"] else source
            )
            names = [f"source-{pool}-{aliases['tp4']}.json"] + [
                f"tp_rank_{rank}-{source}.json" for rank in range(2)
            ]
            if any(
                receipts.get(name, {}).get("status") != "WARMUP_COMPLETE"
                or receipts[name].get("channel_generation") != channel["generation"]
                for name in names
            ):
                missing.append(source)
        if not missing:
            return receipts
        if time.monotonic() >= until:
            raise TimeoutError(
                "six-card warm channel proof incomplete: " + str(missing)
            )
        time.sleep(0.2)


def finalize(plan, common):
    from tools.bridge_tp.run_atomic_takeover import _post_json
    from tools.bridge_tp.verify_performance_trace import verify_pool

    common.write_json(
        plan["run_dir"].parent / "native_empty_receipt.json", wait_empty(plan)
    )
    for row in instances(plan):
        name = row["trace_name"]
        receipt = _post_json(
            row["endpoint"],
            "/bridge_tp/v1/performance_trace/finalize",
            {"token": row["finalize_token"]},
            60,
        )
        common.write_json(
            plan["run_dir"].parent / f"{name}_trace_finalization.json", receipt
        )
        if receipt.get("status") != "FINALIZED":
            raise ValueError("six-card trace did not finalize: " + name)
        common.write_json(
            plan["run_dir"].parent / f"{name}_trace_acceptance.json",
            verify_pool(plan["run_dir"] / "performance_trace" / name, row["tp_degree"]),
        )


def assess(plan, proof, expected_requests):
    """Mechanism coverage is independent of performance scores."""
    directory = plan["run_dir"]
    status = pc.read_json(directory / "layout_status.json") or {}
    active = status.get("instances", {})
    expected = plan["instances"]["instances"]
    requests = pc.read_json(directory / "pressure_request_results.json") or []
    ids = {r["instance_id"] for r in expected}
    complete = (
        len(requests) == expected_requests
        and len({r["job_id"] for r in requests}) == len(requests)
        and all(
            r["source_done"]
            and r["source_error"] is None
            and len(r["response"]["emitted"]) == r["request"]["max_tokens"]
            for r in requests
        )
    )
    observed = set(active) == ids and all(
        row.get("lifecycle") == "ACTIVE" and row.get("inventory") is not None
        for row in active.values()
    )
    migrated = any(
        s["state"] == "COMMITTED" and s.get("source_release") for s in proof["sessions"]
    )
    required = (
        set(plan["instances"]["legacy_pool_aliases"]) == {"tp1", "tp4"}
        and plan.get("receiver_stage") not in (2, 3)
        and plan.get("migration_coverage_required", True)
    )
    receiver_targets = {row.get("target_instance") for row in proof["sessions"]}
    if plan.get("receiver_stage") == 2:
        allowed = plan["instances"]["legacy_pool_aliases"]["tp4"]
        if receiver_targets - {allowed}:
            raise ValueError("stage 2 executed migration outside B")
    elif plan.get("receiver_stage") == 3:
        allowed = set(plan["migration_plan"]["execution_targets"])
        if receiver_targets - allowed:
            raise ValueError("migration executed outside the allowed receivers")
    return dict(
        passed=complete and observed and (migrated or not required),
        all_instances_active=observed,
        requests_complete=complete,
        migration_required=required,
        committed_migration_observed=migrated,
        migration_target=(
            "B only; A/B preconnected"
            if plan.get("receiver_stage") == 2
            else "chosen per migration among "
            + ",".join(plan["migration_plan"]["execution_targets"])
            if plan.get("receiver_stage") == 3
            else "B only; multi-target stage not implemented"
        ),
        migration_targets_used=sorted(receiver_targets - {None}),
        performance_validated=False,
    )
