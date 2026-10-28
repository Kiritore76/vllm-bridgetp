# SPDX-License-Identifier: Apache-2.0
"""Retained/retired/restarted target controls after one completed migration."""

import json
import time

from tools.bridge_tp.qwen3_migration_diagnostics import control_batch
from tools.bridge_tp.qwen3_online_validation import latest_inventory


def restart_plan(plan, directory):
    argv = list(plan["target_argv"])
    i = argv.index("--kv-transfer-config") + 1
    config = json.loads(argv[i])
    extra = config["kv_connector_extra_config"]
    extra.update(
        bridgetp_stream_manifest=str(directory / "session_manifest.json"),
        bridgetp_stream_receipt_dir=str(directory / "receiver_receipts"),
        bridgetp_takeover_control_path=str(directory / "takeover_state.json"),
        bridgetp_preconnect_gpu_direct=False,
    )
    argv[i] = json.dumps(config)
    env = {k: v for k, v in plan["target_env"].items() if not k.startswith("BRIDGETP_")}
    env.update(
        BRIDGETP_PRESSURE_POOL="tp4",
        BRIDGETP_PRESSURE_INVENTORY_PATH=str(directory / "target_inventory.jsonl"),
    )
    return argv, env


def wait_target_empty(directory, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = latest_inventory(directory / "target_inventory.jsonl")
        if row and (row.get("writer_error") or row.get("dropped_before")):
            raise RuntimeError("restarted target inventory evidence invalid")
        if row and not row["requests"] and row["running"] == row["waiting"] == 0:
            return row
        time.sleep(0.1)
    raise TimeoutError("restarted TP2 did not drain")


def run_ablation(args, plan, processes, inputs, pool, io, common, config):
    from tools.bridge_tp.run_atomic_takeover import _post_json
    from tools.bridge_tp.run_pressure_calibration import wait_empty
    from tools.bridge_tp.run_tp2_migration_smoke import migration_request

    directory = plan["run_dir"]
    from tools.bridge_tp.pressure_session import channel_idle

    state = json.loads((directory / "takeover_state.json").read_text())
    if not channel_idle(directory, "mechanism", state["source_request_id"]):
        raise RuntimeError("matching channel IDLE required before retirement")
    native = wait_empty(directory)
    arm = dict(
        migration_id="mechanism",
        source_api_pid=processes[1].process.pid,
        target_api_pid=processes[0].process.pid,
        matching_channel_idle_verified=True,
        armed_unix_s=time.time(),
        native_generations=[r["generation"] for r in native],
    )
    common.write_json(directory / "channel_retirement_arm.json", arm)
    for role in ("source", "target"):
        token = plan[role + "_env"]["BRIDGETP_CHANNEL_DIAGNOSTIC_TOKEN"]
        result = _post_json(
            getattr(config, role + "_url"),
            "/bridge_tp/v1/diagnostic/retire_channel",
            dict(token=token),
            timeout_s=90,
        )
        if result.get("status") != "RETIRED":
            raise RuntimeError("retirement incomplete")
        common.write_json(directory / (role + "_channel_retirement.json"), result)
    control_batch(io, config.target_url, inputs, directory, pool, 5, "retired")
    wait_empty(directory)
    old = processes[0]
    old_pid = old.process.pid
    common.stop_processes([old])
    if common.process_group_alive(old):
        raise RuntimeError("old target group still alive")
    fresh = directory / "restarted"
    fresh.mkdir()
    argv, env = restart_plan(plan, fresh)
    common.write_json(
        fresh / "effective_plan.json",
        dict(
            argv=argv,
            configuration_difference=(
                "same connector/model/service flags; "
                "fresh session, no migration preconnect"
            ),
        ),
    )
    new = common.start_process(
        "target restarted", argv, env, args.out_root / "target_restarted.log"
    )
    processes.append(new)
    common.write_json(
        args.out_root / "diagnostic_processes.json",
        [dict(pid=p.process.pid, name=p.name) for p in processes],
    )
    common.wait_healthy(config.target_url, new, 900)
    io.post_streaming_completion(
        config.target_url,
        migration_request(inputs, 16, "restart-warm"),
        900,
        lambda *values: None,
        fresh / "warm_failure.json",
        retain_chunks=False,
    )
    initial = wait_target_empty(fresh)
    control_batch(io, config.target_url, inputs, fresh, pool, 5, "restarted")
    final = wait_target_empty(fresh)
    if (
        initial["free_blocks"] != final["free_blocks"]
        or final["generation"] <= initial["generation"]
    ):
        raise RuntimeError("restarted target KV did not return")
    if old_pid == new.process.pid:
        raise RuntimeError("target API pid was not replaced")
    common.write_json(
        directory / "channel_ablation_receipt.json",
        dict(
            status="COMPLETED",
            retained_and_retired_target_api_pid=old_pid,
            restarted_target_api_pid=new.process.pid,
            retired_kept_model_process=True,
            restarted_native_initial=initial,
            restarted_native_final=final,
            scope=("retire both endpoints; afterwards only ordinary target inference"),
        ),
    )
