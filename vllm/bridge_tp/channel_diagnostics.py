# SPDX-License-Identifier: Apache-2.0
"""Opt-in, one-way retirement of quiescent BridgeTP migration channels only."""

import contextlib
import os
import socket

from vllm.bridge_tp.channel_runtime import receiver_snapshot


def idle_lifecycle(channel):
    lifecycle = getattr(channel, "lifecycle", None)
    return (
        lifecycle is not None
        and lifecycle.state.value == "IDLE"
        and lifecycle.active_session is None
    )


def retire_senders(pool, active, lock):
    with lock:
        if (
            active
            or len(pool) != 1
            or any(not idle_lifecycle(s) for s in pool.values())
        ):
            raise RuntimeError(
                "source migration pool must contain exactly one IDLE channel"
            )
        senders = list(pool.values())
        if any(not s.comms for s in senders):
            raise RuntimeError("source communicators already absent")
        count = sum(len(s.comms) for s in senders)
        for sender in senders:
            # IDLE peers are not in a receive loop to ACK CLOSE. Do not call
            # the graceful wire close and wait for an ACK that cannot arrive.
            for connection in sender.connections:
                with contextlib.suppress(OSError):
                    connection.shutdown(socket.SHUT_RDWR)
                connection.close()
            sender.connections = []
            sender.abort()
            if sender.comms or sender.streams or sender.nccl is not None:
                raise RuntimeError("source migration resources still present")
        pool.clear()
    return dict(
        status="RETIRED", channels=len(senders), communicators=count, role="source"
    )


def retire_receiver(connector):
    if not connector.persistent_channel or connector._retained_gpu_receivers:
        raise RuntimeError("requires only the persistent target migration channel")
    if any(t.is_alive() for t in connector._load_threads.values()):
        raise RuntimeError("target restore thread still active")
    receiver = connector._persistent_gpu_receiver
    if receiver is None:
        raise RuntimeError("target persistent receiver is absent")
    if not idle_lifecycle(receiver):
        raise RuntimeError("target migration lifecycle is not IDLE")
    if receiver.comm is None:
        raise RuntimeError("target migration communicator is absent")
    if connector._prebound_gpu_receiver not in (None, receiver):
        raise RuntimeError("unexpected independent prebound receiver")
    connector._prebind_receiver_stop.set()
    thread = connector._prebind_receiver_thread
    if thread is not None:
        thread.join(timeout=10)
        if thread.is_alive():
            raise RuntimeError("target prebind thread did not stop")
    receiver.abort()
    if (
        receiver.comm is not None
        or receiver.stream is not None
        or receiver.nccl is not None
    ):
        raise RuntimeError("target migration resources still present")
    connector._persistent_gpu_receiver = None
    connector._prebound_gpu_receiver = None
    connector._prebound_gpu_history = None
    return dict(status="RETIRED", channels=1, communicators=1, role="target")


def worker_retire(worker):
    role = os.environ.get("BRIDGETP_CHANNEL_DIAGNOSTIC_ROLE")
    if role not in {"source", "target"} or not os.environ.get(
        "BRIDGETP_CHANNEL_DIAGNOSTIC_TOKEN"
    ):
        raise RuntimeError("channel retirement diagnostic is not enabled")
    from vllm.distributed.parallel_state import get_tp_group

    group = get_tp_group()
    before = (id(worker.model_runner.model), id(group), id(group.device_communicator))
    if role == "source":
        from vllm.bridge_tp import kv_stream

        result = retire_senders(
            kv_stream._persistent_gpu_senders,
            kv_stream._active_persistent_gpu_senders,
            kv_stream._persistent_gpu_senders_lock,
        )
    else:
        from vllm.distributed.kv_transfer import get_kv_transfer_group

        connector = get_kv_transfer_group()
        snapshot = receiver_snapshot(connector)
        try:
            result = retire_receiver(connector)
        except RuntimeError as error:
            # Expected guard refusals must not escape the worker RPC and
            # terminate the model executor. Preserve the state for diagnosis.
            return dict(
                status="REJECTED",
                error=str(error),
                role=role,
                pid=os.getpid(),
                channel_runtime=snapshot,
            )
        result["channel_runtime_before"] = snapshot
    after = (
        id(worker.model_runner.model),
        id(get_tp_group()),
        id(get_tp_group().device_communicator),
    )
    if before != after:
        raise RuntimeError("model or tensor-parallel communicator changed")
    return dict(
        **result,
        pid=os.getpid(),
        model_id=before[0],
        tp_group_id=before[1],
        tp_device_communicator_id=before[2],
        identity_unchanged=True,
    )


def validate_arm(directory, role, pid):
    import json

    from vllm.bridge_tp.controller.native_journal import latest_inventory

    arm = json.loads((directory / "channel_retirement_arm.json").read_text())
    if arm.get(role + "_api_pid") != pid or arm.get("migration_id") != "mechanism":
        raise ValueError("retirement arm identity differs")
    state = json.loads((directory / "takeover_state.json").read_text())
    release = json.loads((directory / "source_kv_release_receipt.json").read_text())
    if (
        state.get("state") != "COMMITTED"
        or state.get("source_abort_dispatched") is not True
        or release.get("status") != "SOURCE_KV_RELEASED"
        or release.get("request_id") != state.get("source_request_id")
    ):
        raise ValueError("source ownership/release proof missing")
    if not arm.get("matching_channel_idle_verified"):
        raise ValueError("matching channel IDLE was not confirmed")
    generations = arm.get("native_generations")
    if not isinstance(generations, list) or len(generations) != 2:
        raise ValueError("native generation arm missing")
    for index, name in enumerate(("source", "target")):
        native = latest_inventory(directory / (name + "_inventory.jsonl"))
        if (
            not native
            or native["requests"]
            or native["running"]
            or native["waiting"]
            or native.get("writer_error")
            or native.get("dropped_before")
            or native.get("generation", -1) < generations[index]
        ):
            raise ValueError("both native pools must be empty")
    return arm
