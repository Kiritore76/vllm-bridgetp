# SPDX-License-Identifier: Apache-2.0
"""Opt-in urgent Shadow admission without increasing the decode sequence cap."""


def urgent_waiter(connector, waiting, skipped, finished):
    if not getattr(connector, "urgent_shadow_admission", False):
        return None
    loads = []
    for queue in (skipped, waiting):
        for request in queue:
            stage = connector.urgent_shadow_stage(request)
            if stage == "RESUME" and (
                request.request_id in finished
                or getattr(request.status, "name", str(request.status))
                in {"WAITING", "PREEMPTED"}
            ):
                return queue, request, stage
            if stage == "LOAD":
                loads.append((queue, request, stage))
    return loads[0] if loads else None


def ordinary_limit(connector, maximum):
    # One engine slot is reserved for urgent continuation. Existing requests
    # are never preempted; after a continuation joins it drains back naturally.
    return (
        maximum - 1
        if (getattr(connector, "urgent_shadow_admission", False) and maximum > 1)
        else maximum
    )
