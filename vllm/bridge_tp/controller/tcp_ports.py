# SPDX-License-Identifier: Apache-2.0
"""Task-local port planning and bounded Linux socket conflict evidence.

Probes are not reservations. Both peers use the published plan and the actual
listener must still handle bind failure; nothing here changes host sysctls or
terminates processes.
"""

import errno
import os
import socket
import sys
from contextlib import ExitStack
from pathlib import Path

PORT_FIRST = 18000
PORT_LAST = 27999


def ephemeral_range(proc=Path("/proc")):
    if sys.platform != "linux":
        return None
    try:
        low, high = map(
            int, (proc / "sys/net/ipv4/ip_local_port_range").read_text().split()
        )
        if not 1 <= low <= high <= 65535:
            raise ValueError("invalid range")
        return [low, high]
    except (OSError, ValueError) as error:
        raise RuntimeError("cannot establish Linux ephemeral port range") from error


def environment_snapshot(proc=Path("/proc")):
    value = dict(
        platform=sys.platform,
        ephemeral_range=ephemeral_range(proc),
        candidate_range=[PORT_FIRST, PORT_LAST],
        probes_are_reservations=False,
    )
    try:
        value["reserved_ports"] = (
            (proc / "sys/net/ipv4/ip_local_reserved_ports").read_text().strip()
        )
    except OSError:
        value["reserved_ports"] = None
    return value


def bind_snapshot(port, proc=Path("/proc"), max_fds=4096):
    """Observe matching local sockets, with bounded best-effort PID lookup."""
    value = dict(port=port, sockets=[], owners=[], errors=[])
    try:
        value["ephemeral_range"] = ephemeral_range(proc)
    except RuntimeError as error:
        value["errors"].append(str(error))
    for name in ("tcp", "tcp6", "udp", "udp6"):
        try:
            rows = (proc / "net" / name).read_text().splitlines()[1:]
        except OSError as error:
            value["errors"].append(f"{name}: {error}")
            continue
        for row in rows:
            fields = row.split()
            try:
                if int(fields[1].rsplit(":", 1)[1], 16) == port:
                    value["sockets"].append(
                        dict(
                            protocol=name,
                            local=fields[1],
                            remote=fields[2],
                            state=fields[3],
                            uid=fields[7],
                            inode=fields[9],
                        )
                    )
            except (IndexError, ValueError):
                continue
    inodes = {r["inode"] for r in value["sockets"]} - {"0"}
    scanned = 0
    if inodes:
        try:
            processes = [p for p in proc.iterdir() if p.name.isdigit()]
        except OSError:
            processes = []
        for process in processes:
            if scanned >= max_fds:
                break
            try:
                for fd in (process / "fd").iterdir():
                    if scanned >= max_fds:
                        break
                    scanned += 1
                    try:
                        target = os.readlink(fd)
                    except OSError:
                        continue
                    if target.startswith("socket:[") and target[8:-1] in inodes:
                        value["owners"].append(
                            dict(pid=int(process.name), fd=fd.name, inode=target[8:-1])
                        )
            except OSError:
                continue
    value["fd_scan_limit_reached"] = scanned >= max_fds
    return value


def probe_block(first, width=10):
    """Probe the entire block for TCP and UDP, then release every probe."""
    port, protocol = first, "tcp"
    try:
        with ExitStack() as stack:
            for port in range(first, first + width):
                for kind, protocol in (
                    (socket.SOCK_STREAM, "tcp"),
                    (socket.SOCK_DGRAM, "udp"),
                ):
                    listener = stack.enter_context(socket.socket(socket.AF_INET, kind))
                    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                        listener.setsockopt(
                            socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1
                        )
                    listener.bind(("0.0.0.0", port))
        return True, None
    except OSError as error:
        if error.errno not in (errno.EADDRINUSE, errno.EACCES, 10048, 10013):
            raise
        return False, dict(port=port, protocol=protocol, error=repr(error))


def state():
    return dict(next_port=PORT_FIRST, port_policy=environment_snapshot())


def take(state, width=10, probe=None):
    """Allocate a non-overlapping task block outside the measured OS range."""
    probe = probe or probe_block
    if type(width) is not int or width <= 0:
        raise ValueError("invalid port block width")
    policy = state["port_policy"]
    ephemeral = policy["ephemeral_range"]
    reserved = []
    for item in (policy.get("reserved_ports") or "").split(","):
        if not item:
            continue
        bounds = list(map(int, item.split("-")))
        if len(bounds) == 1:
            bounds *= 2
        if len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1] <= 65535:
            raise ValueError("invalid reserved port range")
        reserved.append(bounds)
    first = max(state["next_port"], PORT_FIRST)
    while first + width - 1 <= PORT_LAST:
        last = first + width - 1
        if ephemeral is not None and first <= ephemeral[1] and last >= ephemeral[0]:
            first = ephemeral[1] + 1
            continue
        overlaps = [high for low, high in reserved if first <= high and last >= low]
        if overlaps:
            first = max(overlaps) + 1
            continue
        if any(
            first <= b["last"] and last >= b["first"]
            for b in state.get("port_blocks", [])
        ):
            first += width
            continue
        available, failure = probe(first, width)
        if available:
            state["next_port"] = first + width
            state.setdefault("port_blocks", []).append(
                dict(owner=f"six_conversion_{first}", first=first, last=last)
            )
            return first
        failures = state.setdefault("port_probe_failures", [])
        if len(failures) < 32:
            failures.append(failure)
        first += width
    raise ValueError("no usable six-card port block outside OS ephemeral range")
