# SPDX-License-Identifier: Apache-2.0
"""Capture evidence when a vLLM engine stops returning (BridgeTP, 11-12).

Runs beside a scene. When any engine log under ``--root`` prints vLLM's
"No available shared memory broadcast block" (its workers have not answered
for 60 s), it records, for every Python process that registered
``stack_dump/sitecustomize.py`` (``PY_STACK_DUMP_ON_USR1=1``): command line,
status, wchan and kernel stack, then sends SIGUSR1 so each prints all its
Python thread stacks into its own log. nvidia-smi is recorded too. At most
``--max-dumps`` captures, ``--spacing-s`` apart. Linux only; it never stops
or restarts anything.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

PATTERN = "No available shared memory broadcast block"


def read(path, limit=None):
    try:
        data = Path(path).read_bytes()
    except OSError as error:
        return f"<unreadable: {error}>"
    if limit is not None:
        data = data[:limit]
    return data.decode("utf-8", "replace")


def candidates(own=os.getpid(), proc=Path("/proc")):
    """Python processes that registered the SIGUSR1 stack dump."""
    rows = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own:
            continue
        try:
            environ = (entry / "environ").read_bytes().split(b"\0")
            cmdline = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if b"PY_STACK_DUMP_ON_USR1=1" not in environ:
            continue
        if not cmdline or b"python" not in os.path.basename(cmdline[0]):
            continue
        pool = next(
            (
                v.split(b"=", 1)[1].decode("utf-8", "replace")
                for v in environ
                if v.startswith(b"BRIDGETP_PRESSURE_POOL=")
            ),
            None,
        )
        rows.append(
            dict(
                pid=int(entry.name),
                pool=pool,
                cmdline=" ".join(c.decode("utf-8", "replace") for c in cmdline if c),
            )
        )
    return rows


def capture(output, number, trigger, kill=os.kill, proc=Path("/proc")):
    folder = Path(output) / f"hang_{number}"
    folder.mkdir(parents=True, exist_ok=True)
    rows = candidates(proc=proc)
    for row in rows:
        base = proc / str(row["pid"])
        row.update(
            status=read(base / "status", 4096),
            wchan=read(base / "wchan", 256),
            kernel_stack=read(base / "stack", 16384),
        )
    try:
        smi = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=30
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        smi = f"<nvidia-smi failed: {error}>"
    (folder / "nvidia_smi.txt").write_text(smi, encoding="utf-8")
    signalled = []
    usr1 = getattr(signal, "SIGUSR1", None)  # POSIX only
    for row in rows if usr1 is not None else ():
        try:
            kill(row["pid"], usr1)
            signalled.append(row["pid"])
        except OSError:
            pass
    (folder / "processes.json").write_text(
        json.dumps(
            dict(
                unix_s=time.time(),
                trigger=trigger,
                signalled=signalled,
                note="Python thread stacks went to each process's own log",
                processes=rows,
            ),
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return folder


def scan(root, offsets):
    """New matching lines since the last scan: [(log name, line)]."""
    found = []
    for path in sorted(Path(root).rglob("*.log")):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        start = offsets.get(path, 0)
        if size <= start:
            offsets[path] = min(start, size)
            continue
        with path.open("rb") as stream:
            stream.seek(start)
            chunk = stream.read(size - start)
        offsets[path] = size
        for line in chunk.decode("utf-8", "replace").splitlines():
            if PATTERN in line:
                found.append((path.name, line[:300]))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument("--spacing-s", type=float, default=30.0)
    parser.add_argument("--max-dumps", type=int, default=3)
    args = parser.parse_args(argv)
    offsets, dumps, last = {}, 0, -float("inf")
    while dumps < args.max_dumps:
        if args.root.exists():
            found = scan(args.root, offsets)
            if found and time.monotonic() - last >= args.spacing_s:
                dumps += 1
                last = time.monotonic()
                folder = capture(args.output, dumps, found[:5])
                print(f"[hang_watch] capture {dumps}: {folder}", flush=True)
        time.sleep(args.interval_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
