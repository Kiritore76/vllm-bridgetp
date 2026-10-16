# SPDX-License-Identifier: Apache-2.0
"""Low-priority Linux host evidence, independent of serving and SSE threads."""

import argparse
import contextlib
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def optional_text(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


def capture(proc_root, cgroup_root, filesystem):
    stat = optional_text(proc_root / "stat") or ""
    mem = optional_text(proc_root / "meminfo") or ""
    space = os.statvfs(filesystem)
    return dict(
        unix_s=time.time(),
        monotonic_s=time.monotonic(),
        loadavg=optional_text(proc_root / "loadavg"),
        cpu_stat=[
            line
            for line in stat.splitlines()
            if line.startswith(("cpu ", "ctxt ", "procs_running ", "procs_blocked "))
        ],
        memory=[
            line
            for line in mem.splitlines()
            if line.split(":")[0]
            in {"MemAvailable", "MemFree", "SwapFree", "Dirty", "Writeback"}
        ],
        pressure={
            kind: optional_text(proc_root / "pressure" / kind)
            for kind in ("cpu", "io", "memory")
        },
        diskstats=optional_text(proc_root / "diskstats"),
        filesystem_free_bytes=space.f_bavail * space.f_frsize,
        cgroup={
            name: optional_text(cgroup_root / name)
            for name in (
                "cpu.stat",
                "memory.events",
                "memory.current",
                "pids.current",
                "cpu/cpu.stat",
                "memory/memory.failcnt",
            )
        },
    )


def gpu_sample():
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return dict(rows=list(csv.reader(result.stdout.splitlines())), error=None)
    except (OSError, subprocess.SubprocessError) as error:
        return dict(rows=None, error=repr(error))


def validate_journal(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if (
        not rows
        or rows[0].get("schema") != "PRESSURE_HOST_MONITOR_V1"
        or rows[-1].get("kind") != "monitor_footer"
        or rows[-1].get("status") != "COMPLETED"
        or rows[-1]["samples"] != len(rows) - 2
        or rows[-1]["samples"] < 1
    ):
        raise RuntimeError("host monitor did not finalize; retain evidence")
    samples = rows[1:-1]
    return dict(
        status="COMPLETED",
        samples=len(samples),
        largest_sample_gap_s=max(r["sample_gap_s"] for r in samples),
        gpu_query_errors=sum(bool(r.get("gpu", {}).get("error")) for r in samples),
        note="sampling gaps are evidence; no slow sample is removed",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    args = parser.parse_args()
    if os.name != "posix":
        raise RuntimeError("Linux monitor required")
    with contextlib.suppress(OSError):
        os.nice(10)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    samples = 0
    last = None
    last_gpu = -float("inf")
    with args.output.open("x", encoding="utf-8") as stream:

        def write(row):
            stream.write(json.dumps(row) + "\n")
            stream.flush()

        write(
            dict(
                schema="PRESSURE_HOST_MONITOR_V1",
                kind="monitor_header",
                host_interval_s=2,
                gpu_interval_s=10,
                pid=os.getpid(),
            )
        )
        status = "FAILED"
        try:
            while not stopping and not args.stop_file.exists():
                row = capture(Path("/proc"), Path("/sys/fs/cgroup"), args.output.parent)
                stamp = row["monotonic_s"]
                row.update(
                    kind="host_sample",
                    sample_gap_s=stamp - last if last is not None else 0,
                )
                last = stamp
                if stamp - last_gpu >= 10:
                    row["gpu"] = gpu_sample()
                    last_gpu = stamp
                write(row)
                samples += 1
                deadline = stamp + 2
                while (
                    not stopping
                    and not args.stop_file.exists()
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.2)
            status = "COMPLETED"
        finally:
            write(
                dict(
                    kind="monitor_footer",
                    status=status,
                    samples=samples,
                    unix_s=time.time(),
                    monotonic_s=time.monotonic(),
                )
            )


if __name__ == "__main__":
    main()
