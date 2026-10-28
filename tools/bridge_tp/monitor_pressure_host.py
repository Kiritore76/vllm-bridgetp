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
                "cpu.max",
                "cpuset.cpus.effective",
                "memory.events",
                "memory.current",
                "memory.max",
                "pids.current",
                "cpu/cpu.stat",
                "memory/memory.failcnt",
            )
        },
    )


def gpu_sample(include_processes=False):
    started = time.monotonic()
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
        sample = dict(
            rows=list(csv.reader(result.stdout.splitlines())),
            error=None,
            duration_s=time.monotonic() - started,
        )
        if include_processes:
            processes = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            sample["compute_processes"] = list(
                csv.reader(processes.stdout.splitlines())
            )
            sample["process_query_error"] = (
                processes.stderr if processes.returncode else None
            )
            clocks = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,clocks.sm,clocks.mem,pstate,power.limit",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3,
            )
            sample["clocks_and_limits"] = list(csv.reader(clocks.stdout.splitlines()))
            sample["clock_query_error"] = clocks.stderr if clocks.returncode else None
        return sample
    except (OSError, subprocess.SubprocessError) as error:
        return dict(rows=None, error=repr(error), duration_s=time.monotonic() - started)


def process_samples(registry, proc_root=Path("/proc")):
    """Follow only registered owned roots and their current Linux children."""
    if registry is None or not registry.exists():
        return []
    roots = json.loads(registry.read_text())
    pending = [(r["pid"], r["name"]) for r in roots]
    seen, result = set(), []
    while pending:
        pid, role = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        directory = proc_root / str(pid)
        stat = optional_text(directory / "stat")
        if stat is None:
            continue
        status = optional_text(directory / "status") or ""
        result.append(
            dict(
                pid=pid,
                role=role,
                stat=stat,
                wchan=optional_text(directory / "wchan"),
                schedstat=optional_text(directory / "schedstat"),
                io=optional_text(directory / "io"),
                status=[
                    line
                    for line in status.splitlines()
                    if line.startswith(
                        (
                            "State:",
                            "VmRSS:",
                            "Threads:",
                            "voluntary_ctxt_switches:",
                            "nonvoluntary_ctxt_switches:",
                        )
                    )
                ],
            )
        )
        children = optional_text(directory / "task" / str(pid) / "children") or ""
        pending.extend((int(child), role + ":child") for child in children.split())
    return result


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
    parser.add_argument("--process-file", type=Path)
    parser.add_argument("--gpu-processes", action="store_true")
    parser.add_argument("--gpu-interval-s", type=int, choices=(2, 10), default=10)
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
                gpu_interval_s=args.gpu_interval_s,
                pid=os.getpid(),
            )
        )
        status = "FAILED"
        try:
            while not stopping and not args.stop_file.exists():
                row = capture(Path("/proc"), Path("/sys/fs/cgroup"), args.output.parent)
                process_started = time.monotonic()
                row["processes"] = process_samples(args.process_file)
                row["process_capture_s"] = time.monotonic() - process_started
                stamp = row["monotonic_s"]
                row.update(
                    kind="host_sample",
                    sample_gap_s=stamp - last if last is not None else 0,
                )
                last = stamp
                if stamp - last_gpu >= args.gpu_interval_s:
                    row["gpu"] = gpu_sample(args.gpu_processes)
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
