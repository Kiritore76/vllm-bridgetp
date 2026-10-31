# SPDX-License-Identifier: Apache-2.0
"""Record GPU identity, topology and the NCCL transport each GPU pair really uses.

Read-only with respect to serving: it refuses to run while any compute process
is present and it starts only its own two-rank NCCL workers, one pair at a time.
Each rank sees only its own GPU (CUDA_VISIBLE_DEVICES), as source and target
engines do, so NCCL must pick its transport across process boundaries.

The parsed transport is what NCCL logged (for example ``P2P/CUMEM`` or ``SHM``).
It is evidence about this NCCL build and machine, not a guarantee for other
communicators, message sizes or driver settings.
"""

import argparse
import itertools
import json
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VIA = re.compile(r"\bvia\s+(\S+)")


def transports(log_text):
    """Distinct NCCL channel transports, e.g. {'P2P/CUMEM', 'SHM/direct/direct'}."""
    found = set()
    for line in log_text.splitlines():
        if "Channel" not in line or " via " not in line:
            continue
        match = VIA.search(line)
        if match:
            found.add(match.group(1))
    return sorted(found)


def classify(found):
    """Coarse class; anything not plainly P2P is reported as non-direct."""
    if not found:
        return "UNKNOWN"
    if all(t.startswith("P2P") for t in found):
        return "P2P"
    if any(t.startswith("P2P") for t in found):
        return "MIXED"
    return "NON_P2P"


def pairs(gpus):
    if len(set(gpus)) != len(gpus) or len(gpus) < 2:
        raise ValueError("at least two distinct GPUs required")
    return list(itertools.combinations(gpus, 2))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def worker(rank, port, size_mb, repeats):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=2,
    )
    elements = size_mb * 1024 * 1024 // 2
    tensor = torch.full((elements,), rank + 1, dtype=torch.bfloat16, device="cuda")
    results = {}
    for direction, sender in (("0to1", 0), ("1to0", 1)):
        timings = []
        for index in range(repeats + 1):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            if rank == sender:
                dist.send(tensor, 1 - rank)
            else:
                dist.recv(tensor, 1 - rank)
            torch.cuda.synchronize()
            if index:  # first transfer establishes the connection
                timings.append(time.perf_counter() - start)
        if rank != sender and not bool(torch.all(tensor == sender + 1).item()):
            raise RuntimeError("received payload differs")
        tensor.fill_(rank + 1)
        median = sorted(timings)[len(timings) // 2]
        results[direction] = dict(seconds=timings, median_GBps=size_mb / 1024 / median)
    dist.destroy_process_group()
    if rank == 1:
        print("PROBE_RESULT " + json.dumps(results), flush=True)


def probe_pair(src, dst, out_dir, size_mb, repeats, timeout_s):
    port = free_port()
    logs = []
    procs = []
    for rank, gpu in enumerate((src, dst)):
        env = dict(os.environ)
        env.update(
            CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_VISIBLE_DEVICES=str(gpu),
            NCCL_DEBUG="INFO",
            NCCL_DEBUG_SUBSYS="INIT,P2P,SHM,GRAPH",
            OMP_NUM_THREADS="1",
            PYTHONUNBUFFERED="1",
        )
        log = out_dir / f"pair_{src}_{dst}_rank{rank}.log"
        logs.append(log)
        handle = log.open("wb")
        procs.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).resolve()),
                    "--worker",
                    "--rank",
                    str(rank),
                    "--port",
                    str(port),
                    "--size-mb",
                    str(size_mb),
                    "--repeats",
                    str(repeats),
                ],
                cwd=ROOT,
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        )
    deadline = time.monotonic() + timeout_s
    codes = []
    for proc in procs:
        try:
            codes.append(proc.wait(max(1, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            proc.kill()
            codes.append(proc.wait())
    text = "\n".join(log.read_text(errors="replace") for log in logs)
    found = transports(text)
    measured = None
    for line in text.splitlines():
        if line.startswith("PROBE_RESULT "):
            measured = json.loads(line[len("PROBE_RESULT ") :])
    return dict(
        gpus=[src, dst],
        exit_codes=codes,
        ok=codes == [0, 0] and measured is not None,
        transports=found,
        transport_class=classify(found),
        size_mb=size_mb,
        bandwidth=measured,
        logs=[log.name for log in logs],
    )


def smi(*args):
    return subprocess.check_output(["nvidia-smi", *args], text=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--size-mb", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--pair-timeout-s", type=float, default=180)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--port", type=int)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.rank, args.port, args.size_mb, args.repeats)
        return 0
    if args.out is None:
        parser.error("--out is required")
    if not 1 <= args.size_mb <= 4096 or not 1 <= args.repeats <= 50:
        raise ValueError("bounded probe size/repeats required")
    gpus = [int(g) for g in args.gpus.split(",")]
    if smi("--query-compute-apps=pid", "--format=csv,noheader").strip():
        raise RuntimeError("GPUs busy; nothing stopped")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "nvidia_smi_L.txt").write_text(smi("-L"))
    (args.out / "gpu_topology.txt").write_text(smi("topo", "-m"))
    (args.out / "gpu_query.csv").write_text(
        smi(
            "--query-gpu=index,uuid,pci.bus_id,name,memory.total,driver_version",
            "--format=csv",
        )
    )
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # before CUDA initializes
    import torch

    peer = {
        f"{a}->{b}": bool(torch.cuda.can_device_access_peer(a, b))
        for a in gpus
        for b in gpus
        if a != b
    }
    try:
        nccl = list(torch.cuda.nccl.version())
    except Exception:
        nccl = None
    rows = [
        probe_pair(a, b, args.out, args.size_mb, args.repeats, args.pair_timeout_s)
        for a, b in pairs(gpus)
    ]
    result = dict(
        status="COMPLETED" if all(r["ok"] for r in rows) else "TECHNICAL_FAILURE",
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        torch_nccl=nccl,
        nccl_environment={k: v for k, v in os.environ.items() if k.startswith("NCCL_")},
        peer_access=peer,
        pairs=rows,
        note="torch.distributed NCCL; transport of vLLM pynccl channels is "
        "expected to match only if it loads the same libnccl",
    )
    (args.out / "link_probe.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in rows:
        keys = ("gpus", "ok", "transport_class", "transports")
        print(json.dumps({k: row[k] for k in keys}))
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
