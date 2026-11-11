#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Two-GPU vLLM Sleep Mode (level 1) round-trip probe (user 11-11).

Question: can a six-card group switch TP1 x2 <-> TP2 by keeping the inactive
engines asleep instead of stopping and cold-starting them? Plain vLLM only, no
BridgeTP connector, on GPUs 0 and 1, with the production serving flags
(bf16, 32K, util 0.88, eager, 16 seqs, 2048 batched tokens, block 16, no prefix
cache) plus --enable-sleep-mode and VLLM_SERVER_DEV_MODE=1.

Steps:
  1. cold-start two TP1 (one per GPU); reference output; short TTFT
  2. sleep both TP1 (level 1): time, GPU residual, host memory
  3. cold-start TP2 on both GPUs next to the sleeping TP1s (util 0.88, lower
     only if it cannot start); KV size
  4. round trips x3: sleep TP2 -> wake TP1 x2 -> serve; sleep TP1 x2 -> wake
     TP2 -> serve. Times, first-request TTFT, output equal to the reference
  5. cold baseline: everything stopped, cold TP2 alone (KV without residual)
Writes probe_result.json and engine logs into --output. No performance claim
beyond these timings; greedy output equality is checked, not assumed.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

FLAGS = [
    "--served-model-name",
    "probe",
    "--dtype",
    "bfloat16",
    "--max-model-len",
    "32768",
    "--no-enable-prefix-caching",
    "--disable-hybrid-kv-cache-manager",
    "--no-async-scheduling",
    "--enforce-eager",
    "--max-num-seqs",
    "16",
    "--max-num-batched-tokens",
    "2048",
    "--block-size",
    "16",
    "--enable-sleep-mode",
]
REF_PROMPT = "Explain in three sentences why the sky appears blue during the day."
SHORT_PROMPT_TOKENS = [9707] * 512  # fixed token ids: TTFT only, not quality


def now():
    return time.time()


def http(method, url, body=None, timeout=600):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        return resp.status, raw


class Engine:
    def __init__(self, name, gpus, tp, port, util, args, log_dir):
        self.name, self.gpus, self.tp, self.port, self.util = name, gpus, tp, port, util
        self.base = f"http://127.0.0.1:{port}"
        env = dict(
            os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus), VLLM_SERVER_DEV_MODE="1"
        )
        cmd = [
            args.python,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            args.model,
            "--tensor-parallel-size",
            str(tp),
            "--gpu-memory-utilization",
            str(util),
            "--port",
            str(port),
            *FLAGS,
        ]
        self.log_path = log_dir / f"{name}_util{util}.log"
        self.log = self.log_path.open("w")
        self.started = now()
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.ready_s = None

    def wait_ready(self, timeout=900):
        deadline = now() + timeout
        while now() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                if http("GET", self.base + "/health", timeout=5)[0] == 200:
                    self.ready_s = now() - self.started
                    return True
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(0.5)
        return False

    def kv_tokens(self):
        text = self.log_path.read_text(errors="replace")
        m = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens", text)
        return int(m[-1].replace(",", "")) if m else None

    def sleep(self):
        t = now()
        http("POST", self.base + "/sleep?level=1")
        while (
            json.loads(http("GET", self.base + "/is_sleeping")[1])["is_sleeping"]
            is not True
        ):
            time.sleep(0.05)
        return now() - t

    def wake(self):
        t = now()
        http("POST", self.base + "/wake_up")
        while (
            json.loads(http("GET", self.base + "/is_sleeping")[1])["is_sleeping"]
            is not False
        ):
            time.sleep(0.05)
        return now() - t

    def stop(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.log.close()


def ttft(engine, prompt, max_tokens):
    """Streamed completion: first-token latency, total, output token ids."""
    body = dict(
        model="probe",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0,
        stream=True,
        return_token_ids=True,
        ignore_eos=True,
    )
    req = urllib.request.Request(
        engine.base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0, first, ids = now(), None, []
    with urllib.request.urlopen(req, timeout=600) as resp:
        for line in resp:
            line = line.decode(errors="replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            choice = (json.loads(line[6:]).get("choices") or [{}])[0]
            got = choice.get("token_ids") or []
            if got and first is None:
                first = now() - t0
            ids += got
    return dict(ttft_s=first, total_s=now() - t0, tokens=len(ids), ids=ids)


def gpu_memory():
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return {
        r.split(",")[0].strip(): dict(
            used_mib=int(r.split(",")[1]), total_mib=int(r.split(",")[2])
        )
        for r in out.strip().splitlines()
    }


def host_memory(engines):
    info = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k, v = line.split(":", 1)
        if k in ("MemTotal", "MemAvailable", "Mlocked"):
            info[k + "_gib"] = round(int(v.split()[0]) / 2**20, 2)
    rss = {}
    children = {}
    for p in Path("/proc").iterdir():
        if p.name.isdigit():
            try:
                stat = (p / "stat").read_text().rsplit(")", 1)[1].split()
                children.setdefault(int(stat[1]), []).append(int(p.name))
            except OSError:
                pass
    for e in engines:
        total, todo = 0, [e.proc.pid]
        while todo:
            pid = todo.pop()
            todo += children.get(pid, [])
            try:
                for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        total += int(line.split()[1])
            except OSError:
                pass
        rss[e.name] = round(total / 2**20, 2)
    info["engine_rss_gib"] = rss
    return info


def parallel(fns):
    out = [None] * len(fns)

    def run(i, f):
        out[i] = f()

    ts = [threading.Thread(target=run, args=(i, f)) for i, f in enumerate(fns)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        print(
            "TP1 cmd:",
            [
                args.python,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                args.model,
                "--tensor-parallel-size",
                "1",
                "--gpu-memory-utilization",
                "0.88",
                "--port",
                "8401",
                *FLAGS,
            ],
        )
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    res = dict(schema="SLEEP_MODE_TWO_GPU_PROBE_V1", flags=FLAGS, steps=[], errors=[])
    engines = []

    def note(step, **v):
        v.update(step=step, gpu=gpu_memory(), host=host_memory(engines))
        res["steps"].append(v)
        print(
            json.dumps({k: v[k] for k in v if k not in ("ids",)}, ensure_ascii=False)[
                :600
            ],
            flush=True,
        )

    def save():
        (args.output / "probe_result.json").write_text(json.dumps(res, indent=1) + "\n")

    try:
        res["baseline_gpu"] = gpu_memory()
        a0 = Engine("tp1_gpu0", ["0"], 1, 8401, 0.88, args, args.output)
        a1 = Engine("tp1_gpu1", ["1"], 1, 8402, 0.88, args, args.output)
        engines += [a0, a1]
        if not all(parallel([a0.wait_ready, a1.wait_ready])):
            raise RuntimeError("TP1 cold start failed")
        ref1 = [ttft(e, REF_PROMPT, 64) for e in (a0, a1)]
        shorts = [ttft(a0, SHORT_PROMPT_TOKENS, 128) for _ in range(3)]
        note(
            "1_tp1_cold",
            cold_ready_s=[a0.ready_s, a1.ready_s],
            kv_tokens=[a0.kv_tokens(), a1.kv_tokens()],
            ref_ttft_s=[r["ttft_s"] for r in ref1],
            short_ttft_s=[s["ttft_s"] for s in shorts],
        )
        res["ref_tp1_ids"] = [r["ids"] for r in ref1]
        save()

        note("2_tp1_sleep", sleep_s=parallel([a0.sleep, a1.sleep]))
        save()

        tp2 = None
        for util in (0.88, 0.86, 0.84, 0.82, 0.80):
            cand = Engine("tp2_util", ["0", "1"], 2, 8403, util, args, args.output)
            engines.append(cand)
            if cand.wait_ready():
                tp2 = cand
                break
            engines.remove(cand)
            cand.stop()
            res["errors"].append(
                f"TP2 could not start at util {util} next to sleeping TP1s"
            )
        if tp2 is None:
            raise RuntimeError("TP2 did not start next to sleeping TP1s")
        ref2 = ttft(tp2, REF_PROMPT, 64)
        res["ref_tp2_ids"] = ref2["ids"]
        note(
            "3_tp2_cold_beside_sleep",
            util=tp2.util,
            cold_ready_s=tp2.ready_s,
            kv_tokens=tp2.kv_tokens(),
            ref_ttft_s=ref2["ttft_s"],
            short_ttft_s=[
                ttft(tp2, SHORT_PROMPT_TOKENS, 128)["ttft_s"] for _ in range(3)
            ],
        )
        save()

        for r in range(args.rounds):
            note(f"4.{r}a_tp2_sleep", sleep_s=tp2.sleep())
            wake = parallel([a0.wake, a1.wake])
            first = [ttft(e, REF_PROMPT, 64) for e in (a0, a1)]
            note(
                f"4.{r}b_tp1_wake",
                wake_s=wake,
                first_ttft_s=[f["ttft_s"] for f in first],
                same_output=[
                    f["ids"] == ref for f, ref in zip(first, res["ref_tp1_ids"])
                ],
                short_ttft_s=[
                    ttft(a0, SHORT_PROMPT_TOKENS, 128)["ttft_s"] for _ in range(3)
                ],
            )
            note(f"4.{r}c_tp1_sleep", sleep_s=parallel([a0.sleep, a1.sleep]))
            w2 = tp2.wake()
            f2 = ttft(tp2, REF_PROMPT, 64)
            note(
                f"4.{r}d_tp2_wake",
                wake_s=w2,
                first_ttft_s=f2["ttft_s"],
                same_output=f2["ids"] == res["ref_tp2_ids"],
                short_ttft_s=[
                    ttft(tp2, SHORT_PROMPT_TOKENS, 128)["ttft_s"] for _ in range(3)
                ],
            )
            save()

        t = now()
        for e in list(engines):
            e.stop()
        engines.clear()
        stop_s = now() - t
        cold = Engine("tp2_alone", ["0", "1"], 2, 8404, 0.88, args, args.output)
        engines.append(cold)
        ok = cold.wait_ready()
        note(
            "5_tp2_cold_alone",
            stop_all_s=stop_s,
            started=ok,
            cold_ready_s=cold.ready_s,
            kv_tokens=cold.kv_tokens(),
        )
        res["passed"] = ok
    except Exception as error:  # keep every measurement taken so far
        res["errors"].append(repr(error))
        res["passed"] = False
    finally:
        for e in engines:
            e.stop()
        save()
    return 0 if res.get("passed") else 1


if __name__ == "__main__":
    sys.exit(main())
