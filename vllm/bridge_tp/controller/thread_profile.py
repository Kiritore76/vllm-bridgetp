# SPDX-License-Identifier: Apache-2.0
"""Optional low-rate sampling of where controller threads are and what they cost.

Off by default. Every ``sample_s`` it records each thread's top frames, and every
``report_s`` it appends one JSON line: CPU seconds per thread group since the last
line (from /proc, Linux only) and the most frequent stacks. A sample is only
where a thread stood when the sampler held the GIL, so stacks show proportions,
not exact CPU; the /proc numbers are the CPU.
"""

import collections
import json
import os
import re
import sys
import threading
import time
from pathlib import Path

IDLE = {
    "wait",
    "acquire",
    "select",
    "poll",
    "sleep",
    "recv_into",
    "readinto",
    "readline",
    "get",
    "_worker",
    "accept",
}
TICKS = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def group_name(name):
    """Thread pool members share a group: 'pool-3_17' and 'pool-3_18' count once."""
    return re.sub(r"[-_ ]?\d+", "", name) or "unnamed"


def stack_of(frame, depth=4):
    """Innermost first: 'file:function:line' for the top ``depth`` frames."""
    rows = []
    while frame is not None and len(rows) < depth:
        code = frame.f_code
        rows.append(f"{Path(code.co_filename).name}:{code.co_name}:{frame.f_lineno}")
        frame = frame.f_back
    return tuple(rows)


def is_idle(stack):
    return bool(stack) and stack[0].split(":")[1] in IDLE


def thread_cpu_seconds(root=Path("/proc/self/task")):
    """{native thread id: CPU seconds}; empty where /proc does not exist."""
    out = {}
    if not root.is_dir():
        return out
    for task in root.iterdir():
        try:
            fields = (task / "stat").read_text().rsplit(")", 1)[1].split()
            out[int(task.name)] = (int(fields[11]) + int(fields[12])) / TICKS
        except (OSError, ValueError, IndexError):
            continue  # a thread may exit while the directory is read
    return out


class ThreadProfile:
    def __init__(self, path, sample_s=0.2, report_s=30.0, top=40, clock=time.time):
        self.path, self.sample_s, self.report_s = Path(path), sample_s, report_s
        self.top, self.clock = top, clock
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="thread-profile", daemon=True
        )
        self.stacks = collections.Counter()
        self.idle_stacks = collections.Counter()
        self.samples = 0
        self.last_cpu = {}
        self.names = {}

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join(5)
        return False

    def sample(self):
        me = threading.get_ident()
        threads = {t.ident: t for t in threading.enumerate()}
        for t in threads.values():
            if t.native_id is not None:
                self.names[t.native_id] = group_name(t.name)
        for ident, frame in sys._current_frames().items():
            if ident == me:
                continue
            thread = threads.get(ident)
            group = group_name(thread.name) if thread else "unknown"
            stack = stack_of(frame)
            target = self.idle_stacks if is_idle(stack) else self.stacks
            target[(group, stack)] += 1
        self.samples += 1

    def report(self):
        cpu = thread_cpu_seconds()
        groups = collections.Counter()
        counts = collections.Counter()
        for tid, used in cpu.items():
            name = self.names.get(tid, "unknown")
            groups[name] += used - self.last_cpu.get(tid, 0.0)
            counts[name] += 1
        self.last_cpu = cpu
        row = dict(
            unix_s=self.clock(),
            samples=self.samples,
            thread_cpu_s={k: round(v, 2) for k, v in groups.most_common()},
            threads=dict(counts),
            busy_stacks=[
                dict(group=g, stack=list(s), count=n)
                for (g, s), n in self.stacks.most_common(self.top)
            ],
            idle_stacks=[
                dict(group=g, stack=list(s), count=n)
                for (g, s), n in self.idle_stacks.most_common(10)
            ],
        )
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        self.stacks.clear()
        self.idle_stacks.clear()
        self.samples = 0

    def _run(self):
        next_report = time.monotonic() + self.report_s
        while not self.stop_event.wait(self.sample_s):
            self.sample()
            if time.monotonic() >= next_report:
                self.report()
                next_report += self.report_s
        self.report()
