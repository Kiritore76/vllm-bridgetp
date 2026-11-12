# SPDX-License-Identifier: Apache-2.0
"""Direct-to-KV switch in the six-card plans and per-rank memory samples."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from tests.bridge_tp.test_runtime_instance_ids import canonical_plan
from tools.bridge_tp import run_qwen3_stay_migrate as runner
from tools.bridge_tp import six_conversion as six
from vllm.bridge_tp import streaming_connector as sc
from vllm.bridge_tp.gpu_memory_log import GpuMemoryLog


def extra_config(argv):
    return json.loads(argv[argv.index("--kv-transfer-config") + 1])[
        "kv_connector_extra_config"
    ]


class PlanTests(unittest.TestCase):
    def test_switch_reaches_every_tp2_receiver(self):
        with tempfile.TemporaryDirectory() as folder:
            plan = canonical_plan(Path(folder), "L2C")
            runner.set_connector_extra(
                plan["target_argv"],
                bridgetp_history_direct_to_kv=True,
                bridgetp_history_direct_ring_slots=16,
                bridgetp_gpu_memory_log_s=1.0,
            )
            six.configure(plan)
            receivers = [plan["target_argv"]] + [
                e["receiver_argv"] for e in plan["extra_engines"] if "receiver_argv" in e
            ]
            self.assertEqual(len(receivers), 2)  # L2C: A and B TP2
            for argv in receivers:
                config = extra_config(argv)
                self.assertIs(config["bridgetp_history_direct_to_kv"], True)
                self.assertEqual(config["bridgetp_history_direct_ring_slots"], 16)
                self.assertEqual(config["bridgetp_gpu_memory_log_s"], 1.0)
                # Unchanged existing receiver options.
                self.assertIn("bridgetp_gpu_direct_channel_plan", config)

    def test_no_options_leave_the_argv_untouched(self):
        argv = ["serve", "--kv-transfer-config", json.dumps({"a": 1})]
        before = list(argv)
        runner.set_connector_extra(argv)
        self.assertEqual(argv, before)
        with self.assertRaises(ValueError):
            runner.set_connector_extra(["serve"], bridgetp_history_direct_to_kv=True)

    def test_cli_requires_the_chunked_sender(self):
        base = [
            "--model", "m", "--output", "unused", "--expected-revision", "CPU",
            "--cards", "6", "--static-layout", "L1AC", "--phase-shift",
            "--phase-scenario", "mixed_guard_six_v1", "--scheduler", "unified",
            "--hold-queue", "--kv-short-exempt", "--overload-migrate",
            "--migrate-current-length", "--current-machine", "--repetitions", "1",
            "--history-direct-to-kv", "--gpu-memory-log-s", "1", "--validate-only",
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(base + ["--expanded-context"]), 0)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(base)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main(base + ["--expanded-context", "--history-direct-ring-slots", "0"])


class FakeCuda:
    def __init__(self):
        self.allocated, self.peak, self.resets = 100, 300, 0

    def mem_get_info(self, device):
        return (5_000, 24_000)

    def memory_allocated(self, device):
        return self.allocated

    def memory_reserved(self, device):
        return 200

    def max_memory_allocated(self, device):
        return self.peak

    def max_memory_reserved(self, device):
        return 400

    def reset_peak_memory_stats(self, device):
        self.resets += 1
        self.peak = self.allocated


class MemoryLogTests(unittest.TestCase):
    def test_periodic_samples_reset_peaks_events_do_not(self):
        with tempfile.TemporaryDirectory() as folder:
            now = [100.0]
            cuda = FakeCuda()
            log = GpuMemoryLog(
                Path(folder) / "x" / "gpu_memory_A-g6-tp2_tp1.jsonl",
                interval_s=1.0,
                device="cuda:0",
                pool="A-g6-tp2",
                tp_rank=1,
                cuda=cuda,
                clock=lambda: now[0],
            )
            self.assertTrue(log.maybe_sample())
            self.assertFalse(log.maybe_sample())  # within the interval
            log.sample("direct_history_begin", history_bytes=123)
            now[0] += 1.5
            self.assertTrue(log.maybe_sample())
            rows = [
                json.loads(line)
                for line in log.path.read_text().splitlines()
            ]
            self.assertEqual(
                [r["event"] for r in rows],
                ["periodic", "direct_history_begin", "periodic"],
            )
            self.assertEqual(cuda.resets, 2)
            self.assertEqual(rows[0]["max_allocated_bytes"], 300)
            self.assertEqual(rows[1]["history_bytes"], 123)
            self.assertEqual(rows[2]["max_allocated_bytes"], 100)  # peak was reset
            self.assertEqual(rows[0]["device_free_bytes"], 5_000)
            self.assertEqual((rows[0]["pool"], rows[0]["tp_rank"]), ("A-g6-tp2", 1))
        with self.assertRaises(ValueError):
            GpuMemoryLog(Path("x"), interval_s=0, device=None, pool=None, tp_rank=None,
                         cuda=FakeCuda())


class ConnectorSampleTests(unittest.TestCase):
    def connector(self, interval):
        value = object.__new__(sc.BridgeTPStreamingConnector)
        value.gpu_memory_log_s = interval
        value._gpu_memory_log = None
        value._registered_kv_caches = {
            "layer.0": NS(device=torch.device("cuda", 0))
        }
        value.manifest_path = Path("unused") / "session_manifest.json"
        return value

    def test_hook_creates_the_rank_log_next_to_the_inventory(self):
        made = []

        class Log:
            def __init__(self, path, **kwargs):
                made.append((path, kwargs))
                self.calls = []

            def maybe_sample(self):
                self.calls.append("periodic")

            def sample(self, event, **extra):
                self.calls.append((event, extra))

        value = self.connector(1.0)
        with tempfile.TemporaryDirectory() as folder, patch.dict(
            os.environ,
            {
                "BRIDGETP_PRESSURE_POOL": "A-g6-tp2",
                "BRIDGETP_PRESSURE_INVENTORY_PATH": str(
                    Path(folder) / "inventory_A-g6-tp2.jsonl"
                ),
            },
        ), patch("vllm.bridge_tp.gpu_memory_log.GpuMemoryLog", Log), patch.object(
            sc, "get_tp_group", return_value=NS(rank_in_group=1)
        ):
            value._sample_gpu_memory()
            value._sample_gpu_memory("direct_history_done", chunks=3)
            path, kwargs = made[0]
            self.assertEqual(path, Path(folder) / "gpu_memory_A-g6-tp2_tp1.jsonl")
            self.assertEqual((kwargs["pool"], kwargs["tp_rank"]), ("A-g6-tp2", 1))
        self.assertEqual(len(made), 1)  # one log per rank, reused
        self.assertEqual(
            value._gpu_memory_log.calls,
            ["periodic", ("direct_history_done", {"chunks": 3})],
        )

    def test_off_cpu_or_broken_logging_never_fails_serving(self):
        value = self.connector(0)
        value._sample_gpu_memory()
        self.assertIsNone(value._gpu_memory_log)
        value = self.connector(1.0)
        value._registered_kv_caches = {"layer.0": NS(device=torch.device("cpu"))}
        value._sample_gpu_memory()
        self.assertIsNone(value._gpu_memory_log)
        value = self.connector(1.0)
        with patch.object(sc, "get_tp_group", side_effect=RuntimeError("no group")):
            value._sample_gpu_memory()
        self.assertEqual(value.gpu_memory_log_s, 0)  # disabled after a failure
        bare = object.__new__(sc.BridgeTPStreamingConnector)
        bare._sample_gpu_memory()  # old connectors: attribute absent


if __name__ == "__main__":
    unittest.main()
