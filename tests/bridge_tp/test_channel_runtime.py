# SPDX-License-Identifier: Apache-2.0
"""Regression coverage for real preconnect publication and worker cleanup."""

import ast
import json
import os
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from vllm.bridge_tp.channel_runtime import (
    ManifestPollCache,
    publish_prebound_receiver,
    receiver_snapshot,
)


def methods(*names):
    source = (
        Path(__file__).resolve().parents[2] / "vllm/bridge_tp/streaming_connector.py"
    )
    cls = next(
        n
        for n in ast.parse(source.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "BridgeTPStreamingConnector"
    )
    body = [ast.parse("from __future__ import annotations").body[0]]
    body += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(
        publish_prebound_receiver=publish_prebound_receiver,
        get_tp_group=lambda: NS(rank_in_group=0),
        _atomic_json_dump=Mock(),
        time=time,
        logger=Mock(),
    )
    exec(
        compile(ast.Module(body=body, type_ignores=[]), str(source), "exec"), namespace
    )
    return namespace


class RuntimeTests(unittest.TestCase):
    def test_runtime_probe_is_one_stage_without_retirement(self):
        from tools.bridge_tp import run_qwen3_online_loaded_validation as runner

        stages = runner.stages(True, False, True)
        self.assertEqual(len(stages), 1)
        self.assertEqual(stages[0]["background"], 4)
        args = NS(
            model=Path("m"),
            inputs=Path("i"),
            output=Path("o"),
            checkpoint=Path("c"),
            expected_revision="r",
            diagnostic_probe=True,
            runtime_fix_probe=True,
            channel_ablation=False,
        )
        argv = runner.child_arguments(args, stages[0])
        self.assertIn("--diagnostic-probe", argv)
        self.assertNotIn("--channel-ablation", argv)
        self.assertNotIn("--natural-probe", argv)

    def test_runtime_receipts_require_both_ranks_and_registered_idle_owner(self):
        from tools.bridge_tp import run_qwen3_online_loaded_validation as runner

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            c = d / "mechanism/controller/persistent_payload_cleanup"
            c.mkdir(parents=True)
            for rank in range(2):
                receipt = dict(
                    status="PERSISTENT_SESSION_PAYLOAD_RELEASED",
                    tp_rank=rank,
                    channel_runtime=dict(
                        same_receiver=True,
                        lifecycle_state="IDLE",
                        active_session=False,
                        communicator_present=True,
                        live_load_threads=0,
                        manifest_polling=dict(polls=200, parses=1),
                    ),
                )
                (c / f"tp_rank_{rank}.json").write_text(json.dumps(receipt))
            self.assertEqual(runner.runtime_proof(d)["status"], "COMPLETED")
            receipt["channel_runtime"]["same_receiver"] = False
            (c / "tp_rank_1.json").write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                runner.runtime_proof(d)

    def test_real_preconnect_registers_both_owners_before_open(self):
        c = NS(
            persistent_channel=True,
            preconnect_gpu_direct=True,
            preconnect_gpu_direct_host="127.0.0.1",
            preconnect_gpu_direct_base_port=1,
            channel_generation=1,
            _prebind_receiver_thread=None,
            _registered_kv_caches={"layer": NS(device="cuda")},
            _prebound_gpu_receiver=None,
            _persistent_gpu_receiver=None,
            _prebound_gpu_receiver_lock=threading.Lock(),
            _persistent_gpu_receiver_lock=threading.Lock(),
            _prebind_receiver_ready=threading.Event(),
            _prebind_receiver_stop=threading.Event(),
            manifest_path=Path("unused/session.json"),
        )
        receiver = NS(
            listener=Mock(),
            connection=Mock(),
            channel_generation=1,
            warmup_completed_unix_s=1,
            close=Mock(),
        )

        def opened(**kwargs):
            self.assertIs(c._persistent_gpu_receiver, receiver)
            self.assertIs(c._prebound_gpu_receiver, receiver)

        receiver._open = opened

        def warm(**kwargs):
            c._prebind_receiver_stop.set()
            return 16

        receiver.warmup_preconnected_channel = warm
        fake = types.ModuleType("vllm.bridge_tp.gpu_direct_history")
        fake.GpuDirectHistoryReceiver = Mock(return_value=receiver)
        namespace = methods("_start_gpu_direct_receiver_prebind")
        namespace["threading"] = NS(Thread=lambda **kw: NS(start=kw["target"]))
        with patch.dict(sys.modules, {fake.__name__: fake}):
            namespace["_start_gpu_direct_receiver_prebind"](c)
        receiver.close.assert_not_called()

    def test_publication_refuses_independent_owner_without_mutation(self):
        old, new = object(), object()
        c = NS(
            persistent_channel=True,
            _prebound_gpu_receiver=None,
            _persistent_gpu_receiver=old,
            _prebound_gpu_receiver_lock=threading.Lock(),
            _persistent_gpu_receiver_lock=threading.Lock(),
        )
        with self.assertRaises(RuntimeError):
            publish_prebound_receiver(c, new)
        self.assertIsNone(c._prebound_gpu_receiver)
        self.assertIs(c._persistent_gpu_receiver, old)

    def test_idle_manifest_is_parsed_once_and_atomic_replacement_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text('{"migration_id":"one"}')
            load = Mock(side_effect=lambda p: json.loads(p.read_text()))
            cache = ManifestPollCache()
            for _ in range(200):
                self.assertEqual(cache.load(path, load)["migration_id"], "one")
            self.assertEqual(load.call_count, 1)
            replacement = path.with_suffix(".tmp")
            replacement.write_text('{"migration_id":"two"}')
            os.replace(replacement, path)
            self.assertEqual(cache.load(path, load)["migration_id"], "two")
            self.assertEqual(load.call_count, 2)
            self.assertEqual(cache.snapshot()["polls"], 201)

    def test_failed_read_is_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            path.write_text("bad")
            cache = ManifestPollCache()
            with self.assertRaises(ValueError):
                cache.load(path, lambda p: json.loads(p.read_text()))
            self.assertIsNone(cache.signature)
            self.assertEqual(
                cache.load(path, lambda p: {"retry": True}), {"retry": True}
            )

    def test_worker_completion_defers_live_loader_then_cleans_only_its_request(self):
        thread = Mock()
        thread.is_alive.return_value = True
        c = NS(
            _load_errors={},
            _load_threads={"migrated": thread},
            _completed_recvs={"migrated"},
            _reported_recvs=set(),
            _finished_worker_requests=set(),
            _release_persistent_payload_buffer=Mock(),
            _gpu_kv_lock=threading.Lock(),
            _gpu_restore_quiescence_lock=threading.Lock(),
            _copy_done_events={"migrated": 1, "other": 2},
            _model_wait_streams={("migrated", 0), ("other", 0)},
            _model_wait_pending_requests={"migrated", "other"},
            _gpu_restore_quiescence={"migrated": 1, "other": 2},
        )
        call = methods("get_finished")["get_finished"]
        self.assertEqual(call(c, {"migrated", "ordinary"}), (None, {"migrated"}))
        c._release_persistent_payload_buffer.assert_not_called()
        thread.is_alive.return_value = False
        self.assertEqual(call(c, set()), (None, None))
        c._release_persistent_payload_buffer.assert_called_once_with("migrated")
        self.assertEqual(c._copy_done_events, {"other": 2})
        self.assertEqual(c._gpu_restore_quiescence, {"other": 2})
        self.assertFalse(c._load_threads)
        self.assertFalse(c._finished_worker_requests)

    def test_old_completion_cannot_release_new_session_payload(self):
        c = NS(
            persistent_channel=True,
            release_persistent_payload_buffer=True,
            _claimed_target_request_id="new",
        )
        methods("_release_persistent_payload_buffer")[
            "_release_persistent_payload_buffer"
        ](c, "old")

    def test_worker_releases_its_idle_payload_and_writes_runtime_receipt(self):
        import contextlib

        receiver = NS(
            lifecycle=NS(state=NS(value="IDLE"), active_session=None),
            device="cuda",
            comm=object(),
            release_session_payload_buffer=Mock(return_value=303169536),
        )
        c = NS(
            persistent_channel=True,
            release_persistent_payload_buffer=True,
            _claimed_target_request_id="migrated",
            _persistent_gpu_receiver=receiver,
            _prebound_gpu_receiver=receiver,
            _load_threads={},
            _persistent_gpu_receiver_lock=threading.Lock(),
            manifest_path=Path("unused/session.json"),
        )
        namespace = methods("_release_persistent_payload_buffer")
        namespace.update(
            receiver_snapshot=receiver_snapshot,
            torch=NS(
                cuda=NS(
                    device=lambda d: contextlib.nullcontext(),
                    memory_allocated=lambda d: 1,
                    memory_reserved=lambda d: 2,
                )
            ),
        )
        namespace["_release_persistent_payload_buffer"](c, "migrated")
        receiver.release_session_payload_buffer.assert_called_once()
        receipt = namespace["_atomic_json_dump"].call_args.args[0]
        self.assertEqual(receipt["released_payload_bytes"], 303169536)
        self.assertTrue(receipt["channel_runtime"]["same_receiver"])

    def test_worker_guard_refusal_returns_evidence_without_rpc_exception(self):
        from vllm.bridge_tp import channel_diagnostics as diag

        group = NS(device_communicator=object())
        parallel = types.ModuleType("vllm.distributed.parallel_state")
        parallel.get_tp_group = lambda: group
        kv = types.ModuleType("vllm.distributed.kv_transfer")
        c = NS(
            persistent_channel=True,
            _retained_gpu_receivers={},
            _load_threads={},
            _persistent_gpu_receiver=None,
            _prebound_gpu_receiver=object(),
        )
        kv.get_kv_transfer_group = lambda: c
        with (
            patch.dict(sys.modules, {parallel.__name__: parallel, kv.__name__: kv}),
            patch.dict(
                os.environ,
                BRIDGETP_CHANNEL_DIAGNOSTIC_ROLE="target",
                BRIDGETP_CHANNEL_DIAGNOSTIC_TOKEN="test",
            ),
        ):
            result = diag.worker_retire(NS(model_runner=NS(model=object())))
        self.assertEqual(result["status"], "REJECTED")
        self.assertIn("receiver is absent", result["error"])
        self.assertTrue(result["channel_runtime"]["prebound_receiver_present"])

    def test_snapshot_distinguishes_absent_receiver_from_active_channel(self):
        c = NS(
            _persistent_gpu_receiver=None,
            _prebound_gpu_receiver=object(),
            _load_threads={},
        )
        snap = receiver_snapshot(c)
        self.assertFalse(snap["persistent_receiver_present"])
        self.assertTrue(snap["prebound_receiver_present"])
        self.assertIsNone(snap["lifecycle_state"])
