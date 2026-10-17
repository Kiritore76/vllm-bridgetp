# SPDX-License-Identifier: Apache-2.0
"""Lossless compact pressure histories and transport retention contracts."""

import gc
import io
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tests.bridge_tp.test_pressure_ingress import DeferredExecutor, reading
from tests.bridge_tp.test_pressure_policy import config
from tests.bridge_tp.test_pressure_session import real_helper
from tools.bridge_tp.pressure_cpu import load

p, online, ingress, runtime = (
    load(name)
    for name in ("response_proxy", "online_io", "pressure_ingress", "pressure_runtime")
)


class CompactHistoryTests(unittest.TestCase):
    def recorder(self, compact, mode=None):
        return online.ProxyRecorder(
            "same", mode or p.ProxyMode.HOLD_BACK, compact_history=compact
        )

    def test_commit_and_early_target_race_preserve_full_evidence(self):
        outputs = []
        for compact in (False, True):
            emitted = []
            r = online.ProxyRecorder(
                "same",
                p.ProxyMode.HOLD_BACK,
                emission_sink=emitted.append,
                compact_history=compact,
            )
            r.on_source_token(0, 42, 10.0)
            r.on_source_token(1, 43, 10.1)
            r.set_cutover(2, 10.2)
            r.on_source_token(2, 44, 10.3)
            r.on_target_token(0, 44, 10.4)
            r.on_commit(10.5)
            r.on_target_token(1, 45, 10.6)
            r.on_source_token(3, 45, 10.7)
            outputs.append((r.stats(), emitted, r.progress()))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[1][0]["token_ids"], [42, 43, 44, 45])

    def test_rollback_and_cancel_retry_preserve_indices_and_original_times(self):
        outputs = []
        for compact in (False, True):
            r = self.recorder(compact)
            r.on_source_token(0, 42, 10)
            r.set_cutover(1, 10.1)
            r.on_source_token(1, 43, 10.2)
            r.on_source_token(2, 44, 10.3)
            r.on_rollback(10.5, "cancel")
            r.set_cutover(3, 10.6)
            r.on_commit(10.7)
            r.on_target_token(0, 45, 10.8)
            outputs.append(r.stats())
        self.assertEqual(*outputs)
        self.assertEqual(
            [t["unix_s"] for t in outputs[1]["emitted"]], [10, 10.2, 10.3, 10.8]
        )

    def test_greedy_overlap_and_divergence_checks_are_unchanged(self):
        outputs = []
        for compact in (False, True):
            r = self.recorder(compact, p.ProxyMode.GREEDY_FASTPATH)
            for n in range(4):
                r.on_source_token(n, n + 42, 10 + n / 10)
            r.set_cutover(2, 10.4)
            r.on_commit(10.5)
            r.on_target_token(0, 44, 10.6)
            with self.assertRaises(p.StreamViolation):
                r.on_target_token(1, 99, 10.7)
            r.on_target_token(1, 45, 10.8)
            r.on_target_token(2, 46, 10.9)
            outputs.append(r.stats())
        self.assertEqual(*outputs)

    def test_live_compact_history_does_not_retain_per_token_gc_objects(self):
        r = self.recorder(True)
        for n in range(20000):
            r.on_source_token(n, 2**31 + n, 10 + n / 100)
        history = r.proxy.emitted
        self.assertIsInstance(history, p.CompactTokens)
        self.assertEqual(len(history), 20000)
        self.assertEqual(
            history[-1], p.EmittedToken(19999, 2**31 + 19999, 209.99, "source")
        )
        self.assertEqual(history[1:3], [history[1], history[2]])
        for index in (-20001, 20000):
            with self.assertRaises(IndexError):
                history[index]
        # Arrays contain primitive values rather than retained token dataclasses.
        referents = gc.get_referents(history.ids, history.times, history.origins)
        self.assertFalse(any(isinstance(v, p.EmittedToken) for v in referents))
        self.assertLess(len(referents), 10)

    def test_sse_without_chunks_retains_ids_timing_and_errors(self):
        chunks = [
            dict(
                id="cmpl-test", choices=[dict(token_ids=[42, 43], finish_reason=None)]
            ),
            dict(id="cmpl-test", choices=[dict(token_ids=[], finish_reason="stop")]),
        ]
        body = (
            "".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
            + "data: [DONE]\n\n"
        ).encode()
        for retain in (True, False):
            tokens = []
            with patch.object(
                online.urllib.request, "urlopen", return_value=io.BytesIO(body)
            ):
                result = online.post_streaming_completion(
                    "http://unused",
                    dict(prompt=[1], max_tokens=8),
                    1,
                    lambda n, t, stamp, tokens=tokens: tokens.append((n, t)),
                    retain_chunks=retain,
                )
            self.assertEqual(tokens, [(0, 42), (1, 43)])
            self.assertEqual(result["token_ids"], [42, 43])
            self.assertEqual(result["finish_reason"], "stop")
            self.assertEqual(result["chunks"], chunks if retain else [])
            self.assertEqual(result["chunks_retained"], retain)
            self.assertLessEqual(
                result["request_started_unix_s"], result["completed_unix_s"]
            )
        with (
            patch.object(
                online.urllib.request, "urlopen", return_value=io.BytesIO(body[:-14])
            ),
            self.assertRaisesRegex(RuntimeError, "ended before completion"),
        ):
            online.post_streaming_completion(
                "http://unused", dict(prompt=[1]), 1, Mock(), retain_chunks=False
            )

    def test_pressure_source_and_target_disable_only_unused_chunk_retention(self):
        executor = DeferredExecutor()
        with patch.object(
            ingress, "post_streaming_completion", return_value=dict(completed_unix_s=11)
        ) as post:
            router = ingress.PressureIngress(
                runtime.PressureCoordinator(config(), lambda row: None),
                load("pressure_observation").NativeServiceWindow(),
                executor,
                "source",
                "target",
                clock=lambda: 10,
            )
            value = router.submit(
                dict(prompt=[1], max_tokens=8), (reading("tp1"), reading("tp4"))
            )
            executor.function()
            self.assertFalse(post.call_args.kwargs["retain_chunks"])
            self.assertTrue(value.recorder.compact_history)
        context = dict(
            post_streaming_completion=Mock(return_value={}),
            atomic_json_dump=Mock(),
            Path=Path,
            ProxyRecorder=online.ProxyRecorder,
        )
        helper = real_helper("recorded_completion", context)

        for compact in (False, True):
            recorder = self.recorder(compact)
            helper("unused", {}, 1, Mock(), Path("target_response.json"), recorder)
            self.assertEqual(
                context["post_streaming_completion"].call_args.kwargs["retain_chunks"],
                not compact,
            )
            self.assertEqual(
                context["atomic_json_dump"].call_args.args[0], recorder.stats()
            )


if __name__ == "__main__":
    unittest.main()
