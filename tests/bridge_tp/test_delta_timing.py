# SPDX-License-Identifier: Apache-2.0
"""Ensure diagnostic observers preserve exact KV restoration and never wait."""

import unittest
from unittest.mock import MagicMock, patch

import torch

from vllm.bridge_tp.delta_timing import DeltaTiming, delta_timing
from vllm.bridge_tp.kv_restore import inject_rank_delta


class TestDeltaTiming(unittest.TestCase):
    def test_cpu_observer_preserves_strided_restore_and_reports_stages(self):
        plain = torch.full((2, 12, 16, 2, 5), -9.0)
        observed = plain.clone()
        delta = {"layer": torch.randn(6, 2, 2, 5)}
        kwargs = dict(start_token=13, end_token=19, block_axis=1, block_size=16)
        inject_rank_delta({"layer": plain}, delta, [7, 2, 10], **kwargs)
        trace = DeltaTiming(torch.device("cpu"))
        inject_rank_delta(
            {"layer": observed}, delta, [7, 2, 10], timing_hook=trace.mark, **kwargs
        )
        trace.mark("FINAL_RESULT_RETURNED", cuda=False)
        self.assertTrue(torch.equal(plain, observed))
        report = trace.result()
        self.assertIn("SCATTER", report["cpu_stage_ms"])
        self.assertIn("FINAL_RESULT_WAIT", report["cpu_stage_ms"])
        self.assertEqual(report["cuda_stream_stage_ms"], {})
        self.assertFalse(report["extra_cuda_synchronize"])

    def test_disabled_observer_and_cuda_events_do_not_synchronize(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(delta_timing(torch.device("cuda:0")))
        event = MagicMock()
        event.query.return_value = False
        with patch("torch.cuda.Event", return_value=event), patch(
            "torch.cuda.current_stream", return_value=object()
        ), patch("torch.cuda.synchronize") as synchronize:
            trace = DeltaTiming(torch.device("cuda:0"))
            trace.mark("SCATTER")
            trace.mark("READBACK")
            report = trace.result()
        self.assertEqual(report["incomplete_cuda_intervals"], 1)
        event.synchronize.assert_not_called()
        event.wait.assert_not_called()
        synchronize.assert_not_called()

    def test_diagnostic_hook_does_not_bypass_readback_failure(self):
        trace = DeltaTiming(torch.device("cpu"))
        with (
            patch("torch.count_nonzero", return_value=torch.tensor(1)),
            self.assertRaisesRegex(ValueError, "readback differs"),
        ):
            inject_rank_delta(
                {"layer": torch.zeros(2, 12, 16, 2, 5)},
                {"layer": torch.ones(6, 2, 2, 5)},
                [7, 2, 10], start_token=13, end_token=19,
                block_axis=1, block_size=16, timing_hook=trace.mark,
            )
