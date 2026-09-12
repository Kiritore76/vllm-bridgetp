# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from vllm.bridge_tp.experiment_timeline import emit_event, merge_parts
from vllm.bridge_tp.request_freeze import RequestFreezeGate, request_freeze


class TestRequestFreezeGate(unittest.TestCase):
    def test_freeze_is_request_scoped_and_records_actual_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            request = SimpleNamespace(
                request_id="anchor-1",
                num_prompt_tokens=2048,
                output_token_ids=[1] * 64,
                num_computed_tokens=2111,
            )
            request_freeze(
                root,
                request.request_id,
                output_tokens=64,
                num_computed_tokens=2111,
            )
            gate = RequestFreezeGate(root)
            self.assertTrue(gate.is_frozen("anchor-1"))
            self.assertFalse(gate.is_frozen("peer-1"))
            gate.record_frozen(request, scheduler_step=77)
            gate.record_released(request)
            frozen = json.loads(
                (root / "request_frozen_receipt.json").read_text(encoding="utf-8")
            )
            released = json.loads(
                (root / "source_kv_release_receipt.json").read_text(encoding="utf-8")
            )
            self.assertEqual(frozen["status"], "FROZEN")
            self.assertEqual(frozen["num_output_tokens"], 64)
            self.assertEqual(released["status"], "SOURCE_KV_RELEASED")

    def test_from_env_is_dormant_by_default(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(RequestFreezeGate.from_env())


class TestExperimentTimeline(unittest.TestCase):
    def test_merge_process_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            emit_event(directory, "client", "REQUEST_SENT", request_id="r")
            emit_event(directory, "client", "REQUEST_COMPLETE", request_id="r")
            result = merge_parts(directory)
            self.assertEqual(result["status"], "PASS")
            self.assertEqual(result["events"], 2)
            rows = (
                (Path(directory) / "timeline.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
