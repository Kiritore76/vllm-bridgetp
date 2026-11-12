# SPDX-License-Identifier: Apache-2.0
"""Direct-to-KV deferred history must open the dormant target's admission gate.

11-12 directkv run 3: prebind wrote INITIAL_HISTORY_DEFERRED_TO_ADMISSION, the
controller's admission gate accepted only buffered/resident receipts, so the
target was never admitted and the receiver never received (cancelled 115 s
later). These tests feed the receipt the connector actually writes.
"""

import ast
import json
import tempfile
import unittest
from pathlib import Path

from vllm.bridge_tp.controller.action_adapter import ActionAdapter

CONNECTOR = (
    Path(__file__).resolve().parents[2] / "vllm/bridge_tp/streaming_connector.py"
)


def connector_deferred_receipt():
    """The constant fields of the deferred receipt in the connector source."""
    tree = ast.parse(CONNECTOR.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        fields = {
            k.value: v.value
            for k, v in zip(node.keys, node.values)
            if isinstance(k, ast.Constant) and isinstance(v, ast.Constant)
        }
        if fields.get("status") == "INITIAL_HISTORY_DEFERRED_TO_ADMISSION":
            return fields
    raise AssertionError("connector writes no deferred receipt")


def session(folder, tp=2, migration="m"):
    run = Path(folder)
    (run / "session_manifest.json").write_text(
        json.dumps(
            dict(
                migration_id=migration,
                session_token="s",
                source_request_id="r",
                target_tp_size=tp,
            )
        )
    )
    (run / "gpu_initial_receipts").mkdir()
    return run


def write(run, rank, receipt):
    (run / "gpu_initial_receipts" / f"tp_rank_{rank}.json").write_text(
        json.dumps(receipt)
    )


class GateTests(unittest.TestCase):
    def setUp(self):
        self.deferred = dict(connector_deferred_receipt(), migration_id="m")

    def test_the_connector_receipt_admits_the_dormant_target(self):
        self.assertEqual(self.deferred["storage"], "TARGET_KV_BLOCKS")
        with tempfile.TemporaryDirectory() as folder:
            run = session(folder)
            write(run, 0, self.deferred)
            adapter = ActionAdapter("http://source", run)
            buffered, ranks, _ = adapter.poll_initial_history_gpu_buffered()
            self.assertFalse(buffered)  # one of two ranks bound
            write(run, 1, self.deferred)
            buffered, ranks, detail = adapter.poll_initial_history_gpu_buffered()
            self.assertTrue(buffered, detail)
            self.assertEqual(ranks, {0, 1})
            # Admission only: the source cutover still needs resident history.
            ready, _, _ = adapter.poll_initial_history_gpu_ready()
            self.assertFalse(ready)

    def test_resident_direct_receipts_then_pass_the_final_gate(self):
        with tempfile.TemporaryDirectory() as folder:
            run = session(folder)
            for rank in (0, 1):
                write(run, rank, dict(
                    migration_id="m", status="INITIAL_HISTORY_GPU_RESIDENT",
                    exact_readback=True, end_token=100, history_path="DIRECT_TO_KV",
                ))
            adapter = ActionAdapter("http://source", run)
            self.assertTrue(adapter.poll_initial_history_gpu_buffered()[0])
            self.assertTrue(adapter.poll_initial_history_gpu_ready()[0])

    def test_mixed_wrong_storage_or_wrong_migration_never_admit(self):
        cases = {
            "mixed": [self.deferred, dict(migration_id="m",
                                          status="INITIAL_HISTORY_GPU_BUFFERED",
                                          gpu_ready=True)],
            "storage": [dict(self.deferred, storage="NONE")] * 2,
        }
        for name, receipts in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as folder:
                run = session(folder)
                for rank, receipt in enumerate(receipts):
                    write(run, rank, receipt)
                buffered, _, _ = ActionAdapter(
                    "http://source", run
                ).poll_initial_history_gpu_buffered()
                self.assertFalse(buffered)
        with tempfile.TemporaryDirectory() as folder:
            run = session(folder)
            for rank in (0, 1):
                write(run, rank, dict(self.deferred, migration_id="other"))
            buffered, _, detail = ActionAdapter(
                "http://source", run
            ).poll_initial_history_gpu_buffered()
            self.assertFalse(buffered)
            self.assertIn("migration ID differs", detail)


if __name__ == "__main__":
    unittest.main()
