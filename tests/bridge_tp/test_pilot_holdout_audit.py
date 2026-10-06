# SPDX-License-Identifier: Apache-2.0
"""Ensure a split label alone cannot hide duplicate pilot content."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.audit_pilot_holdout import audit


class TestPilotHoldoutAudit(unittest.TestCase):
    def test_rejects_duplicate_messages_across_train_and_test(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "requests.jsonl"
            common = [{"role": "user", "content": "same prompt"}]
            training = {
                "id": "train", "split": "train", "source_tree_id": "tree-a",
                "source_message_id": "message-a", "messages": common,
            }
            heldout = {
                "id": "test", "split": "test", "source_tree_id": "tree-b",
                "source_message_id": "message-b", "messages": common,
                "workload_group": "natural",
            }
            path.write_text(
                "\n".join(json.dumps(row) for row in (training, heldout))
                + "\n", encoding="utf-8",
            )
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with patch(
                "tools.bridge_tp.audit_pilot_holdout.select_inputs",
                return_value=[heldout],
            ):
                with self.assertRaisesRegex(ValueError, "message content"):
                    audit(path, digest, [1])


if __name__ == "__main__":
    unittest.main()
