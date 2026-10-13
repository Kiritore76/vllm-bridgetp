# SPDX-License-Identifier: Apache-2.0
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from vllm.bridge_tp.controller.action_adapter import ActionAdapter, ActionError


class SourceCleanupOrderTests(unittest.TestCase):
    def setup_adapter(self, root):
        for name, data in {
            'session_manifest.json': {'migration_id': 'm', 'session_token': 's',
                                      'source_request_id': 'r'},
            'rolling_reservation.json': {},
            'cleanup_request.json': {'migration_id': 'm', 'source_request_id': 'r',
                                     'abort_source': False, 'reason': 'cancel',
                                     'requested_unix_s': 10.0},
        }.items():
            (root / name).write_text(json.dumps(data), encoding='utf-8')
        return ActionAdapter('http://source', root)

    def receipt(self, root, **changes):
        row = {'status': 'CLEANED', 'component': 'source_delta_mirror',
               'migration_id': 'm', 'reason': 'cancel', 'updated_unix_s': 11.0}
        row.update(changes)
        (root / 'source_cleanup_receipt.json').write_text(
            json.dumps(row), encoding='utf-8')

    def test_cancel_waits_for_delayed_drain_before_returning(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = self.setup_adapter(root)
            calls = []

            def drained(_seconds):
                calls.append('drained')
                self.receipt(root)

            with patch('vllm.bridge_tp.controller.action_adapter._post',
                       return_value={'state': 'CANCELLED'}) as post, patch(
                           'vllm.bridge_tp.controller.action_adapter.time.sleep',
                           side_effect=drained):
                self.assertEqual(adapter.cancel('cancel', abort_source=False),
                                 {'state': 'CANCELLED'})
                self.assertEqual(post.call_count, 1)
                self.assertEqual(calls, ['drained'])

    def test_wrong_or_stale_receipt_cannot_release_target(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = self.setup_adapter(root)
            for changes in [{'migration_id': 'other'}, {'updated_unix_s': 9.0},
                            {'status': 'FAILED'}, {'reason': 'other'}]:
                self.receipt(root, **changes)
                with self.assertRaisesRegex(ActionError, 'receiver retained'):
                    adapter.wait_for_source_cleanup(timeout_s=0)

    def test_successful_matching_receipt_and_legacy_bypass(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter = self.setup_adapter(root)
            self.receipt(root)
            self.assertEqual(adapter.wait_for_source_cleanup(0)['status'],
                             'CLEANED')
            (root / 'rolling_reservation.json').unlink()
            with patch('vllm.bridge_tp.controller.action_adapter._post',
                       return_value={'state': 'CANCELLED'}), patch.object(
                           adapter, 'wait_for_source_cleanup') as wait:
                adapter.cancel('cancel', abort_source=False)
                wait.assert_not_called()