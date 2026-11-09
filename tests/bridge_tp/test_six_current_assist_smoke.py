# SPDX-License-Identifier: Apache-2.0
"""A published route without business requests is insufficient coverage."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp import six_current_assist_smoke as smoke


class RoutingCoverageTests(unittest.TestCase):
    def test_publication_and_warmup_do_not_count_as_business(self):
        steps = [dict(plan_id="p", plan=dict(state="COMPLETED"))]
        events = [
            dict(kind="ROUTE_PUBLISHED", plan_id="p", unix_s=10, instances=["new"])
        ]
        self.assertFalse(smoke.joined_routing_coverage(steps, events, [])["passed"])
        row = dict(
            initial_pool="new",
            source_done=True,
            source_error=None,
            source_transport=dict(request_started_unix_s=11),
        )
        self.assertTrue(smoke.joined_routing_coverage(steps, events, [row])["passed"])
        row["source_transport"]["request_started_unix_s"] = 9
        self.assertFalse(smoke.joined_routing_coverage(steps, events, [row])["passed"])

    def test_missing_scene_preserves_technical_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            self.assertEqual(smoke.check(Path(folder) / "trial"), 1)
            d = json.loads(
                (Path(folder) / "current_assist_smoke_check.json").read_text()
            )
            self.assertFalse(d["passed"])

    def test_no_assist_does_not_fake_coverage_or_prevent_routing_check(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "trial"
            scene = root / "scene"
            (scene / "controller").mkdir(parents=True)
            sequence = dict(steps=[dict(plan_id="p", plan=dict(state="COMPLETED"))])
            (scene / "conversion_sequence_result.json").write_text(json.dumps(sequence))
            (scene / "conversion_events.jsonl").write_text(
                json.dumps(
                    dict(
                        kind="ROUTE_PUBLISHED",
                        plan_id="p",
                        unix_s=10,
                        instances=["new"],
                    )
                )
                + "\n"
            )
            (scene / "controller/pressure_audit.jsonl").write_text("")
            rows = [
                dict(
                    initial_pool="new",
                    source_done=True,
                    source_error=None,
                    source_transport=dict(request_started_unix_s=11),
                )
            ]
            path = scene / "controller/pressure_request_results.json"
            path.write_text(json.dumps(rows))
            (root.parent / "smoke_check.json").write_text(json.dumps(dict(passed=True)))
            with patch.object(smoke.six_auto_smoke, "check", return_value=0):
                self.assertEqual(smoke.check(root), 0)
                d = json.loads(
                    (root.parent / "current_assist_smoke_check.json").read_text()
                )
                self.assertFalse(d["assist_start_coverage"])
                path.write_text("[]")
                self.assertEqual(smoke.check(root), 2)


if __name__ == "__main__":
    unittest.main()
