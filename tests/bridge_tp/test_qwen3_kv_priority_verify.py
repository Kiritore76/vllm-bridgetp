# SPDX-License-Identifier: Apache-2.0
"""Single fixed-work rerun and explicit mechanism versus performance acceptance."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.bridge_tp import qwen3_kv_priority_verify as verify
from tools.bridge_tp import qwen3_pressure_four_mode as pressure


def evidence():
    raw = [dict(server_id="s", response=dict(committed=True, handoff_stall_s=0.5))]
    audit = [
        dict(
            kind="pressure_decision",
            execution="RESERVED",
            unix_s=4.0,
            decision=dict(
                capacity_path=True,
                request_id="s",
                candidates=[
                    dict(
                        selected=True,
                        kv_priority_override=True,
                        soft_warnings=["emergency target service budget unavailable"],
                    )
                ],
            ),
        ),
        dict(
            kind="pressure_actual_start", request_id="s", migration_id="m", unix_s=5.0
        ),
        dict(kind="pressure_source_release_confirmed", migration_id="m"),
    ]
    return raw, audit


class KVPriorityVerifyTests(unittest.TestCase):
    def test_only_one_migrate_scene_same_prefill_jobs_and_budget(self):
        inputs = dict(
            requests={str(n): dict(prompt=list(range(n))) for n in (2048, 6144)}
        )
        original = pressure.config(inputs)["recipes"][-1]
        with tempfile.TemporaryDirectory() as tmp:
            prior = Path(tmp)
            prior.joinpath("result.json").write_text(
                json.dumps(
                    dict(
                        runs=[
                            dict(
                                name="pressure-four-prefill_competition-r1-migrate",
                                status="COMPLETED",
                                input_sha256=original["input_sha256"],
                                summary=dict(migration_starts=0),
                            )
                        ]
                    )
                )
            )
            cfg, baseline = verify.config(inputs, prior)
        self.assertEqual(cfg["arms"], ["migrate"])
        self.assertEqual(len(cfg["recipes"]), 1)
        self.assertEqual(cfg["recipes"][0]["jobs"], original["jobs"])
        self.assertEqual(len(cfg["recipes"][0]["jobs"]), 375)
        self.assertEqual(
            cfg["policy_overrides"],
            dict(capacity_guard_blocks=125, kv_emergency_priority=True),
        )
        self.assertEqual(baseline["input_sha256"], original["input_sha256"])

    def test_actual_bypass_commit_free_and_one_second_handoff_are_required(self):
        raw, audit = evidence()
        self.assertTrue(verify.assess(raw, audit)["verification_pass"])
        self.assertFalse(verify.assess(raw, audit[:-1])["verification_pass"])
        bad = copy.deepcopy(raw)
        bad[0]["response"]["handoff_stall_s"] = 1.001
        self.assertTrue(verify.assess(bad, audit)["mechanism_pass"])
        self.assertFalse(verify.assess(bad, audit)["verification_pass"])
        no_bypass = copy.deepcopy(audit)
        no_bypass[0]["decision"]["candidates"][0]["soft_warnings"] = []
        self.assertFalse(verify.assess(raw, no_bypass)["mechanism_pass"])
        self.assertFalse(verify.assess(raw, [])["verification_pass"])

    def test_ordinary_or_legacy_start_cannot_count_as_fix_coverage(self):
        for key in ("capacity_path", "kv_priority_override"):
            raw, audit = evidence()
            if key == "capacity_path":
                audit[0]["decision"][key] = False
            else:
                audit[0]["decision"]["candidates"][0][key] = False
            with self.assertRaises(ValueError):
                verify.assess(raw, audit)
