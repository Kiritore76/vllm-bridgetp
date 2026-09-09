# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
protocol_path = REPO_ROOT / "vllm" / "bridge_tp" / "shadow_transfer_protocol.py"
protocol_spec = importlib.util.spec_from_file_location(
    "_test_shadow_transfer_protocol", protocol_path
)
if protocol_spec is None or protocol_spec.loader is None:
    raise RuntimeError(f"cannot load protocol module from {protocol_path}")
protocol_module = importlib.util.module_from_spec(protocol_spec)
sys.modules[protocol_spec.name] = protocol_module
protocol_spec.loader.exec_module(protocol_module)
build_shadow_transfer_plan = protocol_module.build_shadow_transfer_plan
group_units_by_step = protocol_module.group_units_by_step
reverse_history_blocks = protocol_module.reverse_history_blocks


class TestShadowTransferProtocol(unittest.TestCase):
    def test_history_moves_backward_from_boundary(self) -> None:
        self.assertEqual(
            reverse_history_blocks(64, 16),
            [(48, 64), (32, 48), (16, 32), (0, 16)],
        )

    def test_new_only_keeps_full_history_backlog(self) -> None:
        plan = build_shadow_transfer_plan(
            strategy="S_NEW",
            outcome="COMMIT",
            history_tokens=64,
            shadow_steps=2,
        )
        self.assertEqual(plan.history_tokens_copied_in_shadow, 0)
        self.assertEqual(plan.bridge_entry_history_backlog_tokens, 64)
        self.assertEqual(plan.bridge_steps, 4)
        self.assertTrue(all(unit.kind == "NEW" for unit in plan.shadow_units))

    def test_new_old_prioritizes_new_then_history(self) -> None:
        plan = build_shadow_transfer_plan(
            strategy="S_NEW_OLD",
            outcome="COMMIT",
            history_tokens=64,
            shadow_steps=2,
        )
        groups = group_units_by_step(plan.shadow_units)
        self.assertEqual(
            [[unit.kind for unit in group] for group in groups],
            [["NEW", "HISTORY"], ["NEW", "HISTORY"]],
        )
        history = [unit for unit in plan.shadow_units if unit.kind == "HISTORY"]
        self.assertEqual(
            [(unit.token_start, unit.token_end) for unit in history],
            [(48, 64), (32, 48)],
        )
        self.assertEqual(plan.history_tokens_copied_in_shadow, 32)
        self.assertEqual(plan.bridge_entry_history_backlog_tokens, 32)

    def test_cancel_never_enters_bridge(self) -> None:
        plan = build_shadow_transfer_plan(
            strategy="S_NEW_OLD",
            outcome="CANCEL",
            history_tokens=64,
            shadow_steps=3,
        )
        self.assertEqual(plan.bridge_units, ())
        self.assertEqual(plan.bridge_steps, 0)

    def test_fully_prefilled_history_has_zero_length_bridge(self) -> None:
        plan = build_shadow_transfer_plan(
            strategy="S_NEW_OLD",
            outcome="COMMIT",
            history_tokens=32,
            shadow_steps=2,
        )
        self.assertEqual(plan.bridge_entry_history_backlog_tokens, 0)
        self.assertEqual(plan.bridge_units, ())

    def test_rejects_non_block_aligned_boundary(self) -> None:
        with self.assertRaisesRegex(ValueError, "align"):
            build_shadow_transfer_plan(
                strategy="S_NEW",
                outcome="COMMIT",
                history_tokens=63,
                shadow_steps=2,
            )


if __name__ == "__main__":
    unittest.main()
