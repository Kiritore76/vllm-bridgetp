# SPDX-License-Identifier: Apache-2.0
"""Phase 9 online plumbing tests that do not require a GPU."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

from tools.bridge_tp.run_phase9_controller import (
    _prepare_source_request,
    step_handoff,
    step_local,
)
from vllm.bridge_tp.controller.action_adapter import ActionAdapter
from vllm.bridge_tp.controller.config import ControllerConfig
from vllm.bridge_tp.controller.events import (
    Action,
    MigrationState,
    SourceRequestView,
)
from vllm.bridge_tp.controller.online_io import (
    ProxyRecorder,
    build_target_request,
    honored_generation,
)
from vllm.bridge_tp.controller.policy import InterferenceModel, TpotModel
from vllm.bridge_tp.controller.response_proxy import ProxyMode
from vllm.bridge_tp.controller.sampling_contract import (
    STRICT_GREEDY_SAMPLING_CONTRACT,
    freeze_strict_greedy_sampling,
    strict_greedy_sampling_errors,
)
from vllm.bridge_tp.controller.state_machine import MigrationStateMachine
from vllm.bridge_tp.controller.token_equivalence import (
    classify_token_equivalence,
    first_divergence,
)


class TestProxyRecorder(unittest.TestCase):
    def test_holdback_buffers_target_acknowledgement_race(self) -> None:
        externally_visible: list[dict] = []
        recorder = ProxyRecorder(
            "ext",
            ProxyMode.HOLD_BACK,
            emission_sink=externally_visible.append,
        )
        for index in range(5):
            recorder.on_source_token(index, 100 + index, index * 0.01)
        recorder.set_cutover(5, 0.05)
        recorder.on_source_token(5, 105, 0.06)
        recorder.on_target_token(0, 105, 0.07)
        recorder.on_commit(0.08)
        recorder.on_target_token(1, 106, 0.09)

        stats = recorder.stats()
        self.assertEqual(stats["token_ids"], [100, 101, 102, 103, 104, 105, 106])
        self.assertEqual(stats["source_origin_tokens"], 5)
        self.assertEqual(stats["target_origin_tokens"], 2)
        self.assertEqual(stats["discarded_source_tokens"], 1)
        self.assertEqual(
            [row["token_id"] for row in externally_visible],
            stats["token_ids"],
        )

    def test_fastpath_verifies_buffered_overlap(self) -> None:
        recorder = ProxyRecorder("ext", ProxyMode.GREEDY_FASTPATH)
        for index in range(7):
            recorder.on_source_token(index, 200 + index, index * 0.01)
        recorder.set_cutover(5, 0.05)
        recorder.on_target_token(0, 205, 0.07)
        recorder.on_target_token(1, 206, 0.08)
        recorder.on_commit(0.09)
        recorder.on_target_token(2, 207, 0.10)

        stats = recorder.stats()
        self.assertEqual(stats["token_ids"], list(range(200, 208)))
        self.assertEqual(stats["verified_overlap_tokens"], 2)
        self.assertEqual(stats["target_origin_tokens"], 1)


class TestOnlineArtifacts(unittest.TestCase):
    def test_target_request_uses_staging_boundary(self) -> None:
        source = freeze_strict_greedy_sampling(
            {"model": "m", "max_tokens": 100, "logprobs": 20}
        )
        staging = {
            "snapshot_num_output_tokens": 40,
            "all_known_token_ids": [1, 2, 3],
            "migration_id": "migration",
        }
        request, cutover = build_target_request(source, staging, "run")
        self.assertEqual(cutover, 40)
        self.assertEqual(request["max_tokens"], 60)
        self.assertEqual(request["logprobs"], 20)
        self.assertEqual(
            request["kv_transfer_params"]["bridgetp_migration_id"],
            "migration",
        )
        self.assertFalse(strict_greedy_sampling_errors(request))

    def test_source_request_freezes_model_sampling_defaults(self) -> None:
        source = {
            "model": "m",
            "prompt": "p",
            "max_tokens": 100,
            "temperature": 0.0,
        }
        request = _prepare_source_request(source, Path("run"))
        for key, expected in STRICT_GREEDY_SAMPLING_CONTRACT.items():
            self.assertEqual(request[key], expected)
        self.assertFalse(strict_greedy_sampling_errors(request))

    def test_source_request_rejects_explicit_repetition_penalty(self) -> None:
        source = {
            "model": "m",
            "prompt": "p",
            "max_tokens": 100,
            "repetition_penalty": 1.05,
        }
        with self.assertRaisesRegex(ValueError, "repetition_penalty"):
            _prepare_source_request(source, Path("run"))

    def test_honored_marker_is_generation_specific(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "runtime_control_honored"
            path.write_text(
                json.dumps({"format_version": 1, "generation": 7}),
                encoding="utf-8",
            )
            self.assertEqual(honored_generation(path), 7)


class TestTokenEquivalence(unittest.TestCase):
    @staticmethod
    def completion(
        token_ids: list[int],
        texts: list[str],
        tops: list[dict[str, float]],
    ) -> dict:
        return {
            "choices": [
                {
                    "token_ids": token_ids,
                    "logprobs": {
                        "tokens": texts,
                        "top_logprobs": tops,
                    },
                }
            ]
        }

    @staticmethod
    def probe_request(prompt: list[int]) -> dict:
        return freeze_strict_greedy_sampling(
            {"prompt": prompt, "max_tokens": 1}
        )

    def test_exact_match_needs_no_tie_evidence(self) -> None:
        result = classify_token_equivalence(
            emitted=[1, 2],
            control=[1, 2],
            emitted_rows=[],
            cutover_index=1,
            target_response=None,
            control_response=None,
            token_text_map=None,
            probe_request=None,
            tp1_probe=None,
            tp4_probe=None,
            expected_probe_prompt=None,
        )
        self.assertEqual(result["classification"], "EXACT")
        self.assertTrue(result["acceptable"])

    def test_equal_maximum_is_certified_on_both_topologies(self) -> None:
        tied = {" necessary": -1.0, " essential": -1.0, " other": -3.0}
        control = self.completion(
            [10, 20, 40],
            [" a", " b", " necessary"],
            [{" a": 0.0}, {" b": 0.0}, tied],
        )
        target = {
            "token_ids": [30],
            "chunks": [
                {
                    "choices": [
                        {
                            "token_ids": [30],
                            "logprobs": {
                                "tokens": [" essential"],
                                "top_logprobs": [tied],
                            },
                        }
                    ]
                }
            ]
        }
        probe = self.completion([40], [" necessary"], [tied])
        prompt = [101, 102, 10, 20]
        result = classify_token_equivalence(
            emitted=[10, 20, 30],
            control=[10, 20, 40],
            emitted_rows=[
                {"index": 0, "origin": "source"},
                {"index": 1, "origin": "source"},
                {"index": 2, "origin": "target"},
            ],
            cutover_index=2,
            target_response=target,
            control_response=control,
            token_text_map={
                "control_token_id": 40,
                "target_token_id": 30,
                "tokens": {"40": " necessary", "30": " essential"},
            },
            probe_request=self.probe_request(prompt),
            tp1_probe=probe,
            tp4_probe=probe,
            expected_probe_prompt=prompt,
            require_strict_sampling_contract=True,
        )
        self.assertEqual(
            result["classification"],
            "TIE_EQUIVALENT_DIVERGENCE",
        )
        self.assertTrue(result["tie_certified"])
        self.assertTrue(result["acceptable"])

    def test_non_tied_divergence_fails_closed(self) -> None:
        tied = {" necessary": -1.0, " essential": -1.1}
        control = self.completion([40], [" necessary"], [tied])
        target = {
            "token_ids": [30],
            "chunks": [
                {
                    "choices": [
                        {
                            "token_ids": [30],
                            "logprobs": {
                                "tokens": [" essential"],
                                "top_logprobs": [tied],
                            },
                        }
                    ]
                }
            ]
        }
        probe = self.completion([40], [" necessary"], [tied])
        result = classify_token_equivalence(
            emitted=[30],
            control=[40],
            emitted_rows=[{"index": 0, "origin": "target"}],
            cutover_index=0,
            target_response=target,
            control_response=control,
            token_text_map={
                "control_token_id": 40,
                "target_token_id": 30,
                "tokens": {"40": " necessary", "30": " essential"},
            },
            probe_request=self.probe_request([9]),
            tp1_probe=probe,
            tp4_probe=probe,
            expected_probe_prompt=[9],
            require_strict_sampling_contract=True,
        )
        self.assertEqual(result["classification"], "UNPROVEN_DIVERGENCE")
        self.assertFalse(result["acceptable"])

    def test_probe_with_inherited_penalty_fails_closed(self) -> None:
        tied = {" necessary": -1.0, " essential": -1.0}
        control = self.completion([40], [" necessary"], [tied])
        target = {
            "token_ids": [30],
            "chunks": [
                {
                    "choices": [
                        {
                            "token_ids": [30],
                            "logprobs": {
                                "tokens": [" essential"],
                                "top_logprobs": [tied],
                            },
                        }
                    ]
                }
            ],
        }
        probe = self.completion([40], [" necessary"], [tied])
        contaminated = self.probe_request([9])
        contaminated["repetition_penalty"] = 1.05
        result = classify_token_equivalence(
            emitted=[30],
            control=[40],
            emitted_rows=[{"index": 0, "origin": "target"}],
            cutover_index=0,
            target_response=target,
            control_response=control,
            token_text_map={
                "control_token_id": 40,
                "target_token_id": 30,
                "tokens": {"40": " necessary", "30": " essential"},
            },
            probe_request=contaminated,
            tp1_probe=probe,
            tp4_probe=probe,
            expected_probe_prompt=[9],
            require_strict_sampling_contract=True,
        )
        self.assertEqual(result["classification"], "UNPROVEN_DIVERGENCE")
        self.assertIn("sampling contract differs", result["reason"])

    def test_phase8_classifier_keeps_legacy_probe_boundary(self) -> None:
        tied = {" necessary": -1.0, " essential": -1.0}
        control = self.completion([40], [" necessary"], [tied])
        target = {
            "token_ids": [30],
            "chunks": [
                {
                    "choices": [
                        {
                            "token_ids": [30],
                            "logprobs": {
                                "tokens": [" essential"],
                                "top_logprobs": [tied],
                            },
                        }
                    ]
                }
            ],
        }
        probe = self.completion([40], [" necessary"], [tied])
        result = classify_token_equivalence(
            emitted=[30],
            control=[40],
            emitted_rows=[{"index": 0, "origin": "target"}],
            cutover_index=0,
            target_response=target,
            control_response=control,
            token_text_map={
                "control_token_id": 40,
                "target_token_id": 30,
                "tokens": {"40": " necessary", "30": " essential"},
            },
            probe_request={
                "prompt": [9],
                "max_tokens": 1,
                "temperature": 0.0,
            },
            tp1_probe=probe,
            tp4_probe=probe,
            expected_probe_prompt=[9],
        )
        self.assertEqual(
            result["classification"],
            "TIE_EQUIVALENT_DIVERGENCE",
        )

    def test_first_divergence_includes_length_only_mismatch(self) -> None:
        self.assertEqual(first_divergence([1], [1, 2]), 1)


class TestDynamicRateProvider(unittest.TestCase):
    def test_rate_is_reloaded_for_every_frame(self) -> None:
        if importlib.util.find_spec("torch") is None:
            sys.modules.setdefault("torch", types.ModuleType("torch"))
        from vllm.bridge_tp.stream_protocol import send_payload_frames

        class Connection:
            def __init__(self) -> None:
                self.parts: list[bytes] = []

            def sendall(self, value: bytes) -> None:
                self.parts.append(value)

        rates = iter((1e12, 2e12, 0.0))
        observed: list[float] = []

        def provider() -> float:
            value = next(rates)
            observed.append(value)
            return value

        result = send_payload_frames(
            Connection(),
            b"0123456789",
            chunk_bytes=4,
            rate_provider=provider,
        )
        self.assertEqual(result["num_frames"], 3)
        self.assertEqual(observed, [1e12, 2e12, 0.0])


class TestLazyActionBinding(unittest.TestCase):
    def test_runtime_control_precedes_session_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            adapter = ActionAdapter("http://source", run_dir)
            control = adapter.arm_shadow(33, 0.5, cutover_output_tokens=65)
            self.assertTrue(control.armed)
            self.assertEqual(control.trigger_output_tokens, 33)
            self.assertEqual(control.cutover_output_tokens, 65)
            self.assertIsNone(adapter.refresh_binding())

            (run_dir / "session_manifest.json").write_text(
                json.dumps(
                    {
                        "migration_id": "m",
                        "session_token": "s",
                        "source_request_id": "r",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(adapter.refresh_binding().migration_id, "m")


class TestConfigFailClosed(unittest.TestCase):
    def test_fill_in_calibration_is_rejected(self) -> None:
        config = ControllerConfig(
            tp1_total_kv_blocks=1,
            tp4_total_kv_blocks=1,
            tpot_tp1=TpotModel(0.03, 0.0, "FILL IN: run"),
            tpot_tp4=TpotModel(0.02, 0.0, "real run"),
            interference=InterferenceModel(1.0, calibration_source="real run"),
        )
        with self.assertRaisesRegex(ValueError, "uncalibrated"):
            config.validate()


class TestRunnerTransitions(unittest.TestCase):
    def test_local_arm_and_four_rank_commit(self) -> None:
        class Decision:
            action = Action.START_SHADOW
            reason = "worth migrating"

            @staticmethod
            def to_json() -> dict:
                return {"action": Action.START_SHADOW.value, "reason": "test"}

        class Policy:
            @staticmethod
            def evaluate(*_args, **_kwargs) -> Decision:
                return Decision()

        class Audit:
            def __init__(self) -> None:
                self.records: list[dict] = []

            def write(self, value: dict) -> None:
                self.records.append(value)

        class Adapter:
            def __init__(self) -> None:
                self.armed: tuple[int, float, int] | None = None

            def arm_shadow(
                self,
                trigger: int,
                rate: float,
                cutover_output_tokens: int,
                note: str,
            ) -> None:
                del note
                self.armed = (trigger, rate, cutover_output_tokens)

            @staticmethod
            def poll_target_ready() -> tuple[bool, set[int], str]:
                return True, {0, 1, 2, 3}, "ready"

            @staticmethod
            def commit() -> dict:
                return {"state": "COMMITTED"}

        class Rate:
            rate_bytes_s = 1024.0
            rate_gib_s = 0.5

        audit = Audit()
        machine = MigrationStateMachine(audit_sink=audit.write)
        record = machine.create("migration", "request")
        adapter = Adapter()
        recorder = ProxyRecorder("external", ProxyMode.HOLD_BACK)
        for index in range(10):
            recorder.on_source_token(index, 100 + index, float(index))
        request = SourceRequestView(
            request_id="request",
            prompt_tokens=8,
            output_tokens=10,
            computed_tokens=17,
            pending_tokens=1,
            arrival_unix_s=0.0,
            last_token_unix_s=1.0,
        )
        config = ControllerConfig(handoff_output_tokens=32)

        step_local(
            Policy(),
            machine,
            adapter,
            audit,
            record,
            request,
            object(),
            object(),
            0.1,
            Rate(),
            10.0,
            False,
            config,
            recorder,
            100,
        )
        self.assertEqual(adapter.armed, (11, 0.5, 43))
        self.assertEqual(record.state, MigrationState.SHADOW)

        for index in range(10, 43):
            recorder.on_source_token(index, 100 + index, float(index))
        machine.transition("migration", MigrationState.HANDOFF, 11.0, "ready")
        recorder.on_target_token(0, 143, 11.1)
        step_handoff(
            machine,
            adapter,
            audit,
            record,
            recorder,
            12.0,
            False,
        )
        self.assertEqual(record.state, MigrationState.TAKEOVER)
        self.assertEqual(record.ranks_ready, {0, 1, 2, 3})
        self.assertEqual(recorder.stats()["target_origin_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
