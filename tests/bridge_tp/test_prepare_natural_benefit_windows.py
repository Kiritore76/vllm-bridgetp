"""Trace preparation preserves timing and refuses unsafe input shortcuts."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.bridge_tp.prepare_natural_benefit_windows import freeze_windows, main


class TestNaturalBenefitWindows(unittest.TestCase):
    def test_preserves_arrival_spacing_and_does_not_read_final_length(self) -> None:
        requests = [
            {"id": "a", "prompt": "hello", "max_tokens": 1200,
             "final_output_tokens": 999},
            {"id": "b", "messages": [{"role": "user", "content": "x"}],
             "max_tokens": 500},
            {"id": "c", "prompt": "world", "max_tokens": 800},
        ]
        arrivals = [
            {"request_id": "c", "arrival_unix_s": 112.0, "pool": "target"},
            {"request_id": "a", "arrival_unix_s": 100.0, "pool": "source"},
            {"request_id": "b", "arrival_unix_s": 103.25, "pool": "target"},
        ]
        windows = freeze_windows(requests, arrivals, window_s=10.0)
        self.assertEqual(len(windows), 2)
        self.assertEqual(windows[0]["source_arrivals"], 1)
        self.assertEqual(windows[0]["target_arrivals"], 1)
        self.assertEqual([row["arrival_offset_s"]
                          for row in windows[0]["arrivals"]], [0.0, 3.25])
        self.assertEqual(windows[1]["arrivals"][0]["arrival_offset_s"], 2.0)
        self.assertNotIn("final_output_tokens", str(windows))
        self.assertEqual(windows[0]["status"], "TRACE_WINDOW_PLAN_ONLY")

    def test_rejects_duplicate_arrival_and_forced_length(self) -> None:
        requests = [{"id": "a", "prompt": "x", "max_tokens": 20}]
        event = {"request_id": "a", "arrival_unix_s": 10.0,
                 "pool": "source"}
        with self.assertRaisesRegex(ValueError, "duplicate arrival"):
            freeze_windows(requests, [event, event], window_s=5.0)
        with self.assertRaisesRegex(ValueError, "requires EOS"):
            freeze_windows(requests, [{**event, "ignore_eos": True}],
                           window_s=5.0)

    def test_simultaneous_arrivals_keep_trace_line_order(self) -> None:
        requests = [
            {"id": "z", "prompt": "first", "max_tokens": 20},
            {"id": "a", "prompt": "second", "max_tokens": 20},
        ]
        arrivals = [
            {"request_id": "z", "arrival_unix_s": 10.0, "pool": "source"},
            {"request_id": "a", "arrival_unix_s": 10.0, "pool": "target"},
        ]
        windows = freeze_windows(requests, arrivals, window_s=5.0)
        self.assertEqual(
            [row["request_id"] for row in windows[0]["arrivals"]],
            ["z", "a"],
        )

    def test_default_output_cap_is_explicit(self) -> None:
        requests = [{"id": "a", "prompt": "x"}]
        arrivals = [{"request_id": "a", "arrival_unix_s": 10.0,
                     "pool": "source"}]
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            freeze_windows(requests, arrivals, window_s=5.0)
        windows = freeze_windows(requests, arrivals, window_s=5.0,
                                 default_max_tokens=1024)
        self.assertTrue(windows[0]["arrivals"][0]["max_tokens_defaulted"])

    def test_cli_checks_input_hashes_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            requests = root / "requests.jsonl"
            arrivals = root / "arrivals.jsonl"
            output = root / "windows"
            requests.write_text(json.dumps({
                "id": "a", "prompt": "x", "max_tokens": 20,
            }) + "\n", encoding="utf-8")
            arrivals.write_text(json.dumps({
                "request_id": "a", "arrival_unix_s": 10.0,
                "pool": "source",
            }) + "\n", encoding="utf-8")
            args = [
                "prepare_natural_benefit_windows.py",
                "--requests", str(requests), "--arrivals", str(arrivals),
                "--expected-requests-sha256", "wrong",
                "--expected-arrivals-sha256", hashlib.sha256(
                    arrivals.read_bytes()).hexdigest(),
                "--window-s", "5", "--out-dir", str(output),
            ]
            with mock.patch("sys.argv", args):
                with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                    main()
            self.assertFalse(output.exists())
            args[args.index("wrong")] = hashlib.sha256(
                requests.read_bytes()).hexdigest()
            with mock.patch("sys.argv", args):
                main()
            summary = json.loads((output / "summary.json").read_text())
            self.assertFalse(summary["exact_replay_supported_by_current_runner"])


if __name__ == "__main__":
    unittest.main()
