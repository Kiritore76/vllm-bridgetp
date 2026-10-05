"""Guard composite arrival preparation against future-length leakage."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from tools.bridge_tp.prepare_tracea_oasst_arrivals import (
    make_composite,
    read_arrival_slice,
)
from tools.bridge_tp.prepare_natural_benefit_windows import freeze_windows


class TestTraceAOasstArrivals(unittest.TestCase):
    def test_preserves_order_without_using_future_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.csv"
            with path.open("w", newline="", encoding="utf-8") as out:
                writer = csv.DictWriter(out, fieldnames=(
                    "timestamp", "input_len", "output_len", "type"
                ))
                writer.writeheader()
                writer.writerows([
                    {"timestamp": "3", "input_len": "100",
                     "output_len": "10", "type": "text"},
                    {"timestamp": "3.1", "input_len": "200",
                     "output_len": "8000", "type": "search"},
                    {"timestamp": "3.3", "input_len": "300",
                     "output_len": "1", "type": "text"},
                ])
            trace = read_arrival_slice(path, start_row=0, count=3)
        self.assertTrue(all("output_len" not in row for row in trace))
        requests = [
            {"id": f"r{i}", "messages": [{"role": "user", "content": "hi"}],
             "split": "test"} for i in range(4)
        ]
        paired, arrivals, audit = make_composite(
            trace, requests, seed=3, time_scale=10.0,
            source_fraction=0.5, max_tokens=100, max_model_len=1000,
            token_count=lambda _row: 50,
        )
        self.assertEqual([row["arrival_unix_s"] for row in arrivals],
                         [0.0, 1.0, 3.0])
        self.assertEqual(arrivals[0]["pool"], "source")
        self.assertFalse(audit["trace_output_len_used"])
        self.assertFalse(audit["original_trace_content_replayed"])
        self.assertNotIn("output_len", json.dumps([paired, arrivals]))
        windows = freeze_windows(paired, arrivals, window_s=5.0)
        self.assertEqual(len(windows), 1)
        self.assertEqual(windows[0]["source_arrivals"], 2)

    def test_rejects_nonmonotonic_arrivals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.csv"
            path.write_text(
                "timestamp,input_len,type\n2,10,text\n1,10,text\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not monotonic"):
                read_arrival_slice(path, start_row=0, count=2)


if __name__ == "__main__":
    unittest.main()
