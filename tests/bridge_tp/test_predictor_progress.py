"""Check that progress is accurate and full-log filtering keeps errors."""

import io
import sqlite3
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools/bridge_tp"))
from predictor_console import display
from predictor_progress import EpochProgress, RequestProgress


class ProgressTests(unittest.TestCase):
    def test_sqlite_progress_ignores_previous_request_and_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            directory = Path(tmp) / "features with spaces"
            directory.mkdir()
            connection = sqlite3.connect(directory / "features.sqlite3")
            stack.callback(connection.close)
            connection.execute(
                "CREATE TABLE samples(sample_id INTEGER PRIMARY KEY, generated_tokens INTEGER)"
            )
            connection.execute("INSERT INTO samples VALUES(1, 900)")
            connection.commit()
            output = io.StringIO()
            with (
                redirect_stdout(output),
                RequestProgress(2, 12, features=directory, interval=3600) as progress,
            ):
                progress.heartbeat()
                self.assertNotIn("900 token", output.getvalue())
                connection.execute("INSERT INTO samples VALUES(2, 20)")
                connection.commit()
                progress.heartbeat()
                progress.finish(39, "stop")
            self.assertIn("第2/12条", output.getvalue())
            self.assertIn("已生成约20 token", output.getvalue())
            self.assertIn("输出39 token，自然结束", output.getvalue())
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0], 2
            )

    def test_request_failure_propagates_and_stops_heartbeat(self):
        progress = RequestProgress(1, 12, interval=3600)
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(ValueError, "failed"):
            with progress:
                raise ValueError("failed")
        self.assertIn("[异常]", output.getvalue())
        self.assertFalse(progress._thread.is_alive())

    def test_epoch_batches_are_throttled_and_last_batch_is_shown(self):
        output = io.StringIO()
        with (
            redirect_stdout(output),
            patch("predictor_progress.time.monotonic", side_effect=[0, 0, 1, 2]),
        ):
            progress = EpochProgress(1, 40, 5, 2, 2, interval=15)
            progress.update(1, 2)
            progress.update(2, 4)
            progress.update(3, 5)
        self.assertIn("2条训练请求", output.getvalue())
        self.assertNotIn("批次2/3", output.getvalue())
        self.assertIn("批次3/3", output.getvalue())
        self.assertIn("状态样本5/5", output.getvalue())

    def test_console_hides_engine_noise_and_retains_progress_and_traceback(self):
        output = io.StringIO()
        display(
            [
                "Loading safetensors shards: 100%\r\n",
                "(EngineCore) INFO Available KV cache memory: 4.88 GiB\n",
                "(EngineCore) WARNING Triton compilation\n",
                "stage=long_coverage_12requests natural EOS\n",
                "[进度] 请求采集 第1/12条：开始生成\n",
                "Traceback (most recent call last):\n",
                "  File test.py, line 1\n",
                "ValueError: failed\n",
                "(EngineCore) INFO cleanup done\n",
            ],
            output,
        )
        visible = output.getvalue()
        self.assertNotIn("safetensors", visible)
        self.assertNotIn("KV cache", visible)
        self.assertNotIn("Triton", visible)
        self.assertNotIn("cleanup done", visible)
        self.assertIn("阶段3/3", visible)
        self.assertIn("第1/12条", visible)
        self.assertIn("File test.py", visible)
        self.assertIn("ValueError: failed", visible)


if __name__ == "__main__":
    unittest.main()
