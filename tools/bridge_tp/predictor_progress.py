"""Human-readable predictor progress without changing generation scheduling."""

import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path


def emit(message):
    print(f"[进度] {message}", flush=True)


class RequestProgress:
    """Show request start, periodic read-only capture counts, and completion."""

    def __init__(
        self,
        number,
        total,
        *,
        features: Path | None = None,
        label="请求采集",
        interval=15,
    ):
        self.number, self.total = number, total
        self.features, self.label, self.interval = features, label, interval
        self._stop = threading.Event()
        self._baseline = {}
        self._thread = None

    def _latest(self):
        result = {}
        if self.features:
            for file in self.features.rglob("features.sqlite3"):
                try:
                    with closing(
                        sqlite3.connect(
                            file.resolve().as_uri() + "?mode=ro", uri=True, timeout=0.1
                        )
                    ) as connection:
                        row = connection.execute(
                            "SELECT sample_id, generated_tokens FROM samples "
                            "ORDER BY sample_id DESC LIMIT 1"
                        ).fetchone()
                        if row:
                            result[file] = row
                except (sqlite3.Error, OSError):
                    pass  # Optional status reads must not affect collection.
        return result

    def __enter__(self):
        self.started = time.monotonic()
        self._baseline = {file: row[0] for file, row in self._latest().items()}
        emit(f"{self.label} 第{self.number}/{self.total}条：开始生成")
        self._thread = threading.Thread(target=self._heartbeat, daemon=True)
        self._thread.start()
        return self

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            self.heartbeat()

    def heartbeat(self):
        counts = [
            row[1]
            for file, row in self._latest().items()
            if row[0] > self._baseline.get(file, 0)
        ]
        tokens = f"，已生成约{max(counts)} token" if counts else "，生成中"
        emit(
            f"{self.label} 第{self.number}/{self.total}条{tokens}"
            f"，耗时{time.monotonic() - self.started:.0f}s"
        )

    def finish(self, tokens, reason):
        self._stop.set()
        ending = (
            "自然结束"
            if reason == "stop"
            else "达到上限"
            if reason == "length"
            else reason
        )
        emit(
            f"{self.label} 第{self.number}/{self.total}条：完成，输出{tokens} token，"
            f"{ending}，耗时{time.monotonic() - self.started:.1f}s"
        )

    def __exit__(self, kind, error, traceback):
        self._stop.set()
        self._thread.join(timeout=1)
        if error:
            print(
                f"[异常] {self.label} 第{self.number}/{self.total}条失败：{error}",
                flush=True,
            )


class EpochProgress:
    """Throttle batch updates, distinguishing state samples from requests."""

    def __init__(self, epoch, epochs, samples, batch_size, requests, interval=15):
        self.epoch, self.epochs, self.samples = epoch, epochs, samples
        self.batches = (samples + batch_size - 1) // batch_size
        self.interval = interval
        self.started = self.last = time.monotonic()
        emit(
            f"续训 第{epoch}/{epochs}轮：{requests}条训练请求，"
            f"{samples}个状态样本，共{self.batches}批"
        )

    def update(self, batch, processed):
        now = time.monotonic()
        if batch in (1, self.batches) or now - self.last >= self.interval:
            emit(
                f"续训 第{self.epoch}/{self.epochs}轮，批次{batch}/{self.batches}，"
                f"状态样本{processed}/{self.samples}，耗时{now - self.started:.0f}s"
            )
            self.last = now
