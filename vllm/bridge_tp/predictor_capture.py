"""Opt-in offline feature capture or live remaining-length observation.

The GPU runner calls this only after selecting the last state used for logits.
Live inference is enabled only by an explicit frozen-checkpoint environment.
"""

import hashlib
import json
import math
import os
import sqlite3
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CaptureRow:
    request_id: str
    generated_tokens: int
    phase: str
    batch_index: int


def decoder_layer_index(feature_layer: str) -> int | None:
    """Parse a zero-based residual/MLP layer index, or final normalized states."""
    if feature_layer == "final":
        return None
    prefix, separator, value = feature_layer.partition(":")
    if prefix not in ("decoder", "mlp") or not separator or not value.isdecimal():
        raise ValueError("feature layer must be final, decoder:<index> or mlp:<index>")
    return int(value)


def select_capture_rows(
    request_ids: Sequence[str],
    generated_tokens: Sequence[int],
    computed_prompt_tokens: Sequence[int],
    scheduled_tokens: Sequence[int],
    prompt_tokens: Sequence[int],
    interval: int,
) -> list[CaptureRow]:
    """Choose completed-prefill and periodic decode states for each request."""
    if interval <= 0:
        raise ValueError("prediction capture interval must be positive")
    lengths = {
        len(request_ids),
        len(generated_tokens),
        len(computed_prompt_tokens),
        len(scheduled_tokens),
        len(prompt_tokens),
    }
    if len(lengths) != 1:
        raise ValueError("capture metadata must have one entry per request")
    selected = []
    for index, request_id in enumerate(request_ids):
        output_len = int(generated_tokens[index])
        if output_len < 0:
            raise ValueError("generated token count cannot be negative")
        if int(computed_prompt_tokens[index]) + int(scheduled_tokens[index]) < int(
            prompt_tokens[index]
        ):
            continue
        if output_len == 0:
            phase = "PREFILL_COMPLETE"
        elif output_len % interval == 0:
            phase = "DECODE"
        else:
            continue
        selected.append(CaptureRow(str(request_id), output_len, phase, index))
    return selected


class PredictorFeatureCapture:
    """Commit selected hidden states to one durable SQLite feature file."""

    def __init__(
        self, directory: Path, interval: int, feature_layer: str = "final"
    ) -> None:
        if interval <= 0:
            raise ValueError("prediction capture interval must be positive")
        self.directory = directory
        self.interval = interval
        self.feature_layer = feature_layer
        self.layer_index = decoder_layer_index(feature_layer)
        self._intermediate_states = None
        self._layer_hook = None
        directory.mkdir(parents=True, exist_ok=True)
        self.database = sqlite3.connect(directory / "features.sqlite3")
        self.database.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                sample_id INTEGER PRIMARY KEY,
                request_id TEXT NOT NULL,
                generated_tokens INTEGER NOT NULL,
                phase TEXT NOT NULL,
                hidden_size INTEGER NOT NULL,
                hidden_fp16 BLOB NOT NULL,
                captured_unix_ns INTEGER NOT NULL,
                UNIQUE (request_id, generated_tokens)
            )
        """)
        self.database.commit()

    @classmethod
    def from_environment(
        cls,
    ) -> "PredictorFeatureCapture | MultiLayerFeatureCapture | None":
        directory = os.environ.get("BRIDGETP_PREDICTOR_CAPTURE_DIR")
        if not directory:
            return None
        interval = int(os.environ.get("BRIDGETP_PREDICTOR_CAPTURE_INTERVAL", "20"))
        layers = os.environ.get("BRIDGETP_PREDICTOR_CAPTURE_LAYERS")
        if layers:
            return MultiLayerFeatureCapture(
                Path(directory), interval, layers.split(",")
            )
        feature_layer = os.environ.get("BRIDGETP_PREDICTOR_CAPTURE_LAYER", "final")
        return cls(Path(directory), interval, feature_layer)

    def attach_model(self, model: "torch.nn.Module") -> None:
        """Observe a Qwen2 decoder block without modifying its output."""
        if self.layer_index is None:
            return
        if self._layer_hook is not None:
            raise RuntimeError("intermediate capture hook is already installed")
        suffix = f"layers.{self.layer_index}"
        candidates = [
            module
            for name, module in model.named_modules()
            if (name == suffix or name.endswith("." + suffix))
            and module.__class__.__name__ == "Qwen2DecoderLayer"
        ]
        if len(candidates) != 1:
            raise ValueError(
                "intermediate capture needs one matching Qwen2 decoder block"
            )
        self._layer_hook = candidates[0].register_forward_hook(self._observe_decoder)

    def _observe_decoder(self, module, inputs, output) -> None:
        # vLLM returns the MLP branch and residual separately. Copy their sum now:
        # the next fused RMSNorm may mutate the residual tensor in place.
        if not isinstance(output, tuple) or len(output) != 2:
            raise ValueError("Qwen2 decoder output must be (hidden, residual)")
        hidden, residual = output
        if residual is None or hidden.ndim != 2 or hidden.shape != residual.shape:
            raise ValueError("invalid Qwen2 residual-stream shape")
        if self.feature_layer.startswith("mlp:"):
            # Keep only the branch returned by the decoder, before residual add.
            # Clone because a later forward operation may reuse its storage.
            self._intermediate_states = hidden.detach().clone()
        else:
            self._intermediate_states = (hidden.detach() + residual.detach()).detach()

    def begin_forward(self) -> None:
        """Discard any state from profiling/dummy or preceding forward passes."""
        self._intermediate_states = None

    def sample_states(
        self, final_states: "torch.Tensor", logits_indices
    ) -> "torch.Tensor":
        """Select identical token positions from the requested representation."""
        if self.layer_index is None:
            return final_states
        states = self._intermediate_states
        self._intermediate_states = None
        if states is None:
            raise RuntimeError("intermediate layer did not run in this forward pass")
        selected = states[logits_indices]
        if selected.shape != final_states.shape:
            raise ValueError("intermediate and final sample positions differ")
        return selected

    def capture(
        self,
        hidden_states: "torch.Tensor",
        request_ids: Sequence[str],
        generated_tokens: Sequence[int],
        computed_prompt_tokens: Sequence[int],
        scheduled_tokens: Sequence[int],
        prompt_tokens: Sequence[int],
    ) -> None:
        """Copy only selected rows; this opt-in collection synchronizes GPU."""
        rows = select_capture_rows(
            request_ids,
            generated_tokens,
            computed_prompt_tokens,
            scheduled_tokens,
            prompt_tokens,
            self.interval,
        )
        if not rows:
            return
        if hidden_states.ndim != 2 or hidden_states.shape[0] != len(request_ids):
            raise ValueError("sample hidden states must be [request, hidden_size]")
        import torch

        states = (
            hidden_states[[row.batch_index for row in rows]]
            .detach()
            .to(device="cpu", dtype=torch.float16)
            .numpy()
        )
        captured_at = time.time_ns()
        with self.database:
            self.database.executemany(
                """INSERT INTO samples
                   (request_id, generated_tokens, phase, hidden_size,
                    hidden_fp16, captured_unix_ns) VALUES (?, ?, ?, ?, ?, ?)""",
                [
                    (
                        row.request_id,
                        row.generated_tokens,
                        row.phase,
                        states.shape[1],
                        states[index].tobytes(),
                        captured_at,
                    )
                    for index, row in enumerate(rows)
                ],
            )


class MultiLayerFeatureCapture:
    """Collect paired representations during one unchanged generation."""

    def __init__(self, directory: Path, interval: int, layers: list[str]) -> None:
        if not layers or len(set(layers)) != len(layers):
            raise ValueError("capture layers must be nonempty and unique")
        self.writers = [
            PredictorFeatureCapture(
                directory / layer.replace(":", "") / "features", interval, layer
            )
            for layer in layers
        ]
        self.layer_index = next(
            (x.layer_index for x in self.writers if x.layer_index is not None), None
        )
        self._logits_indices = None

    def attach_model(self, model: "torch.nn.Module") -> None:
        for writer in self.writers:
            writer.attach_model(model)

    def begin_forward(self) -> None:
        self._logits_indices = None
        for writer in self.writers:
            writer.begin_forward()

    def sample_states(
        self, final_states: "torch.Tensor", logits_indices
    ) -> "torch.Tensor":
        self._logits_indices = logits_indices
        return final_states

    def capture(self, final_states, *metadata) -> None:
        if self._logits_indices is None:
            raise RuntimeError("multi-layer sample positions are unavailable")
        indices = self._logits_indices
        self._logits_indices = None
        for writer in self.writers:
            writer.capture(writer.sample_states(final_states, indices), *metadata)


class PredictorLiveObserver(PredictorFeatureCapture):
    """Opt-in GPU inference with bounded asynchronous event publication.

    The decoder hook is shared with the offline capture. A CUDA copy stream
    moves only category probabilities to pinned host memory; later forwards
    publish completed copies without waiting for the device in the hot path.
    """

    def __init__(
        self,
        checkpoint: Path,
        checkpoint_sha256: str,
        event_path: Path,
        model_path: Path,
        device: "torch.device",
        *,
        interval: int = 20,
        max_pending: int = 256,
    ) -> None:
        import torch

        from vllm.bridge_tp.controller.distribution_predictor import (
            DistributionPredictor,
        )

        if interval != 20 or max_pending <= 0:
            raise ValueError("live predictor requires interval=20 and a queue")
        config_path = model_path / "config.json"
        digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
        self.predictor = DistributionPredictor(
            checkpoint,
            checkpoint_sha256=checkpoint_sha256,
            model_config_sha256=digest,
            feature_layer="decoder:31",
            device=device,
        )
        self.feature_layer = "decoder:31"
        self.layer_index = 31
        self._intermediate_states = None
        self._layer_hook = None
        self.interval = interval
        self.max_pending = max_pending
        self._pending = deque()
        self._latest_generated: dict[str, int] = {}
        self._copy_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        event_path.parent.mkdir(parents=True, exist_ok=True)
        self._events = event_path.open("x", encoding="utf-8", buffering=1)
        self._events.write(
            json.dumps(
                {
                    "kind": "predictor_header",
                    "format_version": 1,
                    "checkpoint_sha256": checkpoint_sha256,
                    "capture_input_sha256": self.predictor.capture_input_sha256,
                    "model_config_sha256": digest,
                    "feature_layer": self.feature_layer,
                    "interval": interval,
                    "category_upper_edges": (
                        self.predictor.category_upper_edges.cpu().tolist()
                    ),
                }
            )
            + "\n"
        )

    def begin_forward(self) -> None:
        self.poll_ready()
        super().begin_forward()

    def capture(
        self,
        hidden_states: "torch.Tensor",
        request_ids: Sequence[str],
        generated_tokens: Sequence[int],
        computed_prompt_tokens: Sequence[int],
        scheduled_tokens: Sequence[int],
        prompt_tokens: Sequence[int],
    ) -> None:
        import torch

        self.poll_ready()
        rows = select_capture_rows(
            request_ids,
            generated_tokens,
            computed_prompt_tokens,
            scheduled_tokens,
            prompt_tokens,
            self.interval,
        )
        rows = [
            row
            for row in rows
            if row.generated_tokens > self._latest_generated.get(row.request_id, -1)
        ]
        if not rows:
            return
        if hidden_states.ndim != 2 or hidden_states.shape[0] != len(request_ids):
            raise ValueError("live predictor states must align with requests")
        if len(self._pending) >= self.max_pending:
            raise RuntimeError("live predictor event queue is full")
        captured_unix_ns = time.time_ns()
        counts = torch.tensor(
            [row.generated_tokens for row in rows], dtype=torch.float32
        )
        with torch.no_grad():
            probabilities = self.predictor.probabilities(
                hidden_states[[row.batch_index for row in rows]],
                counts,
                validate_inputs=False,
            )
            if self._copy_stream is None:
                host = probabilities.detach().cpu()
                event = None
            else:
                host = torch.empty(
                    probabilities.shape,
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=True,
                )
                self._copy_stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(self._copy_stream):
                    host.copy_(probabilities, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
                probabilities.record_stream(self._copy_stream)
        for row in rows:
            self._latest_generated[row.request_id] = row.generated_tokens
        self._pending.append((event, host, rows, captured_unix_ns))
        self.poll_ready()

    def poll_ready(self) -> None:
        """Write completed prediction events without synchronizing CUDA."""
        while self._pending:
            event, host, rows, captured_unix_ns = self._pending[0]
            if event is not None and not event.query():
                break
            self._pending.popleft()
            published_unix_ns = time.time_ns()
            for row, probabilities in zip(rows, host.tolist()):
                if (
                    not all(
                        math.isfinite(value) and value >= 0
                        for value in probabilities
                    )
                    or abs(sum(probabilities) - 1.0) > 1e-4
                ):
                    result = {
                        "kind": "predictor_unavailable",
                        "reason": "nonfinite or unnormalized probabilities",
                    }
                else:
                    result = {
                        "kind": "predictor_prediction",
                        "probabilities": probabilities,
                    }
                self._events.write(
                    json.dumps(
                        {
                            **result,
                            "request_id": row.request_id,
                            "generated_tokens": row.generated_tokens,
                            "phase": row.phase,
                            "checkpoint_sha256": self.predictor.checkpoint_sha256,
                            "captured_unix_ns": captured_unix_ns,
                            "published_unix_ns": published_unix_ns,
                        }
                    )
                    + "\n"
                )

    def close(self) -> None:
        """Drain at shutdown; no per-forward CUDA synchronization is added."""
        while self._pending:
            event = self._pending[0][0]
            if event is not None:
                event.synchronize()
            self.poll_ready()
        self._events.close()


def predictor_observer_from_environment(
    model_path: Path, device: "torch.device", tensor_parallel_size: int
) -> PredictorFeatureCapture | MultiLayerFeatureCapture | PredictorLiveObserver | None:
    """Select exactly one opt-in predictor observer for this model runner."""
    checkpoint = os.environ.get("BRIDGETP_PREDICTOR_LIVE_CHECKPOINT")
    if not checkpoint:
        return PredictorFeatureCapture.from_environment()
    if os.environ.get("BRIDGETP_PREDICTOR_CAPTURE_DIR"):
        raise ValueError("live prediction and offline capture cannot run together")
    if tensor_parallel_size != 1:
        return None  # The current experiment observes source TP1 only.
    sha = os.environ.get("BRIDGETP_PREDICTOR_LIVE_SHA256")
    events = os.environ.get("BRIDGETP_PREDICTOR_LIVE_EVENTS")
    if not sha or not events:
        raise ValueError("live predictor requires checkpoint SHA and event path")
    return PredictorLiveObserver(
        Path(checkpoint), sha, Path(events), model_path, device
    )
