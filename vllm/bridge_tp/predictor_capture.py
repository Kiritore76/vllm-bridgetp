"""Opt-in offline feature capture or live remaining-length observation.

The GPU runner calls this only after selecting the last state used for logits.
Live inference is enabled only by an explicit frozen-checkpoint environment.
"""

import hashlib
import json
import math
import os
import queue
import sqlite3
import threading
import time
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
    """Opt-in GPU inference outside the token sampling stream.

    The live path reads Qwen2's graph-returned auxiliary state at decoder:31.
    The runner snapshots only selected rows. A worker executes the predictor on
    a separate CUDA stream and publishes probabilities after its copy completes.
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
        diagnostic = os.environ.get("BRIDGETP_PREDICTOR_DIAGNOSTIC_DIR")
        self._diagnostic_dir = Path(diagnostic) if diagnostic else None
        self._diagnostic_limit = int(
            os.environ.get("BRIDGETP_PREDICTOR_DIAGNOSTIC_LIMIT", "32")
        )
        self._diagnostic_count = 0
        if self._diagnostic_dir:
            if self._diagnostic_limit <= 0:
                raise ValueError("diagnostic limit must be positive")
            self._diagnostic_dir.mkdir(parents=True, exist_ok=True)
        self.feature_layer = "decoder:31"
        self.layer_index = 31
        self._intermediate_states = None
        self._layer_hook = None
        self.interval = interval
        self.max_pending = max_pending
        self._pending: queue.Queue = queue.Queue(maxsize=max_pending)
        self._latest_generated: dict[str, int] = {}
        self._compute_stream = (
            torch.cuda.Stream(device=device, priority=1)
            if device.type == "cuda"
            else None
        )
        self._worker_error: BaseException | None = None
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
        self._worker = threading.Thread(
            target=self._run_predictions, name="bridgetp-predictor", daemon=True
        )
        self._worker.start()

    def attach_model(self, model: "torch.nn.Module") -> None:
        """Check the exact Qwen2 layer contract; do not install a Python hook."""
        if (
            model.__class__.__name__ != "Qwen2ForCausalLM"
            or model.model.__class__.__name__ != "Qwen2Model"
            or not hasattr(model, "set_aux_hidden_state_layers")
            or self.layer_index >= model.model.config.num_hidden_layers
        ):
            raise ValueError("live predictor requires Qwen2 decoder:31 aux output")

    def sample_aux_states(
        self,
        final_states: "torch.Tensor",
        logits_indices: "torch.Tensor",
        aux_hidden_states: list["torch.Tensor"] | None,
    ) -> "torch.Tensor":
        """Use the same hidden + residual state as the offline decoder hook."""
        if aux_hidden_states is None or len(aux_hidden_states) != 1:
            raise RuntimeError("decoder:31 auxiliary output is missing")
        selected = aux_hidden_states[0][logits_indices]
        if selected.shape != final_states.shape:
            raise ValueError("decoder:31 and final sample positions differ")
        return selected

    def begin_forward(self) -> None:
        self._check_worker()
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

        self._check_worker()
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
        captured_unix_ns = time.time_ns()
        # Advanced indexing makes a private snapshot. Graph-returned model
        # buffers may be reused by the next forward before the worker runs.
        snapshot = hidden_states[[row.batch_index for row in rows]].detach()
        ready = None
        if self._compute_stream is not None:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream())
        try:
            self._pending.put_nowait((snapshot, ready, rows, captured_unix_ns))
        except queue.Full as exc:
            raise RuntimeError("live predictor event queue is full") from exc
        for row in rows:
            self._latest_generated[row.request_id] = row.generated_tokens

    def _check_worker(self) -> None:
        if self._worker_error is not None:
            raise RuntimeError("live predictor worker failed") from self._worker_error

    def _run_predictions(self) -> None:
        import torch

        while True:
            item = self._pending.get()
            try:
                if item is None:
                    return
                snapshot, ready, rows, captured_unix_ns = item
                counts = torch.tensor(
                    [row.generated_tokens for row in rows], dtype=torch.float32
                )
                with torch.no_grad():
                    if self._compute_stream is None:
                        probabilities = self.predictor.probabilities(
                            snapshot, counts, validate_inputs=False
                        )
                        host = probabilities.detach().cpu()
                    else:
                        with torch.cuda.stream(self._compute_stream):
                            self._compute_stream.wait_event(ready)
                            probabilities = self.predictor.probabilities(
                                snapshot, counts, validate_inputs=False
                            )
                            host = torch.empty(
                                probabilities.shape,
                                dtype=torch.float32,
                                device="cpu",
                                pin_memory=True,
                            )
                            host.copy_(probabilities, non_blocking=True)
                            done = torch.cuda.Event()
                            done.record(self._compute_stream)
                        done.synchronize()
                self._publish(rows, host.tolist(), captured_unix_ns)
                if (
                    self._diagnostic_dir
                    and self._diagnostic_count < self._diagnostic_limit
                ):
                    self._diagnostic_count += 1
                    # Worker only; optional diagnostic perturbation, not benefit data.
                    if self._compute_stream is not None:
                        with torch.cuda.stream(self._compute_stream):
                            feature = snapshot.to(torch.float16).cpu()
                    else:
                        feature = snapshot.to(torch.float16).cpu()
                    torch.save(
                        {
                            "hidden_fp16": feature,
                            "generated_tokens": counts.cpu(),
                            "probabilities": host,
                            "request_ids": [row.request_id for row in rows],
                            "captured_unix_ns": captured_unix_ns,
                            "checkpoint_sha256": self.predictor.checkpoint_sha256,
                        },
                        self._diagnostic_dir
                        / f"probe-{os.getpid()}-{self._diagnostic_count:04d}.pt",
                    )
            except BaseException as exc:
                self._worker_error = exc
                return
            finally:
                self._pending.task_done()

    def _publish(self, rows, probability_rows, captured_unix_ns: int) -> None:
        published_unix_ns = time.time_ns()
        for row, probabilities in zip(rows, probability_rows):
            if (
                not all(math.isfinite(value) and value >= 0 for value in probabilities)
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
        """Drain predictions when the runner shuts down."""
        if self._worker.is_alive():
            self._pending.put(None)
            self._worker.join()
        try:
            self._check_worker()
        finally:
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
