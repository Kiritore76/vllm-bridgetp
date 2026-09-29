"""Opt-in final-layer feature capture for an offline length predictor.

The GPU runner calls this only after selecting the last state used for logits.
No predictor is trained or invoked in the serving path.
"""

import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class CaptureRow:
    request_id: str
    generated_tokens: int
    phase: str
    batch_index: int


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
        len(request_ids), len(generated_tokens), len(computed_prompt_tokens),
        len(scheduled_tokens), len(prompt_tokens),
    }
    if len(lengths) != 1:
        raise ValueError("capture metadata must have one entry per request")
    selected = []
    for index, request_id in enumerate(request_ids):
        output_len = int(generated_tokens[index])
        if output_len < 0:
            raise ValueError("generated token count cannot be negative")
        if (
            int(computed_prompt_tokens[index]) + int(scheduled_tokens[index])
            < int(prompt_tokens[index])
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
    """Write selected hidden states without retaining tensors across steps."""

    def __init__(self, directory: Path, interval: int) -> None:
        if interval <= 0:
            raise ValueError("prediction capture interval must be positive")
        self.directory = directory
        self.interval = interval
        directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_environment(cls) -> "PredictorFeatureCapture | None":
        directory = os.environ.get("BRIDGETP_PREDICTOR_CAPTURE_DIR")
        if not directory:
            return None
        interval = int(os.environ.get("BRIDGETP_PREDICTOR_CAPTURE_INTERVAL", "20"))
        return cls(Path(directory), interval)

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
            request_ids, generated_tokens, computed_prompt_tokens,
            scheduled_tokens, prompt_tokens, self.interval,
        )
        if not rows:
            return
        if hidden_states.ndim != 2 or hidden_states.shape[0] != len(request_ids):
            raise ValueError("sample hidden states must be [request, hidden_size]")
        import torch

        states = (
            hidden_states[[row.batch_index for row in rows]]
            .detach().to(device="cpu", dtype=torch.float16).numpy()
        )
        name = f"{time.time_ns()}-{os.getpid()}-{uuid.uuid4().hex}"
        temporary = self.directory / f".{name}.npz"
        destination = self.directory / f"{name}.npz"
        try:
            np.savez(
                temporary,
                format_version=np.array(1, dtype=np.int32),
                hidden_states=states,
                request_ids=np.asarray([row.request_id for row in rows]),
                generated_tokens=np.asarray(
                    [row.generated_tokens for row in rows], dtype=np.int32
                ),
                phase=np.asarray([row.phase for row in rows]),
                captured_unix_ns=np.array(time.time_ns(), dtype=np.int64),
            )
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
