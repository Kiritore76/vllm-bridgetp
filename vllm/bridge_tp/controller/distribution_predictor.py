# SPDX-License-Identifier: Apache-2.0
"""Read-only inference for a frozen remaining-length distribution checkpoint.

This module does not schedule migrations or capture hidden states. The caller
must supply the selected decoder representation and its generated-token count.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch
from torch import nn


class DistributionPredictor:
    """Serve a vetted training checkpoint without changing controller actions."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        checkpoint_sha256: str,
        model_config_sha256: str,
        feature_layer: str = "decoder:31",
        device: str | torch.device = "cpu",
    ) -> None:
        path = Path(checkpoint_path)
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != checkpoint_sha256:
            raise ValueError("predictor checkpoint SHA-256 differs")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 2 or checkpoint.get(
            "model_type"
        ) != "remaining_length_categorical":
            raise ValueError("unsupported predictor checkpoint format")
        if checkpoint.get("feature_layer") != feature_layer:
            raise ValueError("predictor feature layer differs")
        if checkpoint.get("model_config_sha256") != model_config_sha256:
            raise ValueError("predictor base model config differs")
        if checkpoint.get("overflow_category") is not True:
            raise ValueError("predictor needs an open tail category")
        self.feature_layer = feature_layer
        self.checkpoint_sha256 = checkpoint_sha256
        self.capture_input_sha256 = checkpoint["capture_input_sha256"]
        self.device = torch.device(device)

        mean = checkpoint["feature_mean"].to(dtype=torch.float32)
        std = checkpoint["feature_std"].to(dtype=torch.float32)
        edges = checkpoint["category_upper_edges"].to(dtype=torch.int64)
        width = int(checkpoint["hidden_width"])
        input_width = int(checkpoint["input_width"])
        if (
            mean.ndim != 1
            or std.shape != mean.shape
            or input_width != mean.numel() + 1
            or edges.ndim != 1
            or edges.numel() < 2
            or edges[0].item() != 0
            or not bool(torch.all(edges[1:] > edges[:-1]))
            or not bool(torch.isfinite(mean).all())
            or not bool(torch.isfinite(std).all())
            or not bool(torch.all(std > 0))
        ):
            raise ValueError("invalid predictor feature or category geometry")
        if checkpoint.get("activation") != "GELU" or width <= 0:
            raise ValueError("unsupported predictor network")
        scale = float(checkpoint["position_log_scale"])
        temperature = float(checkpoint["temperature"])
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("invalid generated-token scale")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("invalid predictor temperature")
        self.position_log_scale = scale
        self.temperature = temperature
        self.feature_mean = mean.to(self.device)
        self.feature_std = std.to(self.device)
        self.category_upper_edges = edges.to(self.device)
        self.model = nn.Sequential(
            nn.Linear(input_width, width),
            nn.GELU(),
            nn.Dropout(float(checkpoint["dropout"])),
            nn.Linear(width, edges.numel() + 1),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()

    @torch.no_grad()
    def probabilities(
        self,
        hidden_states: torch.Tensor,
        generated_tokens: torch.Tensor,
        *,
        validate_inputs: bool = True,
    ) -> torch.Tensor:
        """Return calibrated category probabilities for one or more states.

        Training captured hidden values as fp16 before converting to fp32.
        Reproducing that conversion here gives identical feature semantics.
        """
        if (
            hidden_states.ndim != 2
            or hidden_states.shape[1] != self.feature_mean.numel()
        ):
            raise ValueError("predictor hidden-state width differs")
        if generated_tokens.shape != (hidden_states.shape[0],):
            raise ValueError("generated-token count must match hidden states")
        if validate_inputs and (
            not bool(torch.isfinite(hidden_states).all())
            or not bool(torch.isfinite(generated_tokens).all())
            or not bool(torch.all(generated_tokens >= 0))
        ):
            raise ValueError("predictor input is nonfinite or negative")
        hidden = hidden_states.to(self.device, dtype=torch.float16).float()
        position = generated_tokens.to(self.device, dtype=torch.float32)
        features = torch.cat(
            (
                (hidden - self.feature_mean) / self.feature_std,
                (torch.log1p(position) / self.position_log_scale)[:, None],
            ),
            dim=1,
        )
        logits = self.model(features)
        return torch.softmax(logits / self.temperature, dim=-1)

    @torch.no_grad()
    def probability_gt_bounds(
        self, probabilities: torch.Tensor, horizons: torch.Tensor | float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower and conservative upper bounds for P(remaining > H)."""
        edges = self.category_upper_edges
        if probabilities.ndim != 2 or probabilities.shape[1] != edges.numel() + 1:
            raise ValueError("predictor probability shape differs")
        if probabilities.device != self.device:
            raise ValueError("predictor probability device differs")
        h = torch.as_tensor(horizons, dtype=torch.float64, device=self.device)
        if h.ndim == 0:
            h = h.expand(probabilities.shape[0])
        if h.shape != (probabilities.shape[0],) or bool(torch.isnan(h).any()):
            raise ValueError("one finite or infinite horizon is required per row")
        category = torch.searchsorted(edges.to(torch.float64), h.contiguous())
        count = edges.numel()
        tail = probabilities.flip(-1).cumsum(-1).flip(-1)
        tail = torch.cat((tail, torch.zeros_like(tail[:, :1])), dim=1)
        low = tail.gather(1, (category + 1).clamp(max=count + 1)[:, None])[:, 0]
        upper = tail.gather(1, category.clamp(max=count)[:, None])[:, 0]
        exact = (category < count) & (
            h == edges[category.clamp(max=count - 1)].to(torch.float64)
        )
        upper = torch.where(exact, low, upper)
        low = torch.where(h < 0, torch.ones_like(low), low)
        upper = torch.where(h < 0, torch.ones_like(upper), upper)
        low = torch.where(torch.isposinf(h), torch.zeros_like(low), low)
        upper = torch.where(torch.isposinf(h), torch.zeros_like(upper), upper)
        return low, upper
