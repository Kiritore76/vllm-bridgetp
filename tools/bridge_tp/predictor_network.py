"""Construct offline predictor architectures without loading the base LLM."""


def hidden_widths(width: int, depth: int) -> list[int]:
    """Return hidden widths for two or four affine layers, including the head."""
    if not isinstance(width, int) or width <= 0 or depth not in (2, 4):
        raise ValueError("predictor requires positive width and depth 2 or 4")
    return [width] if depth == 2 else [width, 2 * width, width]


def build_mlp(input_width: int, width: int, categories: int, dropout: float, depth=2):
    """Build a GELU MLP; old two-layer state-dict indices remain unchanged."""
    from torch import nn

    layers = []
    for next_width in hidden_widths(width, depth):
        layers += [nn.Linear(input_width, next_width), nn.GELU(), nn.Dropout(dropout)]
        input_width = next_width
    layers.append(nn.Linear(input_width, categories))
    return nn.Sequential(*layers)
