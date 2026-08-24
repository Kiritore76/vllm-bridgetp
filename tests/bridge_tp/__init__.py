# SPDX-License-Identifier: Apache-2.0
"""BridgeTP tests.

The Phase 9 controller tests are intentionally torch-free. When they run in a
minimal logic-test environment, expose the source tree as a namespace package
without executing vLLM's torch-dependent top-level initializer.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

if importlib.util.find_spec("torch") is None and "vllm" not in sys.modules:
    package = types.ModuleType("vllm")
    package.__path__ = [str(Path(__file__).resolve().parents[2] / "vllm")]
    sys.modules["vllm"] = package
