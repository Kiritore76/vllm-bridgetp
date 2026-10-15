# SPDX-License-Identifier: Apache-2.0
"""Load the pure pressure modules without importing the GPU vLLM package."""

import importlib
import sys
import types
from pathlib import Path


def load(name):
    package = "_bridgetp_pressure_cpu"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [
            str(Path(__file__).resolve().parents[2] / "vllm/bridge_tp/controller")
        ]
        sys.modules[package] = module
    return importlib.import_module(f"{package}.{name}")
