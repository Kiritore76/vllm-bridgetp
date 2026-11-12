# SPDX-License-Identifier: Apache-2.0
"""Hang diagnostics without dependencies (BridgeTP, 11-12).

On PYTHONPATH with ``PY_STACK_DUMP_ON_USR1=1``, every Python process (API
server, EngineCore, TP workers, controller, stagers) prints the Python
stacks of all its threads to stderr on SIGUSR1. ``hang_watch.py`` sends the
signal when an engine stops returning. Any other ``sitecustomize`` on the
path still runs afterwards. Without the variable nothing is registered.
"""

import os
import sys


def _register():
    if os.environ.get("PY_STACK_DUMP_ON_USR1") != "1":
        return
    import faulthandler
    import signal

    try:
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
    except (AttributeError, RuntimeError, ValueError, OSError):
        pass


def _chain():
    """Run the next sitecustomize on sys.path, if one exists."""
    import importlib.machinery
    import importlib.util

    here = os.path.dirname(os.path.abspath(__file__))
    rest = [p for p in sys.path if os.path.abspath(p or ".") != here]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None or spec.origin == os.path.abspath(__file__):
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


_register()
try:
    _chain()
except Exception:  # never break interpreter start-up for diagnostics
    pass
