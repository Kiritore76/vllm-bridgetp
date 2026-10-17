#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Internal CPU-only inventory worker; stdout is reserved for bounded IPC."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.bridge_tp.pressure_cpu import load  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--pool", choices=("tp1", "tp4"), required=True)
    parser.add_argument("--timing", type=Path, required=True)
    args = parser.parse_args()
    return load("pressure_inventory_process").worker(
        args.inventory, args.pool, args.timing
    )


if __name__ == "__main__":
    raise SystemExit(main())
