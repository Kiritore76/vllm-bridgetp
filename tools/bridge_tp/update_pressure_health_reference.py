# SPDX-License-Identifier: Apache-2.0
"""Export offline empirical health references; never start or modify serving."""

import argparse
import hashlib
import json
from pathlib import Path

from tools.bridge_tp.pressure_health_reference import build_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--service-evidence", type=Path, required=True)
    parser.add_argument("--diagnostic-shape", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    diagnostics = set()
    for value in args.diagnostic_shape:
        pool, nd, context_bin = value.split(":")
        diagnostics.add((pool, int(nd), int(context_bin)))
    evidence = read(args.service_evidence)
    release = evidence["source_release_summary"]["start_to_release_s"]["max"]
    provenance = dict(
        sources=read(args.provenance),
        input_sha256={
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in (
                ("cells", args.cells),
                ("windows", args.windows),
                ("service_evidence", args.service_evidence),
            )
        },
        diagnostic_shapes=sorted(diagnostics),
    )
    reference = build_reference(
        read(args.cells),
        read(args.windows),
        provenance=provenance,
        release_budget_s=release,
        diagnostic_shapes=diagnostics,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(reference, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            dict(
                output=str(args.output),
                cells=len(reference["cells"]),
                deployment_ready=False,
            )
        )
    )


if __name__ == "__main__":
    main()
