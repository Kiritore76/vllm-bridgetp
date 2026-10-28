# SPDX-License-Identifier: Apache-2.0
"""Read the last complete native inventory row without scanning the journal."""

import json


def latest_inventory(path):
    if not path.exists():
        return None
    # Read a bounded suffix, not the growing journal on every SSE polling tick.
    with path.open("rb") as source:
        source.seek(0, 2)
        end = source.tell()
        size = min(end, 65536)
        while size:
            source.seek(end - size)
            data = source.read(size)
            if size < end:
                data = data.partition(b"\n")[2]
            complete = data.splitlines(keepends=True)
            complete = [line for line in complete if line.endswith(b"\n")]
            if complete:
                return json.loads(complete[-1])
            if size == end:
                return None
            size = min(end, size * 2)
    return None
