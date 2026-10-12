# SPDX-License-Identifier: Apache-2.0
"""Pack one evidence tree without nested archive copies or Python caches."""

import argparse
import hashlib
import io
import json
import os
import tarfile
import tempfile
from pathlib import Path


def exclusion_reason(relative, root):
    if relative.as_posix() == "archive_contents_manifest.json":
        return "REGENERATED_ARCHIVE_MANIFEST"
    if any(part in {"__pycache__", ".pytest_cache"} for part in relative.parts):
        return "REBUILDABLE_CACHE"
    if relative.suffix in {".pyc", ".pyo"}:
        return "REBUILDABLE_CACHE"
    for suffix in (".tar.gz", ".tgz", ".tar"):
        if relative.name.endswith(suffix):
            directory = root / relative.parent / relative.name[:-len(suffix)]
            if directory.is_dir() and any(
                (directory / marker).is_file()
                for marker in ("pilot_summary.json", "preregistration.json")
            ):
                return "NESTED_ARCHIVE_COPY"
    return None


def pack(root, output):
    """Keep raw features, responses, journals, receipts, configs and failure logs."""
    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir() or output.is_relative_to(root):
        raise ValueError("archive needs a directory and an output outside its tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    included, omitted = [], []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"evidence symlink must be resolved explicitly: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"unexpected special evidence file: {path}")
        relative = path.relative_to(root)
        record = {"path": relative.as_posix(), "bytes": path.stat().st_size}
        reason = exclusion_reason(relative, root)
        if reason:
            omitted.append({**record, "reason": reason})
        else:
            included.append(record)
    manifest = {
        "policy": "SINGLE_EVIDENCE_ARCHIVE_NO_NESTED_COPIES_V1",
        "root": root.name, "included": included, "omitted": omitted,
        "included_bytes": sum(row["bytes"] for row in included),
        "omitted_bytes": sum(row["bytes"] for row in omitted),
        "predictor_features_preserved": True,
    }
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=output.name + ".", suffix=".partial", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with tarfile.open(temporary, "w:gz", compresslevel=9) as archive:
            for row in included:
                archive.add(root / row["path"],
                            arcname=f"{root.name}/{row['path']}", recursive=False)
            payload = json.dumps(manifest, indent=2).encode()
            member = tarfile.TarInfo(f"{root.name}/archive_contents_manifest.json")
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    with output.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "archive": str(output), "sha256": digest,
        "archive_bytes": output.stat().st_size,
        "included_files": len(included), "omitted_files": len(omitted),
        "included_bytes": manifest["included_bytes"],
        "omitted_bytes": manifest["omitted_bytes"], "policy": manifest["policy"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(pack(args.root, args.output)), flush=True)


if __name__ == "__main__":
    main()
