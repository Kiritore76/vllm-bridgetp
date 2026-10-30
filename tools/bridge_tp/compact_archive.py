# SPDX-License-Identifier: Apache-2.0
"""Lossless evidence transport; original logical files are hash-verified on restore."""

import collections
import hashlib
import io
import json
import lzma
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

PREFIX = ".__bridgetp_compact__"
POLICY = "SINGLE_EVIDENCE_ARCHIVE_LOSSLESS_COMPACT_V2"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def safe_path(root, relative):
    name = PurePosixPath(relative)
    if (
        not relative
        or name.is_absolute()
        or ".." in name.parts
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError("unsafe evidence path")
    path = root.joinpath(*name.parts)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("evidence path escaped root")
    return path


def audit_subset(full, durable):
    """Only deduplicate when sorted serialization reproduces every byte."""
    indices = collections.defaultdict(collections.deque)
    try:
        with full.open("rb") as source:
            for i, line in enumerate(source):
                row = json.loads(line)
                canonical = json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
                indices[hashlib.sha256(canonical).digest()].append(i)
        selected = []
        with durable.open("rb") as source:
            for line in source:
                row = json.loads(line)
                canonical = json.dumps(row, ensure_ascii=False, sort_keys=True).encode()
                key = hashlib.sha256(canonical).digest()
                if line != canonical + b"\n" or not indices[key]:
                    return None
                selected.append(indices[key].popleft())
        return selected
    except (ValueError, OSError):
        return None  # Torn/unique failure journals are retained in full.


def pack_compact(root, output):
    from tools.bridge_tp.result_archive import exclusion_reason

    root, output = Path(root).resolve(), Path(output).resolve()
    if not root.is_dir() or output.is_relative_to(root) or (root / PREFIX).exists():
        raise ValueError("compact output must be outside an unencoded evidence tree")
    output.parent.mkdir(parents=True, exist_ok=True)
    included, omitted, stored = [], [], []
    with tempfile.TemporaryDirectory(dir=output.parent, prefix="compact-") as temp:
        temporary_root = Path(temp)
        archive_path = temporary_root / "evidence.tar.gz"
        with tarfile.open(archive_path, "w:gz", compresslevel=1) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise ValueError("evidence symlink must be resolved explicitly")
                if path.is_dir():
                    continue
                if not path.is_file():
                    raise ValueError("special evidence file")
                relative = path.relative_to(root)
                row = dict(path=relative.as_posix(), bytes=path.stat().st_size)
                reason = exclusion_reason(relative, root)
                if reason:
                    omitted.append(dict(row, reason=reason))
                    continue
                row["sha256"] = digest(path)
                included.append(row)
                transform = None
                if path.name == "pressure_audit_durable.jsonl":
                    full = path.with_name("pressure_audit.jsonl")
                    selected = audit_subset(full, path) if full.is_file() else None
                    if selected is not None:
                        payload = json.dumps(selected).encode()
                        member = f"{PREFIX}/{relative.as_posix()}.index.json"
                        info = tarfile.TarInfo(f"{root.name}/{member}")
                        info.size = len(payload)
                        archive.addfile(info, io.BytesIO(payload))
                        transform = dict(
                            encoding="SORTED_JSON_AUDIT_SUBSET_V1",
                            source=full.relative_to(root).as_posix(),
                            member=member,
                        )
                if (
                    transform is None
                    and row["bytes"] >= 1_000_000
                    and (path.suffix in {".json", ".jsonl"})
                ):
                    compressed = temporary_root / "payload.xz"
                    with (
                        path.open("rb") as src,
                        lzma.open(compressed, "wb", preset=6) as dst,
                    ):
                        shutil.copyfileobj(src, dst, length=1024 * 1024)
                    member = f"{PREFIX}/{relative.as_posix()}.xz"
                    archive.add(
                        compressed, arcname=f"{root.name}/{member}", recursive=False
                    )
                    transform = dict(encoding="XZ_V1", member=member)
                if transform is None:
                    archive.add(
                        path,
                        arcname=f"{root.name}/{relative.as_posix()}",
                        recursive=False,
                    )
                else:
                    stored.append(dict(path=relative.as_posix(), **transform))
                if digest(path) != row["sha256"]:
                    raise ValueError("evidence changed during packing")
            manifest = dict(
                policy=POLICY,
                root=root.name,
                included=included,
                omitted=omitted,
                included_bytes=sum(r["bytes"] for r in included),
                omitted_bytes=sum(r["bytes"] for r in omitted),
                transforms=stored,
                predictor_features_preserved=True,
                restore_command="python -m tools.bridge_tp.compact_archive "
                "--archive ARCHIVE --output EMPTY_DIRECTORY",
            )
            payload = json.dumps(manifest, indent=2).encode()
            info = tarfile.TarInfo(f"{root.name}/archive_contents_manifest.json")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
        archive_sha = digest(archive_path)
        size = archive_path.stat().st_size
        os.replace(archive_path, output)
    return dict(
        archive=str(output),
        sha256=archive_sha,
        archive_bytes=size,
        included_files=len(included),
        omitted_files=len(omitted),
        included_bytes=manifest["included_bytes"],
        omitted_bytes=manifest["omitted_bytes"],
        policy=POLICY,
        transformed_files=len(stored),
    )


def restore(archive, output):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("restore requires an empty directory")
    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        members = tar.getmembers()
        roots = set()
        for member in members:
            safe_path(output, member.name)
            if not member.isfile() and not member.isdir():
                raise ValueError("links and special archive members are forbidden")
            roots.add(PurePosixPath(member.name).parts[0])
        if len(roots) != 1:
            raise ValueError("archive must have one evidence root")
        tar.extractall(output, filter="data")
    root = safe_path(output, roots.pop())
    manifest_path = root / "archive_contents_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("root") != root.name:
        raise ValueError("manifest root differs")
    records = {r["path"]: r for r in manifest["included"]}
    transforms = manifest.get("transforms", [])
    for transform in sorted(transforms, key=lambda r: r["encoding"] != "XZ_V1"):
        path = safe_path(root, transform["path"])
        encoded = safe_path(root, transform["member"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if transform["encoding"] == "XZ_V1":
            size = 0
            with lzma.open(encoded, "rb") as src, path.open("wb") as dst:
                while chunk := src.read(1024 * 1024):
                    size += len(chunk)
                    if size > records[transform["path"]]["bytes"]:
                        raise ValueError("restored size exceeds manifest")
                    dst.write(chunk)
        elif transform["encoding"] == "SORTED_JSON_AUDIT_SUBSET_V1":
            selected = json.loads(encoded.read_bytes())
            wanted = set(selected)
            found = {}
            with safe_path(root, transform["source"]).open("rb") as src:
                for i, line in enumerate(src):
                    if i in wanted:
                        found[i] = (
                            json.dumps(
                                json.loads(line), ensure_ascii=False, sort_keys=True
                            ).encode()
                            + b"\n"
                        )
            with path.open("wb") as dst:
                for index in selected:
                    dst.write(found[index])
        else:
            raise ValueError("unknown evidence encoding")
    for row in manifest["included"]:
        path = safe_path(root, row["path"])
        if path.stat().st_size != row["bytes"]:
            raise ValueError(f"restored size differs: {row['path']}")
        if "sha256" in row and digest(path) != row["sha256"]:
            raise ValueError(f"restored hash differs: {row['path']}")
    # No recursive deletion: unlink only the encoded files listed in manifest.
    for transform in transforms:
        encoded = safe_path(root, transform["member"])
        if not encoded.is_relative_to(root / PREFIX):
            raise ValueError("encoded member outside compact directory")
        encoded.unlink()
    for directory in sorted((root / PREFIX).rglob("*"), reverse=True):
        if directory.is_dir():
            directory.rmdir()
    if (root / PREFIX).exists():
        (root / PREFIX).rmdir()
    return dict(
        status="RESTORED_AND_HASH_VERIFIED",
        root=str(root),
        files=len(manifest["included"]),
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(restore(args.archive, args.output)))


if __name__ == "__main__":
    main()
