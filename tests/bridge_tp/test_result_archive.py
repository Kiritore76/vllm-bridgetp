# SPDX-License-Identifier: Apache-2.0
"""Compact packing retains fitting/replay/failure evidence unchanged."""

import hashlib
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.result_archive import pack


class ResultArchiveTests(unittest.TestCase):
    def test_only_archive_copies_and_caches_are_removed(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "batch"
            retained = {
                "pilot/run/predictor_features/feature.pt": b"feature tensor",
                "pilot/run/receipt.json": b'{"dropped":0}',
                "pilot/run/failed.log": b"failed rank diagnostic",
                "pilot/preregistration.json": b"canonical stage evidence",
                "pilot/pair/pilot_summary.json": b"canonical pair evidence",
                "runtime_source.tar.gz": b"unique source package",
                "inputs/requests.jsonl.gz": b"compressed requests",
            }
            removed = {"archive_contents_manifest.json": b"old manifest",
                       "pilot.tar.gz": b"duplicate", "pilot/pair.tar.gz": b"copy",
                       "runtime/__pycache__/module.pyc": b"cache"}
            for name, payload in {**retained, **removed}.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            output = base / "batch.tar.gz"
            result = pack(root, output)
            with tarfile.open(output) as archive:
                for name, payload in retained.items():
                    self.assertEqual(archive.extractfile(f"batch/{name}").read(),
                                     payload)
                self.assertFalse(any(f"batch/{name}" in archive.getnames()
                                     for name in removed
                                     if name != "archive_contents_manifest.json"))
                self.assertEqual(archive.getnames().count(
                    "batch/archive_contents_manifest.json"), 1)
                manifest = json.load(archive.extractfile(
                    "batch/archive_contents_manifest.json"))
            self.assertEqual(result["omitted_files"], 4)
            self.assertEqual(len(manifest["included"]), len(retained))
            self.assertEqual(result["sha256"],
                             hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertTrue(all((root / name).exists() for name in removed))

    def test_output_inside_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with self.assertRaisesRegex(ValueError, "outside its tree"):
                pack(root, root / "recursive.tar.gz")

    def test_failed_pack_preserves_previous_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            root = base / "batch"
            root.mkdir()
            (root / "receipt.json").write_text("evidence")
            output = base / "batch.tar.gz"
            output.write_bytes(b"previous complete archive")
            with (
                patch.object(tarfile.TarFile, "add", side_effect=OSError("disk")),
                self.assertRaises(OSError),
            ):
                pack(root, output)
            self.assertEqual(output.read_bytes(), b"previous complete archive")
            self.assertFalse(list(base.glob("*.partial")))
