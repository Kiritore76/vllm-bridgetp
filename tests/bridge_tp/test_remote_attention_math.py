# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - local documentation hosts
    torch = None

if torch is not None:
    from tools.bridge_tp.run_remote_attention_validation import (
        attention_stats,
        full_attention,
        load_tensor_bundle,
        merge_stats,
        percentile,
    )
    from vllm.bridge_tp.remote_attention_protocol import (
        RemoteAttentionGeometry,
    )


@unittest.skipIf(torch is None, "torch is required")
class TestRemoteAttentionMath(unittest.TestCase):
    def tensors(self, tokens: int = 64):
        generator = torch.Generator().manual_seed(7)
        q = torch.randn(40, 128, generator=generator, dtype=torch.bfloat16)
        k = torch.randn(8, tokens, 128, generator=generator, dtype=torch.bfloat16)
        v = torch.randn(k.shape, generator=generator, dtype=torch.bfloat16)
        return q, k, v

    def test_split_merge_matches_full_gqa_attention(self) -> None:
        q, k, v = self.tensors()
        cut = 32
        local = attention_stats(q, k[:, :cut], v[:, :cut])
        remote = attention_stats(q, k[:, cut:], v[:, cut:])
        merged = merge_stats(local, remote)
        reference = full_attention(q, k, v)
        self.assertTrue(torch.isfinite(merged).all())
        self.assertLess((merged - reference).abs().max().item(), 1e-5)

    def test_empty_local_partition_matches_full_remote(self) -> None:
        q, k, v = self.tensors()
        local = attention_stats(q, k[:, :0], v[:, :0])
        remote = attention_stats(q, k, v)
        merged = merge_stats(local, remote)
        reference = full_attention(q, k, v)
        self.assertTrue(torch.equal(merged, reference))

    def test_head_shards_reassemble_remote_statistics(self) -> None:
        q, k, v = self.tensors()
        maximum = []
        denominator = []
        numerator = []
        for rank in range(4):
            q_slice = slice(rank * 10, (rank + 1) * 10)
            kv_slice = slice(rank * 2, (rank + 1) * 2)
            stats = attention_stats(q[q_slice], k[kv_slice], v[kv_slice])
            maximum.append(stats[0])
            denominator.append(stats[1])
            numerator.append(stats[2])
        sharded = (
            torch.cat(maximum),
            torch.cat(denominator),
            torch.cat(numerator),
        )
        unsharded = attention_stats(q, k, v)
        for actual, expected in zip(sharded, unsharded, strict=True):
            self.assertTrue(torch.equal(actual, expected))

    def test_percentile_uses_observed_higher_value(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0, 4.0], 0.95), 4.0)

    def test_captured_tensor_bundle_contract(self) -> None:
        q, k, v = self.tensors()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "captured.pt"
            torch.save({"q": q, "k": k, "v": v}, path)
            loaded = load_tensor_bundle(path, RemoteAttentionGeometry())
        self.assertEqual(tuple(loaded[0].shape), (40, 128))
        self.assertEqual(tuple(loaded[1].shape), (8, 64, 128))

    def test_captured_tensor_bundle_rejects_wrong_q_shape(self) -> None:
        q, k, v = self.tensors()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "captured.pt"
            torch.save({"q": q[:39], "k": k, "v": v}, path)
            with self.assertRaisesRegex(ValueError, "invalid shape"):
                load_tensor_bundle(path, RemoteAttentionGeometry())


if __name__ == "__main__":
    unittest.main()
