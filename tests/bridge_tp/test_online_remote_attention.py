# SPDX-License-Identifier: Apache-2.0

import math
import socket
import unittest
from unittest.mock import Mock

try:
    import torch

    from vllm.bridge_tp.online_remote_attention import (
        _configure_low_latency_socket,
        RemoteAttentionClient,
        attention_stats,
        gather_paged_kv,
        merge_attention_stats,
    )

    TORCH_AVAILABLE = True
except ModuleNotFoundError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is unavailable")
class TestOnlineRemoteAttention(unittest.TestCase):
    def test_remote_attention_socket_disables_nagle(self) -> None:
        connection = Mock()
        _configure_low_latency_socket(connection)
        connection.setsockopt.assert_called_once_with(
            socket.IPPROTO_TCP,
            socket.TCP_NODELAY,
            1,
        )

    def test_remote_activation_is_latched_for_all_layers_of_a_token(self) -> None:
        client = RemoteAttentionClient.__new__(RemoteAttentionClient)
        client._activation_key = None
        client._remote_enabled_for_activation_key = False

        self.assertFalse(client.remote_enabled_for_forward("anchor", 100, False))
        self.assertFalse(client.remote_enabled_for_forward("anchor", 100, True))
        self.assertTrue(client.remote_enabled_for_forward("anchor", 101, True))
        self.assertTrue(client.remote_enabled_for_forward("anchor", 101, False))

    def test_gathers_physical_blocks_in_logical_order(self) -> None:
        cache = torch.zeros(5, 2, 2, 1, 1)
        for block in range(5):
            for offset in range(2):
                cache[block, 0, offset, 0, 0] = block * 10 + offset
                cache[block, 1, offset, 0, 0] = 100 + block * 10 + offset
        key, value = gather_paged_kv(cache, [3, 1], 1, 4)
        self.assertEqual(key.flatten().tolist(), [31.0, 10.0, 11.0])
        self.assertEqual(value.flatten().tolist(), [131.0, 110.0, 111.0])

    def test_partition_merge_matches_full_attention(self) -> None:
        generator = torch.Generator().manual_seed(7)
        query = torch.randn(4, 8, generator=generator)
        key = torch.randn(2, 11, 8, generator=generator)
        value = torch.randn(2, 11, 8, generator=generator)
        scale = 1 / math.sqrt(8)
        left = attention_stats(query, key[:, :6], value[:, :6], scale)
        right = attention_stats(query, key[:, 6:], value[:, 6:], scale)
        merged = merge_attention_stats(right, left)
        full_stats = attention_stats(query, key, value, scale)
        expected = full_stats[2] / full_stats[1].unsqueeze(-1)
        torch.testing.assert_close(merged, expected, rtol=1e-5, atol=1e-6)

    def test_empty_partition_merges_safely(self) -> None:
        query = torch.ones(2, 4)
        empty = torch.empty(1, 0, 4)
        key = torch.ones(1, 1, 4)
        value = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
        merged = merge_attention_stats(
            attention_stats(query, empty, empty, 0.5),
            attention_stats(query, key, value, 0.5),
        )
        torch.testing.assert_close(merged, value.expand(2, -1, -1).squeeze(1))


if __name__ == "__main__":
    unittest.main()
