# SPDX-License-Identifier: Apache-2.0
"""CPU checks for six-card port selection and conflict snapshots."""

import errno
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.bridge_tp.pressure_cpu import load

ports = load("tcp_ports")


class PortTests(unittest.TestCase):
    def state(self, ephemeral=None):
        return dict(next_port=18000, port_policy=dict(ephemeral_range=ephemeral))

    def test_kernel_range_is_read_and_invalid_linux_evidence_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = Path(folder)
            path = proc / "sys/net/ipv4/ip_local_port_range"
            path.parent.mkdir(parents=True)
            with patch.object(ports.sys, "platform", "linux"):
                with self.assertRaises(RuntimeError):
                    ports.ephemeral_range(proc)
                path.write_text("32768 60999\n")
                self.assertEqual(ports.ephemeral_range(proc), [32768, 60999])
                path.write_text("bad range")
                with self.assertRaises(RuntimeError):
                    ports.ephemeral_range(proc)

    def test_skip_ephemeral_and_occupied_blocks_without_reusing_prior_blocks(self):
        state = self.state([18000, 18019])
        probes = []

        def probe(first, width):
            probes.append(first)
            return (first != 18020, dict(port=first, error="occupied"))

        self.assertEqual(ports.take(state, probe=probe), 18030)
        self.assertEqual(ports.take(state, probe=probe), 18040)
        self.assertEqual(probes, [18020, 18030, 18040])
        self.assertEqual(state["port_probe_failures"][0]["port"], 18020)

    def test_no_non_ephemeral_block_is_a_clear_error_not_a_fallback(self):
        with self.assertRaisesRegex(ValueError, "outside OS ephemeral"):
            ports.take(
                self.state([1024, 65535]), probe=lambda *a: self.fail("must not probe")
            )
        with self.assertRaises(ValueError):
            ports.take(self.state(), 0)

    def test_reserved_ranges_are_not_consumed_even_if_probe_would_pass(self):
        state = self.state()
        state["port_policy"]["reserved_ports"] = "18000-18009,18010"
        self.assertEqual(ports.take(state, probe=lambda *args: (True, None)), 18011)
        state["port_policy"]["reserved_ports"] = "18020-18010"
        with self.assertRaises(ValueError):
            ports.take(state, probe=lambda *args: (True, None))

    def test_real_tcp_probe_detects_an_occupied_port(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
            held.bind(("127.0.0.1", 0))
            held.listen(1)
            port = held.getsockname()[1]
            available, failure = ports.probe_block(port, 1)
            self.assertFalse(available)
            self.assertEqual(failure["port"], port)
            self.assertEqual(failure["protocol"], "tcp")

    def test_probe_failure_closes_all_sockets_and_preserves_unexpected_errors(self):
        created = []

        class Probe:
            def __init__(self, *args):
                self.closed = False
                created.append(self)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.closed = True

            def setsockopt(self, *args):
                pass

            def bind(self, address):
                if address[1] == 18001:
                    raise OSError(errno.EADDRINUSE, "occupied")

        with patch.object(ports.socket, "socket", Probe):
            self.assertFalse(ports.probe_block(18000, 2)[0])
        self.assertEqual(len(created), 3)
        self.assertTrue(all(s.closed for s in created))
        with (
            patch.object(
                ports.socket, "socket", side_effect=OSError(errno.EMFILE, "fds")
            ),
            self.assertRaises(OSError),
        ):
            ports.probe_block(18000, 1)

    def test_snapshot_captures_time_wait_without_inventing_a_pid(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = Path(folder)
            (proc / "net").mkdir()
            (proc / "net/tcp").write_text(
                "header\n0: 0100007F:852A 00000000:0000 06 0 0 0 1000 0 0\n"
            )
            result = ports.bind_snapshot(34090, proc)
            self.assertEqual(result["sockets"][0]["state"], "06")
            self.assertEqual(result["sockets"][0]["inode"], "0")
            self.assertEqual(result["owners"], [])

    def test_snapshot_maps_socket_inode_to_a_bounded_pid_scan(self):
        with tempfile.TemporaryDirectory() as folder:
            proc = Path(folder)
            (proc / "net").mkdir()
            (proc / "net/tcp").write_text(
                "header\n0: 0100007F:852A 00000000:0000 0A 0 0 0 1000 0 123\n"
            )
            fd = proc / "42/fd"
            fd.mkdir(parents=True)
            (fd / "4").touch()
            with patch.object(ports.os, "readlink", return_value="socket:[123]"):
                result = ports.bind_snapshot(34090, proc, max_fds=1)
            self.assertEqual(result["owners"], [dict(pid=42, fd="4", inode="123")])
            self.assertTrue(result["fd_scan_limit_reached"])
