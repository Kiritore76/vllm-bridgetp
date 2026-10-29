# SPDX-License-Identifier: Apache-2.0
"""CPU-side ownership and polling helpers for persistent migration channels."""

import time


def publish_prebound_receiver(connector, receiver):
    """Publish the same receiver to both owners, including the preconnect path."""
    with (
        connector._prebound_gpu_receiver_lock,
        connector._persistent_gpu_receiver_lock,
    ):
        if connector._prebound_gpu_receiver not in (None, receiver):
            raise RuntimeError("independent prebound receiver already exists")
        if connector.persistent_channel:
            if connector._persistent_gpu_receiver not in (None, receiver):
                raise RuntimeError("independent persistent receiver already exists")
            connector._persistent_gpu_receiver = receiver
        connector._prebound_gpu_receiver = receiver


class ManifestPollCache:
    """Cache an atomically replaced manifest, never its changing control state.

    Checking file identity preserves the existing polling interval for new
    sessions without decoding old token arrays on every idle poll. Failed reads
    are not cached. Counters are in memory; no per-poll diagnostic writes.
    """

    def __init__(self):
        self.signature = None
        self.value = None
        self.polls = 0
        self.parses = 0
        self.parse_wall_s = 0.0
        self.parse_thread_cpu_s = 0.0

    def load(self, path, loader):
        self.polls += 1
        stat = path.stat()
        signature = (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
        if signature != self.signature:
            wall = time.perf_counter()
            cpu = time.thread_time()
            try:
                value = loader(path)
            finally:
                self.parse_wall_s += time.perf_counter() - wall
                self.parse_thread_cpu_s += time.thread_time() - cpu
                self.parses += 1
            self.value = value
            self.signature = signature
        return self.value

    def snapshot(self):
        return dict(
            polls=self.polls,
            parses=self.parses,
            parse_wall_s=self.parse_wall_s,
            parse_thread_cpu_s=self.parse_thread_cpu_s,
        )


def receiver_snapshot(connector):
    """Return explicit refusal evidence without querying or synchronizing CUDA."""
    receiver = connector._persistent_gpu_receiver
    prebound = getattr(connector, "_prebound_gpu_receiver", None)
    lifecycle = getattr(receiver, "lifecycle", None)
    cache = getattr(connector, "_prebind_manifest_cache", None)
    return dict(
        persistent_receiver_present=receiver is not None,
        prebound_receiver_present=prebound is not None,
        same_receiver=receiver is not None and receiver is prebound,
        lifecycle_state=getattr(getattr(lifecycle, "state", None), "value", None),
        active_session=getattr(lifecycle, "active_session", None) is not None,
        communicator_present=getattr(receiver, "comm", None) is not None,
        live_load_threads=sum(t.is_alive() for t in connector._load_threads.values()),
        manifest_polling=cache.snapshot() if cache is not None else None,
    )


def worker_state_snapshot(connector):
    """Observe retained worker state and stream readiness without waiting on CUDA."""
    names = (
        "_load_threads",
        "_copy_done_events",
        "_model_wait_streams",
        "_model_wait_pending_requests",
        "_gpu_restore_quiescence",
        "_completed_recvs",
        "_reported_recvs",
        "_finished_worker_requests",
    )
    receiver = connector._persistent_gpu_receiver
    streams = dict(
        migration_receive=getattr(receiver, "stream", None),
        restore=getattr(connector, "_persistent_restore_stream", None),
    )
    ready = {}
    for name, stream in streams.items():
        try:
            ready[name] = None if stream is None else bool(stream.query())
        except RuntimeError as error:
            ready[name] = dict(error=str(error))
    return dict(
        counts={
            name.removeprefix("_"): len(getattr(connector, name, ())) for name in names
        },
        receiver_payload_present=getattr(receiver, "_packed_buffer", None) is not None,
        receiver_payload_capacity_elements=getattr(
            receiver, "buffer_capacity_elements", 0
        ),
        stream_ready=ready,
    )
