#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run paired Shadow-policy or Bridge/Shadow-only live TP1/TP4 experiments.

This experiment uses the real Phase 8 KV export, TCP staging, TP4 restore,
atomic takeover, and unified response proxy.  It measures target-request TPOT
in paired pre-Shadow/Shadow/Bridge/post-commit windows.  The current online
decode path still waits for complete history before TP4 continuation, so this
runner does not claim online remote-attention execution.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tools.bridge_tp import run_phase9_cap0_calibration as common  # noqa: E402
from tools.bridge_tp import run_phase9_cap0_noop as scenario_runner  # noqa: E402
from tools.bridge_tp import run_phase9_cap0_rescue as rescue  # noqa: E402
from tools.bridge_tp.run_phase9_capacity_background import (  # noqa: E402
    load_manifest,
)
from vllm.bridge_tp.online_shadow_strategy_protocol import (  # noqa: E402
    percentile,
    summarize_background_windows,
    validate_strategy_timing,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--survival-table", type=Path, required=True)
    parser.add_argument("--guard-file", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-survival-sha256", required=True)
    parser.add_argument("--expected-guard-sha256", required=True)
    parser.add_argument("--expected-guard", type=int, required=True)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--tp1-blocks", type=int, required=True)
    parser.add_argument("--tp4-blocks", type=int, required=True)
    parser.add_argument("--phase", choices=["smoke", "formal"], default="smoke")
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--managed-formal-subrun",
        action="store_true",
        help=(
            "allow one formal repetition when a parent matrix runner owns "
            "interleaving, repetition counting, and final acceptance"
        ),
    )
    parser.add_argument("--strategy-order", nargs=2, default=["S_NEW", "S_NEW_OLD"])
    parser.add_argument(
        "--bridge-only",
        action="store_true",
        help="run only the original S_NEW Bridge system on this branch",
    )
    parser.add_argument(
        "--architecture-comparison",
        action="store_true",
        help=(
            "pair the original Bridge path (S_NEW) with direct Shadow-only "
            "takeover (S_NEW_OLD) instead of comparing copy policies alone"
        ),
    )
    parser.add_argument(
        "--shadow-only-only",
        action="store_true",
        help="run only the Shadow-only system on this branch",
    )
    parser.add_argument(
        "--gpu-resident-shadow",
        action="store_true",
        help=(
            "reserve TP4 blocks at Shadow start and inject history/deltas "
            "before direct takeover"
        ),
    )
    parser.add_argument(
        "--gpu-direct-history",
        action="store_true",
        help="move initial Shadow history by NCCL without a CPU tensor relay",
    )
    parser.add_argument("--gpu-direct-base-port", type=int, default=30400)
    parser.add_argument(
        "--gpu-direct-history-pacing",
        action="store_true",
        help="pace 16 MiB aggregate NCCL history chunks with background sleeps",
    )
    parser.add_argument(
        "--gpu-direct-delta",
        action="store_true",
        help="stream batched Shadow deltas over the retained NCCL session",
    )
    parser.add_argument("--gpu-direct-delta-batch-tokens", type=int, default=16)
    parser.add_argument("--gpu-direct-delta-flush-ms", type=float, default=25.0)
    parser.add_argument(
        "--persistent-channel",
        action="store_true",
        help=(
            "reuse one GPU-direct communicator across migration sessions; "
            "C0 initially permits only one active session"
        ),
    )
    parser.add_argument("--channel-generation", type=int, default=0)
    parser.add_argument(
        "--preconnect-persistent-channel",
        action="store_true",
        help=(
            "open and warm the TP1-to-TP4 persistent NCCL links during "
            "service startup before the anchor request begins"
        ),
    )
    parser.add_argument(
        "--persistent-sequential-reuse",
        action="store_true",
        help=(
            "keep one TP1/TP4 server pair alive across all repetitions so "
            "C0 measures real cross-request communicator reuse"
        ),
    )
    parser.add_argument(
        "--persistent-session-gap-s",
        type=float,
        default=0.0,
        help="delay after each session has returned to IDLE",
    )
    parser.add_argument(
        "--ready-sync-mode",
        choices=["DEVICE_WIDE", "STREAM_EVENT"],
        default="DEVICE_WIDE",
        help="select the old whole-device or P0 stream/event target-ready path",
    )
    parser.add_argument(
        "--ready-sync-comparison",
        action="store_true",
        help=(
            "interleave DEVICE_WIDE and STREAM_EVENT Shadow-only runs within "
            "each repetition"
        ),
    )
    parser.add_argument(
        "--ready-notification-mode",
        choices=["FILE_POLL", "UDP"],
        default="FILE_POLL",
        help="P1 controller wake-up path",
    )
    parser.add_argument(
        "--ready-notification-comparison",
        action="store_true",
        help="interleave FILE_POLL and UDP ready notification runs",
    )
    parser.add_argument("--ready-notification-host", default="127.0.0.1")
    parser.add_argument("--ready-notification-port", type=int, default=30500)
    parser.add_argument("--ready-latch-poll-ms", type=float, default=5.0)
    parser.add_argument(
        "--deferred-comm-destroy",
        action="store_true",
        help=(
            "retain source and target GPU-direct NCCL communicators until "
            "their worker processes shut down"
        ),
    )
    parser.add_argument(
        "--deferred-comm-destroy-comparison",
        action="store_true",
        help=(
            "interleave synchronous communicator destruction and the "
            "process-lifetime communicator pool within every repetition"
        ),
    )
    parser.add_argument(
        "--post-takeover-comm-destroy",
        action="store_true",
        help=(
            "experiment-only: keep terminal-close off the ready path, then "
            "destroy target communicators asynchronously after ownership commit"
        ),
    )
    parser.add_argument(
        "--post-takeover-destroy-comparison",
        action="store_true",
        help=(
            "interleave process-lifetime pooling and post-takeover async "
            "target communicator destruction within every repetition"
        ),
    )
    parser.add_argument(
        "--stop-and-copy-only",
        action="store_true",
        help=(
            "run the request-level frozen full-KV Stop-and-Copy baseline; "
            "only the selected anchor is stopped"
        ),
    )
    parser.add_argument(
        "--online-remote-attention",
        action="store_true",
        help="execute TP4-prefix attention in the live Bridge token data path",
    )
    parser.add_argument("--remote-attention-base-port", type=int, default=30200)
    parser.add_argument("--slo-tpot-ms", type=float, default=50.0)
    parser.add_argument("--slo-ttft-ms", type=float, default=1000.0)
    parser.add_argument("--slo-e2e-ms", type=float, default=60000.0)
    parser.add_argument("--slo-handoff-ms", type=float, default=1000.0)
    parser.add_argument("--trigger-output-tokens", type=int, default=128)
    parser.add_argument(
        "--bridge-output-tokens",
        type=int,
        default=None,
        help="SHADOW-to-BRIDGE boundary; default is the window midpoint",
    )
    parser.add_argument("--cutover-output-tokens", type=int, default=160)
    parser.add_argument(
        "--commit-timing",
        choices=("FIXED", "EARLIEST_READY"),
        default="FIXED",
        help=(
            "FIXED freezes at --cutover-output-tokens. EARLIEST_READY waits "
            "for exact initial history GPU residency on all TP4 ranks, then "
            "publishes the first safe dynamic cutover."
        ),
    )
    parser.add_argument(
        "--manager-m0-shadow",
        action="store_true",
        help="record M0 advisory decisions in each controller audit",
    )
    parser.add_argument(
        "--manager-m1-auto-start",
        action="store_true",
        help="let M1 select Shadow start from live evidence",
    )
    parser.add_argument(
        "--m1-min-output-tokens",
        type=int,
        default=None,
        help="experimental earliest M1 Shadow start; M1 safety gates still apply",
    )
    parser.add_argument(
        "--m1-source-release-tail-s", type=float,
        help="diagnostic tail allowance after estimated history copy until TP1 KV release",
    )
    parser.add_argument("--manager-m2-rate", action="store_true")
    parser.add_argument("--manager-m3-commit", action="store_true")
    parser.add_argument("--manager-m4-cancel", action="store_true")
    parser.add_argument("--manager-m5-predictor-shadow", action="store_true")
    parser.add_argument(
        "--paired-stay", action="store_true",
        help="paired GoodOutput control arm: retain M1/M5 observation but suppress migration",
    )
    parser.add_argument("--predictor-checkpoint", type=Path)
    parser.add_argument("--predictor-checkpoint-sha256")
    parser.add_argument("--manager-m4-expect-cancel", action="store_true")
    parser.add_argument("--manager-m2-force-initial-high", action="store_true")
    parser.add_argument("--manager-m2-require-source-high", action="store_true")
    parser.add_argument("--manager-m2-require-low-to-high", action="store_true")
    parser.add_argument("--m2-low-gib-s", type=float)
    parser.add_argument("--m2-medium-gib-s", type=float)
    parser.add_argument("--m2-high-gib-s", type=float)
    parser.add_argument(
        "--manager-m2-expected-profile",
        choices=("LOW", "MEDIUM", "HIGH"),
    )
    parser.add_argument(
        "--manager-m2-min-history-byte-frac", type=float, default=0.0
    )
    parser.add_argument(
        "--manager-m1-expect-stay",
        action="store_true",
        help="accept an M1 request completed entirely on TP1",
    )
    parser.add_argument(
        "--manager-m1-stay-reason",
        choices=("short-budget", "target-load"),
        default="short-budget",
        help="the specific M1 refusal that the STAY smoke must observe",
    )
    parser.add_argument("--anchor-max-tokens", type=int, default=1024)
    parser.add_argument("--anchor-prompt-tokens", type=int, default=None)
    parser.add_argument("--anchor-request-file", type=Path)
    parser.add_argument("--expected-anchor-request-sha256")
    parser.add_argument("--natural-eos-anchor", action="store_true")
    parser.add_argument("--minimum-ready-target-jobs", type=int, default=2)
    parser.add_argument("--minimum-ready-source-jobs", type=int, default=0)
    parser.add_argument("--background-lead-s", type=float, default=2.0)
    parser.add_argument("--minimum-window-samples", type=int, default=4)
    parser.add_argument(
        "--source-pressure",
        action="store_true",
        help="allow real TP1 peer jobs in the background manifest for A4-P",
    )
    parser.add_argument(
        "--minimum-source-kv-usage-frac",
        type=float,
        default=0.0,
        help="require this observed TP1 KV usage peak for A4-P",
    )
    parser.add_argument(
        "--fixed-rate-gib-s",
        type=float,
        default=None,
        help=(
            "Pin aggregate migration bandwidth to this value. Zero means "
            "unlimited. Omit to retain the adaptive Phase 9 rate controller."
        ),
    )
    parser.add_argument("--tp1-gpu", default="0")
    parser.add_argument("--tp4-gpus", default="1,2,3,4")
    parser.add_argument("--tp4-max-num-seqs", type=int, default=None)
    parser.add_argument("--tp1-port", type=int, default=8001)
    parser.add_argument("--tp4-port", type=int, default=8200)
    parser.add_argument("--snapshot-port", type=int, default=29800)
    parser.add_argument("--delta-port", type=int, default=29900)
    parser.add_argument("--delivery-port", type=int, default=30000)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    parser.add_argument("--server-start-timeout-s", type=float, default=900)
    parser.add_argument("--run-timeout-s", type=float, default=2400)
    parser.add_argument("--stager-timeout-s", type=float, default=1800)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def validate_inputs(args: argparse.Namespace) -> tuple[str, int, dict[str, Any]]:
    if os.name == "nt":
        raise RuntimeError("online Shadow validation requires Linux and five GPUs")
    if (
        args.phase == "formal"
        and args.repetitions < 3
        and not args.managed_formal_subrun
    ):
        raise ValueError("formal online validation requires at least three runs")
    if args.managed_formal_subrun and (
        args.phase != "formal" or args.repetitions != 1
    ):
        raise ValueError(
            "managed formal subruns require --phase formal --repetitions 1"
        )
    if args.repetitions <= 0:
        raise ValueError("repetitions must be positive")
    selected_modes = sum(
        bool(value)
        for value in (
            args.bridge_only,
            args.architecture_comparison,
            args.shadow_only_only,
            args.stop_and_copy_only,
        )
    )
    if selected_modes > 1:
        raise ValueError(
            "select at most one of Bridge-only, architecture comparison, "
            "Shadow-only, or Stop-and-Copy"
        )
    if args.online_remote_attention and not (
        args.bridge_only or args.architecture_comparison
    ):
        raise ValueError(
            "online remote attention requires Bridge-only or architecture comparison"
        )
    if args.gpu_resident_shadow and not (
        args.shadow_only_only
        or args.stop_and_copy_only
        or args.online_remote_attention
    ):
        raise ValueError(
            "GPU-resident staging requires Shadow-only or online Bridge mode"
        )
    if args.gpu_direct_history and not args.gpu_resident_shadow:
        raise ValueError("GPU-direct history requires --gpu-resident-shadow")
    if args.gpu_direct_delta and not args.gpu_direct_history:
        raise ValueError("GPU-direct delta requires --gpu-direct-history")
    if args.gpu_direct_history_pacing and not (
        args.gpu_direct_history and args.persistent_channel
        and (
            (args.fixed_rate_gib_s is not None and args.fixed_rate_gib_s > 0)
            or args.manager_m2_rate
        )
    ):
        raise ValueError(
            "GPU-direct history pacing needs a persistent channel and "
            "a positive fixed migration rate"
        )
    if args.gpu_direct_delta and not args.shadow_only_only:
        raise ValueError("GPU-direct delta batch sweep currently requires Shadow-only")
    if args.persistent_channel and not (
        args.shadow_only_only
        and args.gpu_resident_shadow
        and args.gpu_direct_history
        and args.gpu_direct_delta
    ):
        raise ValueError(
            "persistent channel requires the GPU-resident GPU-direct "
            "Shadow-only path"
        )
    if args.persistent_sequential_reuse and not args.persistent_channel:
        raise ValueError(
            "persistent sequential reuse requires --persistent-channel"
        )
    if args.preconnect_persistent_channel and not args.persistent_channel:
        raise ValueError(
            "persistent preconnect requires --persistent-channel"
        )
    if args.persistent_session_gap_s < 0:
        raise ValueError("persistent session gap cannot be negative")
    if args.channel_generation < 0:
        raise ValueError("channel generation cannot be negative")
    if args.persistent_channel and (
        args.deferred_comm_destroy
        or args.post_takeover_comm_destroy
        or args.deferred_comm_destroy_comparison
        or args.post_takeover_destroy_comparison
    ):
        raise ValueError(
            "persistent channel cannot be combined with a destroy experiment"
        )
    if args.deferred_comm_destroy and not args.gpu_direct_delta:
        raise ValueError(
            "deferred communicator destroy requires persistent GPU-direct delta"
        )
    if args.post_takeover_comm_destroy and not (
        args.shadow_only_only
        and args.gpu_resident_shadow
        and args.gpu_direct_history
        and args.gpu_direct_delta
        and args.ready_sync_mode == "STREAM_EVENT"
        and args.ready_notification_mode == "UDP"
    ):
        raise ValueError(
            "post-takeover destroy requires the STREAM_EVENT/UDP persistent "
            "GPU-direct Shadow-only path"
        )
    if args.post_takeover_destroy_comparison and not (
        args.shadow_only_only
        and args.gpu_resident_shadow
        and args.gpu_direct_history
        and args.gpu_direct_delta
        and args.ready_sync_mode == "STREAM_EVENT"
        and args.ready_notification_mode == "UDP"
    ):
        raise ValueError(
            "post-takeover destroy comparison requires the STREAM_EVENT/UDP "
            "persistent GPU-direct Shadow-only path"
        )
    if args.deferred_comm_destroy_comparison and not (
        args.shadow_only_only
        and args.gpu_resident_shadow
        and args.gpu_direct_history
        and args.gpu_direct_delta
        and args.ready_sync_mode == "STREAM_EVENT"
        and args.ready_notification_mode == "UDP"
    ):
        raise ValueError(
            "deferred destroy comparison requires the P1 STREAM_EVENT/UDP "
            "GPU-direct Shadow-only path"
        )
    if args.deferred_comm_destroy and args.deferred_comm_destroy_comparison:
        raise ValueError(
            "select either fixed deferred destroy or its comparison"
        )
    if args.post_takeover_comm_destroy and args.deferred_comm_destroy:
        raise ValueError(
            "post-takeover destroy selects deferred terminal-close handling "
            "automatically; do not also pass --deferred-comm-destroy"
        )
    if args.post_takeover_destroy_comparison and (
        args.deferred_comm_destroy
        or args.deferred_comm_destroy_comparison
        or args.post_takeover_comm_destroy
    ):
        raise ValueError(
            "post-takeover destroy comparison cannot be combined with another "
            "communicator-lifecycle selector"
        )
    if args.ready_sync_mode == "STREAM_EVENT" and not (
        args.gpu_resident_shadow and args.gpu_direct_history
    ):
        raise ValueError(
            "STREAM_EVENT ready synchronization requires GPU-resident "
            "GPU-direct history"
        )
    if args.ready_sync_comparison and not (
        args.shadow_only_only
        and args.gpu_resident_shadow
        and args.gpu_direct_history
        and args.gpu_direct_delta
    ):
        raise ValueError(
            "ready sync comparison requires the GPU-direct Shadow-only path"
        )
    if args.ready_notification_comparison and not (
        args.shadow_only_only
        and args.gpu_resident_shadow
        and args.gpu_direct_history
        and args.gpu_direct_delta
        and args.ready_sync_mode == "STREAM_EVENT"
    ):
        raise ValueError(
            "ready notification comparison requires the P0 STREAM_EVENT "
            "GPU-direct Shadow-only path"
        )
    comparison_count = sum(
        bool(value)
        for value in (
            args.ready_sync_comparison,
            args.ready_notification_comparison,
            args.deferred_comm_destroy_comparison,
            args.post_takeover_destroy_comparison,
        )
    )
    if comparison_count > 1:
        raise ValueError("select only one P0/P1/teardown comparison at a time")
    if (
        args.ready_notification_mode == "UDP"
        or args.ready_notification_comparison
    ) and not (0 < args.ready_notification_port <= 65535):
        raise ValueError("UDP ready notification port is invalid")
    if args.ready_latch_poll_ms < 0:
        raise ValueError("ready latch poll interval cannot be negative")
    if args.gpu_direct_delta_batch_tokens <= 0:
        raise ValueError("GPU-direct delta batch tokens must be positive")
    if args.gpu_direct_delta_flush_ms < 0:
        raise ValueError("GPU-direct delta flush time cannot be negative")
    if args.gpu_direct_history and not (
        1024 <= args.gpu_direct_base_port <= 65530
    ):
        raise ValueError("GPU-direct base port must leave five valid ports")
    if set(args.strategy_order) != {"S_NEW", "S_NEW_OLD"}:
        raise ValueError("strategy order must contain S_NEW and S_NEW_OLD once")
    earliest_ready = args.commit_timing == "EARLIEST_READY"
    if earliest_ready:
        if not 0 < args.trigger_output_tokens < args.anchor_max_tokens - 64:
            raise ValueError(
                "earliest-ready trigger must leave at least 64 target-owned tokens"
            )
    elif not 0 < args.trigger_output_tokens < args.cutover_output_tokens:
        raise ValueError("trigger/cutover boundaries are invalid")
    bridge_output_tokens = args.bridge_output_tokens
    if bridge_output_tokens is None:
        bridge_output_tokens = (
            args.trigger_output_tokens + args.cutover_output_tokens
        ) // 2
        args.bridge_output_tokens = bridge_output_tokens
    if not earliest_ready and not (
        args.trigger_output_tokens
        < bridge_output_tokens
        < args.cutover_output_tokens
    ):
        raise ValueError("Bridge boundary must be strictly inside the Shadow window")
    if not earliest_ready and args.anchor_max_tokens <= args.cutover_output_tokens + 64:
        raise ValueError("anchor must leave at least 64 target-owned tokens")
    if (
        args.anchor_prompt_tokens is not None
        and args.anchor_prompt_tokens + args.anchor_max_tokens > args.max_model_len
    ):
        raise ValueError("anchor prompt plus output exceeds max model length")
    if args.natural_eos_anchor and (
        args.anchor_request_file is None
        or not args.expected_anchor_request_sha256
        or args.anchor_prompt_tokens is None
    ):
        raise ValueError(
            "natural EOS anchor requires a pinned request and prompt length"
        )
    if args.anchor_request_file is not None and not args.natural_eos_anchor:
        raise ValueError("custom anchor request requires natural EOS mode")
    if args.minimum_ready_target_jobs < 0 or args.minimum_window_samples < 0:
        raise ValueError("online sample thresholds cannot be negative")
    if not 0 <= args.minimum_source_kv_usage_frac <= 1:
        raise ValueError("minimum source KV usage fraction must be in [0, 1]")
    if args.minimum_source_kv_usage_frac and not args.source_pressure:
        raise ValueError("source KV usage gate requires --source-pressure")
    if (
        min(
            args.slo_tpot_ms,
            args.slo_ttft_ms,
            args.slo_e2e_ms,
            args.slo_handoff_ms,
        )
        <= 0
    ):
        raise ValueError("SLO thresholds must be positive")
    if args.fixed_rate_gib_s is not None and args.fixed_rate_gib_s < 0:
        raise ValueError("fixed migration rate cannot be negative")
    if args.manager_m2_rate:
        profiles = (args.m2_low_gib_s, args.m2_medium_gib_s, args.m2_high_gib_s)
        if not (
            args.manager_m1_auto_start
            and args.manager_m0_shadow
            and args.fixed_rate_gib_s is None
            and all(value is not None and math.isfinite(value) for value in profiles)
            and 0 < profiles[0] < profiles[1] < profiles[2]
        ):
            raise ValueError(
                "M2 requires M1/M0, no fixed rate, and explicit LOW < MEDIUM "
                "< HIGH positive profiles"
            )
    elif any(
        value is not None
        for value in (args.m2_low_gib_s, args.m2_medium_gib_s, args.m2_high_gib_s)
    ):
        raise ValueError("M2 profiles require --manager-m2-rate")
    if args.manager_m3_commit and not (
        args.manager_m2_rate
        and args.commit_timing == "EARLIEST_READY"
        and args.shadow_only_only
    ):
        raise ValueError(
            "M3 requires M2 Shadow-only EARLIEST_READY"
        )
    if args.manager_m4_cancel and not args.manager_m3_commit:
        raise ValueError("M4 requires M3 earliest-ready commit")
    if args.manager_m5_predictor_shadow:
        if args.predictor_checkpoint is None or not args.predictor_checkpoint_sha256:
            raise ValueError("M5 requires checkpoint path and SHA-256")
        if args.repetitions != 1 or args.persistent_sequential_reuse:
            raise ValueError("M5 smoke requires one fresh source server session")
        if not args.predictor_checkpoint.is_file():
            raise FileNotFoundError(args.predictor_checkpoint)
        if common.sha256(args.predictor_checkpoint) != args.predictor_checkpoint_sha256:
            raise ValueError("M5 predictor checkpoint SHA-256 differs")
    if args.manager_m4_expect_cancel and not args.manager_m4_cancel:
        raise ValueError("M4 cancellation smoke requires M4 enabled")
    if args.paired_stay and not (
        args.manager_m1_auto_start
        and args.shadow_only_only
        and args.repetitions == 1
        and not args.manager_m1_expect_stay
        and not args.manager_m4_expect_cancel
        and not args.persistent_sequential_reuse
    ):
        raise ValueError("paired STAY requires one fresh M1 Shadow-only run")
    if args.manager_m2_expected_profile and not args.manager_m2_rate:
        raise ValueError("M2 expected profile requires --manager-m2-rate")
    if not 0 <= args.manager_m2_min_history_byte_frac <= 1:
        raise ValueError("M2 history byte fraction must be in [0, 1]")
    if args.manager_m2_min_history_byte_frac and not (
        args.manager_m2_rate and args.manager_m2_expected_profile
    ):
        raise ValueError("M2 history fraction requires an expected profile")
    if args.manager_m2_force_initial_high and not (
        args.manager_m2_rate
        and args.phase == "smoke"
        and args.manager_m2_expected_profile == "HIGH"
    ):
        raise ValueError("forced initial HIGH is only for M2 HIGH smoke")
    if args.manager_m2_require_source_high and not (
        args.manager_m2_rate
        and args.source_pressure
        and not args.manager_m2_force_initial_high
        and args.manager_m2_expected_profile == "HIGH"
    ):
        raise ValueError(
            "source-driven HIGH requires M2, source peers, and no force flag"
        )
    if args.manager_m2_require_low_to_high and not (
        args.manager_m2_rate
        and args.source_pressure
        and args.phase == "smoke"
        and args.manager_m2_expected_profile is None
        and args.minimum_ready_source_jobs == 0
    ):
        raise ValueError(
            "M2 LOW-to-HIGH smoke requires M2, delayed source peers, "
            "and no fixed expected profile"
        )
    if args.manager_m1_auto_start and not (
        args.shadow_only_only
        and args.gpu_resident_shadow
        and args.gpu_direct_history
        and args.gpu_direct_delta
        and args.gpu_direct_history_pacing
        and (
            args.manager_m2_rate
            or (args.fixed_rate_gib_s is not None and args.fixed_rate_gib_s > 0)
        )
        and args.commit_timing == "EARLIEST_READY"
    ):
        raise ValueError(
            "M1 requires paced GPU-direct Shadow-only, a positive rate, "
            "and EARLIEST_READY cutover"
        )
    if args.m1_min_output_tokens is not None and (
        not args.manager_m1_auto_start
        or not 0 <= args.m1_min_output_tokens < args.anchor_max_tokens - 64
    ):
        raise ValueError(
            "M1 minimum output requires auto-start and at least 64 remaining tokens"
        )
    if args.manager_m1_auto_start and (
        args.m1_source_release_tail_s is None
        or not math.isfinite(args.m1_source_release_tail_s)
        or args.m1_source_release_tail_s <= 0
    ):
        raise ValueError("M1 requires a positive TP1 KV release tail allowance")
    if not args.manager_m1_auto_start and args.m1_source_release_tail_s is not None:
        raise ValueError("M1 release tail allowance requires auto-start")
    if args.tp4_max_num_seqs is not None and args.tp4_max_num_seqs <= 0:
        raise ValueError("TP4 max-num-seqs must be positive")
    if args.manager_m1_expect_stay and not (
        args.manager_m1_auto_start
        and args.shadow_only_only
        and not args.persistent_sequential_reuse
    ):
        raise ValueError(
            "M1 STAY smoke requires M1 Shadow-only without sequential reuse"
        )
    if args.manager_m1_expect_stay:
        if args.manager_m1_stay_reason == "short-budget":
            if args.anchor_max_tokens != 96 or args.tp4_max_num_seqs is not None:
                raise ValueError("short-budget STAY requires 96 output tokens")
        elif args.anchor_max_tokens != 1024 or args.tp4_max_num_seqs != 2:
            raise ValueError(
                "target-load STAY requires 1024 output tokens and TP4 max-num-seqs 2"
            )
    if not args.python_bin.is_file():
        raise FileNotFoundError(f"Python executable is missing: {args.python_bin}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"model path is missing: {args.model_path}")
    common.model_kv_geometry(args.model_path, args.dtype)
    required_files = {
        "manifest": args.manifest,
        "survival table": args.survival_table,
        "guard file": args.guard_file,
        "controller template": common.CONFIG_TEMPLATE,
        "source request": common.SOURCE_REQUEST,
    }
    for label, path in required_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"{label} is not a file: {path}")
    revision = common.git("rev-parse", "HEAD")
    expected = common.git("rev-parse", args.expected_revision)
    if revision != expected:
        raise RuntimeError(f"HEAD {revision} differs from expected {expected}")
    if (
        subprocess.run(
            ["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--"]
        ).returncode
        != 0
    ):
        raise RuntimeError("tracked working-tree changes are present")
    expected_hashes = (
        (args.manifest, args.expected_manifest_sha256, "manifest"),
        (args.survival_table, args.expected_survival_sha256, "survival table"),
        (args.guard_file, args.expected_guard_sha256, "guard file"),
    )
    for path, expected_sha, label in expected_hashes:
        if common.sha256(path) != expected_sha:
            raise RuntimeError(f"{label} SHA-256 differs from expected")
    if args.natural_eos_anchor:
        assert args.anchor_request_file is not None
        if (
            not args.anchor_request_file.is_file()
            or common.sha256(args.anchor_request_file)
            != args.expected_anchor_request_sha256
        ):
            raise RuntimeError("natural anchor request SHA-256 differs")
        anchor_request = common.read_json(args.anchor_request_file)
        prompt = anchor_request.get("prompt")
        if (
            not isinstance(prompt, list)
            or len(prompt) != args.anchor_prompt_tokens
            or not all(isinstance(token, int) and not isinstance(token, bool)
                       for token in prompt)
            or anchor_request.get("ignore_eos") is not False
            or anchor_request.get("max_tokens") != args.anchor_max_tokens
        ):
            raise ValueError("natural anchor request differs from pinned contract")
    guard = int(args.guard_file.read_text(encoding="utf-8").strip())
    if guard != args.expected_guard:
        raise RuntimeError(f"frozen guard {guard} differs from expected")
    manifest = load_manifest(args.manifest)
    jobs = manifest["jobs"]
    target_jobs = [job for job in jobs if job.get("pool") == "target"]
    source_jobs = [job for job in jobs if job.get("pool") == "source"]
    if args.manager_m2_require_low_to_high and (
        not source_jobs
        or any(job.get("start_after_event") != "M2_INITIAL_RATE"
               for job in source_jobs)
    ):
        raise ValueError("M2 LOW-to-HIGH smoke requires event-start source peers")
    if source_jobs and not args.source_pressure:
        raise ValueError("online Shadow manifest must contain target jobs only")
    if args.source_pressure and (not source_jobs or not target_jobs):
        raise ValueError("A4-P requires source peers and target background jobs")
    if args.source_pressure and not (
        args.shadow_only_only
        and args.gpu_direct_history
        and args.gpu_direct_delta
        and args.persistent_channel
    ):
        raise ValueError("A4-P requires persistent GPU-direct Shadow-only")
    if len(target_jobs) < args.minimum_ready_target_jobs:
        raise ValueError("manifest has too few target jobs for the readiness gate")
    if not 0 <= args.minimum_ready_source_jobs <= len(source_jobs):
        raise ValueError("source readiness count exceeds manifest source jobs")
    if args.manager_m2_require_source_high and (
        args.minimum_ready_source_jobs < 1
    ):
        raise ValueError("source-driven HIGH requires a source readiness gate")
    if (
        args.manager_m1_expect_stay
        and args.manager_m1_stay_reason == "target-load"
        and len(target_jobs) < 16
    ):
        raise ValueError("target-load STAY requires at least 16 target jobs")
    for job in target_jobs:
        prompt = job["request"].get("prompt")
        if not isinstance(prompt, list) or not all(isinstance(x, int) for x in prompt):
            raise ValueError("online target jobs require exact prompt token IDs")
        if len(prompt) + int(job["request"]["max_tokens"]) > args.max_model_len:
            raise ValueError(f"target job {job['job_id']} exceeds max model length")
    for job in source_jobs:
        prompt = job["request"].get("prompt")
        if not isinstance(prompt, list) or not all(isinstance(x, int) for x in prompt):
            raise ValueError("A4-P source peers require exact prompt token IDs")
        if len(prompt) + int(job["request"]["max_tokens"]) > args.max_model_len:
            raise ValueError(f"source job {job['job_id']} exceeds max model length")
    if args.manager_m2_require_source_high:
        if args.anchor_prompt_tokens is None:
            raise ValueError("source-driven HIGH needs an explicit anchor prompt")
        capped_source_tokens = sum(
            len(job["request"]["prompt"])
            + int(job["request"]["max_tokens"])
            for job in source_jobs
        ) + args.anchor_prompt_tokens + args.anchor_max_tokens
        if capped_source_tokens >= args.tp1_blocks * 16:
            raise ValueError(
                "source-driven HIGH peers plus anchor exceed TP1 KV capacity"
            )
    if args.out_root.exists():
        raise FileExistsError(f"refusing to reuse output root {args.out_root}")
    return (
        revision,
        guard,
        {
            "target_jobs": len(target_jobs),
            "source_jobs": len(source_jobs),
            "trigger_output_tokens": args.trigger_output_tokens,
            "commit_timing": args.commit_timing,
            "cutover_output_tokens": (
                None if earliest_ready else args.cutover_output_tokens
            ),
            "bridge_output_tokens": (
                None if earliest_ready else args.bridge_output_tokens
            ),
            "shadow_window_output_tokens": (
                None
                if earliest_ready
                else args.cutover_output_tokens - args.trigger_output_tokens
            ),
            "minimum_ready_target_jobs": args.minimum_ready_target_jobs,
            "fixed_rate_gib_s": args.fixed_rate_gib_s,
        },
    )


def build_controller_config_overrides(
    *,
    trigger_output_tokens: int,
    cutover_output_tokens: int,
    fixed_rate_gib_s: float | None,
    m2_profiles_gib_s: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Keep the controller's configured Shadow window equal to the CLI design."""
    overrides: dict[str, Any] = {
        "handoff_output_tokens": cutover_output_tokens - trigger_output_tokens
    }
    if fixed_rate_gib_s is not None:
        fixed_rate_bytes_s = fixed_rate_gib_s * 1024**3
        overrides["rate"] = {
            "b_min_bytes_s": fixed_rate_bytes_s,
            "b_max_bytes_s": fixed_rate_bytes_s,
            "b_start_bytes_s": fixed_rate_bytes_s,
            "b_hard_max_bytes_s": fixed_rate_bytes_s,
        }
    if m2_profiles_gib_s is not None:
        low, medium, high = m2_profiles_gib_s
        overrides["rate"] = {
            "b_min_bytes_s": low * 1024**3,
            "b_start_bytes_s": medium * 1024**3,
            "b_max_bytes_s": high * 1024**3,
            "b_hard_max_bytes_s": high * 1024**3,
        }
    return overrides


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize_slo(
    results: list[dict[str, Any]],
    *,
    tpot_ms: float,
    ttft_ms: float,
    e2e_ms: float,
) -> dict[str, Any]:
    completed = [row for row in results if row.get("status") == "COMPLETED"]

    def exceeds(row: dict[str, Any], field: str, threshold: float) -> bool:
        value = row.get(field)
        # A one-token warm-up has no inter-token interval, so its TPOT
        # percentile is correctly undefined rather than an SLO violation.
        return isinstance(value, (int, float)) and float(value) > threshold

    intervals: list[float] = []
    for row in completed:
        times = [float(value) for value in row.get("token_times_unix_s", [])]
        intervals.extend(
            (current - previous) * 1000 for previous, current in zip(times, times[1:])
        )
    violating_intervals = sum(value > tpot_ms for value in intervals)
    return {
        "thresholds": {
            "tpot_ms": tpot_ms,
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
        },
        "completed_requests": len(completed),
        "token_intervals": len(intervals),
        "tpot_interval_violations": violating_intervals,
        "tpot_interval_violation_rate": (
            violating_intervals / len(intervals) if intervals else None
        ),
        "request_p99_tpot_violations": sum(
            exceeds(row, "tpot_p99_ms", tpot_ms) for row in completed
        ),
        "ttft_violations": sum(
            exceeds(row, "ttft_ms", ttft_ms) for row in completed
        ),
        "e2e_violations": sum(
            exceeds(row, "e2e_ms", e2e_ms) for row in completed
        ),
    }


def summarize_emitted_intervals(
    emitted: list[dict[str, Any]],
    *,
    origin: str | None = None,
) -> dict[str, Any]:
    """Summarize visible token gaps, including the outliers percentiles hide."""

    selected = [
        row
        for row in emitted
        if row.get("unix_s") is not None
        and (origin is None or row.get("origin") == origin)
    ]
    intervals = [
        (float(current["unix_s"]) - float(previous["unix_s"])) * 1000
        for previous, current in zip(selected, selected[1:])
    ]
    return {
        "samples": len(intervals),
        "mean_ms": sum(intervals) / len(intervals) if intervals else None,
        "p50_ms": percentile(intervals, 0.50),
        "p95_ms": percentile(intervals, 0.95),
        "p99_ms": percentile(intervals, 0.99),
        "max_ms": max(intervals, default=None),
    }


def emitted_boundary_gap_ms(
    emitted: list[dict[str, Any]],
    *,
    origin: str,
    output_tokens: int,
) -> float | None:
    """Return the visible gap ending at a one-based output-token boundary."""

    selected = [
        row
        for row in emitted
        if row.get("origin") == origin and row.get("unix_s") is not None
    ]
    current = output_tokens - 1
    previous = current - 1
    if previous < 0 or current >= len(selected):
        return None
    return (
        float(selected[current]["unix_s"])
        - float(selected[previous]["unix_s"])
    ) * 1000


def capture_persistent_memory(
    processes: list[Any],
) -> dict[str, Any]:
    """Capture process-tree RSS and per-process GPU memory after one session."""
    roots = {int(item.process.pid) for item in processes}
    rows: list[dict[str, int]] = []
    selected = set(roots)
    proc = Path("/proc")
    if proc.is_dir():
        process_table: dict[int, tuple[int, int]] = {}
        for child in proc.iterdir():
            if not child.name.isdigit():
                continue
            try:
                status = (child / "status").read_text(encoding="utf-8")
            except OSError:
                continue
            values: dict[str, str] = {}
            for line in status.splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    values[key] = value.strip()
            try:
                parent = int(values.get("PPid", "-1"))
                rss_kib = int(values.get("VmRSS", "0 kB").split()[0])
                process_table[int(child.name)] = (parent, rss_kib)
            except (ValueError, IndexError):
                continue
        changed = True
        while changed:
            changed = False
            for pid, (parent, _) in process_table.items():
                if parent in selected and pid not in selected:
                    selected.add(pid)
                    changed = True
        rows = [
            {
                "pid": pid,
                "ppid": process_table[pid][0],
                "rss_kib": process_table[pid][1],
            }
            for pid in sorted(selected & process_table.keys())
        ]
    gpu_rows: list[dict[str, int]] = []
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
        for line in completed.stdout.splitlines():
            pid_text, memory_text = (part.strip() for part in line.split(",", 1))
            pid = int(pid_text)
            if pid in selected:
                gpu_rows.append(
                    {"pid": pid, "used_memory_mib": int(memory_text)}
                )
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {
        "format_version": 1,
        "captured_unix_s": time.time(),
        "root_pids": sorted(roots),
        "processes": rows,
        "process_tree_rss_kib": sum(row["rss_kib"] for row in rows),
        "gpu_processes": gpu_rows,
        "gpu_used_memory_mib": sum(row["used_memory_mib"] for row in gpu_rows),
    }


def write_measurements(out_root: Path, runs: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        acceptance = run["acceptance"]
        ready_event_waits = [
            float(value)
            for value in (acceptance.get("target_ready_event_wait_ms") or [])
            if value is not None
        ]
        communicator_destroy_ms = [
            float(value)
            for value in (acceptance.get("communicator_destroy_ms") or [])
            if value is not None
        ]
        lifetime_path = Path(run.get("root", "")) / "process_lifetimes.json"
        lifetimes = (
            common.read_json(lifetime_path).get("processes", [])
            if lifetime_path.is_file()
            else []
        )
        lifetime_by_name = {
            str(item.get("name")): item.get("wall_time_s") for item in lifetimes
        }
        row: dict[str, Any] = {
            "repetition": run["repetition"],
            "strategy": run["strategy"],
            "architecture": run.get("architecture", "BRIDGE"),
            "ready_sync_mode": run.get("ready_sync_mode"),
            "ready_notification_mode": run.get("ready_notification_mode"),
            "status": acceptance["status"],
            "shadow_duration_ms": acceptance["shadow_duration_ms"],
            "bridge_to_commit_ms": acceptance["bridge_to_commit_ms"],
            "final_sync_to_commit_ms": acceptance.get("final_sync_to_commit_ms"),
            "handoff_stall_ms": acceptance["handoff_stall_ms"],
            "freeze_to_final_delta_ack_ms": acceptance.get(
                "freeze_to_final_delta_ack_ms"
            ),
            "freeze_to_first_rank_ready_ms": acceptance.get(
                "freeze_to_first_rank_ready_ms"
            ),
            "freeze_to_all_rank_ready_ms": acceptance.get(
                "freeze_to_all_rank_ready_ms"
            ),
            "rank_ready_skew_ms": acceptance.get("rank_ready_skew_ms"),
            "last_rank_ready_to_controller_wakeup_ms": acceptance.get(
                "last_rank_ready_to_controller_wakeup_ms"
            ),
            "controller_wakeup_to_commit_ms": acceptance.get(
                "controller_wakeup_to_commit_ms"
            ),
            "last_rank_ready_to_commit_ms": acceptance.get(
                "last_rank_ready_to_commit_ms"
            ),
            "ready_notification_count": acceptance.get(
                "ready_notification_count"
            ),
            "ready_notification_delivery_ms": acceptance.get(
                "ready_notification_delivery_ms"
            ),
            "ready_latch_poll_ms": acceptance.get("ready_latch_poll_ms"),
            "ready_latch_poll_count": acceptance.get(
                "ready_latch_poll_count"
            ),
            "ready_latch_to_authoritative_ready_ms": acceptance.get(
                "ready_latch_to_authoritative_ready_ms"
            ),
            "request_frozen_unix_s": acceptance.get("request_frozen_unix_s"),
            "source_kv_released_unix_s": acceptance.get(
                "source_kv_released_unix_s"
            ),
            "source_kv_release_after_commit_ms": acceptance.get(
                "source_kv_release_after_commit_ms"
            ),
            "trigger_to_source_kv_release_ms": acceptance.get(
                "trigger_to_source_kv_release_ms"
            ),
            "source_origin_tokens": acceptance["source_origin_tokens"],
            "target_origin_tokens": acceptance["target_origin_tokens"],
            "fixed_rate_gib_s": acceptance.get("fixed_rate_gib_s"),
            "history_payload_bytes": acceptance.get("history_payload_bytes"),
            "history_observed_aggregate_gib_s": acceptance.get(
                "history_observed_aggregate_gib_s"
            ),
            "history_max_stage_ms": acceptance.get("history_max_stage_ms"),
            "history_ready_before_freeze_ms": acceptance.get(
                "history_ready_before_freeze_ms"
            ),
            "history_gpu_ready_before_freeze_ms": acceptance.get(
                "history_gpu_ready_before_freeze_ms"
            ),
            "gpu_resident_shadow": acceptance.get("gpu_resident_shadow"),
            "gpu_direct_delta": acceptance.get("gpu_direct_delta"),
            "deferred_comm_destroy": acceptance.get(
                "deferred_comm_destroy"
            ),
            "post_takeover_comm_destroy": acceptance.get(
                "post_takeover_comm_destroy"
            ),
            "communicator_destroy_ms_max": (
                max(communicator_destroy_ms)
                if communicator_destroy_ms
                else None
            ),
            "communicator_destroy_statuses": "|".join(
                str(value)
                for value in (
                    acceptance.get("communicator_destroy_statuses") or []
                )
            ),
            "gpu_direct_delta_batch_tokens": acceptance.get(
                "gpu_direct_delta_batch_tokens"
            ),
            "gpu_direct_delta_flush_ms": acceptance.get(
                "gpu_direct_delta_flush_ms"
            ),
            "gpu_direct_delta_batches": acceptance.get(
                "gpu_direct_delta_batches"
            ),
            "gpu_direct_delta_logical_submissions": acceptance.get(
                "gpu_direct_delta_logical_submissions"
            ),
            "gpu_direct_delta_coalesced_submissions": acceptance.get(
                "gpu_direct_delta_coalesced_submissions"
            ),
            "gpu_direct_delta_tokens": acceptance.get(
                "gpu_direct_delta_tokens"
            ),
            "gpu_direct_delta_payload_bytes": acceptance.get(
                "gpu_direct_delta_payload_bytes"
            ),
            "gpu_direct_delta_total_ms": acceptance.get(
                "gpu_direct_delta_total_ms"
            ),
            "gpu_direct_delta_pack_ms": acceptance.get(
                "gpu_direct_delta_pack_ms"
            ),
            "gpu_direct_delta_receiver_ready_ms": acceptance.get(
                "gpu_direct_delta_receiver_ready_ms"
            ),
            "gpu_direct_delta_nccl_send_ms": acceptance.get(
                "gpu_direct_delta_nccl_send_ms"
            ),
            "gpu_direct_delta_target_apply_ack_ms": acceptance.get(
                "gpu_direct_delta_target_apply_ack_ms"
            ),
            "gpu_direct_delta_max_batch_ms": acceptance.get(
                "gpu_direct_delta_max_batch_ms"
            ),
            "gpu_history_block_acks": acceptance.get("gpu_history_block_acks"),
            "gpu_delta_acks": acceptance.get("gpu_delta_acks"),
            "target_ready_sync_scopes": "|".join(
                str(value)
                for value in (acceptance.get("target_ready_sync_scopes") or [])
            ),
            "target_ready_event_wait_ms_mean": (
                sum(ready_event_waits) / len(ready_event_waits)
                if ready_event_waits
                else None
            ),
            "target_ready_event_wait_ms_max": (
                max(ready_event_waits) if ready_event_waits else None
            ),
            "target_device_wide_synchronize": any(
                value is True
                for value in (
                    acceptance.get("target_device_wide_synchronize") or []
                )
            ),
            "target_receive_dependency_scopes": "|".join(
                str(value)
                for value in (
                    acceptance.get("target_receive_dependency_scopes") or []
                )
            ),
            "target_model_stream_wait_event": all(
                value is True
                for value in (
                    acceptance.get("target_model_stream_wait_event") or []
                )
            ),
            "remote_attention_calls": acceptance.get("remote_attention_calls"),
            "remote_attention_layers": acceptance.get("remote_attention_layers"),
            "remote_attention_token_forwards": acceptance.get(
                "remote_attention_token_forwards"
            ),
            "remote_attention_visible_calls": acceptance.get(
                "remote_attention_visible_calls"
            ),
            "remote_attention_visible_token_forwards": acceptance.get(
                "remote_attention_visible_token_forwards"
            ),
            "remote_attention_speculative_calls": acceptance.get(
                "remote_attention_speculative_calls"
            ),
            "remote_attention_speculative_token_forwards": acceptance.get(
                "remote_attention_speculative_token_forwards"
            ),
            "remote_attention_p50_ms": acceptance.get(
                "remote_attention_total_ms", {}
            ).get("p50"),
            "remote_attention_p95_ms": acceptance.get(
                "remote_attention_total_ms", {}
            ).get("p95"),
            "remote_attention_p99_ms": acceptance.get(
                "remote_attention_total_ms", {}
            ).get("p99"),
            "remote_attention_visible_p50_ms": acceptance.get(
                "remote_attention_visible_total_ms", {}
            ).get("p50"),
            "remote_attention_visible_p95_ms": acceptance.get(
                "remote_attention_visible_total_ms", {}
            ).get("p95"),
            "remote_attention_visible_p99_ms": acceptance.get(
                "remote_attention_visible_total_ms", {}
            ).get("p99"),
            "source_process_wall_time_s": lifetime_by_name.get("source TP1"),
            "target_process_wall_time_s": lifetime_by_name.get("target TP4"),
            "stager_process_wall_time_s": lifetime_by_name.get("stager"),
            "controller_process_wall_time_s": lifetime_by_name.get("controller"),
            "anchor_tpot_p50_ms": acceptance.get("anchor_tpot", {}).get("p50_ms"),
            "anchor_ttft_ms": acceptance.get("anchor_ttft_ms"),
            "anchor_e2e_ms": acceptance.get("anchor_e2e_ms"),
            "anchor_mean_itl_ms": acceptance.get("anchor_tpot", {}).get("mean_ms"),
            "anchor_tpot_p95_ms": acceptance.get("anchor_tpot", {}).get("p95_ms"),
            "anchor_tpot_p99_ms": acceptance.get("anchor_tpot", {}).get("p99_ms"),
            "anchor_tpot_max_ms": acceptance.get("anchor_tpot", {}).get("max_ms"),
            "source_tpot_p50_ms": acceptance.get("source_tpot", {}).get("p50_ms"),
            "source_tpot_p95_ms": acceptance.get("source_tpot", {}).get("p95_ms"),
            "source_tpot_p99_ms": acceptance.get("source_tpot", {}).get("p99_ms"),
            "source_tpot_max_ms": acceptance.get("source_tpot", {}).get("max_ms"),
            "snapshot_trigger_stall_ms": acceptance.get(
                "snapshot_trigger_stall_ms"
            ),
            "freeze_boundary_stall_ms": acceptance.get(
                "freeze_boundary_stall_ms"
            ),
            "final_delta_enqueue_ms": acceptance.get(
                "final_delta_enqueue_ms"
            ),
            "final_delta_drain_ms": acceptance.get("final_delta_drain_ms"),
            "cutover_hook_to_delta_drain_ms": acceptance.get(
                "cutover_hook_to_delta_drain_ms"
            ),
            "output_throughput_tokens_s": acceptance.get("workload", {}).get(
                "output_throughput_tokens_s"
            ),
            "slo_tpot_interval_violation_rate": acceptance.get("slo", {}).get(
                "tpot_interval_violation_rate"
            ),
            "slo_ttft_violations": acceptance.get("slo", {}).get("ttft_violations"),
            "slo_e2e_violations": acceptance.get("slo", {}).get("e2e_violations"),
            "slo_handoff_violation": acceptance.get("slo", {}).get("handoff_violation"),
            "anchor_slo_success": acceptance.get("anchor_slo", {}).get("success"),
            "anchor_slo_itl_violation_rate": acceptance.get(
                "anchor_slo", {}
            ).get("itl_violation_rate"),
        }
        for window, metrics in acceptance["target_tpot_windows"].items():
            prefix = window.lower()
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    if not rows:
        return
    # Architecture comparisons intentionally expose different semantic windows:
    # Bridge runs report BRIDGE while direct Shadow-only runs additionally report
    # FINAL_SYNC.  Build a stable union instead of assuming the first row defines
    # every later row; alternating pair order makes that assumption invalid.
    fieldnames = list(rows[0])
    known_fields = set(fieldnames)
    for row in rows[1:]:
        for field in row:
            if field not in known_fields:
                fieldnames.append(field)
                known_fields.add(field)
    with (out_root / "measurements.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    stage_fields = [
        "repetition",
        "architecture",
        "ready_sync_mode",
        "ready_notification_mode",
        "freeze_to_final_delta_ack_ms",
        "freeze_to_first_rank_ready_ms",
        "freeze_to_all_rank_ready_ms",
        "rank_ready_skew_ms",
        "last_rank_ready_to_controller_wakeup_ms",
        "controller_wakeup_to_commit_ms",
        "last_rank_ready_to_commit_ms",
        "ready_notification_count",
        "ready_notification_delivery_ms",
        "ready_latch_poll_ms",
        "ready_latch_poll_count",
        "ready_latch_to_authoritative_ready_ms",
        "handoff_stall_ms",
    ]
    with (out_root / "handoff_stage_breakdown.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=stage_fields)
        writer.writeheader()
        writer.writerows(
            {field: row.get(field) for field in stage_fields} for row in rows
        )

    sync_comparisons: list[dict[str, Any]] = []
    sync_metrics = stage_fields[4:]
    for repetition in sorted({int(row["repetition"]) for row in rows}):
        selected = {
            str(row.get("ready_sync_mode")): row
            for row in rows
            if int(row["repetition"]) == repetition
        }
        if set(selected) != {"DEVICE_WIDE", "STREAM_EVENT"}:
            continue
        old = selected["DEVICE_WIDE"]
        new = selected["STREAM_EVENT"]
        comparison: dict[str, Any] = {"repetition": repetition}
        for metric in sync_metrics:
            old_value = old.get(metric)
            new_value = new.get(metric)
            comparison[f"old_{metric}"] = old_value
            comparison[f"new_{metric}"] = new_value
            comparison[f"saved_{metric}"] = (
                float(old_value) - float(new_value)
                if old_value is not None and new_value is not None
                else None
            )
        sync_comparisons.append(comparison)
    if sync_comparisons:
        with (out_root / "p0_ready_sync_comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(sync_comparisons[0])
            )
            writer.writeheader()
            writer.writerows(sync_comparisons)

    notification_comparisons: list[dict[str, Any]] = []
    for repetition in sorted({int(row["repetition"]) for row in rows}):
        selected = {
            str(row.get("ready_notification_mode")): row
            for row in rows
            if int(row["repetition"]) == repetition
        }
        if set(selected) != {"FILE_POLL", "UDP"}:
            continue
        old = selected["FILE_POLL"]
        new = selected["UDP"]
        comparison = {"repetition": repetition}
        for metric in sync_metrics:
            old_value = old.get(metric)
            new_value = new.get(metric)
            comparison[f"old_{metric}"] = old_value
            comparison[f"new_{metric}"] = new_value
            comparison[f"saved_{metric}"] = (
                float(old_value) - float(new_value)
                if old_value is not None and new_value is not None
                else None
            )
        notification_comparisons.append(comparison)
    if notification_comparisons:
        with (out_root / "p1_ready_notification_comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(notification_comparisons[0])
            )
            writer.writeheader()
            writer.writerows(notification_comparisons)

    destroy_comparisons: list[dict[str, Any]] = []
    for repetition in sorted({int(row["repetition"]) for row in rows}):
        selected = {
            bool(row.get("deferred_comm_destroy")): row
            for row in rows
            if int(row["repetition"]) == repetition
        }
        if set(selected) != {False, True}:
            continue
        old = selected[False]
        new = selected[True]
        comparison = {"repetition": repetition}
        for metric in sync_metrics:
            old_value = old.get(metric)
            new_value = new.get(metric)
            comparison[f"old_{metric}"] = old_value
            comparison[f"new_{metric}"] = new_value
            comparison[f"saved_{metric}"] = (
                float(old_value) - float(new_value)
                if old_value is not None and new_value is not None
                else None
            )
        comparison["new_communicator_destroy_ms_max"] = new.get(
            "communicator_destroy_ms_max"
        )
        destroy_comparisons.append(comparison)
    if destroy_comparisons:
        with (out_root / "deferred_destroy_comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(destroy_comparisons[0])
            )
            writer.writeheader()
            writer.writerows(destroy_comparisons)

    post_destroy_comparisons: list[dict[str, Any]] = []
    post_destroy_metrics = sync_metrics + [
        "post_commit_tpot_p50_ms",
        "post_commit_tpot_p95_ms",
        "post_commit_tpot_p99_ms",
        "communicator_destroy_ms_max",
    ]
    for repetition in sorted({int(row["repetition"]) for row in rows}):
        selected = {
            str(row.get("architecture")): row
            for row in rows
            if int(row["repetition"]) == repetition
            and row.get("architecture")
            in {"SHADOW_ONLY_POOL", "SHADOW_ONLY_POST_DESTROY"}
        }
        if set(selected) != {
            "SHADOW_ONLY_POOL",
            "SHADOW_ONLY_POST_DESTROY",
        }:
            continue
        pool = selected["SHADOW_ONLY_POOL"]
        destroy = selected["SHADOW_ONLY_POST_DESTROY"]
        comparison = {"repetition": repetition}
        for metric in post_destroy_metrics:
            pool_value = pool.get(metric)
            destroy_value = destroy.get(metric)
            comparison[f"pool_{metric}"] = pool_value
            comparison[f"post_destroy_{metric}"] = destroy_value
            comparison[f"post_destroy_minus_pool_{metric}"] = (
                float(destroy_value) - float(pool_value)
                if pool_value is not None and destroy_value is not None
                else None
            )
        post_destroy_comparisons.append(comparison)
    if post_destroy_comparisons:
        with (out_root / "post_takeover_destroy_comparison.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(post_destroy_comparisons[0])
            )
            writer.writeheader()
            writer.writerows(post_destroy_comparisons)

    paired: list[dict[str, Any]] = []
    repetitions = sorted({int(row["repetition"]) for row in rows})
    for repetition in repetitions:
        architecture_comparison = any(
            row.get("architecture") == "SHADOW_ONLY" for row in rows
        )
        label_field = "architecture" if architecture_comparison else "strategy"
        by_strategy = {
            str(row[label_field]): row
            for row in rows
            if int(row["repetition"]) == repetition
        }
        expected_labels = (
            {
                "BRIDGE_RA" if "BRIDGE_RA" in by_strategy else "BRIDGE",
                "SHADOW_ONLY",
            }
            if architecture_comparison
            else {"S_NEW", "S_NEW_OLD"}
        )
        if set(by_strategy) != expected_labels:
            continue
        new = by_strategy[
            ("BRIDGE_RA" if "BRIDGE_RA" in by_strategy else "BRIDGE")
            if architecture_comparison
            else "S_NEW"
        ]
        old = by_strategy["SHADOW_ONLY" if architecture_comparison else "S_NEW_OLD"]
        required = (
            "bridge_to_commit_ms",
            "handoff_stall_ms",
            "shadow_tpot_p99_ms",
            "bridge_tpot_p99_ms",
        )
        if any(new.get(key) is None or old.get(key) is None for key in required):
            continue
        paired.append(
            {
                "repetition": repetition,
                "bridge_ms_saved_by_history_precopy": (
                    float(new["bridge_to_commit_ms"])
                    - float(old["bridge_to_commit_ms"])
                ),
                "handoff_stall_ms_saved_by_history_precopy": (
                    float(new["handoff_stall_ms"]) - float(old["handoff_stall_ms"])
                ),
                "shadow_target_p99_ms_extra_from_history_precopy": (
                    float(old["shadow_tpot_p99_ms"]) - float(new["shadow_tpot_p99_ms"])
                ),
                "bridge_target_p99_ms_delta_history_precopy": (
                    float(old["bridge_tpot_p99_ms"]) - float(new["bridge_tpot_p99_ms"])
                ),
            }
            if not architecture_comparison
            else {
                "repetition": repetition,
                "final_sync_ms_saved_by_shadow_only": (
                    float(new["bridge_to_commit_ms"])
                    - float(old["bridge_to_commit_ms"])
                ),
                "handoff_stall_ms_saved_by_shadow_only": (
                    float(new["handoff_stall_ms"]) - float(old["handoff_stall_ms"])
                ),
                "shadow_target_p99_ms_delta_shadow_only": (
                    float(old["shadow_tpot_p99_ms"]) - float(new["shadow_tpot_p99_ms"])
                ),
                "final_sync_target_p99_ms_delta_shadow_only": (
                    float(old["bridge_tpot_p99_ms"]) - float(new["bridge_tpot_p99_ms"])
                ),
            }
        )
    if paired:
        with (out_root / "paired_comparisons.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)


def controller_completion_errors(
    end_rows: list[dict[str, Any]], manager_m1_auto_start: bool
) -> list[str]:
    """Check takeover and the trigger path selected for this run."""
    if len(end_rows) != 1 or end_rows[0].get("final_state") != "TAKEOVER":
        return ["controller did not finish in TAKEOVER"]
    expected = (
        "MANAGER_M1_START"
        if manager_m1_auto_start
        else "DIAGNOSTIC_FIXED_BOUNDARY"
    )
    if end_rows[0].get("trigger_path") != expected:
        return [f"controller trigger path differs from {expected}"]
    return []


def accept_m1_stay(
    controller_dir: Path,
    background_dir: Path,
    expected_jobs: int,
    expected_anchor_tokens: int,
    stay_reason: str = "short-budget",
) -> dict[str, Any]:
    """Accept a fully observed M1 refusal that never entered Shadow."""
    background = common.read_json(background_dir / "background_summary.json")
    source = common.read_json(controller_dir / "source_response.json")
    audit = _load_rows(controller_dir / "phase9_audit.jsonl")
    decision_rows = [
        row for row in audit
        if row.get("kind") == "manager_m1_start_decision"
    ]
    decisions = [row["decision"] for row in decision_rows]
    endings = [row for row in audit if row.get("kind") == "run_end"]
    transitions = [row.get("to") for row in audit if row.get("kind") == "transition"]
    tokens = source.get("token_ids") or []
    errors: list[str] = []
    if background.get("jobs") != expected_jobs or (
        background.get("completed") != expected_jobs
        or background.get("failed") != 0
    ):
        errors.append("background jobs did not all complete")
    if len(tokens) != expected_anchor_tokens:
        errors.append("source did not emit the full capped output")
    if source.get("finish_reason") != "length":
        errors.append("source did not finish at the output cap")
    if len(endings) != 1 or endings[0].get("final_state") != "COMPLETED_ON_TP1":
        errors.append("controller did not complete on TP1")
    elif endings[0].get("trigger_path") is not None:
        errors.append("controller recorded a migration trigger")
    if transitions != ["COMPLETED_ON_TP1"]:
        errors.append("controller entered a migration state")
    if not decisions or any(row.get("action") != "STAY" for row in decisions):
        errors.append("M1 did not consistently choose STAY")
    expected_reason = {
        "short-budget": "insufficient target output budget",
        "target-load": "target load exceeds admission guard",
    }[stay_reason]
    guarded = [
        row for row in decision_rows
        if row["decision"].get("reason") == expected_reason
    ]
    if not guarded:
        errors.append(f"M1 never observed the {stay_reason} guard")
    if stay_reason == "target-load" and not any(
        (
            (row.get("snapshot", {}).get("target_waiting") or 0) > 4
            or (row.get("snapshot", {}).get("target_kv_usage_frac") or 0) > 0.85
        )
        and (row.get("snapshot", {}).get("source_free_kv_tokens") or 0)
        > (row.get("snapshot", {}).get("source_guard_free_kv_tokens") or 0)
        for row in guarded
    ):
        errors.append("target-load refusal lacked live load and source slack")
    if (controller_dir / "session_manifest.json").exists():
        errors.append("a Shadow session was created")
    if (controller_dir / "cutover_manifest.json").exists():
        errors.append("a migration cutover was recorded")
    if (controller_dir / "target_response.json").exists():
        errors.append("a target response was recorded")
    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "expected_anchor_tokens": expected_anchor_tokens,
        "source_origin_tokens": len(tokens),
        "target_origin_tokens": 0,
        "m1_decisions": len(decisions),
        "m1_reasons": sorted({str(row.get("reason")) for row in decisions}),
        "stay_reason": stay_reason,
        "guarded_decisions": len(guarded),
        "peak_target_waiting": max(
            (row.get("snapshot", {}).get("target_waiting") or 0)
            for row in decision_rows
        ) if decision_rows else None,
        "final_state": endings[0].get("final_state") if endings else None,
        "errors": errors,
    }


def accept_paired_stay(
    controller_dir: Path,
    background_dir: Path,
    expected_jobs: int,
    expected_anchor_tokens: int,
    natural_eos_anchor: bool = False,
) -> dict[str, Any]:
    """Require a complete TP1 response with M1 evidence and no migration actuation."""
    background = common.read_json(background_dir / "background_summary.json")
    source = common.read_json(controller_dir / "source_response.json")
    proxy = common.read_json(controller_dir / "response_proxy_stats.json")
    audit = _load_rows(controller_dir / "phase9_audit.jsonl")
    decisions = [row for row in audit if row.get("kind") == "manager_m1_start_decision"]
    interventions = [row for row in audit if row.get("kind") == "paired_stay_intervention"]
    endings = [row for row in audit if row.get("kind") == "run_end"]
    transitions = [row.get("to") for row in audit if row.get("kind") == "transition"]
    errors: list[str] = []
    if (background.get("jobs") != expected_jobs
            or background.get("completed") != expected_jobs
            or background.get("failed") != 0):
        errors.append("background jobs did not all complete")
    source_tokens = len(source.get("token_ids") or [])
    if natural_eos_anchor:
        if (source.get("finish_reason") != "stop"
                or not 1 <= source_tokens < expected_anchor_tokens):
            errors.append("source did not naturally finish before its cap")
    elif (source.get("finish_reason") != "length"
          or source_tokens != expected_anchor_tokens):
        errors.append("source did not finish its full capped output")
    if (proxy.get("emitted_tokens") != source_tokens
            or proxy.get("source_origin_tokens") != source_tokens
            or proxy.get("target_origin_tokens") != 0
            or proxy.get("committed") is not False):
        errors.append("visible response was not entirely from TP1")
    if len(endings) != 1 or endings[0].get("final_state") != "COMPLETED_ON_TP1":
        errors.append("controller did not complete on TP1")
    elif endings[0].get("trigger_path") is not None:
        errors.append("paired STAY recorded a migration trigger")
    if transitions != ["COMPLETED_ON_TP1"]:
        errors.append("controller entered a migration state")
    if not decisions:
        errors.append("M1 did not produce online decisions")
    natural_starts = sum(
        row.get("decision", {}).get("action") == "START_SHADOW"
        for row in decisions
    )
    if natural_starts != len(interventions):
        errors.append("paired STAY intervention did not cover M1 starts")
    if any((controller_dir / name).exists() for name in (
        "session_manifest.json", "cutover_manifest.json", "target_response.json",
    )):
        errors.append("migration artifact exists in paired STAY arm")
    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "expected_anchor_tokens": expected_anchor_tokens,
        "source_origin_tokens": proxy.get("source_origin_tokens"),
        "target_origin_tokens": proxy.get("target_origin_tokens"),
        "m1_decisions": len(decisions),
        "natural_start_decisions": natural_starts,
        "interventions": len(interventions),
        "final_state": endings[0].get("final_state") if endings else None,
        "background_completed": background.get("completed"),
        "errors": errors,
    }


def active_source_peer_count(
    unix_s: float | None, source_peers: list[dict[str, Any]]
) -> int:
    if unix_s is None:
        return 0
    return sum(
        isinstance(peer.get("request_started_unix_s"), (int, float))
        and isinstance(peer.get("request_ended_unix_s"), (int, float))
        and peer["request_started_unix_s"] <= unix_s
        < peer["request_ended_unix_s"]
        for peer in source_peers
    )


def has_measured_source_high(
    audit: list[dict[str, Any]], source_peers: list[dict[str, Any]]
) -> bool:
    """Check that an M2 HIGH action used live pre-guard source telemetry."""
    for row in audit:
        if row.get("kind") == "manager_m2_initial_rate":
            decision = row.get("decision") or {}
            snapshot = row.get("snapshot") or {}
        elif row.get("kind") == "rate":
            decision = row.get("manager_m2_decision") or {}
            snapshot = row.get("manager_m2_snapshot") or {}
        else:
            continue
        horizon = decision.get("source_time_to_guard_s")
        free = snapshot.get("source_free_kv_tokens")
        guard = snapshot.get("source_guard_free_kv_tokens")
        reserved = snapshot.get("source_prefill_pending_kv_tokens")
        reserved_guard_risk = (
            decision.get("source_capacity_model")
            == "prefill_reservation_plus_decode_growth"
            and isinstance(reserved, (int, float))
            and isinstance(free, (int, float))
            and isinstance(guard, (int, float))
            and reserved >= free - guard
        )
        decision_time = row.get("unix_s")
        active_peers = active_source_peer_count(
            decision_time if isinstance(decision_time, (int, float)) else None,
            source_peers,
        )
        if (
            decision.get("action") == "SET_RATE"
            and decision.get("profile") == "HIGH"
            and decision.get("reason") == "source guard horizon is short"
            and isinstance(horizon, (int, float))
            and (
                0 < horizon <= 30.0
                or (horizon == 0 and reserved_guard_risk)
            )
            and isinstance(free, (int, float))
            and isinstance(guard, (int, float))
            and free > guard
            and isinstance(snapshot.get("source_running"), (int, float))
            and snapshot["source_running"] >= 2
            and active_peers >= 2
        ):
            return True
    return False


def m2_expected_profile_used(
    initial_rows: list[dict[str, Any]],
    rate_rows: list[dict[str, Any]],
    expected_profile: str | None,
    profiles_gib_s: tuple[float, float, float],
) -> bool:
    """Accept an effective initial profile without requiring a transition."""
    if not rate_rows:
        return False
    decisions = [row.get("decision") or {} for row in initial_rows] + [
        row.get("manager_m2_decision") or {} for row in rate_rows
    ]
    if any(
        decision.get("action") == "SET_RATE"
        and (expected_profile is None
             or decision.get("profile") == expected_profile)
        for decision in decisions
    ):
        return True
    if expected_profile is None:
        return False
    expected_rate = dict(zip(
        ("LOW", "MEDIUM", "HIGH"), profiles_gib_s
    ))[expected_profile]
    return any(
        (row.get("decision") or {}).get("action") == "HOLD"
        and row["decision"].get("profile") == expected_profile
        and abs(float(row["decision"].get("rate_bytes_s", 0))
                / 1024**3 - expected_rate) <= 1e-9
        for row in initial_rows
    ) and any(
        abs(float(row.get("rate_gib_s", 0)) - expected_rate) <= 1e-9
        for row in rate_rows
    )


def accept_m5_shadow(
    accepted: dict[str, Any], controller_dir: Path,
    event_path: Path, checkpoint_sha256: str,
) -> dict[str, Any]:
    """Require real source inference and manager consumption in M5 smoke."""
    errors = list(accepted.get("errors", []))
    audit_path = controller_dir / "phase9_audit.jsonl"
    predictions = 0
    available = 0
    measured_blocks = None
    sampled_peak_mib = None
    source_log = controller_dir / "source_tp1.log"
    if not source_log.is_file():
        errors.append("M5 source server log is missing")
    else:
        source_config = source_log.read_text(encoding="utf-8", errors="replace")
        if (
            "enforce_eager=False" not in source_config
            or "CompilationMode.VLLM_COMPILE" not in source_config
            or "CUDAGraphMode.FULL_AND_PIECEWISE" not in source_config
            or "vLLM's torch.compile cache is disabled" not in source_config
        ):
            errors.append(
                "M5 source did not retain uncached compilation and CUDA graphs"
            )
    provenance = controller_dir.parent / "provenance"
    try:
        capacity = common.read_json(provenance / "source_kv_capacity.json")
        config = common.read_json(provenance / "controller_config.json")
        memory = common.read_json(provenance / "source_gpu_memory_summary.json")
        measured_blocks = int(capacity["measured_blocks"])
        sampled_peak_mib = int(memory["sampled_peak_used_mib"])
        if (
            measured_blocks <= 0
            or measured_blocks != int(config["tp1_total_kv_blocks"])
            or measured_blocks * int(config["block_size"])
            != int(capacity["measured_tokens"])
        ):
            errors.append("M5 controller did not use measured TP1 KV capacity")
        if int(memory["sample_count"]) < 5:
            errors.append("M5 TP1 runtime GPU memory samples are incomplete")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"M5 capacity or runtime GPU memory evidence is missing: {exc}")
    if not event_path.is_file() or not audit_path.is_file():
        errors.append("M5 source events or manager audit are missing")
    else:
        events = [json.loads(line) for line in event_path.read_text(
            encoding="utf-8"
        ).splitlines() if line]
        if not events or events[0].get("checkpoint_sha256") != checkpoint_sha256:
            errors.append("M5 source event header has wrong checkpoint SHA")
        predictions = sum(
            row.get("kind") == "predictor_prediction" for row in events
        )
        audit = [json.loads(line) for line in audit_path.read_text(
            encoding="utf-8"
        ).splitlines() if line]
        available_rows = [
            row for row in audit
            if row.get("kind") == "manager_m5_predictor_shadow"
            and row.get("status") == "AVAILABLE"
        ]
        available = len(available_rows)
        if predictions == 0 or available == 0:
            errors.append("M5 did not produce and consume an online prediction")
        for row in available_rows:
            cap = row.get("max_remaining_output_tokens")
            headroom = row.get("headroom_tokens")
            runtime_bounds = row.get("p_remaining_gt_headroom_runtime_bounds")
            if (
                not isinstance(cap, int)
                or not isinstance(headroom, int)
                or not isinstance(runtime_bounds, list)
                or len(runtime_bounds) != 2
            ):
                errors.append("M5 runtime output-cap audit is missing")
                break
            if row.get("ignore_eos") is True:
                expected = float(cap > headroom)
                if not all(abs(float(value) - expected) <= 1e-6
                           for value in runtime_bounds):
                    errors.append("M5 forced-output cap risk bound differs")
                    break
            elif headroom >= cap and any(
                abs(float(value)) > 1e-6 for value in runtime_bounds
            ):
                errors.append("M5 output-cap risk bound differs")
                break
    return {
        **accepted,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "m5_source_predictions": predictions,
        "m5_manager_available_ticks": available,
        "m5_measured_source_kv_blocks": measured_blocks,
        "m5_sampled_tp1_gpu_peak_mib": sampled_peak_mib,
    }


def accept_m4_cancel(
    controller_dir: Path,
    _background_dir: Path,
    _expected_jobs: int,
    expected_anchor_tokens: int,
) -> dict[str, Any]:
    """Check that pre-freeze cancel retains the complete TP1 response."""
    errors: list[str] = []
    audit_path = controller_dir / "phase9_audit.jsonl"
    source_path = controller_dir / "source_response.json"
    unified_path = controller_dir / "unified_response.jsonl"
    if not all(path.is_file() for path in (
        audit_path, source_path, unified_path
    )):
        return {
            "status": "FAIL",
            "errors": ["M4 cancellation evidence is incomplete"],
        }
    audit = [json.loads(line) for line in audit_path.read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]
    source = common.read_json(source_path)
    unified = [json.loads(line) for line in unified_path.read_text(
        encoding="utf-8"
    ).splitlines() if line.strip()]
    decisions = [row for row in audit if (
        row.get("kind") == "manager_m4_cancel_decision"
        and (row.get("decision") or {}).get("action") == "CANCEL_SHADOW"
    )]
    end = [row for row in audit if row.get("kind") == "run_end"]
    cleanup = [row for row in audit if row.get("kind") == "manager_m4_source_cleanup"]
    if not decisions or not any(
        row.get("kind") == "manager_m4_cancelled" for row in audit
    ):
        errors.append("M4 did not cancel Shadow")
    if not end or end[-1].get("final_state") != "CANCELLED":
        errors.append("M4 did not reach CANCELLED")
    if controller_dir.joinpath("request_frozen_receipt.json").exists():
        errors.append("M4 cancelled after source freeze")
    if any(row.get("to") == "TAKEOVER" for row in audit):
        errors.append("M4 committed after cancellation")
    if not cleanup or cleanup[-1].get("source_abort_dispatched") is not False:
        errors.append("M4 did not preserve TP1 ownership")
    takeover_path = controller_dir / "takeover_state.json"
    takeover = (
        common.read_json(takeover_path) if takeover_path.is_file() else {}
    )
    if (
        takeover.get("state") != "CANCELLED"
        or takeover.get("source_continues_on_tp1") is not True
    ):
        errors.append("M4 source cleanup state is incomplete")
    control_path = controller_dir / "runtime_control.json"
    control = common.read_json(control_path) if control_path.is_file() else {}
    if control.get("target_request_admitted"):
        target_path = controller_dir / "target_cleanup_receipt.json"
        target = common.read_json(target_path) if target_path.is_file() else {}
        if target.get("status") != "CLEANED":
            errors.append("M4 did not release the dormant TP4 request")
    elif (controller_dir / "session_manifest.json").is_file():
        for rank in range(4):
            path = (controller_dir / "gpu_cancel_cleanup_receipts"
                    / f"tp_rank_{rank}.json")
            receipt = common.read_json(path) if path.is_file() else {}
            if (
                receipt.get("status") != "CANCELLED_PREBOUND_HISTORY_RELEASED"
                or receipt.get("channel_state") != "IDLE"
            ):
                errors.append(f"M4 rank {rank} kept prebound GPU history")
        sender_path = controller_dir / "gpu_direct_sender.json"
        sender = (
            common.read_json(sender_path) if sender_path.is_file() else {}
        )
        if sender.get("communicator_lifecycle") != "PERSISTENT_CHANNEL_IDLE":
            errors.append("M4 source GPU channel did not return to IDLE")
    source_ids = source.get("token_ids") or []
    unified_ids = [row.get("token_id") for row in unified]
    if len(source_ids) != expected_anchor_tokens:
        errors.append("TP1 did not complete its output budget")
    if source_ids != unified_ids or any(
        row.get("origin") != "source" for row in unified
    ):
        errors.append("unified response differs from TP1 output")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "m4_cancel_output_tokens": (
            decisions[0]["decision"].get("output_tokens") if decisions else None
        ),
        "source_output_tokens": len(source_ids),
        "unified_output_tokens": len(unified_ids),
    }


def accept_online(
    controller_dir: Path,
    background_dir: Path,
    expected_jobs: int,
    expected_anchor_tokens: int,
    *,
    natural_eos_anchor: bool = False,
    strategy: str,
    minimum_window_samples: int,
    fixed_rate_gib_s: float | None = None,
    manager_m2_rate: bool = False,
    manager_m3_commit: bool = False,
    m2_profiles_gib_s: tuple[float, float, float] | None = None,
    manager_m2_expected_profile: str | None = None,
    manager_m2_min_history_byte_frac: float = 0.0,
    manager_m2_require_source_high: bool = False,
    manager_m2_require_low_to_high: bool = False,
    gpu_direct_history_pacing_expected: bool = False,
    handoff_mode: str = "bridge",
    require_remote_attention: bool = False,
    slo_tpot_ms: float = 50.0,
    slo_ttft_ms: float = 1000.0,
    slo_e2e_ms: float = 60000.0,
    slo_handoff_ms: float = 1000.0,
    stop_and_copy: bool = False,
    ready_sync_mode: str = "DEVICE_WIDE",
    ready_notification_mode: str = "FILE_POLL",
    deferred_comm_destroy: bool = False,
    post_takeover_comm_destroy: bool = False,
    persistent_channel: bool = False,
    preconnect_persistent_channel: bool = False,
    persistent_expected_session_count: int = 1,
    commit_timing: str = "FIXED",
    manager_m1_auto_start: bool = False,
    manager_m1_expect_stay: bool = False,
    manager_m1_stay_reason: str = "short-budget",
    source_pressure_expected_jobs: int = 0,
    minimum_source_kv_usage_frac: float = 0.0,
) -> dict[str, Any]:
    if manager_m1_expect_stay:
        return accept_m1_stay(
            controller_dir, background_dir, expected_jobs, expected_anchor_tokens,
            manager_m1_stay_reason,
        )
    background = common.read_json(background_dir / "background_summary.json")
    session = common.read_json(controller_dir / "session_manifest.json")
    cutover = common.read_json(controller_dir / "cutover_manifest.json")
    staging = common.read_json(controller_dir / "staging_manifest.json")
    takeover = common.read_json(controller_dir / "takeover_state.json")
    proxy = common.read_json(controller_dir / "response_proxy_stats.json")
    source_response_path = controller_dir / "source_response.json"
    target_response_path = controller_dir / "target_response.json"
    source_response = (
        common.read_json(source_response_path)
        if source_response_path.is_file()
        else {}
    )
    target_response = (
        common.read_json(target_response_path)
        if target_response_path.is_file()
        else {}
    )
    audit = _load_rows(controller_dir / "phase9_audit.jsonl")
    source_peers = [
        row for row in background.get("results", [])
        if row.get("pool") == "source"
    ]
    target_background = [
        row for row in background.get("results", [])
        if row.get("pool") == "target"
    ]
    source_telemetry = [
        row["tp1"] for row in audit
        if row.get("kind") == "telemetry"
        and isinstance(row.get("tp1"), dict)
    ]
    source_usage = [
        float(row["kv_usage_frac"]) for row in source_telemetry
        if isinstance(row.get("kv_usage_frac"), (int, float))
    ]
    source_preemptions = [
        int(row["preemptions_total"]) for row in source_telemetry
        if isinstance(row.get("preemptions_total"), (int, float))
    ]
    source_pressure_evidence = {
        "peer_jobs": len(source_peers),
        "peer_completed": sum(
            row.get("status") == "COMPLETED" for row in source_peers
        ),
        "peer_failed": sum(row.get("status") == "FAILED" for row in source_peers),
        "peer_output_tokens": sum(
            int(row.get("output_tokens", 0)) for row in source_peers
        ),
        "peer_e2e_ms": [
            row.get("e2e_ms") for row in source_peers
            if row.get("status") == "COMPLETED"
        ],
        "peer_tpot_p95_ms": [
            row.get("tpot_p95_ms") for row in source_peers
            if row.get("status") == "COMPLETED"
        ],
        "telemetry_samples": len(source_telemetry),
        "overlap_samples": sum(
            int(row.get("num_running", 0)) >= 2 for row in source_telemetry
        ),
        "active_peers_at_m2_initial_rate": max(
            (
                active_source_peer_count(row.get("unix_s"), source_peers)
                for row in audit
                if row.get("kind") == "manager_m2_initial_rate"
            ),
            default=0,
        ),
        "peak_tp1_kv_usage_frac": max(source_usage, default=None),
        "minimum_free_kv_tokens": min(
            (
                int(row["free_kv_blocks"]) * int(row["block_size"])
                for row in source_telemetry
                if isinstance(row.get("free_kv_blocks"), (int, float))
                and isinstance(row.get("block_size"), (int, float))
            ),
            default=None,
        ),
        "preemption_delta": (
            max(source_preemptions) - source_preemptions[0]
            if source_preemptions else None
        ),
    }
    earliest_ready_armed = [
        row
        for row in audit
        if row.get("kind") == (
            "manager_m1_earliest_ready_armed"
            if manager_m1_auto_start
            else "diagnostic_earliest_ready_armed"
        )
    ]
    m1_starts = [
        row for row in audit
        if row.get("kind") == "manager_m1_start_decision"
        and row.get("decision", {}).get("action") == "START_SHADOW"
    ]
    earliest_ready_selected = [
        row
        for row in audit
        if row.get("kind") == "earliest_ready_cutover_selected"
    ]
    urgent_prearmed = [
        row for row in audit
        if row.get("kind") == "urgent_cutover_prearmed"
    ]
    urgent_history_wait_allowed = (
        len(urgent_prearmed) == 1
        and manager_m2_rate
        and commit_timing == "EARLIEST_READY"
    )
    earliest_ready_candidates = [
        row
        for row in audit
        if row.get("kind") == "earliest_ready_candidate_published"
    ]
    if manager_m3_commit:
        m3_candidates = [
            row for row in audit
            if row.get("kind") == "manager_m3_candidate_decision"
        ]
        if len(m3_candidates) != 1 or len(earliest_ready_candidates) != 1:
            errors.append("M3 did not plan exactly one candidate before admission")
        elif (
            m3_candidates[0].get("decision", {}).get("candidate_output_tokens")
            != earliest_ready_candidates[0].get("cutover_output_tokens")
        ):
            errors.append("M3 planned candidate differs from admitted target")
        if (
            m3_candidates
            and m3_candidates[0].get("decision", {}).get("action")
            != "COMMIT_EARLIEST"
        ):
            errors.append("M3 did not choose the earliest safe boundary")
        if (
            m3_candidates
            and m3_candidates[0].get("decision", {}).get(
                "candidate_output_tokens"
            ) != m3_candidates[0].get("base_candidate_output_tokens")
        ):
            errors.append("M3 added a delay to the safe base candidate")
    end_rows = [row for row in audit if row.get("kind") == "run_end"]
    transitions = [row.get("to") for row in audit if row.get("kind") == "transition"]
    receipts, receipt_errors = rescue.receipt_evidence(controller_dir)
    target_receipts = [
        common.read_json(path)
        for path in sorted(
            (controller_dir / "receiver_receipts").glob("*/*.json")
        )
    ]
    communicator_destroy_receipts = [
        common.read_json(path)
        for path in sorted(
            (controller_dir / "gpu_communicator_destroy_receipts").glob(
                "tp_rank_*.json"
            )
        )
    ]
    initial_stage_receipts = [
        common.read_json(path)
        for path in sorted((controller_dir / "initial_stage_receipts").glob("*.json"))
    ]
    gpu_initial_receipts = [
        common.read_json(path)
        for path in sorted((controller_dir / "gpu_initial_receipts").glob("*.json"))
    ]
    frozen_receipt_path = controller_dir / "request_frozen_receipt.json"
    frozen_receipt = (
        common.read_json(frozen_receipt_path)
        if frozen_receipt_path.is_file()
        else None
    )
    remote_attention_path = controller_dir / "online_remote_attention.jsonl"
    remote_attention_rows = (
        _load_rows(remote_attention_path) if remote_attention_path.is_file() else []
    )
    cutover_context_tokens = int(cutover["num_prompt_tokens"]) + int(
        cutover["cutover_num_output_tokens"]
    )
    remote_attention_visible_rows = [
        row
        for row in remote_attention_rows
        if int(row.get("sequence_tokens", -1)) < cutover_context_tokens
    ]
    remote_attention_speculative_rows = [
        row
        for row in remote_attention_rows
        if int(row.get("sequence_tokens", -1)) >= cutover_context_tokens
    ]

    shadow_start = float(session["shadow_started_unix_s"])
    freeze_unix_s = (
        int(frozen_receipt["frozen_unix_ns"]) / 1e9
        if frozen_receipt is not None
        else float(cutover["updated_unix_s"])
    )
    bridge_marker_path = controller_dir / "remote_attention_bridge.json"
    bridge_start = (
        float(common.read_json(bridge_marker_path)["started_unix_s"])
        if require_remote_attention and bridge_marker_path.is_file()
        else freeze_unix_s
    )
    committed = float(takeover["updated_unix_s"])
    rank_ready_times = [
        float(row["target_ready_unix_s"])
        for row in target_receipts
        if row.get("target_ready_unix_s") is not None
    ]
    delta_apply_times = [
        float(common.read_json(path)["completed_unix_s"])
        for path in sorted(
            (controller_dir / "gpu_delta_receipts").glob("**/*.json")
        )
    ]
    commit_rows = [
        row for row in audit if row.get("kind") == "shadow_only_commit"
    ]
    commit_row = commit_rows[-1] if commit_rows else {}
    controller_wakeup = commit_row.get("controller_wakeup_unix_s")
    commit_completed = commit_row.get("commit_completed_unix_s", committed)
    ready_notification = commit_row.get("ready_notification") or {}
    if strategy == "S_NEW":
        history_start = float(
            common.read_json(controller_dir / "history_transfer_start.json")[
                "started_unix_s"
            ]
        )
    else:
        history_start = float(session["history_transfer_started_unix_s"])
    windows = summarize_background_windows(
        background.get("results", []),
        shadow_start_unix_s=shadow_start,
        bridge_start_unix_s=bridge_start,
        committed_unix_s=committed,
    )

    errors: list[str] = []
    if len(urgent_prearmed) > 1:
        errors.append("urgent source cutover was prearmed more than once")
    if require_remote_attention:
        if not remote_attention_rows:
            errors.append("online Bridge recorded no remote-attention data-path calls")
        elif any(row.get("status") != "PASS" for row in remote_attention_rows):
            errors.append("online Bridge contains a failed remote-attention call")
        layer_names = {str(row.get("layer_name")) for row in remote_attention_rows}
        if len(layer_names) != 48:
            errors.append(
                "online Bridge exercised "
                f"{len(layer_names)} attention layers, expected 48"
            )
        token_layers: dict[tuple[str, int], list[str]] = {}
        for row in remote_attention_rows:
            token_key = (
                str(row.get("request_id")),
                int(row.get("sequence_tokens", -1)),
            )
            token_layers.setdefault(token_key, []).append(str(row.get("layer_name")))
        incomplete_tokens = {
            f"{request_id}@{sequence_tokens}": {
                "calls": len(names),
                "unique_layers": len(set(names)),
            }
            for (request_id, sequence_tokens), names in token_layers.items()
            if len(names) != 48 or len(set(names)) != 48
        }
        if incomplete_tokens:
            errors.append(
                "online Bridge changed attention paths within a decode token: "
                f"{incomplete_tokens}"
            )
        if any(len(row.get("ranks", [])) != 4 for row in remote_attention_rows):
            errors.append("online Bridge did not use all four TP4 ranks per layer")
        if any(
            int(row.get("remote_prefix_tokens", 0)) <= 0
            for row in remote_attention_rows
        ):
            errors.append("online Bridge used an empty TP4 prefix")
        verification_rows = [
            row for row in remote_attention_rows if row.get("max_abs_error") is not None
        ]
        if len({str(row.get("layer_name")) for row in verification_rows}) != 48:
            errors.append("online Bridge did not numerically verify all 48 layers")
        if any(float(row["max_abs_error"]) > 0.02 for row in verification_rows):
            errors.append(
                "online Bridge exceeded the split-attention max-abs tolerance"
            )
        if any(float(row["cosine_similarity"]) < 0.999 for row in verification_rows):
            errors.append("online Bridge exceeded the split-attention cosine tolerance")
    if background.get("jobs") != expected_jobs:
        errors.append("background job count differs from manifest")
    if background.get("completed") != expected_jobs or background.get("failed") != 0:
        errors.append("target background workload did not complete")
    if source_pressure_expected_jobs:
        if len(source_peers) != source_pressure_expected_jobs:
            errors.append("source pressure peer count differs from manifest")
        if not source_usage:
            errors.append("source pressure has no TP1 KV telemetry")
        elif max(source_usage) < minimum_source_kv_usage_frac:
            errors.append(
                "source pressure TP1 KV usage peak below required "
                f"{minimum_source_kv_usage_frac:.3f}"
            )
        if not source_pressure_evidence["overlap_samples"]:
            errors.append("source peers did not overlap anchor decode")
    if session.get("shadow_strategy") != strategy:
        errors.append("session recorded the wrong Shadow strategy")
    if staging.get("shadow_strategy") != strategy:
        errors.append("staging recorded the wrong Shadow strategy")
    expected_order = "TOKEN_ASCENDING_FROM_REQUEST_START"
    if session.get("history_copy_order") != expected_order:
        errors.append("history KV was not recorded as head-first token order")
    if not stop_and_copy:
        errors.extend(
            validate_strategy_timing(
                strategy,
                shadow_start_unix_s=shadow_start,
                bridge_start_unix_s=bridge_start,
                history_start_unix_s=history_start,
            )
        )
    expected_transitions = (
        ["SHADOW", "READY_NOT_COMMITTED", "TAKEOVER"]
        if manager_m3_commit
        else ["SHADOW", "TAKEOVER"]
        if handoff_mode == "shadow-only"
        else ["SHADOW", "HANDOFF", "TAKEOVER"]
    )
    if transitions[-len(expected_transitions) :] != expected_transitions:
        errors.append(f"unexpected migration transitions: {transitions!r}")
    if manager_m1_auto_start:
        if len(m1_starts) != 1:
            errors.append("M1 did not choose exactly one autonomous Shadow start")
        if any(row.get("kind") == "diagnostic_boundary_forced" for row in audit):
            errors.append("M1 run used a diagnostic fixed start")
    if handoff_mode == "shadow-only" and not stop_and_copy:
        if frozen_receipt is None:
            errors.append("scheduler did not acknowledge the Shadow-only freeze")
        elif int(frozen_receipt.get("num_output_tokens", -1)) != int(
            cutover["cutover_num_output_tokens"]
        ):
            errors.append("scheduler froze Shadow-only at the wrong token boundary")
    if commit_timing == "EARLIEST_READY":
        if len(earliest_ready_armed) != 1:
            errors.append("earliest-ready cutover was not armed exactly once")
        if len(earliest_ready_selected) != 1:
            errors.append("earliest-ready cutover was not selected exactly once")
        elif int(earliest_ready_selected[0].get("cutover_output_tokens", -1)) != int(
            cutover["cutover_num_output_tokens"]
        ):
            errors.append("selected earliest-ready boundary differs from source freeze")
        if earliest_ready_selected:
            selected = earliest_ready_selected[0]
            lag = selected.get("delta_lag_tokens")
            progress = selected.get("rank_gpu_resident_end_tokens")
            if not isinstance(lag, int) or lag > 16:
                errors.append("earliest-ready selected with excessive delta lag")
            if not isinstance(progress, dict) or len(progress) != 4:
                errors.append("earliest-ready lacks four-rank delta progress")
    history_completed = [
        float(row.get("completed_unix_s", float("inf")))
        for row in initial_stage_receipts
    ]
    if (
        handoff_mode == "shadow-only"
        and not stop_and_copy
        and (
            len(history_completed) != 4
            or (
                not urgent_history_wait_allowed
                and any(value > freeze_unix_s for value in history_completed)
            )
        )
    ):
        errors.append(
            "Shadow-only history did not finish staging on all ranks before "
            "the source freeze boundary"
        )
    history_ready_before_freeze_ms = (
        (freeze_unix_s - max(history_completed)) * 1000
        if len(history_completed) == 4
        else None
    )
    gpu_resident_shadow = staging.get("gpu_resident_shadow") is True
    gpu_direct_history = staging.get("gpu_direct_history") is True
    gpu_direct_delta = staging.get("gpu_direct_delta") is True
    direct_sender: dict[str, Any] = {}
    m2_history_profile_byte_fraction: float | None = None
    target_channel_receipts: list[dict[str, Any]] = []
    preconnect_receipts: list[dict[str, Any]] = []
    if gpu_direct_history:
        direct_sender_path = controller_dir / "gpu_direct_sender.json"
        if not direct_sender_path.is_file():
            errors.append("GPU-direct history sender receipt is missing")
        else:
            direct_sender = common.read_json(direct_sender_path)
            direct_ranks = direct_sender.get("ranks", [])
            if gpu_direct_history_pacing_expected:
                paced = direct_ranks[0] if direct_ranks else {}
                chunks = paced.get("history_pacing_chunks", [])
                allowed_rates = (
                    m2_profiles_gib_s if manager_m2_rate
                    else (fixed_rate_gib_s,)
                )
                if (
                    len(direct_ranks) != 4
                    or any(row.get("history_pacing_enabled") is not True
                           for row in direct_ranks)
                    or len(chunks) < 2
                    or len(chunks) != paced.get("history_pacing_chunk_count")
                    or any(
                        all(
                            abs(float(row.get("requested_rate_gib_s", 0))
                                - float(profile or 0)) > 1e-9
                            for profile in allowed_rates
                        )
                        for row in chunks
                    )
                ):
                    errors.append("GPU-direct history pacing was not applied")
                else:
                    total_bytes = sum(int(row.get("raw_tensor_bytes", 0))
                                      for row in direct_ranks)
                    burst_bytes = int(chunks[-1].get("aggregate_bytes", 0))
                    # Slow sends can satisfy the rate cap without sleeping.
                    minimum_ms = (
                        (total_bytes - burst_bytes)
                        / (max(float(rate) for rate in allowed_rates) * 1024**3)
                        * 1000
                    )
                    if float(paced.get("history_pacing_span_ms", 0)) < minimum_ms * 0.98:
                        errors.append("GPU-direct history pacing span is too short")
                if manager_m2_rate and manager_m2_expected_profile is not None:
                    assert m2_profiles_gib_s is not None
                    profile_rate = dict(zip(
                        ("LOW", "MEDIUM", "HIGH"), m2_profiles_gib_s
                    ))[manager_m2_expected_profile]
                    if not any(
                        abs(float(row.get("requested_rate_gib_s", 0))
                            - profile_rate) <= 1e-9
                        for row in chunks
                    ):
                        errors.append(
                            "M2 expected rate was not used by paced GPU history"
                        )
                    total_paced_bytes = sum(
                        int(row.get("aggregate_bytes", 0)) for row in chunks
                    )
                    expected_paced_bytes = sum(
                        int(row.get("aggregate_bytes", 0))
                        for row in chunks
                        if abs(float(row.get("requested_rate_gib_s", 0))
                               - profile_rate) <= 1e-9
                    )
                    actual_fraction = (
                        expected_paced_bytes / total_paced_bytes
                        if total_paced_bytes else 0.0
                    )
                    m2_history_profile_byte_fraction = actual_fraction
                    if actual_fraction < manager_m2_min_history_byte_frac:
                        errors.append(
                            "M2 expected profile paced too few history bytes: "
                            f"{actual_fraction:.3f} < "
                            f"{manager_m2_min_history_byte_frac:.3f}"
                        )
            if (
                direct_sender.get("status") != "READY"
                or len(direct_ranks) != 4
                or [int(row.get("target_tp_rank", -1)) for row in direct_ranks]
                != list(range(4))
            ):
                errors.append("GPU-direct history sender did not complete all ranks")
            if deferred_comm_destroy and direct_sender.get(
                "communicator_lifecycle"
            ) != "POOLED_UNTIL_PROCESS_SHUTDOWN":
                errors.append(
                    "GPU-direct source communicators did not enter the "
                    "process-lifetime pool"
                )
            if persistent_channel:
                if preconnect_persistent_channel:
                    if any(
                        row.get("channel_reused") is not True
                        for row in direct_ranks
                    ):
                        errors.append(
                            "GPU-direct sender did not reuse a preconnected channel"
                        )
                    if persistent_expected_session_count == 1:
                        preconnect_dir = (
                            controller_dir / "gpu_channel_preconnect_receipts"
                        )
                        paths = [preconnect_dir / "source.json"] + [
                            preconnect_dir / f"tp_rank_{rank}.json"
                            for rank in range(4)
                        ]
                        if not all(path.is_file() for path in paths):
                            errors.append(
                                "preconnect receipts are incomplete"
                            )
                        else:
                            preconnect_receipts = [
                                common.read_json(path) for path in paths
                            ]
                            anchor_start = (
                                float(source_response["request_started_unix_s"])
                                if source_response is not None
                                else 0.0
                            )
                            if any(
                                row.get("status") != "WARMUP_COMPLETE"
                                or int(row.get("channel_generation", -1))
                                != int(session.get("channel_generation", -1))
                                or int(row.get("warmup_payload_bytes", 0)) <= 0
                                or float(row.get("completed_unix_s", 0.0))
                                >= anchor_start
                                for row in preconnect_receipts
                            ):
                                errors.append(
                                    "channel warmup was not complete before anchor"
                                )
                if direct_sender.get("communicator_lifecycle") != (
                    "PERSISTENT_CHANNEL_IDLE"
                ):
                    errors.append(
                        "GPU-direct source channel did not return to IDLE"
                    )
                if int(direct_sender.get("channel_create_count", -1)) != 1:
                    errors.append(
                        "GPU-direct source channel create count differs"
                    )
                if int(direct_sender.get("channel_destroy_count", -1)) != 0:
                    errors.append(
                        "GPU-direct source channel was destroyed during request"
                    )
                if int(direct_sender.get("channel_session_count", -1)) != int(
                    persistent_expected_session_count
                ):
                    errors.append(
                        "GPU-direct source channel session count differs"
                    )
                target_channel_receipts = [
                    common.read_json(path)
                    for path in sorted(
                        (
                            controller_dir
                            / "persistent_channel_receipts"
                        ).glob("tp_rank_*.json")
                    )
                ]
                if len(target_channel_receipts) != 4:
                    errors.append(
                        "persistent target channel receipts are incomplete"
                    )
                for row in target_channel_receipts:
                    if row.get("status") != "PERSISTENT_CHANNEL_IDLE":
                        errors.append(
                            "persistent target channel did not return to IDLE"
                        )
                    if int(row.get("channel_create_count", -1)) != 1:
                        errors.append(
                            "persistent target channel create count differs"
                        )
                    if int(row.get("channel_destroy_count", -1)) != 0:
                        errors.append(
                            "persistent target channel was destroyed during request"
                        )
                    if int(row.get("channel_session_count", -1)) != int(
                        persistent_expected_session_count
                    ):
                        errors.append(
                            "persistent target channel session count differs"
                        )
                    if row.get("session_request_id") != session.get(
                        "source_request_id"
                    ):
                        errors.append(
                            "persistent target session request identity differs"
                        )
        expected_sync_scope = (
            "BRIDGETP_RESTORE_STREAM_EVENT"
            if ready_sync_mode == "STREAM_EVENT"
            else "CUDA_DEVICE_WIDE_SYNCHRONIZE"
        )
        expected_device_sync = ready_sync_mode == "DEVICE_WIDE"
        if receipts.get("ready_sync_scopes") != [expected_sync_scope] * 4:
            errors.append("TP4 ranks used the wrong target-ready synchronization")
        if receipts.get("device_wide_synchronize") != [
            expected_device_sync
        ] * 4:
            errors.append("TP4 ranks reported the wrong device synchronization")
        if ready_sync_mode == "STREAM_EVENT":
            if receipts.get("receive_dependency_scopes") != [
                "NCCL_RECEIVE_EVENT_TO_RESTORE_STREAM"
            ] * 4:
                errors.append("TP4 ranks did not chain NCCL receive events to restore")
            if receipts.get("model_stream_wait_event") != [True] * 4:
                errors.append("TP4 model streams did not wait for copy-done events")
        if deferred_comm_destroy:
            ready_by_rank = {
                int(row.get("tp_rank", -1)): float(row["target_ready_unix_s"])
                for row in target_receipts
                if row.get("target_ready_unix_s") is not None
            }
            destroy_by_rank = {
                int(row.get("tp_rank", -1)): row
                for row in communicator_destroy_receipts
            }
            if sorted(destroy_by_rank) != list(range(4)):
                errors.append(
                    "communicator lifecycle receipts are incomplete"
                )
            else:
                for rank, destroy in sorted(destroy_by_rank.items()):
                    status = destroy.get("status")
                    if post_takeover_comm_destroy:
                        if status != "DESTROYED":
                            errors.append(
                                f"TP4 rank {rank} post-takeover communicator "
                                "destruction did not complete"
                            )
                        started = destroy.get("destroy_started_unix_s")
                        commit_observed = destroy.get("commit_observed_unix_s")
                        if (
                            started is None
                            or commit_observed is None
                            or float(started) < float(commit_observed)
                        ):
                            errors.append(
                                f"TP4 rank {rank} communicator destruction "
                                "started before ownership commit"
                            )
                    elif status not in {
                        "POOLED_UNTIL_CONNECTOR_SHUTDOWN",
                        "ABORTED_AT_CONNECTOR_SHUTDOWN",
                    }:
                        errors.append(
                            f"TP4 rank {rank} communicator did not enter the "
                            "connector-lifetime pool"
                        )
                    if destroy.get("deferred_until_after_target_ready") is not True:
                        errors.append(
                            f"TP4 rank {rank} communicator teardown was not deferred"
                        )
                    lifecycle_unix_s = (
                        destroy.get("destroy_started_unix_s")
                        if post_takeover_comm_destroy
                        else destroy.get("retained_unix_s")
                    )
                    if (
                        lifecycle_unix_s is None
                        or rank not in ready_by_rank
                        or float(lifecycle_unix_s) < ready_by_rank[rank]
                    ):
                        errors.append(
                            f"TP4 rank {rank} communicator lifecycle action "
                            "started before TARGET_READY"
                        )
                    started_unix_s = destroy.get("destroy_started_unix_s")
                    if (
                        not post_takeover_comm_destroy
                        and status == "POOLED_UNTIL_CONNECTOR_SHUTDOWN"
                        and started_unix_s is not None
                    ):
                        errors.append(
                            f"TP4 rank {rank} communicator destruction started "
                            "before connector shutdown"
                        )
    gpu_history_completed = [
        float(
            row.get(
                "resident_completed_unix_s",
                row.get(
                    "ready_completed_unix_s",
                    row.get("completed_unix_s", float("inf")),
                ),
            )
        )
        for row in gpu_initial_receipts
    ]
    if (
        gpu_resident_shadow
        and not stop_and_copy
        and (
            len(gpu_history_completed) != 4
            or (
                not urgent_history_wait_allowed
                and any(value > freeze_unix_s for value in gpu_history_completed)
            )
            or not all(
                row.get("exact_readback") is True for row in gpu_initial_receipts
            )
        )
    ):
        errors.append(
            "initial history was not GPU-resident on all ranks before cutover"
        )
    if commit_timing == "EARLIEST_READY" and earliest_ready_selected:
        selected_unix_s = float(earliest_ready_selected[0].get("unix_s", 0.0))
        # The receipt is rewritten when the temporary receive buffer becomes
        # resident. Compare against the immutable resident-ready timestamp,
        # rather than a later mutable ``completed_unix_s`` value.
        gpu_history_ready = [
            float(
                row.get(
                    "ready_completed_unix_s",
                    row.get(
                        "resident_completed_unix_s",
                        row.get("completed_unix_s", float("inf")),
                    ),
                )
            )
            for row in gpu_initial_receipts
        ]
        if (
            len(gpu_history_ready) != 4
            or any(value > selected_unix_s for value in gpu_history_ready)
            or not all(
                row.get("exact_readback") is True for row in gpu_initial_receipts
            )
        ):
            errors.append(
                "earliest-ready boundary was selected before all initial GPU "
                "history receipts completed"
            )
    if gpu_resident_shadow:
        initial_end = int(session["num_computed_tokens"])
        final_end = int(cutover["num_computed_tokens"])
        expected_blocks = int(session["num_blocks"])
        for rank in range(4):
            block_paths = sorted(
                (controller_dir / "gpu_block_receipts" / f"tp_rank_{rank}").glob(
                    "*.json"
                )
            )
            blocks = [common.read_json(path) for path in block_paths]
            if (
                len(blocks) != expected_blocks
                or [int(row.get("logical_block", -1)) for row in blocks]
                != list(range(expected_blocks))
                or not all(row.get("exact_readback") is True for row in blocks)
            ):
                errors.append(f"TP4 rank {rank} history block ACKs are incomplete")
            delta_paths = sorted(
                (controller_dir / "gpu_delta_receipts" / f"tp_rank_{rank}").glob(
                    "*.json"
                )
            )
            expected_start = initial_end
            for path in delta_paths:
                delta = common.read_json(path)
                start = int(delta.get("start_token", -1))
                end = int(delta.get("end_token", -1))
                if (
                    start != expected_start
                    or end <= start
                    or delta.get("exact_readback") is not True
                ):
                    errors.append(f"TP4 rank {rank} delta ACK coverage is invalid")
                    break
                expected_start = end
            if expected_start != final_end:
                errors.append(f"TP4 rank {rank} final GPU watermark is incomplete")
    history_gpu_ready_before_freeze_ms = (
        (freeze_unix_s - max(gpu_history_completed)) * 1000
        if len(gpu_history_completed) == 4
        else None
    )
    urgent_history_wait_ms = (
        max(0.0, -history_gpu_ready_before_freeze_ms)
        if urgent_history_wait_allowed
        and history_gpu_ready_before_freeze_ms is not None
        else None
    )
    if urgent_history_wait_ms is not None and urgent_history_wait_ms > 5000:
        errors.append("urgent source freeze waited over 5 seconds for history")
    errors.extend(controller_completion_errors(end_rows, manager_m1_auto_start))
    if takeover.get("state") != "COMMITTED":
        errors.append("takeover state is not COMMITTED")
    if handoff_mode == "shadow-only":
        observed_notification_mode = str(
            ready_notification.get("mode", "FILE_POLL")
        )
        if observed_notification_mode != ready_notification_mode:
            errors.append("controller ready notification mode differs")
        if (
            ready_notification_mode == "UDP"
            and int(ready_notification.get("notification_count", 0)) <= 0
        ):
            errors.append("controller did not receive a UDP ready notification")
    if proxy.get("committed") is not True:
        errors.append("unified response proxy did not commit")
    emitted_tokens = proxy.get("emitted_tokens")
    if natural_eos_anchor:
        if (
            not isinstance(emitted_tokens, int)
            or isinstance(emitted_tokens, bool)
            or not 1 <= emitted_tokens < expected_anchor_tokens
            or target_response.get("finish_reason") != "stop"
        ):
            errors.append("unified response did not naturally finish before cap")
    elif emitted_tokens != expected_anchor_tokens:
        errors.append("unified response length differs from anchor budget")
    if (
        int(proxy.get("source_origin_tokens", 0)) <= 0
        or int(proxy.get("target_origin_tokens", 0)) <= 0
    ):
        errors.append("unified response does not contain tokens from both owners")
    if proxy.get("handoff_stall_s") is None:
        errors.append("unified response did not record a handoff stall")
    emitted = proxy.get("emitted", [])
    if [row.get("index") for row in emitted] != list(range(len(emitted))):
        errors.append("unified response indices are not contiguous")
    if len(emitted) != emitted_tokens:
        errors.append("unified response record count differs from emitted tokens")
    required_windows = (
        ("PRE_SHADOW",)
        if stop_and_copy
        else (
            "PRE_SHADOW",
            "SHADOW",
            "BRIDGE",
        )
    )
    for window in required_windows:
        if int(windows[window]["samples"]) < minimum_window_samples:
            errors.append(
                f"{window} has {windows[window]['samples']} TPOT samples, "
                f"requires {minimum_window_samples}"
            )
    errors.extend(receipt_errors)
    if stop_and_copy:
        frozen_path = controller_dir / "request_frozen_receipt.json"
        release_path = controller_dir / "source_kv_release_receipt.json"
        if not frozen_path.is_file():
            errors.append("scheduler did not acknowledge the per-request freeze")
        if not release_path.is_file():
            errors.append("source KV release was not observed after takeover")
        if session.get("stop_and_copy") is not True:
            errors.append("session did not record Stop-and-Copy mode")
        if int(cutover.get("delta_tokens", -1)) != 0:
            errors.append("Stop-and-Copy unexpectedly transferred live deltas")
    observed_rates = [
        float(row["rate_gib_s"])
        for row in audit
        if row.get("kind") == "rate" and row.get("rate_gib_s") is not None
    ]
    if fixed_rate_gib_s is not None:
        if not observed_rates:
            errors.append("controller did not record fixed-rate actuation")
        elif any(abs(value - fixed_rate_gib_s) > 1e-9 for value in observed_rates):
            errors.append("controller deviated from the requested fixed rate")
    if manager_m2_rate:
        assert m2_profiles_gib_s is not None
        initial_rows = [
            row for row in audit
            if row.get("kind") == "manager_m2_initial_rate"
        ]
        m2_rows = [
            row for row in audit
            if row.get("kind") == "rate"
            and row.get("manager_m2_decision") is not None
        ]
        if not m2_rows:
            errors.append("M2 did not record active Shadow rate decisions")
        elif not m2_expected_profile_used(
            initial_rows, m2_rows, manager_m2_expected_profile,
            m2_profiles_gib_s,
        ):
            errors.append("M2 did not use the expected profile")
        if any(
            all(abs(float(row["rate_gib_s"]) - profile) > 1e-9
                for profile in m2_profiles_gib_s)
            for row in m2_rows
        ):
            errors.append("M2 used a rate outside the three configured profiles")
        if manager_m2_require_source_high:
            if not has_measured_source_high(audit, source_peers):
                errors.append(
                    "M2 HIGH lacked measured pre-guard pressure with active "
                    "source peers"
                )
        if manager_m2_require_low_to_high:
            initial_low = (
                len(initial_rows) == 1
                and initial_rows[0].get("decision", {}).get("profile") == "LOW"
            )
            if not initial_low:
                errors.append("M2 did not start Shadow at LOW")
            if not has_measured_source_high(audit, source_peers):
                errors.append("M2 did not upshift on measured source pressure")
            first_initial_s = (
                float(initial_rows[0]["unix_s"]) if initial_rows else None
            )
            if first_initial_s is not None and any(
                peer.get("request_started_unix_s", 0) < first_initial_s
                for peer in source_peers
            ):
                errors.append("source peers started before M2 initial decision")
            ranks = direct_sender.get("ranks") or []
            paced_chunks = (
                ranks[0].get("history_pacing_chunks") or [] if ranks else []
            )
            chunk_rates = [
                float(row.get("requested_rate_gib_s", 0))
                for row in paced_chunks
            ]
            low, _, high = m2_profiles_gib_s
            low_before_high = any(
                abs(rate - low) <= 1e-9
                and any(abs(later - high) <= 1e-9
                        for later in chunk_rates[index + 1:])
                for index, rate in enumerate(chunk_rates)
            )
            if not low_before_high:
                errors.append("M2 LOW-to-HIGH did not reach paced history bytes")
    slo = summarize_slo(
        background.get("results", []),
        tpot_ms=slo_tpot_ms,
        ttft_ms=slo_ttft_ms,
        e2e_ms=slo_e2e_ms,
    )
    handoff_stall_ms = (
        float(proxy["handoff_stall_s"]) * 1000
        if proxy.get("handoff_stall_s") is not None
        else None
    )
    slo["handoff_ms"] = handoff_stall_ms
    slo["handoff_threshold_ms"] = slo_handoff_ms
    slo["handoff_violation"] = (
        handoff_stall_ms is None or handoff_stall_ms > slo_handoff_ms
    )
    emitted = proxy.get("emitted", [])
    anchor_tpot = summarize_emitted_intervals(emitted)
    source_tpot = summarize_emitted_intervals(emitted, origin="source")
    snapshot_trigger_stall_ms = emitted_boundary_gap_ms(
        emitted,
        origin="source",
        output_tokens=int(session["snapshot_num_output_tokens"]),
    )
    freeze_boundary_stall_ms = emitted_boundary_gap_ms(
        emitted,
        origin="source",
        output_tokens=int(cutover["cutover_num_output_tokens"]),
    )
    anchor_started_unix_s = source_response.get("request_started_unix_s")
    anchor_completed_unix_s = target_response.get("completed_unix_s")
    anchor_ttft_ms = (
        (float(emitted[0]["unix_s"]) - float(anchor_started_unix_s)) * 1000
        if emitted and anchor_started_unix_s is not None
        else source_response.get("ttft_ms")
    )
    anchor_e2e_ms = (
        (float(anchor_completed_unix_s) - float(anchor_started_unix_s)) * 1000
        if anchor_started_unix_s is not None
        and anchor_completed_unix_s is not None
        else None
    )
    workload_start = float(background.get("start_unix_s", 0.0))
    workload_end = float(background.get("end_unix_s", workload_start))
    workload_seconds = max(0.0, workload_end - workload_start)
    workload_tokens = sum(
        int(row.get("output_tokens", 0))
        for row in background.get("results", [])
        if row.get("status") == "COMPLETED"
    )
    bridge_to_commit_ms = (committed - bridge_start) * 1000
    release_receipt_path = controller_dir / "source_kv_release_receipt.json"
    release_receipt = (
        common.read_json(release_receipt_path)
        if release_receipt_path.is_file()
        else None
    )
    source_kv_released_unix_s = (
        int(release_receipt["released_unix_ns"]) / 1e9
        if release_receipt is not None
        else None
    )
    source_rows = [
        row
        for row in emitted
        if row.get("origin") == "source" and row.get("unix_s") is not None
    ]
    trigger_index = int(session["snapshot_num_output_tokens"]) - 1
    trigger_unix_s = (
        float(source_rows[trigger_index]["unix_s"])
        if 0 <= trigger_index < len(source_rows)
        else None
    )
    trigger_to_source_kv_release_ms = (
        (source_kv_released_unix_s - trigger_unix_s) * 1000
        if source_kv_released_unix_s is not None and trigger_unix_s is not None
        else None
    )
    anchor_slo = {
        "ttft_violation": (
            anchor_ttft_ms is None or float(anchor_ttft_ms) > slo_ttft_ms
        ),
        "e2e_violation": (
            anchor_e2e_ms is None or float(anchor_e2e_ms) > slo_e2e_ms
        ),
        "itl_violations": None,
        "itl_violation_rate": None,
    }
    anchor_source_times = [
        float(row["unix_s"])
        for row in emitted
        if row.get("unix_s") is not None
    ]
    anchor_itls = [
        (current - previous) * 1000
        for previous, current in zip(anchor_source_times, anchor_source_times[1:])
    ]
    anchor_slo["itl_violations"] = sum(
        value > slo_tpot_ms for value in anchor_itls
    )
    anchor_slo["itl_violation_rate"] = (
        anchor_slo["itl_violations"] / len(anchor_itls)
        if anchor_itls
        else None
    )
    anchor_slo["success"] = not (
        anchor_slo["ttft_violation"]
        or anchor_slo["e2e_violation"]
        or bool(anchor_slo["itl_violations"])
        or slo["handoff_violation"]
    )
    reported_windows = dict(windows)
    if handoff_mode == "shadow-only":
        # Preserve the legacy BRIDGE key for old result readers while naming
        # the actual Shadow-only interval accurately for new analyses.
        reported_windows["FINAL_SYNC"] = dict(windows["BRIDGE"])
    return {
        "format_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "migration_id": session.get("migration_id"),
        "source_request_id": session.get("source_request_id"),
        "evidence_class": (
            "ONLINE_VLLM_REQUEST_LEVEL_STOP_AND_COPY"
            if stop_and_copy
            else "ONLINE_VLLM_BRIDGE_REMOTE_ATTENTION"
            if require_remote_attention
            else "ONLINE_VLLM_GPU_RESIDENT_SHADOW_TAKEOVER"
            if gpu_resident_shadow
            else "ONLINE_VLLM_SHADOW_ONLY_TAKEOVER"
            if handoff_mode == "shadow-only"
            else "ONLINE_VLLM_PHASE8_STRATEGY_COMPARISON"
        ),
        "evidence_boundary": (
            "Real vLLM TP1 suffix attention merged with four TP4 ranks' "
            "GPU-resident prefix softmax statistics inside every migrated "
            "attention-layer output, followed by Phase 8 takeover."
            if require_remote_attention
            else (
                "Real vLLM TP1-to-TP4 NCCL GPU-direct historical transfer, "
                + (
                    "persistent batched NCCL GPU-direct delta streaming, "
                    if gpu_direct_delta
                    else "frozen full-image transfer with no live delta, "
                    if stop_and_copy
                    else "incremental CPU delta relay, "
                )
                + "four-rank GPU exact readback and "
                "watermark, atomic takeover, unified response, and target-"
                "request TPOT."
                if gpu_direct_history
                else "Real vLLM TP1/TP4 export, incremental CPU relay into "
                "reserved TP4 GPU blocks, four-rank watermark, atomic "
                "takeover, unified response, and target-request TPOT. TP4 "
                "performs no migrated-request forward before commit; online "
                "remote attention is not executed."
            )
            if gpu_resident_shadow
            else "Real vLLM TP1/TP4 export, transfer, restore, takeover, "
            "unified response, and target-request TPOT. The Shadow-only "
            "variant skips the controller Bridge/Handoff state, but stages "
            "history on CPU and restores TP4 after the final source freeze. "
            "The Bridge baseline defers history until that boundary. Online "
            "remote attention is not executed."
        ),
        "strategy": strategy,
        "commit_timing": commit_timing,
        "earliest_ready_cutover_output_tokens": (
            earliest_ready_selected[0].get("cutover_output_tokens")
            if earliest_ready_selected
            else None
        ),
        "earliest_ready_selection_output_tokens": (
            earliest_ready_selected[0].get("selection_output_tokens")
            if earliest_ready_selected
            else None
        ),
        "earliest_ready_selection_unix_s": (
            earliest_ready_selected[0].get("unix_s")
            if earliest_ready_selected
            else None
        ),
        "earliest_ready_candidate_outstanding_delta_tokens": (
            earliest_ready_candidates[0].get("outstanding_delta_tokens")
            if earliest_ready_candidates
            else None
        ),
        "earliest_ready_selection_delta_lag_tokens": (
            earliest_ready_selected[0].get("delta_lag_tokens")
            if earliest_ready_selected
            else None
        ),
        "earliest_ready_selection_rank_gpu_resident_end_tokens": (
            earliest_ready_selected[0].get("rank_gpu_resident_end_tokens")
            if earliest_ready_selected
            else None
        ),
        "ready_sync_mode": ready_sync_mode,
        "handoff_mode": handoff_mode,
        "stop_and_copy": stop_and_copy,
        "fixed_rate_gib_s": fixed_rate_gib_s,
        "observed_controller_rates_gib_s": sorted(set(observed_rates)),
        "history_payload_bytes": sum(
            int(row.get("payload_bytes", 0)) for row in initial_stage_receipts
        ),
        "history_observed_aggregate_gib_s": sum(
            float(row.get("observed_gib_s", 0.0)) for row in initial_stage_receipts
        ),
        "history_max_stage_ms": max(
            (float(row.get("stage_ms", 0.0)) for row in initial_stage_receipts),
            default=0.0,
        ),
        "history_ready_before_freeze_ms": history_ready_before_freeze_ms,
        "history_gpu_ready_before_freeze_ms": history_gpu_ready_before_freeze_ms,
        "urgent_history_wait_ms": urgent_history_wait_ms,
        "gpu_resident_shadow": gpu_resident_shadow,
        "gpu_direct_history": gpu_direct_history,
        "gpu_direct_history_pacing": gpu_direct_history_pacing_expected,
        "m2_history_profile_byte_fraction": (
            m2_history_profile_byte_fraction
        ),
        "gpu_direct_history_pacing_evidence": (
            next(iter(direct_sender.get("ranks", [])), {}).get(
                "history_pacing_chunks"
            )
            if gpu_direct_history_pacing_expected
            else None
        ),
        "gpu_direct_delta": gpu_direct_delta,
        "persistent_channel": persistent_channel,
        "channel_generation": direct_sender.get("channel_generation"),
        "channel_create_count": direct_sender.get("channel_create_count"),
        "channel_destroy_count": direct_sender.get("channel_destroy_count"),
        "channel_session_count": direct_sender.get("channel_session_count"),
        "channel_buffer_high_water_bytes": direct_sender.get(
            "buffer_high_water_bytes"
        ),
        "deferred_comm_destroy": deferred_comm_destroy,
        "post_takeover_comm_destroy": post_takeover_comm_destroy,
        "communicator_destroy_statuses": [
            row.get("status") for row in communicator_destroy_receipts
        ],
        "source_communicator_lifecycle": direct_sender.get(
            "communicator_lifecycle"
        ),
        "source_communicator_pool_size": direct_sender.get(
            "communicator_pool_size"
        ),
        "communicator_destroy_ms": [
            row.get("destroy_ms") for row in communicator_destroy_receipts
        ],
        "communicator_destroy_started_after_target_ready_ms": [
            (
                float(row["destroy_started_unix_s"])
                - float(
                    next(
                        receipt["target_ready_unix_s"]
                        for receipt in target_receipts
                        if int(receipt.get("tp_rank", -1))
                        == int(row.get("tp_rank", -2))
                    )
                )
            )
            * 1000
            for row in communicator_destroy_receipts
            if row.get("destroy_started_unix_s") is not None
            and any(
                int(receipt.get("tp_rank", -1))
                == int(row.get("tp_rank", -2))
                and receipt.get("target_ready_unix_s") is not None
                for receipt in target_receipts
            )
        ],
        "communicator_destroy_started_after_commit_ms": [
            row.get("destroy_started_after_commit_ms")
            for row in communicator_destroy_receipts
            if row.get("destroy_started_after_commit_ms") is not None
        ],
        "communicator_pool_retained_after_target_ready_ms": [
            (
                float(row["retained_unix_s"])
                - float(
                    next(
                        receipt["target_ready_unix_s"]
                        for receipt in target_receipts
                        if int(receipt.get("tp_rank", -1))
                        == int(row.get("tp_rank", -2))
                    )
                )
            )
            * 1000
            for row in communicator_destroy_receipts
            if row.get("retained_unix_s") is not None
            and any(
                int(receipt.get("tp_rank", -1))
                == int(row.get("tp_rank", -2))
                and receipt.get("target_ready_unix_s") is not None
                for receipt in target_receipts
            )
        ],
        "gpu_direct_delta_batch_tokens": session.get(
            "gpu_direct_delta_batch_tokens"
        ),
        "gpu_direct_delta_flush_ms": session.get("gpu_direct_delta_flush_ms"),
        "gpu_direct_delta_batches": direct_sender.get("delta_batches"),
        "gpu_direct_delta_logical_submissions": direct_sender.get(
            "delta_logical_submissions"
        ),
        "gpu_direct_delta_coalesced_submissions": direct_sender.get(
            "delta_coalesced_submissions"
        ),
        "gpu_direct_delta_tokens": direct_sender.get("delta_tokens"),
        "gpu_direct_delta_payload_bytes": direct_sender.get(
            "delta_payload_bytes"
        ),
        "gpu_direct_delta_total_ms": sum(
            float(row.get("transfer_ms", 0.0))
            for row in direct_sender.get("delta_records", [])
        ),
        "gpu_direct_delta_pack_ms": sum(
            float(row.get("pack_ms", 0.0))
            for row in direct_sender.get("delta_records", [])
        ),
        "gpu_direct_delta_receiver_ready_ms": sum(
            float(row.get("receiver_ready_ms", 0.0))
            for row in direct_sender.get("delta_records", [])
        ),
        "gpu_direct_delta_nccl_send_ms": sum(
            float(row.get("nccl_send_ms", 0.0))
            for row in direct_sender.get("delta_records", [])
        ),
        "gpu_direct_delta_target_apply_ack_ms": sum(
            float(row.get("target_apply_ack_ms", 0.0))
            for row in direct_sender.get("delta_records", [])
        ),
        "gpu_direct_delta_max_batch_ms": max(
            (
                float(row.get("transfer_ms", 0.0))
                for row in direct_sender.get("delta_records", [])
            ),
            default=None,
        ),
        "gpu_history_block_acks": sum(
            1 for _ in (controller_dir / "gpu_block_receipts").glob("**/*.json")
        ),
        "gpu_delta_acks": sum(
            1 for _ in (controller_dir / "gpu_delta_receipts").glob("**/*.json")
        ),
        "remote_attention_calls": len(remote_attention_rows),
        "remote_attention_layers": len(
            {str(row.get("layer_name")) for row in remote_attention_rows}
        ),
        "remote_attention_token_forwards": len(
            {
                (str(row.get("request_id")), int(row.get("sequence_tokens", -1)))
                for row in remote_attention_rows
            }
        ),
        "remote_attention_visible_calls": len(remote_attention_visible_rows),
        "remote_attention_visible_token_forwards": len(
            {
                (str(row.get("request_id")), int(row.get("sequence_tokens", -1)))
                for row in remote_attention_visible_rows
            }
        ),
        "remote_attention_speculative_calls": len(remote_attention_speculative_rows),
        "remote_attention_speculative_token_forwards": len(
            {
                (str(row.get("request_id")), int(row.get("sequence_tokens", -1)))
                for row in remote_attention_speculative_rows
            }
        ),
        "remote_attention_total_ms": {
            "p50": percentile(
                [float(row["total_ms"]) for row in remote_attention_rows], 0.50
            ),
            "p95": percentile(
                [float(row["total_ms"]) for row in remote_attention_rows], 0.95
            ),
            "p99": percentile(
                [float(row["total_ms"]) for row in remote_attention_rows], 0.99
            ),
        },
        "remote_attention_visible_total_ms": {
            "p50": percentile(
                [float(row["total_ms"]) for row in remote_attention_visible_rows],
                0.50,
            ),
            "p95": percentile(
                [float(row["total_ms"]) for row in remote_attention_visible_rows],
                0.95,
            ),
            "p99": percentile(
                [float(row["total_ms"]) for row in remote_attention_visible_rows],
                0.99,
            ),
        },
        "history_copy_order": session.get("history_copy_order"),
        "target_jobs_completed": sum(
            row.get("status") == "COMPLETED" for row in target_background
        ),
        "source_pressure_evidence": (
            source_pressure_evidence if source_pressure_expected_jobs else None
        ),
        "shadow_duration_ms": (bridge_start - shadow_start) * 1000,
        "bridge_to_commit_ms": bridge_to_commit_ms,
        "final_sync_to_commit_ms": (
            bridge_to_commit_ms if handoff_mode == "shadow-only" else None
        ),
        "history_transfer_started_unix_s": history_start,
        "history_transfer_phase": session.get("history_transfer_phase"),
        "handoff_stall_ms": handoff_stall_ms,
        "freeze_to_final_delta_ack_ms": (
            (max(delta_apply_times) - freeze_unix_s) * 1000
            if delta_apply_times
            else None
        ),
        "freeze_to_first_rank_ready_ms": (
            (min(rank_ready_times) - freeze_unix_s) * 1000
            if len(rank_ready_times) == 4
            else None
        ),
        "freeze_to_all_rank_ready_ms": (
            (max(rank_ready_times) - freeze_unix_s) * 1000
            if len(rank_ready_times) == 4
            else None
        ),
        "rank_ready_skew_ms": (
            (max(rank_ready_times) - min(rank_ready_times)) * 1000
            if len(rank_ready_times) == 4
            else None
        ),
        "last_rank_ready_to_controller_wakeup_ms": (
            (float(controller_wakeup) - max(rank_ready_times)) * 1000
            if len(rank_ready_times) == 4 and controller_wakeup is not None
            else None
        ),
        "controller_wakeup_to_commit_ms": (
            (float(commit_completed) - float(controller_wakeup)) * 1000
            if controller_wakeup is not None
            else None
        ),
        "last_rank_ready_to_commit_ms": (
            (float(commit_completed) - max(rank_ready_times)) * 1000
            if len(rank_ready_times) == 4
            else None
        ),
        "request_frozen_unix_s": (
            int(frozen_receipt["frozen_unix_ns"]) / 1e9
            if frozen_receipt is not None
            else None
        ),
        "source_kv_released_unix_s": source_kv_released_unix_s,
        "source_kv_release_after_commit_ms": (
            (source_kv_released_unix_s - committed) * 1000
            if source_kv_released_unix_s is not None
            else None
        ),
        "trigger_to_source_kv_release_ms": trigger_to_source_kv_release_ms,
        "slo": slo,
        "anchor_slo": anchor_slo,
        "anchor_ttft_ms": anchor_ttft_ms,
        "anchor_e2e_ms": anchor_e2e_ms,
        "anchor_tpot": anchor_tpot,
        "source_tpot": source_tpot,
        "snapshot_trigger_stall_ms": snapshot_trigger_stall_ms,
        "freeze_boundary_stall_ms": freeze_boundary_stall_ms,
        "cutover_hook_enter_unix_s": (
            int(cutover["cutover_hook_enter_unix_ns"]) / 1e9
            if cutover.get("cutover_hook_enter_unix_ns") is not None
            else None
        ),
        "final_delta_enqueued_unix_s": (
            int(cutover["final_delta_enqueued_unix_ns"]) / 1e9
            if cutover.get("final_delta_enqueued_unix_ns") is not None
            else None
        ),
        "delta_drain_completed_unix_s": (
            int(cutover["delta_drain_completed_unix_ns"]) / 1e9
            if cutover.get("delta_drain_completed_unix_ns") is not None
            else None
        ),
        "final_delta_enqueue_ms": cutover.get("final_delta_enqueue_ms"),
        "final_delta_drain_ms": cutover.get("final_delta_drain_ms"),
        "cutover_hook_to_delta_drain_ms": cutover.get(
            "cutover_hook_to_delta_drain_ms"
        ),
        "workload": {
            "wall_time_s": workload_seconds,
            "output_tokens": workload_tokens,
            "output_throughput_tokens_s": (
                workload_tokens / workload_seconds if workload_seconds else None
            ),
            "request_throughput_s": (
                background.get("completed", 0) / workload_seconds
                if workload_seconds
                else None
            ),
        },
        "source_origin_tokens": proxy.get("source_origin_tokens"),
        "target_origin_tokens": proxy.get("target_origin_tokens"),
        "receiver_ranks": receipts.get("receiver_ranks"),
        "exact_readback": receipts.get("exact_readback"),
        "preconnect_evidence": (
            preconnect_receipts if preconnect_persistent_channel else None
        ),
        "persistent_channel_evidence": (
            {
                "source": {
                    key: direct_sender.get(key)
                    for key in (
                        "communicator_lifecycle",
                        "channel_generation",
                        "channel_create_count",
                        "channel_destroy_count",
                        "channel_session_count",
                        "buffer_high_water_bytes",
                    )
                },
                "targets": [
                    {
                        key: row.get(key)
                        for key in (
                            "tp_rank",
                            "status",
                            "channel_generation",
                            "channel_create_count",
                            "channel_destroy_count",
                            "channel_session_count",
                            "buffer_high_water_bytes",
                        )
                    }
                    for row in target_channel_receipts
                ],
            }
            if persistent_channel
            else None
        ),
        "target_ready_sync_scopes": receipts.get("ready_sync_scopes"),
        "target_ready_event_wait_ms": receipts.get("ready_event_wait_ms"),
        "target_device_wide_synchronize": receipts.get(
            "device_wide_synchronize"
        ),
        "target_receive_dependency_scopes": receipts.get(
            "receive_dependency_scopes"
        ),
        "target_model_stream_wait_event": receipts.get(
            "model_stream_wait_event"
        ),
        "ready_notification_mode": ready_notification.get(
            "mode", "FILE_POLL"
        ),
        "ready_notification_count": ready_notification.get(
            "notification_count", 0
        ),
        "ready_notification_ranks": ready_notification.get(
            "notified_ranks", []
        ),
        "ready_notification_delivery_ms": ready_notification.get(
            "last_notification_delivery_ms"
        ),
        "ready_latch_poll_ms": ready_notification.get("ready_latch_poll_ms"),
        "ready_latch_poll_count": ready_notification.get(
            "ready_latch_poll_count"
        ),
        "ready_latch_to_authoritative_ready_ms": ready_notification.get(
            "ready_latch_to_authoritative_ready_ms"
        ),
        "target_tpot_windows": reported_windows,
        "errors": errors,
    }


def main() -> None:
    args = parse_args()
    revision, guard, pressure = validate_inputs(args)
    contract = {
        "format_version": 1,
        "phase": args.phase,
        "managed_formal_subrun": args.managed_formal_subrun,
        "revision": revision,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": common.sha256(args.manifest),
        "survival_table_sha256": common.sha256(args.survival_table),
        "guard_file_sha256": common.sha256(args.guard_file),
        "guard_free_kv_tokens": guard,
        "strategies": args.strategy_order,
        "bridge_only": args.bridge_only,
        "architecture_comparison": args.architecture_comparison,
        "shadow_only_only": args.shadow_only_only,
        "stop_and_copy_only": args.stop_and_copy_only,
        "gpu_resident_shadow": args.gpu_resident_shadow,
        "gpu_direct_history": args.gpu_direct_history,
        "gpu_direct_history_pacing": args.gpu_direct_history_pacing,
        "gpu_direct_delta": args.gpu_direct_delta,
        "persistent_channel": args.persistent_channel,
        "preconnect_persistent_channel": args.preconnect_persistent_channel,
        "channel_generation": args.channel_generation,
        "persistent_sequential_reuse": args.persistent_sequential_reuse,
        "persistent_session_gap_s": args.persistent_session_gap_s,
        "gpu_direct_delta_batch_tokens": args.gpu_direct_delta_batch_tokens,
        "gpu_direct_delta_flush_ms": args.gpu_direct_delta_flush_ms,
        "ready_sync_mode": args.ready_sync_mode,
        "ready_sync_comparison": args.ready_sync_comparison,
        "ready_notification_mode": args.ready_notification_mode,
        "ready_notification_comparison": args.ready_notification_comparison,
        "ready_notification_host": args.ready_notification_host,
        "ready_notification_port": args.ready_notification_port,
        "ready_latch_poll_ms": args.ready_latch_poll_ms,
        "deferred_comm_destroy": args.deferred_comm_destroy,
        "deferred_comm_destroy_comparison": (
            args.deferred_comm_destroy_comparison
        ),
        "post_takeover_comm_destroy": args.post_takeover_comm_destroy,
        "post_takeover_destroy_comparison": (
            args.post_takeover_destroy_comparison
        ),
        "gpu_direct_base_port": args.gpu_direct_base_port,
        "online_remote_attention": args.online_remote_attention,
        "remote_attention_base_port": args.remote_attention_base_port,
        "commit_timing": args.commit_timing,
        "manager_m0_shadow": args.manager_m0_shadow,
        "manager_m1_auto_start": args.manager_m1_auto_start,
        "paired_stay": args.paired_stay,
        "m1_source_release_tail_s": args.m1_source_release_tail_s,
        "manager_m2_rate": args.manager_m2_rate,
        "manager_m3_commit": args.manager_m3_commit,
        "manager_m4_cancel": args.manager_m4_cancel,
        "manager_m5_predictor_shadow": args.manager_m5_predictor_shadow,
        "predictor_checkpoint_sha256": args.predictor_checkpoint_sha256,
        "manager_m4_expect_cancel": args.manager_m4_expect_cancel,
        "m3_policy": (
            "COMMIT_EARLIEST_WHEN_READY" if args.manager_m3_commit else None
        ),
        "manager_m2_force_initial_high": args.manager_m2_force_initial_high,
        "manager_m2_require_source_high": args.manager_m2_require_source_high,
        "manager_m2_require_low_to_high": args.manager_m2_require_low_to_high,
        "manager_m2_expected_profile": args.manager_m2_expected_profile,
        "manager_m2_min_history_byte_frac": (
            args.manager_m2_min_history_byte_frac
        ),
        "m2_profiles_gib_s": (
            [args.m2_low_gib_s, args.m2_medium_gib_s, args.m2_high_gib_s]
            if args.manager_m2_rate else None
        ),
        "manager_m1_expect_stay": args.manager_m1_expect_stay,
        "manager_m1_stay_reason": args.manager_m1_stay_reason,
        "anchor_max_tokens": args.anchor_max_tokens,
        "anchor_prompt_tokens": args.anchor_prompt_tokens,
        "natural_eos_anchor": args.natural_eos_anchor,
        "anchor_request_sha256": args.expected_anchor_request_sha256,
        "tp4_max_num_seqs": args.tp4_max_num_seqs,
        "bridge_output_tokens": args.bridge_output_tokens,
        "repetitions": args.repetitions,
        "source_pressure": args.source_pressure,
        "minimum_ready_source_jobs": args.minimum_ready_source_jobs,
        "minimum_source_kv_usage_frac": args.minimum_source_kv_usage_frac,
        "fixed_rate_gib_s": args.fixed_rate_gib_s,
        "slo_thresholds": {
            "tpot_ms": args.slo_tpot_ms,
            "ttft_ms": args.slo_ttft_ms,
            "e2e_ms": args.slo_e2e_ms,
            "handoff_ms": args.slo_handoff_ms,
        },
        "pressure": pressure,
        "evidence_boundary": (
            "online Phase 8 takeover with TP4-prefix attention in the token path"
            if args.online_remote_attention
            else "online Phase 8 takeover without remote attention"
        ),
    }
    if args.validate_only:
        print(json.dumps({"status": "VALID", "contract": contract}, indent=2))
        return

    out_root = args.out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=False)
    common.write_json(out_root / "contract.json", contract)
    batch: dict[str, Any] = {
        "format_version": 1,
        "status": "RUNNING",
        "started_unix_s": time.time(),
        "contract": contract,
        "runs": [],
    }
    common.write_json(out_root / "batch_status.json", batch)
    service_pool: dict[str, Any] | None = None
    if args.persistent_sequential_reuse:
        service_pool = {
            "runtime_root": out_root / "persistent_runtime",
            "processes": [],
        }
    try:
        for repetition in range(1, args.repetitions + 1):
            variants = (
                [
                    (
                        "BRIDGE_RA" if args.online_remote_attention else "BRIDGE",
                        "S_NEW_OLD" if args.online_remote_attention else "S_NEW",
                        "bridge",
                    )
                ]
                if args.bridge_only
                else (
                    [
                        (
                            "BRIDGE_RA" if args.online_remote_attention else "BRIDGE",
                            "S_NEW_OLD" if args.online_remote_attention else "S_NEW",
                            "bridge",
                        ),
                        ("SHADOW_ONLY", "S_NEW_OLD", "shadow-only"),
                    ]
                    if args.architecture_comparison
                    else [("STOP_AND_COPY", "S_NEW_OLD", "shadow-only")]
                    if args.stop_and_copy_only
                    else [("SHADOW_ONLY", "S_NEW_OLD", "shadow-only")]
                    if args.shadow_only_only
                    else [
                        (strategy, strategy, "bridge")
                        for strategy in args.strategy_order
                    ]
                )
            )
            if args.ready_sync_comparison:
                variants = [
                    ("SHADOW_ONLY_SYNC_OLD", "S_NEW_OLD", "shadow-only"),
                    ("SHADOW_ONLY_SYNC_NEW", "S_NEW_OLD", "shadow-only"),
                ]
            if args.ready_notification_comparison:
                variants = [
                    ("SHADOW_ONLY_NOTIFY_OLD", "S_NEW_OLD", "shadow-only"),
                    ("SHADOW_ONLY_NOTIFY_NEW", "S_NEW_OLD", "shadow-only"),
                ]
            if args.deferred_comm_destroy_comparison:
                variants = [
                    ("SHADOW_ONLY_DESTROY_OLD", "S_NEW_OLD", "shadow-only"),
                    ("SHADOW_ONLY_DESTROY_NEW", "S_NEW_OLD", "shadow-only"),
                ]
            if args.post_takeover_destroy_comparison:
                variants = [
                    ("SHADOW_ONLY_POOL", "S_NEW_OLD", "shadow-only"),
                    ("SHADOW_ONLY_POST_DESTROY", "S_NEW_OLD", "shadow-only"),
                ]
            if repetition % 2 == 0:
                variants.reverse()
            for architecture, strategy, handoff_mode in variants:
                label = f"r{repetition:02d}_{architecture.lower()}"
                rep_args = copy.copy(args)
                selected_online_remote_attention = bool(
                    args.online_remote_attention and handoff_mode == "bridge"
                )
                selected_stop_and_copy = architecture == "STOP_AND_COPY"
                rep_args.online_remote_attention = selected_online_remote_attention
                selected_ready_sync_mode = (
                    "DEVICE_WIDE"
                    if architecture == "SHADOW_ONLY_SYNC_OLD"
                    else "STREAM_EVENT"
                    if architecture == "SHADOW_ONLY_SYNC_NEW"
                    else args.ready_sync_mode
                )
                rep_args.ready_sync_mode = selected_ready_sync_mode
                selected_ready_notification_mode = (
                    "FILE_POLL"
                    if architecture == "SHADOW_ONLY_NOTIFY_OLD"
                    else "UDP"
                    if architecture == "SHADOW_ONLY_NOTIFY_NEW"
                    else args.ready_notification_mode
                )
                rep_args.ready_notification_mode = (
                    selected_ready_notification_mode
                )
                selected_deferred_destroy = (
                    False
                    if architecture == "SHADOW_ONLY_DESTROY_OLD"
                    else True
                    if architecture == "SHADOW_ONLY_DESTROY_NEW"
                    else True
                    if architecture
                    in {"SHADOW_ONLY_POOL", "SHADOW_ONLY_POST_DESTROY"}
                    else bool(
                        args.deferred_comm_destroy
                        or args.post_takeover_comm_destroy
                    )
                )
                rep_args.deferred_comm_destroy = selected_deferred_destroy
                selected_post_takeover_destroy = bool(
                    architecture == "SHADOW_ONLY_POST_DESTROY"
                    or (
                        not args.post_takeover_destroy_comparison
                        and args.post_takeover_comm_destroy
                    )
                )
                rep_args.post_takeover_comm_destroy = (
                    selected_post_takeover_destroy
                )
                rep_args.gpu_resident_shadow = bool(
                    args.gpu_resident_shadow
                    or selected_stop_and_copy
                    or selected_online_remote_attention
                    or (
                        args.architecture_comparison
                        and args.online_remote_attention
                        and handoff_mode == "shadow-only"
                    )
                )
                if selected_stop_and_copy:
                    # The target connector's final watermark is the frozen
                    # snapshot boundary, not Shadow's later delta boundary.
                    rep_args.cutover_output_tokens = args.trigger_output_tokens
                elif args.commit_timing == "EARLIEST_READY":
                    # The target request is admitted only after the
                    # controller publishes cutover_manifest.json.  A zero
                    # connector boundary enables that manifest-driven lookup;
                    # it is not used as a runtime cutover value.
                    rep_args.cutover_output_tokens = 0
                rep_args.force_source_eager = bool(args.online_remote_attention)
                rep_args.out_root = out_root / label
                run_id = f"{out_root.name}-{label}"
                predictor_event_path = rep_args.out_root / "predictor_events.jsonl"

                def acceptance(
                    controller_dir: Path,
                    background_dir: Path,
                    expected_jobs: int,
                    expected_anchor_tokens: int,
                    selected: str = strategy,
                    selected_handoff: str = handoff_mode,
                    selected_remote: bool = selected_online_remote_attention,
                    selected_stop: bool = selected_stop_and_copy,
                    selected_ready_sync: str = selected_ready_sync_mode,
                    selected_ready_notification: str = (
                        selected_ready_notification_mode
                    ),
                    selected_deferred_destroy: bool = (
                        selected_deferred_destroy
                    ),
                    selected_post_destroy: bool = (
                        selected_post_takeover_destroy
                    ),
                    selected_predictor_events: Path = predictor_event_path,
                ) -> dict[str, Any]:
                    if args.paired_stay:
                        accepted = accept_paired_stay(
                            controller_dir, background_dir,
                            expected_jobs, expected_anchor_tokens,
                            natural_eos_anchor=args.natural_eos_anchor,
                        )
                    elif args.manager_m4_expect_cancel:
                        accepted = accept_m4_cancel(
                            controller_dir, background_dir,
                            expected_jobs, expected_anchor_tokens,
                        )
                    else:
                        accepted = accept_online(
                            controller_dir,
                            background_dir,
                            expected_jobs,
                            expected_anchor_tokens,
                            natural_eos_anchor=args.natural_eos_anchor,
                            strategy=selected,
                            minimum_window_samples=args.minimum_window_samples,
                            fixed_rate_gib_s=args.fixed_rate_gib_s,
                            manager_m2_rate=args.manager_m2_rate,
                            manager_m3_commit=args.manager_m3_commit,
                            manager_m2_expected_profile=(
                                args.manager_m2_expected_profile
                            ),
                            manager_m2_min_history_byte_frac=(
                                args.manager_m2_min_history_byte_frac
                            ),
                            manager_m2_require_source_high=(
                                args.manager_m2_require_source_high
                            ),
                            manager_m2_require_low_to_high=(
                                args.manager_m2_require_low_to_high
                            ),
                            m2_profiles_gib_s=(
                                (args.m2_low_gib_s, args.m2_medium_gib_s,
                                 args.m2_high_gib_s)
                                if args.manager_m2_rate else None
                            ),
                            gpu_direct_history_pacing_expected=(
                                args.gpu_direct_history_pacing
                            ),
                            handoff_mode=selected_handoff,
                            require_remote_attention=selected_remote,
                            slo_tpot_ms=args.slo_tpot_ms,
                            slo_ttft_ms=args.slo_ttft_ms,
                            slo_e2e_ms=args.slo_e2e_ms,
                            slo_handoff_ms=args.slo_handoff_ms,
                            stop_and_copy=selected_stop,
                            ready_sync_mode=selected_ready_sync,
                            ready_notification_mode=(
                                selected_ready_notification
                            ),
                            deferred_comm_destroy=selected_deferred_destroy,
                            post_takeover_comm_destroy=selected_post_destroy,
                            persistent_channel=args.persistent_channel,
                            preconnect_persistent_channel=(
                                args.preconnect_persistent_channel
                            ),
                            persistent_expected_session_count=(
                                repetition
                                if args.persistent_sequential_reuse
                                else 1
                            ),
                            commit_timing=args.commit_timing,
                            manager_m1_auto_start=args.manager_m1_auto_start,
                            manager_m1_expect_stay=args.manager_m1_expect_stay,
                            manager_m1_stay_reason=args.manager_m1_stay_reason,
                            source_pressure_expected_jobs=(
                                pressure["source_jobs"] if args.source_pressure else 0
                            ),
                            minimum_source_kv_usage_frac=(
                                args.minimum_source_kv_usage_frac
                            ),
                        )
                    if args.manager_m5_predictor_shadow:
                        accepted = accept_m5_shadow(
                            accepted, controller_dir,
                            selected_predictor_events,
                            args.predictor_checkpoint_sha256,
                        )
                    return accepted

                source_env_overrides = {"BRIDGETP_SHADOW_STRATEGY": strategy}
                if args.manager_m5_predictor_shadow:
                    source_env_overrides.update({
                        # The auxiliary-layer selection is set after model
                        # construction and is absent from vLLM's compile-cache
                        # key. A normal cached graph can otherwise be reused.
                        "VLLM_DISABLE_COMPILE_CACHE": "1",
                        "BRIDGETP_PREDICTOR_LIVE_CHECKPOINT": str(
                            args.predictor_checkpoint
                        ),
                        "BRIDGETP_PREDICTOR_LIVE_SHA256": (
                            args.predictor_checkpoint_sha256
                        ),
                        "BRIDGETP_PREDICTOR_LIVE_EVENTS": str(predictor_event_path),
                    })
                if handoff_mode == "shadow-only":
                    source_env_overrides["BRIDGETP_REQUEST_FREEZE_ENABLED"] = "1"
                if selected_stop_and_copy:
                    source_env_overrides.update(
                        {
                            "BRIDGETP_STOP_AND_COPY": "1",
                            "BRIDGETP_REQUEST_FREEZE_ENABLED": "1",
                            "BRIDGETP_STREAM_AFTER_OUTPUT_TOKENS": str(
                                args.trigger_output_tokens
                            ),
                            "BRIDGETP_PHASE8_CUTOVER_OUTPUT_TOKENS": str(
                                args.cutover_output_tokens
                            ),
                        }
                    )
                if selected_online_remote_attention:
                    source_env_overrides.update(
                        {
                            "BRIDGETP_ONLINE_REMOTE_ATTENTION": "1",
                            "BRIDGETP_REMOTE_ATTENTION_BASE_PORT": str(
                                args.remote_attention_base_port
                            ),
                            "BRIDGETP_REMOTE_ATTENTION_STRICT": "1",
                        }
                    )
                controller_config_overrides = build_controller_config_overrides(
                    trigger_output_tokens=args.trigger_output_tokens,
                    cutover_output_tokens=args.cutover_output_tokens,
                    fixed_rate_gib_s=args.fixed_rate_gib_s,
                    m2_profiles_gib_s=(
                        (args.m2_low_gib_s, args.m2_medium_gib_s,
                         args.m2_high_gib_s)
                        if args.manager_m2_rate else None
                    ),
                )
                if args.manager_m1_auto_start:
                    controller_config_overrides["capacity_pilot"] = {
                        "enabled": True,
                        "guard_free_kv_tokens": guard,
                    }
                    if args.m1_min_output_tokens is not None:
                        controller_config_overrides["policy"] = {
                            "min_output_tokens_before_eligible": (
                                args.m1_min_output_tokens
                            ),
                        }
                if args.fixed_rate_gib_s is not None:
                    source_env_overrides["BRIDGETP_STREAM_RATE_GIB_S"] = str(
                        args.fixed_rate_gib_s
                    )
                elif args.manager_m2_rate:
                    source_env_overrides["BRIDGETP_STREAM_RATE_GIB_S"] = str(
                        args.m2_medium_gib_s
                    )
                if args.gpu_direct_history_pacing:
                    source_env_overrides[
                        "BRIDGETP_GPU_DIRECT_HISTORY_PACING"
                    ] = "1"

                controller_extra_args = (
                    [
                        "--manager-m1-auto-start",
                        "--m1-source-release-tail-s",
                        str(args.m1_source_release_tail_s),
                    ]
                    if args.manager_m1_auto_start
                    else [
                        "--diagnostic-trigger-output-tokens",
                        str(args.trigger_output_tokens),
                    ]
                )
                if args.manager_m0_shadow:
                    controller_extra_args.append("--manager-m0-shadow")
                if args.manager_m2_rate:
                    controller_extra_args.append("--manager-m2-rate")
                if args.manager_m3_commit:
                    controller_extra_args.append("--manager-m3-commit")
                if args.manager_m4_cancel:
                    controller_extra_args.append("--manager-m4-cancel")
                if args.manager_m5_predictor_shadow:
                    controller_extra_args.extend([
                        "--manager-m5-predictor-shadow",
                        "--predictor-event-path", str(predictor_event_path),
                        "--predictor-checkpoint-sha256",
                        args.predictor_checkpoint_sha256,
                    ])
                if args.paired_stay:
                    controller_extra_args.append("--paired-stay")
                if args.manager_m2_force_initial_high:
                    controller_extra_args.append(
                        "--manager-m2-force-initial-high"
                    )
                if args.commit_timing == "EARLIEST_READY":
                    controller_extra_args.append(
                        "--diagnostic-earliest-ready-cutover"
                    )
                else:
                    controller_extra_args.extend(
                        [
                            "--diagnostic-cutover-output-tokens",
                            str(args.cutover_output_tokens),
                            "--diagnostic-bridge-output-tokens",
                            str(args.bridge_output_tokens),
                        ]
                    )
                controller_extra_args.extend(
                    [
                        "--handoff-mode",
                        handoff_mode,
                        "--ready-notification-mode",
                        selected_ready_notification_mode,
                        "--ready-notification-host",
                        args.ready_notification_host,
                        "--ready-notification-port",
                        str(args.ready_notification_port),
                        "--ready-latch-poll-ms",
                        str(args.ready_latch_poll_ms),
                    ]
                )

                result = scenario_runner.run(
                    rep_args,
                    revision,
                    guard,
                    pressure,
                    phase="bringup" if args.phase == "smoke" else "formal",
                    repetition=repetition,
                    run_id=run_id,
                    scenario="shadow_online",
                    scenario_title=(
                        f"Online {architecture} architecture comparison"
                        if args.architecture_comparison
                        else f"Online Shadow strategy {strategy}"
                    ),
                    provenance_status=(
                        "SMOKE_NOT_REPORTABLE"
                        if args.phase == "smoke"
                        else "FORMAL_REPETITION"
                    ),
                    platform_note=(
                        f"ONLINE {architecture} / {strategy}; real Phase 8 "
                        "takeover; "
                        + (
                            "TP4-prefix attention is in the token data path"
                            if selected_online_remote_attention
                            else "no online remote attention"
                        )
                    ),
                    success_status="PASS",
                    success_marker="PASS",
                    acceptance_fn=acceptance,
                    allow_clean_stager_exit=True,
                    source_env_overrides=source_env_overrides,
                    controller_config_overrides=controller_config_overrides,
                    controller_extra_args=controller_extra_args
                    + (
                        ["--gpu-resident-shadow"]
                        if rep_args.gpu_resident_shadow
                        else []
                    )
                    + (["--stop-and-copy"] if selected_stop_and_copy else []),
                    background_before_controller=True,
                    background_lead_s=args.background_lead_s,
                    background_ready_jobs=args.minimum_ready_target_jobs,
                    background_ready_source_jobs=(
                        args.minimum_ready_source_jobs
                    ),
                    service_pool=service_pool,
                )
                memory_snapshot = None
                if service_pool is not None:
                    memory_snapshot = capture_persistent_memory(
                        list(service_pool.get("processes", []))
                    )
                    common.write_json(
                        rep_args.out_root / "persistent_memory_snapshot.json",
                        memory_snapshot,
                    )
                batch["runs"].append(
                    {
                        "repetition": repetition,
                        "strategy": strategy,
                        "architecture": architecture,
                        "handoff_mode": handoff_mode,
                        "stop_and_copy": selected_stop_and_copy,
                        "ready_sync_mode": selected_ready_sync_mode,
                        "ready_notification_mode": (
                            selected_ready_notification_mode
                        ),
                        "deferred_comm_destroy": selected_deferred_destroy,
                        "post_takeover_comm_destroy": (
                            selected_post_takeover_destroy
                        ),
                        "persistent_channel": args.persistent_channel,
                        "channel_generation": args.channel_generation,
                        "status": result["status"],
                        "root": str(rep_args.out_root.resolve()),
                        "acceptance": result["acceptance"],
                        "persistent_memory": memory_snapshot,
                    }
                )
                common.write_json(out_root / "batch_status.json", batch)
                if (
                    args.persistent_sequential_reuse
                    and args.persistent_session_gap_s > 0
                ):
                    time.sleep(args.persistent_session_gap_s)

        persistent_summary = None
        if args.persistent_sequential_reuse:
            session_rows = [row["acceptance"] for row in batch["runs"]]
            last_evidence = (
                session_rows[-1].get("persistent_channel_evidence")
                if session_rows
                else None
            ) or {}
            source_channel = last_evidence.get("source") or {}
            target_channels = last_evidence.get("targets") or []
            migration_ids = [row.get("migration_id") for row in session_rows]
            source_request_ids = [
                row.get("source_request_id") for row in session_rows
            ]
            memory_rows = [
                row.get("persistent_memory")
                for row in batch["runs"]
                if row.get("persistent_memory") is not None
            ]
            persistent_errors: list[str] = []
            if len(set(migration_ids)) != len(migration_ids):
                persistent_errors.append("migration IDs are not unique")
            if len(set(source_request_ids)) != len(source_request_ids):
                persistent_errors.append("source request IDs are not unique")
            if int(source_channel.get("channel_create_count", -1)) != 1:
                persistent_errors.append("source communicator was not created once")
            if int(source_channel.get("channel_destroy_count", -1)) != 0:
                persistent_errors.append("source communicator was destroyed")
            if int(source_channel.get("channel_session_count", -1)) != len(
                session_rows
            ):
                persistent_errors.append("source session count is incomplete")
            if len(target_channels) != 4 or any(
                int(row.get("channel_create_count", -1)) != 1
                or int(row.get("channel_destroy_count", -1)) != 0
                or int(row.get("channel_session_count", -1))
                != len(session_rows)
                for row in target_channels
            ):
                persistent_errors.append("target channel lifecycle differs")
            rss_values = [
                int(row.get("process_tree_rss_kib", 0)) for row in memory_rows
            ]
            gpu_values = [
                int(row.get("gpu_used_memory_mib", 0)) for row in memory_rows
            ]
            handoff_values = [
                float(row["handoff_stall_ms"])
                for row in session_rows
                if row.get("handoff_stall_ms") is not None
            ]
            persistent_summary = {
                "format_version": 1,
                "status": "PASS" if not persistent_errors else "FAIL",
                "channel_generation": args.channel_generation,
                "sessions": len(session_rows),
                "migration_ids": migration_ids,
                "source_request_ids": source_request_ids,
                "source_channel": source_channel,
                "target_channels": target_channels,
                "handoff_stall_ms": handoff_values,
                "process_tree_rss_kib": rss_values,
                "gpu_used_memory_mib": gpu_values,
                "rss_growth_kib": (
                    rss_values[-1] - rss_values[0] if rss_values else None
                ),
                "gpu_growth_mib": (
                    gpu_values[-1] - gpu_values[0] if gpu_values else None
                ),
                "errors": persistent_errors,
            }
            common.write_json(
                out_root / "persistent_channel_summary.json",
                persistent_summary,
            )

        errors = [
            f"r{row['repetition']:02d} "
            f"{row.get('architecture', row['strategy'])} failed"
            for row in batch["runs"]
            if row.get("status") != "PASS"
            or row.get("acceptance", {}).get("status") != "PASS"
        ]
        if persistent_summary is not None:
            errors.extend(persistent_summary["errors"])
        final = {
            "format_version": 1,
            "status": "PASS" if not errors else "FAIL",
            "phase": args.phase,
            "expected_runs": args.repetitions
            * (
                2
                if (
                    args.ready_sync_comparison
                    or args.ready_notification_comparison
                    or args.deferred_comm_destroy_comparison
                )
                else 1
                if (
                    args.bridge_only or args.shadow_only_only or args.stop_and_copy_only
                )
                else 2
            ),
            "recorded_runs": len(batch["runs"]),
            "runs": batch["runs"],
            "persistent_channel_summary": persistent_summary,
            "errors": errors,
        }
        # Cancellation has no freeze, commit, or target TPOT windows.  Its
        # acceptance record is the measurement for this diagnostic run.
        if not (args.manager_m1_expect_stay or args.manager_m4_expect_cancel
                or args.paired_stay):
            write_measurements(out_root, batch["runs"])
        common.write_json(out_root / "acceptance.json", final)
        if errors:
            raise RuntimeError("; ".join(errors))
        batch["status"] = "COMPLETE"
        batch["ended_unix_s"] = time.time()
        common.write_json(out_root / "batch_status.json", batch)
        print(f"ONLINE_SHADOW_{args.phase.upper()}_COMPLETE: {out_root}")
    except BaseException as error:
        batch["status"] = "FAILED"
        batch["ended_unix_s"] = time.time()
        batch["error"] = f"{type(error).__name__}: {error}"
        common.write_json(out_root / "batch_status.json", batch)
        raise
    finally:
        if service_pool is not None:
            pooled = list(service_pool.get("processes", []))
            common.stop_processes(pooled)
            common.write_json(
                out_root / "persistent_process_lifetimes.json",
                {
                    "format_version": 1,
                    "channel_generation": args.channel_generation,
                    "sessions_requested": args.repetitions,
                    "processes": [
                        {
                            "name": item.name,
                            "pid": item.process.pid,
                            "started_unix_s": item.started_unix_s,
                            "ended_unix_s": item.ended_unix_s,
                            "returncode": item.returncode,
                            "log_path": str(item.log_path.resolve()),
                        }
                        for item in pooled
                    ],
                },
            )


if __name__ == "__main__":
    main()
