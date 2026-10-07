#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one online BridgeTP Phase 9 controlled migration.

The runner owns both OpenAI-compatible streaming requests, the response seam,
and the fast controller loop. The TP1/TP4 servers and the Phase 8 stager must
already be running with the same run directory and migration ID.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.bridge_tp.experiment_m1_wait import (  # noqa: E402
    M1PredictorRefreshGate,
)
from tools.bridge_tp.experiment_probability_gate import (  # noqa: E402
    ProbabilityGate,
    source_load_eligibility,
    urgency_eligibility,
)
from tools.bridge_tp.risk_observation import build_risk_observation  # noqa: E402
from tools.bridge_tp.risk_urgency import (  # noqa: E402
    DecodeRateTracker,
    build_snapshot,
)
from vllm.bridge_tp.controller.action_adapter import (  # noqa: E402
    ActionAdapter,
    ActionError,
)
from vllm.bridge_tp.controller.audit import AuditLog  # noqa: E402
from vllm.bridge_tp.controller.capacity_signal import (  # noqa: E402
    CapacityHeadroomTracker,
    CapacitySignal,
)
from vllm.bridge_tp.controller.config import ControllerConfig  # noqa: E402
from vllm.bridge_tp.controller.events import (  # noqa: E402
    Action,
    MigrationState,
    SourceRequestView,
    TriggerPath,
)
from vllm.bridge_tp.controller.manager_m0 import (  # noqa: E402
    ChannelRegistry,
    M0ExecutorAdapter,
    M0Proposal,
    MigrationManagerM0,
    RuntimeSnapshot,
    RuntimeStateCollector,
)
from vllm.bridge_tp.controller.manager_m1 import (  # noqa: E402
    M1StartConfig,
    M1StartController,
    M1StartDecision,
)
from vllm.bridge_tp.controller.manager_m2 import (  # noqa: E402
    M2RateConfig,
    M2RateController,
)
from vllm.bridge_tp.controller.manager_m3 import (  # noqa: E402
    M3CommitController,
)
from vllm.bridge_tp.controller.manager_m4 import (  # noqa: E402
    M4CancelController,
)
from vllm.bridge_tp.controller.manager_m5 import (  # noqa: E402
    PredictorEventReader,
)
from vllm.bridge_tp.controller.online_io import (  # noqa: E402
    ProxyRecorder,
    atomic_json_dump,
    build_gpu_resident_shadow_target_request,
    build_target_request,
    honored_generation,
    load_json,
    post_streaming_completion,
)
from vllm.bridge_tp.controller.policy import FastPolicy, RiskTracker  # noqa: E402
from vllm.bridge_tp.controller.predictor import SurvivalTable  # noqa: E402
from vllm.bridge_tp.controller.rate_controller import RateController  # noqa: E402
from vllm.bridge_tp.controller.response_proxy import ProxyMode  # noqa: E402
from vllm.bridge_tp.controller.sampling_contract import (  # noqa: E402
    freeze_strict_greedy_sampling,
)
from vllm.bridge_tp.controller.state_machine import (  # noqa: E402
    IllegalTransition,
    MigrationRecord,
    MigrationStateMachine,
)
from vllm.bridge_tp.controller.telemetry import (  # noqa: E402
    MetricsScraper,
    TelemetryError,
)
from vllm.bridge_tp.rolling_cutover import MECHANISM, read_json  # noqa: E402
from vllm.bridge_tp.runtime_control import RuntimeControl  # noqa: E402

_STOP = False


def _handle_signal(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument('--rolling-cutover', action='store_true')
    parser.add_argument('--rolling-reserve-tokens', type=int, default=512)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-request", type=Path, required=True)
    parser.add_argument(
        "--requested-max-output-tokens", type=int,
        help="client-visible total output budget across TP1 and TP4; "
             "defaults to the TP1 subrequest max_tokens",
    )
    parser.add_argument(
        "--migration-id",
        default=os.getenv("BRIDGETP_STREAM_MIGRATION_ID", "").strip(),
        help="must match BRIDGETP_STREAM_MIGRATION_ID on source and stager",
    )
    parser.add_argument("--max-seconds", type=float, default=900.0)
    parser.add_argument("--preflight-timeout-s", type=float, default=60.0)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--require-runtime-control",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--diagnostic-trigger-output-tokens",
        type=int,
        help=(
            "diagnostic-only fixed trigger boundary; must be supplied with "
            "--diagnostic-cutover-output-tokens"
        ),
    )
    parser.add_argument(
        "--diagnostic-cutover-output-tokens",
        type=int,
        help=(
            "diagnostic-only fixed cutover boundary; must be supplied with "
            "--diagnostic-trigger-output-tokens"
        ),
    )
    parser.add_argument(
        "--diagnostic-bridge-output-tokens",
        type=int,
        help="fixed output-token boundary for SHADOW -> HANDOFF",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--manager-m0-shadow",
        action="store_true",
        help="record advisory M0 decisions alongside the existing controller",
    )
    parser.add_argument(
        "--manager-m1-auto-start",
        action="store_true",
        help="start GPU-resident Shadow from online M1 evidence",
    )
    parser.add_argument(
        "--paired-stay",
        action="store_true",
        help="paired counterfactual: audit M1/M5 but keep the anchor on TP1",
    )
    parser.add_argument(
        "--risk-observation-shadow", action="store_true",
        help="record predecision capacity features without changing actions",
    )
    parser.add_argument("--probability-threshold", type=float)
    parser.add_argument("--probability-min-source-running", type=int, default=0)
    parser.add_argument("--probability-min-urgency", type=float, default=0.0)
    parser.add_argument("--probability-threshold-family", type=float, nargs="*",
                        default=[])
    parser.add_argument("--probability-assigned-action", choices=("START", "STAY"))
    parser.add_argument("--probability-assignment-seed", default="0")
    parser.add_argument("--probability-start-probability", type=float, default=0.5)
    parser.add_argument("--risk-model-config-sha256")
    parser.add_argument("--experiment-m1-action", choices=("NOW", "WAIT"),
                        help="experimental migration timing against M5 refresh")
    parser.add_argument(
        "--diagnostic-m1-max-source-free-kv-tokens", type=int,
        help="experiment-only: defer M1 Shadow until TP1 free KV is at most this",
    )
    parser.add_argument(
        "--m1-source-release-tail-s", type=float,
        help=("diagnostic tail allowance after estimated history copy "
              "until TP1 KV release"),
    )
    parser.add_argument(
        "--manager-m2-rate",
        action="store_true",
        help="apply three-profile M2 rates during active Shadow",
    )
    parser.add_argument(
        "--manager-m3-commit",
        action="store_true",
        help="commit at the first safe boundary after TP4 history readiness",
    )
    parser.add_argument(
        "--manager-m4-cancel",
        action="store_true",
        help="cancel pre-freeze Shadow if TP1 will likely finish soon",
    )
    parser.add_argument(
        "--manager-m5-predictor-shadow", action="store_true",
        help="audit frozen predictor events without changing M1-M4 decisions",
    )
    parser.add_argument("--predictor-event-path", type=Path)
    parser.add_argument("--predictor-checkpoint-sha256")
    parser.add_argument(
        "--manager-m2-force-initial-high",
        action="store_true",
        help="diagnostic smoke: arm HIGH before the first history chunk",
    )
    parser.add_argument(
        "--handoff-mode",
        choices=("bridge", "shadow-only"),
        default="bridge",
        help=(
            "bridge preserves SHADOW->HANDOFF->TAKEOVER; shadow-only keeps "
            "TP1 ownership until all TP4 ranks are ready and commits directly"
        ),
    )
    parser.add_argument(
        "--gpu-resident-shadow",
        action="store_true",
        help="admit a dormant TP4 request at Shadow start and patch its KV blocks",
    )
    parser.add_argument(
        "--stop-and-copy",
        action="store_true",
        help=(
            "freeze the selected source request at the diagnostic trigger; "
            "the following cutover value is only the Phase-8 control sentinel"
        ),
    )
    parser.add_argument(
        "--diagnostic-earliest-ready-cutover",
        action="store_true",
        help=(
            "after the fixed diagnostic trigger, select the first safe "
            "source cutover boundary only after initial history is GPU-ready "
            "with exact readback on all four TP4 ranks"
        ),
    )
    parser.add_argument(
        "--ready-notification-mode",
        choices=("FILE_POLL", "UDP"),
        default="FILE_POLL",
        help="P1 target-ready wake-up path; receipts remain authoritative",
    )
    parser.add_argument("--ready-notification-host", default="127.0.0.1")
    parser.add_argument("--ready-notification-port", type=int, default=0)
    parser.add_argument(
        "--ready-latch-poll-ms",
        type=float,
        default=5.0,
        help=(
            "authoritative receipt polling interval after all four UDP "
            "rank-ready hints have arrived; zero preserves timeout polling"
        ),
    )
    parser.add_argument(
        "--source-request-id",
        help="explicit request identity for persistent sequential sessions",
    )
    args = parser.parse_args()
    trigger = args.diagnostic_trigger_output_tokens
    cutover = args.diagnostic_cutover_output_tokens
    bridge = args.diagnostic_bridge_output_tokens
    if args.rolling_cutover and (
        not args.diagnostic_earliest_ready_cutover
        or not args.gpu_resident_shadow or args.handoff_mode != 'shadow-only'
        or args.stop_and_copy or args.rolling_reserve_tokens < 64
    ):
        raise ValueError('rolling cutover requires GPU Shadow-only earliest-ready')
    if args.diagnostic_earliest_ready_cutover and (
        (trigger is None and not args.manager_m1_auto_start)
        or cutover is not None
    ):
        parser.error(
            "earliest-ready cutover requires a trigger and forbids a fixed cutover"
        )
    if not args.diagnostic_earliest_ready_cutover and (
        (trigger is None) != (cutover is None)
    ):
        parser.error(
            "diagnostic trigger and cutover boundaries must be supplied together"
        )
    if (
        trigger is not None
        and cutover is not None
        and (trigger < 0 or cutover <= trigger)
    ):
        parser.error("diagnostic boundaries require 0 <= trigger < cutover")
    if bridge is not None and (
        trigger is None
        or cutover is None
        or not trigger < bridge < cutover
    ):
        parser.error("diagnostic Bridge boundary must be between trigger and cutover")
    if args.manager_m1_auto_start:
        if trigger is not None or cutover is not None or bridge is not None:
            parser.error("M1 auto-start forbids diagnostic fixed boundaries")
        if not args.diagnostic_earliest_ready_cutover:
            parser.error("M1 auto-start requires earliest-ready cutover")
        if args.handoff_mode != "shadow-only" or not args.gpu_resident_shadow:
            parser.error("M1 auto-start requires GPU-resident Shadow-only mode")
        if args.stop_and_copy:
            parser.error("M1 auto-start does not support Stop-and-Copy")
        if args.m1_source_release_tail_s is None:
            parser.error("M1 auto-start requires source KV release tail allowance")
    if args.paired_stay and not args.manager_m1_auto_start:
        parser.error("paired STAY requires M1 auto-start for comparable evidence")
    if args.probability_min_source_running < 0 or (
        args.probability_min_source_running and args.probability_threshold is None
    ):
        parser.error("source concurrency selection requires probability collection")
    if (not math.isfinite(args.probability_min_urgency)
            or args.probability_min_urgency < 0
            or (args.probability_min_urgency
                and args.probability_threshold is None)):
        parser.error("urgency selection requires finite U and probability mode")
    if args.probability_threshold is not None and (
        not args.manager_m1_auto_start or not args.manager_m5_predictor_shadow
        or args.experiment_m1_action or args.paired_stay
        or args.diagnostic_m1_max_source_free_kv_tokens is not None
    ):
        parser.error("probability collection requires M1/M5 and its own assignment")
    if args.experiment_m1_action and (
        not args.manager_m1_auto_start or not args.manager_m5_predictor_shadow
        or args.paired_stay
    ):
        parser.error("experimental timing requires M1/M5 and forbids paired STAY")
    if args.m1_source_release_tail_s is not None and (
        not args.manager_m1_auto_start
        or not math.isfinite(args.m1_source_release_tail_s)
        or args.m1_source_release_tail_s <= 0
    ):
        parser.error("M1 source release tail requires a positive finite value")
    if args.manager_m2_rate and not (
        args.manager_m1_auto_start and args.manager_m0_shadow
    ):
        parser.error("M2 rate requires M1 auto-start and M0 snapshots")
    if args.manager_m2_force_initial_high and not args.manager_m2_rate:
        parser.error("M2 forced HIGH requires --manager-m2-rate")
    if args.manager_m3_commit and not (
        args.manager_m2_rate
        and args.diagnostic_earliest_ready_cutover
        and args.handoff_mode == "shadow-only"
        and args.gpu_resident_shadow
    ):
        parser.error("M3 requires M1/M2 GPU-resident Shadow-only earliest-ready")
    if args.manager_m4_cancel and not args.manager_m3_commit:
        parser.error("M4 cancel requires M3 earliest-ready commit")
    if args.diagnostic_m1_max_source_free_kv_tokens is not None and (
        not args.manager_m1_auto_start
        or args.diagnostic_m1_max_source_free_kv_tokens <= 0
    ):
        parser.error("diagnostic M1 source-free gate requires M1 and a positive limit")
    if args.manager_m5_predictor_shadow and (
        args.predictor_event_path is None
        or args.predictor_checkpoint_sha256 is None
    ):
        parser.error("M5 shadow requires predictor event path and checkpoint SHA")
    if (
        args.gpu_resident_shadow
        and cutover is None
        and not args.diagnostic_earliest_ready_cutover
    ):
        parser.error("GPU-resident staging requires fixed diagnostic boundaries")
    if args.ready_notification_mode == "UDP" and not (
        0 < args.ready_notification_port <= 65535
    ):
        parser.error("UDP ready notification port is invalid")
    if args.ready_latch_poll_ms < 0:
        parser.error("ready latch poll interval cannot be negative")
    return args


def _prepare_source_request(
    source: dict[str, Any],
    run_dir: Path,
    request_id: str | None = None,
) -> dict[str, Any]:
    request = freeze_strict_greedy_sampling(source)
    request.update(
        {
            "request_id": request_id or f"bridgetp-phase9-{run_dir.name}",
            "stream": True,
            "return_token_ids": True,
        }
    )
    request.setdefault("ignore_eos", True)
    if int(request.get("max_tokens", 0)) <= 0:
        raise ValueError("source request max_tokens must be positive")
    return request


def effective_m1_start_decision(
    decision: M1StartDecision | None, paired_stay: bool,
) -> M1StartDecision | None:
    """Keep the natural decision in audit while suppressing actuation in STAY."""
    if paired_stay:
        return M1StartDecision("STAY", "paired STAY counterfactual")
    return decision


def _assert_fresh_run_dir(run_dir: Path) -> None:
    stale = [
        name
        for name in (
            "phase9_audit.jsonl",
            "earliest_ready_candidate.json",
            "session_manifest.json",
            "staging_manifest.json",
            "takeover_state.json",
            "response_proxy_stats.json",
            "unified_response.jsonl",
        )
        if (run_dir / name).exists()
    ]
    if stale:
        raise FileExistsError(
            f"Phase 9 run directory has stale artifacts: {', '.join(stale)}"
        )


def read_source_progress(
    run_dir: Path,
    source_request: dict[str, Any],
    now: float,
) -> SourceRequestView | None:
    path = run_dir / "source_progress.json"
    try:
        raw = load_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return SourceRequestView(
        request_id=str(raw["source_request_id"]),
        prompt_tokens=int(raw.get("num_prompt_tokens", 0)),
        output_tokens=int(raw["num_output_tokens"]),
        computed_tokens=int(raw["num_computed_tokens"]),
        pending_tokens=int(raw.get("num_pending_tokens", 0)),
        arrival_unix_s=float(raw.get("arrival_unix_s", now)),
        last_token_unix_s=float(raw.get("updated_unix_s", now)),
        group_id=source_request.get("bridgetp_group_id"),
        is_group_longest=bool(source_request.get("bridgetp_group_longest", False)),
    )


def wait_for_runtime_control(
    run_dir: Path,
    generation: int,
    source_request: dict[str, Any],
    source_future: Future[dict[str, Any]],
    timeout_s: float,
    require_marker: bool,
) -> SourceRequestView:
    deadline = time.monotonic() + timeout_s
    marker = run_dir / "runtime_control_honored"
    while time.monotonic() < deadline:
        if source_future.done():
            source_future.result()
            raise RuntimeError("source ended before runtime-control preflight")
        marker_ok = (
            not require_marker or (honored_generation(marker) or -1) >= generation
        )
        progress = read_source_progress(run_dir, source_request, time.time())
        if marker_ok and progress is not None:
            return progress
        time.sleep(0.02)
    raise TimeoutError(
        "source did not honor the Phase 9 control generation or publish "
        f"source_progress.json within {timeout_s:.1f}s"
    )


def _start_target_if_ready(
    *,
    run_dir: Path,
    source_request: dict[str, Any],
    requested_max_output_tokens: int,
    adapter: ActionAdapter,
    recorder: ProxyRecorder,
    executor: ThreadPoolExecutor,
    target_url: str,
    request_timeout_s: float,
    target_future: Future[dict[str, Any]] | None,
    gpu_resident_shadow: bool = False,
    cutover_output_tokens: int | None = None,
    stop_and_copy: bool = False,
    target_request_name: str | None = None,
    rolling_cutover: bool = False,
) -> Future[dict[str, Any]] | None:
    if target_future is not None:
        return target_future
    path = run_dir / (
        "session_manifest.json" if gpu_resident_shadow else "staging_manifest.json"
    )
    if not path.exists():
        return None
    staging = load_json(path)
    if gpu_resident_shadow:
        if cutover_output_tokens is None:
            # Target admission needs a candidate prefix length. The source
            # freeze boundary is published separately after resident readback.
            return None
        if stop_and_copy:
            cutover_output_tokens = int(staging["snapshot_num_output_tokens"])
        target_request, cutover = build_gpu_resident_shadow_target_request(
            source_request,
            staging,
            target_request_name or run_dir.name,
            cutover_output_tokens,
            allow_complete_prefix=stop_and_copy,
            requested_max_output_tokens=requested_max_output_tokens,
        )
    else:
        target_request, cutover = build_target_request(
            source_request,
            staging,
            target_request_name or run_dir.name,
            requested_max_output_tokens=requested_max_output_tokens,
        )
    if not rolling_cutover and recorder.proxy.cutover_index != cutover:
        raise RuntimeError(
            "stager cutover differs from controller cutover: "
            f"{cutover} != {recorder.proxy.cutover_index}"
        )
    atomic_json_dump(target_request, run_dir / "target_request.json")
    adapter.mark_target_request_admitted(note=f"target admitted at cutover {cutover}")
    return executor.submit(
        recorded_completion,
        target_url,
        target_request,
        request_timeout_s,
        recorder.on_target_token,
        run_dir / "target_response.json",
        recorder,
    )


def recorded_completion(
    base_url,
    payload,
    timeout_s,
    token_sink,
    response_path: Path,
    recorder: ProxyRecorder,
):
    """Retain service failures and visible tokens even before the first tick."""
    try:
        return post_streaming_completion(
            base_url, payload, timeout_s, token_sink, failure_path=response_path
        )
    finally:
        atomic_json_dump(
            recorder.stats(), response_path.parent / "response_proxy_stats.json"
        )


def step_local(
    policy: FastPolicy,
    machine: MigrationStateMachine,
    adapter: ActionAdapter,
    audit: AuditLog,
    record: MigrationRecord,
    request: SourceRequestView,
    pool1: Any,
    pool4: Any,
    risk_value: float,
    rate: RateController,
    now: float,
    dry_run: bool,
    config: ControllerConfig,
    recorder: ProxyRecorder,
    max_tokens: int,
    diagnostic_trigger_output_tokens: int | None = None,
    diagnostic_cutover_output_tokens: int | None = None,
    diagnostic_earliest_ready_cutover: bool = False,
    capacity_signal: CapacitySignal | None = None,
    stop_and_copy: bool = False,
    m1_start_decision: M1StartDecision | None = None,
) -> None:
    decision = policy.evaluate(
        request,
        MigrationState.LOCAL,
        pool1,
        pool4,
        risk_value,
        rate.rate_bytes_s,
        now_unix_s=now,
        active_migrations=0,
    )
    audit.write({"kind": "decision", **decision.to_json()})
    diagnostic_boundary = diagnostic_trigger_output_tokens is not None
    capacity_requested = bool(capacity_signal is not None and capacity_signal.active)
    capacity_allowed = False
    if capacity_requested:
        target_unavailable = (
            pool4.kv_usage_frac > config.policy.max_target_kv_usage_frac
            or pool4.num_waiting > config.policy.max_target_waiting
        )
        capacity_allowed = not target_unavailable
        audit.write(
            {
                "kind": "capacity_pilot_decision",
                "action": "START_SHADOW" if capacity_allowed else "STAY",
                "reason": (
                    "measured headroom trigger and target passes current guard"
                    if capacity_allowed
                    else "measured headroom trigger but target fails current guard"
                ),
                "signal": capacity_signal.to_json(),
                "target_kv_usage_frac": pool4.kv_usage_frac,
                "target_waiting": pool4.num_waiting,
                "target_reservation_proven": False,
            }
        )
    performance_allowed = not (
        config.capacity_pilot.enabled and config.capacity_pilot.exclusive_trigger_path
    )
    should_start = (
        m1_start_decision.action == "START_SHADOW"
        if m1_start_decision is not None
        else (
            diagnostic_boundary
            or capacity_allowed
            or (performance_allowed and decision.action is Action.START_SHADOW)
        )
    )
    if dry_run or not should_start:
        return
    if diagnostic_trigger_output_tokens is None:
        trigger = request.output_tokens + 1
        cutover = (
            max_tokens - 1
            if diagnostic_earliest_ready_cutover
            else trigger + config.handoff_output_tokens
        )
    else:
        if (
            diagnostic_cutover_output_tokens is None
            and not diagnostic_earliest_ready_cutover
        ):
            raise RuntimeError("diagnostic cutover boundary is missing")
        trigger = diagnostic_trigger_output_tokens
        # The source still needs an initial upper boundary while its history
        # and deltas stream.  It is replaced once all four initial TP4
        # GPU-history receipts are exact.  Do not expose this sentinel as the
        # response-proxy cutover; that boundary is selected later.
        cutover = (
            max_tokens - 1
            if diagnostic_earliest_ready_cutover
            else diagnostic_cutover_output_tokens
        )
        if request.output_tokens >= trigger:
            audit.write(
                {
                    "kind": "diagnostic_boundary_missed",
                    "observed_output_tokens": request.output_tokens,
                    "trigger_output_tokens": trigger,
                    "cutover_output_tokens": cutover,
                }
            )
            raise RuntimeError(
                "source reached diagnostic trigger before the controller armed it: "
                f"observed={request.output_tokens}, trigger={trigger}"
            )
        audit.write(
            {
                "kind": "diagnostic_boundary_forced",
                "observed_output_tokens": request.output_tokens,
                "trigger_output_tokens": trigger,
                "cutover_output_tokens": cutover,
            }
        )
    if cutover >= max_tokens:
        audit.write(
            {
                "kind": "decision_refused",
                "reason": "not enough generation budget after cutover",
                "trigger_output_tokens": trigger,
                "cutover_output_tokens": cutover,
            }
        )
        return
    if not diagnostic_earliest_ready_cutover:
        recorder.set_cutover(trigger if stop_and_copy else cutover, now)
    if m1_start_decision is not None:
        trigger_path = TriggerPath.MANAGER_M1_START
        trigger_reason = m1_start_decision.reason
    elif diagnostic_boundary:
        trigger_path = TriggerPath.DIAGNOSTIC_FIXED_BOUNDARY
        trigger_reason = "diagnostic fixed boundary"
    elif capacity_allowed:
        trigger_path = TriggerPath.CAPACITY_PILOT
        trigger_reason = "CAP-0 measured source headroom trigger"
    else:
        trigger_path = (
            getattr(
                decision,
                "trigger_path",
                None,
            )
            or TriggerPath.PERFORMANCE_OPPORTUNITY
        )
        trigger_reason = decision.reason
    adapter.arm_shadow(
        trigger,
        rate.rate_gib_s,
        cutover_output_tokens=cutover,
        note=trigger_reason,
    )
    if trigger_reason.startswith("probability collector:"):
        audit.write({
            "kind": "experiment_probability_execution",
            "actual_action": "START_SHADOW", "unix_s": now,
            "observed_output_tokens": request.output_tokens,
            "trigger_output_tokens": trigger,
            "reason": trigger_reason,
        })
    record.trigger_output_tokens = trigger
    record.cutover_output_tokens = (
        None if diagnostic_earliest_ready_cutover else cutover
    )
    record.t_decision = now
    record.trigger_path = trigger_path
    if diagnostic_earliest_ready_cutover:
        audit.write(
            {
                "kind": (
                    "manager_m1_earliest_ready_armed"
                    if m1_start_decision is not None
                    else "diagnostic_earliest_ready_armed"
                ),
                "trigger_output_tokens": trigger,
                "initial_cutover_sentinel": cutover,
                "reason": "wait for four initial GPU-history exact receipts",
            }
        )
    machine.transition(
        record.migration_id,
        MigrationState.SHADOW,
        now,
        trigger_reason,
    )


def _complete_m4_cancel(
    machine: MigrationStateMachine,
    adapter: ActionAdapter,
    audit: AuditLog,
    record: MigrationRecord,
    now: float,
    recorder: ProxyRecorder,
) -> bool:
    """Complete target cleanup after TP1 ownership has been preserved."""
    assert record.m4_source_cleanup_done
    assert record.m4_cancel_reason is not None
    try:
        target_cleanup = adapter.cancel_shadow_target(record.m4_cancel_reason)
    except ActionError as error:
        audit.write({"kind": "manager_m4_cancel_error", "detail": str(error)})
        return True
    audit.write({
        "kind": "manager_m4_target_cleanup",
        "status": (
            target_cleanup.get("status") if target_cleanup is not None else None
        ),
    })
    audit.write({"kind": "manager_m4_cancelled", "reason": record.m4_cancel_reason})
    recorder.on_rollback(now, record.m4_cancel_reason)
    machine.transition(
        record.migration_id, MigrationState.CANCELLED,
        now, record.m4_cancel_reason,
    )
    return True


def step_m4_cancel(
    manager: M4CancelController,
    table: SurvivalTable,
    machine: MigrationStateMachine,
    adapter: ActionAdapter,
    audit: AuditLog,
    record: MigrationRecord,
    request: SourceRequestView,
    *,
    max_output_tokens: int,
    ignore_eos: bool,
    source_free_kv_tokens: int,
    source_guard_free_kv_tokens: int,
    source_capacity_pressure: bool,
    now: float,
    dry_run: bool,
    recorder: ProxyRecorder,
) -> bool:
    """Try pre-freeze cancellation; leave TP1 as the request owner."""
    if record.m4_source_cleanup_done:
        return _complete_m4_cancel(
            machine, adapter, audit, record, now, recorder
        )
    frozen_path = adapter.run_dir / "request_frozen_receipt.json"
    freeze_started = (
        frozen_path.is_file()
        or record.urgent_cutover_prearmed_unix_s is not None
        or (
            record.cutover_output_tokens is not None
            and request.output_tokens + 16 >= record.cutover_output_tokens
        )
    )
    decision = manager.decide(
        request, table,
        max_output_tokens=max_output_tokens,
        ignore_eos=ignore_eos,
        source_free_kv_tokens=source_free_kv_tokens,
        source_guard_free_kv_tokens=source_guard_free_kv_tokens,
        source_capacity_pressure=source_capacity_pressure,
        freeze_started=freeze_started or now - request.last_token_unix_s > 2.0,
    )
    audit.write({"kind": "manager_m4_cancel_decision", "decision": decision.to_json()})
    if decision.action != "CANCEL_SHADOW":
        return False
    if not dry_run:
        # The source can advance while the controller makes its decision.
        # Never cancel after its freeze receipt becomes visible.
        if frozen_path.is_file():
            audit.write({"kind": "manager_m4_cancel_blocked", "reason": "frozen"})
            return False
        try:
            if adapter.refresh_binding() is None:
                audit.write({
                    "kind": "manager_m4_cancel_blocked",
                    "reason": "cleanup binding unavailable",
                })
                return True
            adapter.disarm(decision.reason)
            cleanup = adapter.cancel(decision.reason, abort_source=False)
            if cleanup.get("source_abort_dispatched") is not False:
                raise ActionError(
                    "M4 cleanup did not confirm continued TP1 ownership"
                )
            audit.write({
                "kind": "manager_m4_source_cleanup",
                "state": cleanup.get("state"),
                "source_abort_dispatched": cleanup.get("source_abort_dispatched"),
            })
            record.m4_source_cleanup_done = True
            record.m4_cancel_reason = decision.reason
        except ActionError as error:
            audit.write({"kind": "manager_m4_cancel_error", "detail": str(error)})
            return True
    else:
        record.m4_source_cleanup_done = True
        record.m4_cancel_reason = decision.reason
    return _complete_m4_cancel(
        machine, adapter, audit, record, now, recorder
    )


def step_rolling_shadow(
    *, adapter: ActionAdapter, audit: AuditLog, record: MigrationRecord,
    request: SourceRequestView, max_tokens: int, reserve_tokens: int,
) -> str:
    """Reserve target space; the source owns plans and pre-freeze deferrals."""
    if record.candidate_cutover_output_tokens is None:
        buffered, ranks, detail = adapter.poll_initial_history_gpu_buffered()
        if not buffered:
            return ''
        session = load_json(adapter.run_dir / 'session_manifest.json')
        outstanding = max(0, request.computed_tokens - session['num_computed_tokens'])
        reservation = min(max_tokens - 1,
                          request.output_tokens + outstanding + reserve_tokens)
        if reservation <= request.output_tokens + 16:
            return 'rolling reservation has no remaining safe output budget'
        budgets = load_json(adapter.run_dir / 'request_budgets.json')
        value = {
            'format_version': 1, 'mechanism': MECHANISM,
            'migration_id': session['migration_id'],
            'source_request_id': session['source_request_id'],
            'initial_end_token': session['num_computed_tokens'],
            'num_prompt_tokens': session['num_prompt_tokens'],
            'reservation_output_tokens': reservation,
            'source_max_output_tokens': max_tokens,
            'total_output_budget': budgets['requested_max_output_tokens'],
            'published_unix_s': time.time(),
        }
        atomic_json_dump(value, adapter.run_dir / 'rolling_reservation.json')
        atomic_json_dump(
            {**value, 'cutover_output_tokens': reservation,
             'reservation_only': True},
            adapter.run_dir / 'earliest_ready_candidate.json',
        )
        record.candidate_cutover_output_tokens = reservation
        audit.write({'kind': 'rolling_target_reserved', **value,
                     'ranks': sorted(ranks), 'detail': detail})
    plan = read_json(adapter.run_dir / 'rolling_source_plan.json')
    if plan.get('status') == 'RESERVATION_EXHAUSTED':
        return 'rolling reservation exhausted before safe freeze; source continues'
    if plan and plan.get('version') != getattr(record, 'rolling_seen_version', None):
        if plan.get('migration_id') != record.migration_id:
            raise RuntimeError('source rolling plan migration ID differs')
        record.rolling_seen_version = plan['version']
        audit.write({'kind': 'rolling_cutover_plan_observed', 'plan': plan})
    return ''


def step_shadow(
    policy: FastPolicy,
    machine: MigrationStateMachine,
    adapter: ActionAdapter,
    audit: AuditLog,
    record: MigrationRecord,
    request: SourceRequestView,
    pool1: Any,
    pool4: Any,
    risk_value: float,
    rate: RateController,
    now: float,
    dry_run: bool,
    recorder: ProxyRecorder,
    capacity_signal: CapacitySignal | None = None,
    diagnostic_earliest_ready_cutover: bool = False,
    max_tokens: int | None = None,
    manager_m2: M2RateController | None = None,
    m2_snapshot: RuntimeSnapshot | None = None,
    manager_m3: M3CommitController | None = None,
    rolling_cutover: bool = False,
    rolling_reserve_tokens: int = 512,
) -> None:
    remaining = policy.migration_bytes(request)
    tpot_samples = getattr(pool4, "tpot_samples", 0)
    m2_decision = None
    if manager_m2 is not None:
        if m2_snapshot is None:
            raise RuntimeError("M2 rate requires a current runtime snapshot")
        m2_decision = manager_m2.decide(m2_snapshot)
        rate.rate_bytes_s = m2_decision.rate_bytes_s
        rate.last_reason = m2_decision.reason
        new_rate = rate.rate_bytes_s
    else:
        new_rate = rate.step(
            pool4.p99_tpot_s if tpot_samples > 0 else None,
            remaining,
            seconds_to_deadline=None,
        )
    audit.write(
        {
            "kind": "rate",
            "rate_bytes_s": new_rate,
            "rate_gib_s": rate.rate_gib_s,
            "reason": rate.last_reason,
            "native_p99_tpot_s": pool4.p99_tpot_s,
            "native_tpot_samples": tpot_samples,
            "native_tpot_metric": getattr(pool4, "tpot_metric", None),
            "manager_m2_decision": (
                m2_decision.to_json() if m2_decision is not None else None
            ),
            "manager_m2_snapshot": (
                m2_snapshot.to_json() if m2_decision is not None else None
            ),
        }
    )
    if not dry_run:
        adapter.set_rate(rate.rate_gib_s, note=rate.last_reason)

    late_candidate_reason = ""
    if rolling_cutover:
        if dry_run or max_tokens is None:
            raise ValueError('rolling cutover requires an online output budget')
        late_candidate_reason = step_rolling_shadow(
            adapter=adapter, audit=audit, record=record, request=request,
            max_tokens=max_tokens, reserve_tokens=rolling_reserve_tokens,
        )
    elif (
        diagnostic_earliest_ready_cutover
        and record.candidate_cutover_output_tokens is None
    ):
        buffered, ranks, detail = adapter.poll_initial_history_gpu_buffered()
        audit.write(
            {
                "kind": "earliest_ready_buffered_poll",
                "ready": buffered,
                "ranks": sorted(ranks),
                "detail": detail,
                "observed_output_tokens": request.output_tokens,
            }
        )
        if buffered:
            if max_tokens is None:
                raise RuntimeError("earliest-ready cutover needs source max_tokens")
            # TP4 needs a concrete prefix length before it can allocate KV
            # blocks and turn buffered history into exact resident evidence.
            # Allow more time when source generation has built a large delta
            # backlog during a throttled history transfer. This estimate does
            # not prove catch-up; the exact rank watermarks gate selection.
            initial_end = int(
                load_json(adapter.run_dir / "session_manifest.json")[
                    "num_computed_tokens"
                ]
            )
            outstanding_tokens = max(0, request.computed_tokens - initial_end)
            # The outstanding delta is only the backlog at this snapshot.
            # Keep another 64 output tokens for TP4 admission, exact GPU
            # readback, delta catch-up, and source-control propagation.
            candidate_lead_tokens = max(64, outstanding_tokens + 64)
            candidate = max(
                int(request.output_tokens) + candidate_lead_tokens,
                int(record.trigger_output_tokens or 0) + 1,
            )
            if manager_m3 is not None and candidate < max_tokens:
                base_candidate = candidate
                m3_decision = manager_m3.plan_candidate(
                    output_tokens=request.output_tokens,
                    base_candidate=candidate,
                    max_output_tokens=max_tokens,
                )
                candidate = m3_decision.candidate_output_tokens
                audit.write({
                    "kind": "manager_m3_candidate_decision",
                    "base_candidate_output_tokens": base_candidate,
                    "decision": m3_decision.to_json(),
                })
            if candidate >= max_tokens:
                late_candidate_reason = (
                    "initial history arrived too late for an earliest-ready "
                    f"candidate: candidate={candidate}, max_tokens={max_tokens}"
                )
            else:
                atomic_json_dump(
                    {
                        "format_version": 1,
                        "migration_id": record.migration_id,
                        "cutover_output_tokens": candidate,
                        "buffered_output_tokens": request.output_tokens,
                        "outstanding_delta_tokens": outstanding_tokens,
                        "published_unix_s": time.time(),
                    },
                    adapter.run_dir / "earliest_ready_candidate.json",
                )
                recorder.set_cutover(candidate, now)
                record.candidate_cutover_output_tokens = candidate
                # Under measured source pressure, arm the exact source
                # boundary while there is still room for control propagation.
                # The source may then freeze at this boundary until TP4 has
                # finished restoring history and the delta watermark.
                urgent_wait = (
                    not dry_run
                    and m2_decision is not None
                    and m2_decision.profile == "HIGH"
                    and m2_decision.reason == "source guard horizon is short"
                )
                if urgent_wait:
                    adapter.set_cutover(
                        candidate,
                        note="urgent source pressure: wait for resident history",
                    )
                    record.urgent_cutover_prearmed_unix_s = time.time()
                    audit.write(
                        {
                            "kind": "urgent_cutover_prearmed",
                            "cutover_output_tokens": candidate,
                            "source_time_to_guard_s": (
                                m2_decision.source_time_to_guard_s
                            ),
                        }
                    )
                audit.write(
                    {
                        "kind": "earliest_ready_candidate_published",
                        "cutover_output_tokens": candidate,
                        "buffered_output_tokens": request.output_tokens,
                        "candidate_lead_tokens": candidate - request.output_tokens,
                        "outstanding_delta_tokens": outstanding_tokens,
                        "ranks": sorted(ranks),
                        "detail": detail,
                    }
                )
    elif diagnostic_earliest_ready_cutover and record.cutover_output_tokens is None:
        if (
            manager_m3 is not None
            and record.urgent_cutover_prearmed_unix_s is None
            and m2_decision is not None
            and m2_decision.profile == "HIGH"
            and m2_decision.reason == "source guard horizon is short"
            and not dry_run
        ):
            adapter.set_cutover(
                record.candidate_cutover_output_tokens,
                note="urgent source pressure after M3 candidate publication",
            )
            record.urgent_cutover_prearmed_unix_s = time.time()
            audit.write({
                "kind": "urgent_cutover_prearmed",
                "cutover_output_tokens": record.candidate_cutover_output_tokens,
                "source_time_to_guard_s": m2_decision.source_time_to_guard_s,
            })
        ready, ranks, detail = adapter.poll_initial_history_gpu_ready()
        audit.write(
            {
                "kind": "earliest_ready_poll",
                "ready": ready,
                "ranks": sorted(ranks),
                "detail": detail,
                "observed_output_tokens": request.output_tokens,
            }
        )
        candidate = record.candidate_cutover_output_tokens
        assert candidate is not None
        if ready:
            safe_watermark_lead_tokens = 16
            if (
                request.output_tokens > candidate
                or (
                    record.urgent_cutover_prearmed_unix_s is None
                    and request.output_tokens + safe_watermark_lead_tokens
                    > candidate
                )
            ):
                late_candidate_reason = (
                    "safe cutover publication lead exhausted after history "
                    f"became resident: output={request.output_tokens}, "
                    f"candidate={candidate}"
                )
            else:
                progress_ready, progress, progress_detail = (
                    adapter.poll_delta_gpu_resident_progress()
                )
                delta_lag_tokens = (
                    max(0, request.computed_tokens - min(progress.values()))
                    if progress_ready else None
                )
                audit.write(
                    {
                        "kind": "earliest_ready_delta_progress",
                        "observed_output_tokens": request.output_tokens,
                        "rank_gpu_resident_end_tokens": progress,
                        "delta_lag_tokens": delta_lag_tokens,
                        "maximum_delta_lag_tokens": 16,
                        "detail": progress_detail,
                    }
                )
                if delta_lag_tokens is None or delta_lag_tokens > 16:
                    if (
                        record.urgent_cutover_prearmed_unix_s is None
                        and request.output_tokens + 16 >= candidate
                    ):
                        late_candidate_reason = (
                            "delta did not catch up before the candidate's "
                            f"safe publication point: output={request.output_tokens}, "
                            f"candidate={candidate}, lag={delta_lag_tokens}"
                        )
                    # Continue decoding the source while the four ranks catch up.
                    # The already admitted target owns this fixed candidate.
                else:
                    if not dry_run:
                        adapter.set_cutover(
                            candidate,
                            note=(
                                "earliest-ready: four initial GPU-history "
                                "receipts exact"
                            ),
                        )
                    record.cutover_output_tokens = candidate
                    if manager_m3 is not None:
                        machine.transition(
                            record.migration_id,
                            MigrationState.READY_NOT_COMMITTED,
                            now,
                            "four-rank initial history and live delta ready; "
                            "future cutover scheduled",
                        )
                    audit.write(
                        {
                            "kind": "earliest_ready_cutover_selected",
                            "cutover_output_tokens": candidate,
                            "selection_output_tokens": request.output_tokens,
                            "safety_lead_tokens": candidate - request.output_tokens,
                            "safe_watermark_output_tokens": candidate,
                            "safe_watermark_lead_tokens": safe_watermark_lead_tokens,
                            "delta_lag_tokens": delta_lag_tokens,
                            "rank_gpu_resident_end_tokens": progress,
                            "ranks": sorted(ranks),
                            "detail": detail,
                        }
                    )
        elif request.output_tokens > candidate or (
            record.urgent_cutover_prearmed_unix_s is None
            and request.output_tokens + 16 > candidate
        ):
            late_candidate_reason = (
                "initial history was not resident before the candidate's "
                f"safe publication point: output={request.output_tokens}, "
                f"candidate={candidate}"
            )

        if record.urgent_cutover_prearmed_unix_s is not None:
            frozen_path = adapter.run_dir / "request_frozen_receipt.json"
            if frozen_path.is_file() and record.cutover_output_tokens is None:
                frozen = load_json(frozen_path)
                frozen_s = float(frozen["frozen_unix_ns"]) / 1e9
                wait_s = max(0.0, now - frozen_s)
                audit.write(
                    {
                        "kind": "urgent_history_wait",
                        "cutover_output_tokens": candidate,
                        "wait_s": wait_s,
                        "history_ready": ready,
                    }
                )
                if wait_s > 5.0:
                    late_candidate_reason = (
                        "urgent history wait exceeded 5 seconds at the "
                        f"frozen candidate {candidate}"
                    )

    finalizer_error_path = adapter.run_dir / "cutover_finalize_error.json"
    if finalizer_error_path.is_file():
        failure = load_json(finalizer_error_path)
        late_candidate_reason = (
            "source final delta failed: " + str(failure.get("error", "unknown"))
        )
    diagnostic_path = record.trigger_path is TriggerPath.DIAGNOSTIC_FIXED_BOUNDARY
    safety_path = record.trigger_path in {
        TriggerPath.CAPACITY_PILOT,
        TriggerPath.POLICY_OOM_RISK,
        TriggerPath.MANAGER_M1_START,
    }
    if late_candidate_reason:
        abandon = True
        reason = late_candidate_reason
    elif diagnostic_path:
        # Fixed-boundary experiments isolate mechanism timing.  Re-evaluating
        # the online policy after forcibly entering Shadow would make paired
        # strategy runs follow different state paths and invalidate the
        # comparison.  Safety/readback/commit gates still remain mandatory.
        abandon = bool(late_candidate_reason)
        reason = late_candidate_reason
    elif safety_path:
        abandon = pool4.kv_usage_frac > policy.cfg.max_target_kv_usage_frac + 0.10
        reason = (
            f"target risk too high: kv={pool4.kv_usage_frac:.2f}" if abandon else ""
        )
        if (
            not abandon
            and record.trigger_path is TriggerPath.CAPACITY_PILOT
            and capacity_signal is not None
            and capacity_signal.transition == "CLEAR"
        ):
            abandon = True
            reason = "CAP-0 source headroom recovered before cutover"
    else:
        abandon, reason = policy.should_abandon(
            request,
            pool1,
            pool4,
            rate.rate_bytes_s,
            risk_value,
        )
    if abandon:
        audit.write({"kind": "abandon", "reason": reason})
        if not dry_run:
            adapter.disarm(reason)
            binding = adapter.wait_for_preparing_binding()
            if binding is None:
                audit.write(
                    {
                        "kind": "action_error",
                        "detail": (
                            "timed out waiting for the dynamically armed "
                            "snapshot to publish a PREPARING cleanup binding"
                        ),
                    }
                )
            else:
                try:
                    cleanup = adapter.cancel(reason, abort_source=False)
                    audit.write(
                        {
                            "kind": "cleanup_complete",
                            "state": cleanup.get("state"),
                            "source_abort_dispatched": cleanup.get(
                                "source_abort_dispatched"
                            ),
                        }
                    )
                    cancel_target = getattr(adapter, "cancel_shadow_target", None)
                    if callable(cancel_target):
                        target_cleanup = cancel_target(reason)
                        audit.write(
                            {
                                "kind": "target_cleanup_complete",
                                "status": (
                                    target_cleanup.get("status")
                                    if target_cleanup is not None
                                    else None
                                ),
                            }
                        )
                except ActionError as error:
                    audit.write({"kind": "action_error", "detail": str(error)})
        terminal_now = time.time()
        recorder.on_rollback(terminal_now, reason)
        machine.transition(
            record.migration_id,
            MigrationState.CANCELLED,
            terminal_now,
            reason,
        )
        return


def step_handoff(
    machine: MigrationStateMachine,
    adapter: ActionAdapter,
    audit: AuditLog,
    record: MigrationRecord,
    recorder: ProxyRecorder,
    now: float,
    dry_run: bool,
) -> None:
    ready, ranks, detail = adapter.poll_target_ready()
    for rank in ranks:
        machine.mark_rank_ready(record.migration_id, rank)
    if not ready:
        audit.write({"kind": "handoff_wait", "detail": detail})
        return
    if dry_run:
        return
    try:
        result = adapter.commit()
    except ActionError as error:
        audit.write({"kind": "commit_refused", "detail": str(error)})
        try:
            adapter.rollback(f"commit refused: {error}")
        except ActionError as rollback_error:
            audit.write({"kind": "rollback_failed", "detail": str(rollback_error)})
            machine.transition(
                record.migration_id,
                MigrationState.FAILED,
                now,
                str(error),
            )
            return
        recorder.on_rollback(now, str(error))
        machine.transition(
            record.migration_id,
            MigrationState.ROLLED_BACK,
            now,
            str(error),
        )
        return
    recorder.on_commit(time.time())
    audit.write({"kind": "commit", "server_state": result})
    try:
        machine.transition(
            record.migration_id,
            MigrationState.TAKEOVER,
            now,
            "committed",
        )
    except IllegalTransition as error:
        audit.write({"kind": "invariant_violation", "detail": str(error)})
        raise


def step_shadow_only_takeover(
    machine: MigrationStateMachine,
    adapter: ActionAdapter,
    audit: AuditLog,
    record: MigrationRecord,
    recorder: ProxyRecorder,
    now: float,
    dry_run: bool,
    ready_wait_timeout_s: float = 0.0,
) -> None:
    """Commit directly from Shadow after the four-rank GPU readback gate.

    In GPU-resident Shadow mode, the dormant TP4 request already owns its
    final block table and history/deltas are injected into those blocks before
    this gate succeeds.  The controller never enters Bridge/Handoff.
    """
    ready, ranks, detail = adapter.wait_for_target_ready(ready_wait_timeout_s)
    for rank in ranks:
        machine.mark_rank_ready(record.migration_id, rank)
    if not ready:
        audit.write({"kind": "shadow_only_ready_wait", "detail": detail})
        return
    if dry_run:
        return
    controller_wakeup_at = time.time()
    commit_dispatched_at = time.time()
    try:
        result = adapter.commit()
    except ActionError as error:
        audit.write({"kind": "shadow_only_commit_refused", "detail": str(error)})
        try:
            adapter.rollback(f"shadow-only commit refused: {error}")
        except ActionError as rollback_error:
            audit.write({"kind": "rollback_failed", "detail": str(rollback_error)})
            machine.transition(
                record.migration_id, MigrationState.FAILED, now, str(error)
            )
            return
        recorder.on_rollback(now, str(error))
        machine.transition(
            record.migration_id, MigrationState.ROLLED_BACK, now, str(error)
        )
        return
    committed_at = time.time()
    recorder.on_commit(committed_at)
    audit.write(
        {
            "kind": "shadow_only_commit",
            "server_state": result,
            "direct_transition": f"{record.state.value}->TAKEOVER",
            "controller_wakeup_unix_s": controller_wakeup_at,
            "commit_dispatched_unix_s": commit_dispatched_at,
            "commit_completed_unix_s": committed_at,
            "ready_notification": adapter.ready_notification_evidence(),
        }
    )
    machine.transition(
        record.migration_id,
        MigrationState.TAKEOVER,
        committed_at,
        "shadow fully synchronized; direct atomic takeover",
    )


def _finish_source_without_commit(
    machine: MigrationStateMachine,
    adapter: ActionAdapter,
    audit: AuditLog,
    record: MigrationRecord,
    recorder: ProxyRecorder,
    now: float,
) -> None:
    if record.state is MigrationState.LOCAL:
        machine.transition(
            record.migration_id,
            MigrationState.COMPLETED_ON_TP1,
            now,
            "source reached EOS before migration",
        )
        return
    if record.state in {
        MigrationState.SHADOW,
        MigrationState.READY_NOT_COMMITTED,
    }:
        reason = "source reached EOS before target ready"
        # An admitted target completion may still be deferred in the TP4
        # scheduler.  Leaving its SSE reader alive makes the executor wait for
        # the request timeout even though the source has already finished.
        try:
            target_cleanup = adapter.cancel_shadow_target(reason)
            audit.write({
                "kind": "source_eos_target_cleanup",
                "status": (
                    target_cleanup.get("status")
                    if target_cleanup is not None else None
                ),
            })
        except ActionError as error:
            audit.write({"kind": "action_error", "detail": str(error)})
        binding = adapter.refresh_binding()
        if binding is not None:
            try:
                adapter.cancel(reason, abort_source=False)
            except ActionError as error:
                audit.write({"kind": "action_error", "detail": str(error)})
        recorder.on_rollback(now, "source reached EOS")
        machine.transition(
            record.migration_id,
            MigrationState.COMPLETED_ON_TP1,
            now,
            "source reached EOS before target ready",
        )
        return
    if record.state is MigrationState.HANDOFF:
        adapter.rollback("source reached EOS during handoff")
        recorder.on_rollback(now, "source reached EOS during handoff")
        machine.transition(
            record.migration_id,
            MigrationState.ROLLED_BACK,
            now,
            "source reached EOS during handoff",
        )


def main() -> None:
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    args = parse_args()
    if not args.migration_id and not args.dry_run:
        raise SystemExit(
            "--migration-id is required and must match the source/stager "
            "BRIDGETP_STREAM_MIGRATION_ID"
        )

    config = ControllerConfig.load(args.config)
    diagnostic_trigger = args.diagnostic_trigger_output_tokens
    diagnostic_cutover = args.diagnostic_cutover_output_tokens
    diagnostic_bridge = args.diagnostic_bridge_output_tokens
    if diagnostic_trigger is not None and not args.diagnostic_earliest_ready_cutover:
        assert diagnostic_cutover is not None
        diagnostic_gap = diagnostic_cutover - diagnostic_trigger
        if diagnostic_gap != config.handoff_output_tokens:
            raise SystemExit(
                "diagnostic cutover minus trigger must equal "
                f"handoff_output_tokens ({config.handoff_output_tokens}), got "
                f"{diagnostic_gap}"
            )
    run_dir = args.run_dir.resolve()
    config.run_dir = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _assert_fresh_run_dir(run_dir)
    source_request = _prepare_source_request(
        load_json(args.source_request),
        run_dir,
        args.source_request_id,
    )
    requested_max_output_tokens = (
        args.requested_max_output_tokens
        if args.requested_max_output_tokens is not None
        else int(source_request["max_tokens"])
    )
    if requested_max_output_tokens < int(source_request["max_tokens"]):
        raise ValueError(
            "requested output budget must cover the TP1 subrequest cap"
        )
    atomic_json_dump(source_request, run_dir / "source_request.json")
    atomic_json_dump(
        {
            "format_version": 1,
            "source_subrequest_max_tokens": int(source_request["max_tokens"]),
            "requested_max_output_tokens": requested_max_output_tokens,
        },
        run_dir / "request_budgets.json",
    )

    unified_response_path = run_dir / "unified_response.jsonl"

    def append_unified_token(token: dict[str, Any]) -> None:
        with unified_response_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(token, ensure_ascii=False) + "\n")
            handle.flush()
        from vllm.bridge_tp.experiment_timeline import emit_event

        emit_event(
            run_dir,
            "response_proxy",
            "TOKEN_EMITTED",
            request_id=str(source_request["request_id"]),
            migration_id=args.migration_id or None,
            token_index=int(token["index"]),
            origin=str(token["origin"]),
        )

    recorder = ProxyRecorder(
        str(source_request["request_id"]),
        ProxyMode(config.proxy_mode),
        emission_sink=append_unified_token,
    )
    adapter = ActionAdapter(
        config.source_url,
        run_dir,
        expected_migration_id=args.migration_id or None,
        target_url=config.target_url,
        ready_notification_mode=args.ready_notification_mode,
        ready_notification_host=args.ready_notification_host,
        ready_notification_port=args.ready_notification_port,
        ready_latch_poll_ms=args.ready_latch_poll_ms,
    )
    probe = RuntimeControl(
        armed=False,
        migration_id=args.migration_id or None,
        source_request_id_prefix=str(source_request["request_id"]),
        note="phase 9 preflight",
    ).write(run_dir)

    target_future: Future[dict[str, Any]] | None = None
    audit: AuditLog | None = None
    record: MigrationRecord | None = None
    source_result: dict[str, Any] | None = None
    target_result: dict[str, Any] | None = None
    tick = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        source_future = executor.submit(
            recorded_completion,
            config.source_url,
            source_request,
            args.request_timeout_s,
            recorder.on_source_token,
            run_dir / "source_response.json",
            recorder,
        )
        first_progress = wait_for_runtime_control(
            run_dir,
            probe.generation,
            source_request,
            source_future,
            args.preflight_timeout_s,
            args.require_runtime_control and not args.dry_run,
        )

        table = SurvivalTable.load(config.survival_table_path)
        policy = FastPolicy(
            config.policy,
            table,
            config.tpot_tp1,
            config.tpot_tp4,
            config.interference,
        )
        rate = RateController(config.rate)
        manager_m0 = MigrationManagerM0() if args.manager_m0_shadow else None
        manager_m1 = (
            M1StartController(
                M1StartConfig(
                    min_output_tokens=config.policy.min_output_tokens_before_eligible,
                    min_survivors=config.policy.min_survivors_for_confidence,
                    max_target_kv_usage_frac=(
                        config.policy.max_target_kv_usage_frac
                    ),
                    max_target_waiting=config.policy.max_target_waiting,
                    source_release_tail_s=args.m1_source_release_tail_s,
                )
            )
            if args.manager_m1_auto_start
            else None
        )
        manager_m2 = (
            M2RateController(
                M2RateConfig(
                    low_bytes_s=config.rate.b_min_bytes_s,
                    medium_bytes_s=config.rate.b_start_bytes_s,
                    high_bytes_s=config.rate.b_max_bytes_s,
                ),
                force_initial_high=args.manager_m2_force_initial_high,
            )
            if args.manager_m2_rate
            else None
        )
        manager_m3 = (
            M3CommitController()
            if args.manager_m3_commit
            else None
        )
        manager_m4 = M4CancelController() if args.manager_m4_cancel else None
        manager_m5 = (
            PredictorEventReader(
                args.predictor_event_path, args.predictor_checkpoint_sha256
            )
            if args.manager_m5_predictor_shadow else None
        )
        experiment_gate = (
            M1PredictorRefreshGate(args.experiment_m1_action)
            if args.experiment_m1_action else None
        )
        probability_gate = (
            ProbabilityGate(
                args.probability_threshold, args.probability_assignment_seed,
                args.probability_start_probability, args.probability_assigned_action,
                tuple(args.probability_threshold_family),
            ) if args.probability_threshold is not None else None
        )
        candidate_rate_tracker = DecodeRateTracker()
        m0_collector = (
            RuntimeStateCollector()
            if manager_m0 is not None or manager_m1 is not None
            else None
        )
        m0_registry = (
            ChannelRegistry()
            if manager_m0 is not None or manager_m1 is not None
            else None
        )
        m0_history_total_bytes: int | None = None
        risk = RiskTracker(alpha=config.slow.ewma_alpha)
        capacity = CapacityHeadroomTracker(config.capacity_pilot)
        audit = AuditLog(
            run_dir / "phase9_audit.jsonl",
            run_metadata={
                "phase": "BridgeTP Phase 9",
                "config": config.to_json(),
                "survival_table_source": table.source,
                "survival_table_max_observed": table.max_observed_length,
                "migration_id": args.migration_id or "dry-run",
                "source_request_id": first_progress.request_id,
                "dry_run": args.dry_run,
                "platform_note": config.platform_note,
                "runtime_control_generation": probe.generation,
                "diagnostic_fixed_boundary": (
                    {
                        "trigger_output_tokens": diagnostic_trigger,
                        "cutover_output_tokens": diagnostic_cutover,
                    }
                    if diagnostic_trigger is not None
                    else None
                ),
                "handoff_mode": args.handoff_mode,
                "manager_m1_auto_start": args.manager_m1_auto_start,
                "paired_stay": args.paired_stay,
                "risk_observation_shadow": args.risk_observation_shadow,
                "experiment_m1_action": args.experiment_m1_action,
                "probability_threshold": args.probability_threshold,
                "probability_min_source_running": args.probability_min_source_running,
                "probability_min_urgency": args.probability_min_urgency,
                "probability_threshold_family": args.probability_threshold_family,
                "probability_assigned_action": args.probability_assigned_action,
                "probability_assignment_seed": args.probability_assignment_seed,
                "probability_start_probability": args.probability_start_probability,
                "manager_m3_commit": args.manager_m3_commit,
                'rolling_cutover': args.rolling_cutover,
                'rolling_reserve_tokens': args.rolling_reserve_tokens,
                "manager_m4_cancel": args.manager_m4_cancel,
                "manager_m5_predictor_shadow": args.manager_m5_predictor_shadow,
                "predictor_checkpoint_sha256": args.predictor_checkpoint_sha256,
                "m3_policy": (
                    "COMMIT_EARLIEST_WHEN_READY"
                    if args.manager_m3_commit else None
                ),
                "stop_and_copy": args.stop_and_copy,
                "ready_notification_mode": args.ready_notification_mode,
                "ready_latch_poll_ms": args.ready_latch_poll_ms,
            },
        )
        m0_executor = M0ExecutorAdapter(audit.write) if manager_m0 else None
        machine = MigrationStateMachine(
            audit_sink=audit.write,
            allow_shadow_takeover=args.handoff_mode == "shadow-only",
        )
        record = machine.create(
            args.migration_id or "dry-run",
            first_progress.request_id,
        )
        tp1 = MetricsScraper(
            config.source_url,
            config.block_size,
            config.tp1_total_kv_blocks,
        )
        tp4 = MetricsScraper(
            config.target_url,
            config.block_size,
            config.tp4_total_kv_blocks,
        )
        deadline = time.monotonic() + args.max_seconds
        try:
            while not _STOP and time.monotonic() < deadline and not record.is_terminal:
                tick += 1
                now = time.time()
                ready_notification_waited = False
                if source_future.done():
                    source_result = source_future.result()
                    if record.m4_source_cleanup_done:
                        _complete_m4_cancel(
                            machine, adapter, audit, record, now, recorder
                        )
                        if not record.is_terminal:
                            time.sleep(config.tick_s)
                            continue
                        break
                    _finish_source_without_commit(
                        machine,
                        adapter,
                        audit,
                        record,
                        recorder,
                        now,
                    )
                    break
                if target_future is not None and target_future.done():
                    target_result = target_future.result()
                    if record.state is not MigrationState.TAKEOVER:
                        raise RuntimeError("target ended before committed takeover")

                try:
                    pool1, pool4 = tp1.scrape(), tp4.scrape()
                except TelemetryError as error:
                    audit.write({"kind": "telemetry_error", "detail": str(error)})
                    time.sleep(config.tick_s)
                    continue
                risk_value = risk.update(pool1)
                capacity_signal = capacity.update(
                    pool1.free_kv_tokens,
                    pool1.sampled_unix_s or now,
                    prefill_pending_kv_tokens=pool1.prefill_pending_kv_tokens,
                    prefill_scheduled_tokens_total=(
                        pool1.prefill_scheduled_tokens_total
                    ),
                    decode_scheduled_tokens_total=(
                        pool1.decode_scheduled_tokens_total
                    ),
                )
                request = read_source_progress(run_dir, source_request, now)
                if request is None:
                    time.sleep(config.tick_s)
                    continue
                telemetry_row = {
                    "kind": "telemetry",
                    "tick": tick,
                    "state": record.state.value,
                    "output_tokens": request.output_tokens,
                    "risk_tp1": risk_value,
                    "tp1": pool1.__dict__,
                    "tp4": pool4.__dict__,
                    "rate_bytes_s": rate.rate_bytes_s,
                    "capacity_signal": capacity_signal.to_json(),
                    "unix_s": time.time(),
                }
                audit.write(telemetry_row)
                m5_row: dict[str, Any] | None = None
                if manager_m5 is not None:
                    headroom_tokens = (
                        pool1.free_kv_tokens
                        - config.capacity_pilot.guard_free_kv_tokens
                        - pool1.prefill_pending_kv_tokens
                        if pool1.prefill_pending_kv_tokens is not None else None
                    )
                    try:
                        if headroom_tokens is None:
                            raise ValueError("pending prefill reservation unavailable")
                        m5_row = manager_m5.advisory(
                            request.request_id, request.output_tokens, headroom_tokens,
                            max_output_tokens=int(source_request["max_tokens"]),
                            ignore_eos=bool(source_request["ignore_eos"]),
                        )
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        m5_row = {
                            "kind": "manager_m5_predictor_shadow",
                            "status": "UNAVAILABLE",
                            "reason": f"invalid event stream: {error}",
                            "request_id": request.request_id,
                            "output_tokens": request.output_tokens,
                            "headroom_tokens": headroom_tokens,
                        }
                    m5_row["survival_table_in_support"] = table.in_support(
                        request.output_tokens
                    )
                    m5_row["survival_table_applicable_to_runtime_stop_rule"] = (
                        not bool(source_request["ignore_eos"])
                    )
                    if (m5_row["survival_table_in_support"]
                            and headroom_tokens is not None):
                        m5_row["survival_table_p_remaining_gt_headroom"] = (
                            table.p_remaining_gt(
                                request.output_tokens, headroom_tokens
                            )
                        )
                        m5_row["survival_table_p_remaining_gt_short_window"] = (
                            table.p_remaining_gt(request.output_tokens, 64)
                        )
                    m5_row["prefill_pending_kv_tokens"] = (
                        pool1.prefill_pending_kv_tokens
                    )
                    audit.write(m5_row)
                m1_start_decision: M1StartDecision | None = None
                m2_snapshot: RuntimeSnapshot | None = None
                if manager_m1 is not None and record.state is MigrationState.LOCAL:
                    assert m0_collector is not None
                    assert m0_registry is not None
                    m1_snapshot = m0_collector.collect(
                        telemetry_row,
                        migration_id=record.migration_id,
                        request_id=request.request_id,
                        channel_available=m0_registry.available(
                            "tp4", record.migration_id
                        ),
                        current_context_tokens=(
                            request.computed_tokens + request.pending_tokens
                        ),
                        request_age_s=max(0.0, now - request.arrival_unix_s),
                    )
                    initial_snapshot = None
                    initial_rate_preview = None
                    if manager_m2 is not None:
                        initial_snapshot = replace(
                            m1_snapshot,
                            history_total_bytes=(
                                m1_snapshot.current_context_tokens
                                * config.policy.kv_bytes_per_token
                                if m1_snapshot.current_context_tokens is not None
                                else None
                            ),
                            history_resident_bytes=0,
                        )
                        initial_rate_preview = manager_m2.preview_initial(
                            initial_snapshot
                        )
                    m1_start_decision = manager_m1.decide(
                        m1_snapshot,
                        table,
                        max_output_tokens=int(source_request["max_tokens"]),
                        rate_bytes_s=(
                            initial_rate_preview.rate_bytes_s
                            if initial_rate_preview is not None
                            else rate.rate_bytes_s
                        ),
                        kv_bytes_per_token=config.policy.kv_bytes_per_token,
                    )
                    if (
                        m1_start_decision.action == "START_SHADOW"
                        and args.diagnostic_m1_max_source_free_kv_tokens is not None
                        and m1_snapshot.source_free_kv_tokens is not None
                        and m1_snapshot.source_free_kv_tokens
                        > args.diagnostic_m1_max_source_free_kv_tokens
                    ):
                        m1_start_decision = replace(
                            m1_start_decision, action="STAY",
                            reason="diagnostic source pressure gate not reached",
                        )
                    natural_decision = m1_start_decision
                    risk_snapshot = build_snapshot(
                        snapshot=m1_snapshot.to_json(), prediction=m5_row,
                        candidate_rate=candidate_rate_tracker.update(
                            m1_snapshot.unix_s, request.output_tokens),
                        initial_rate=(initial_rate_preview.to_json()
                                      if initial_rate_preview else {
                                          "rate_bytes_s": rate.rate_bytes_s}),
                        kv_bytes_per_token=config.policy.kv_bytes_per_token,
                        release_tail_s=args.m1_source_release_tail_s,
                        block_size=config.block_size,
                        model_config_sha256=args.risk_model_config_sha256,
                    ).to_json()
                    risk_snapshot["tick"] = tick
                    audit.write(risk_snapshot)
                    if probability_gate is not None:
                        eligibility_errors = source_load_eligibility(
                            m1_snapshot.to_json(), args.probability_min_source_running,
                        ) + urgency_eligibility(
                            risk_snapshot, args.probability_min_urgency,
                        )
                        gate_row = probability_gate.observe(
                            risk_snapshot, tick, eligibility_errors,
                        )
                        # Replace only admission; M2/M3/M4 and rank readiness
                        # still execute unchanged. Old length/load soft rules
                        # are observations, not experimental feasibility.
                        m1_start_decision = replace(
                            natural_decision,
                            action=gate_row["requested_action"],
                            reason="probability collector: " + gate_row["reason"],
                        )
                        audit.write(gate_row)
                    if experiment_gate is not None:
                        allowed, gate_reason = experiment_gate.decide(
                            m1_action=natural_decision.action,
                            m5_row=m5_row,
                            source_time_to_guard_s=(
                                natural_decision.source_time_to_guard_s
                            ),
                            estimated_preparation_s=(
                                natural_decision.estimated_preparation_s
                            ),
                            source_release_tail_s=args.m1_source_release_tail_s,
                        )
                        audit.write({
                            "kind": "experiment_m1_timing",
                            "tick": tick,
                            "output_tokens": request.output_tokens,
                            "assigned_action": experiment_gate.action,
                            "natural_decision": natural_decision.to_json(),
                            "m5_prediction_output_tokens": (
                                m5_row.get("prediction_output_tokens")
                                if m5_row else None
                            ),
                            "gate_reason": gate_reason,
                            "allowed": allowed,
                        })
                        if natural_decision.action == "START_SHADOW" and not allowed:
                            m1_start_decision = replace(
                                natural_decision, action="STAY",
                                reason=f"experimental timing: {gate_reason}",
                            )
                    if args.risk_observation_shadow:
                        effective_decision = (
                            replace(m1_start_decision, action="STAY",
                                    reason="paired stay intervention")
                            if args.paired_stay
                            and m1_start_decision.action == "START_SHADOW"
                            else m1_start_decision
                        )
                        audit.write(build_risk_observation(
                            tick=tick,
                            snapshot=m1_snapshot.to_json(),
                            m5_row=m5_row,
                            natural_m1=natural_decision.to_json(),
                            applied_m1=effective_decision.to_json(),
                            initial_rate=(
                                initial_rate_preview.to_json()
                                if initial_rate_preview is not None else None
                            ),
                            assigned_action=(
                                probability_gate.assigned_action
                                if probability_gate is not None
                                else experiment_gate.action
                                if experiment_gate is not None
                                else "STAY" if args.paired_stay else None
                            ),
                        ))
                    audit.write(
                        {
                            "kind": "manager_m1_start_decision",
                            "tick": tick,
                            "snapshot": m1_snapshot.to_json(),
                            "initial_rate_preview": (
                                initial_rate_preview.to_json()
                                if initial_rate_preview is not None
                                else None
                            ),
                            "decision": m1_start_decision.to_json(),
                        }
                    )
                    if args.paired_stay and m1_start_decision.action == "START_SHADOW":
                        audit.write({
                            "kind": "paired_stay_intervention",
                            "tick": tick,
                            "observed_output_tokens": request.output_tokens,
                            "natural_decision": m1_start_decision.to_json(),
                            "action": "STAY",
                        })
                    if (
                        manager_m2 is not None
                        and m1_start_decision.action == "START_SHADOW"
                        and not args.paired_stay
                    ):
                        assert initial_snapshot is not None
                        initial_rate = manager_m2.decide(
                            initial_snapshot, before_start=True
                        )
                        if initial_rate != initial_rate_preview:
                            raise RuntimeError(
                                "M2 initial rate changed between M1 preview and start"
                            )
                        rate.rate_bytes_s = initial_rate.rate_bytes_s
                        rate.last_reason = initial_rate.reason
                        audit.write(
                            {
                                "kind": "manager_m2_initial_rate",
                                "tick": tick,
                                "snapshot": initial_snapshot.to_json(),
                                "decision": initial_rate.to_json(),
                            }
                        )
                shadow_active = record.state in {
                    MigrationState.SHADOW,
                    MigrationState.READY_NOT_COMMITTED,
                }
                if manager_m0 is not None:
                    assert m0_collector is not None
                    assert m0_registry is not None
                    assert m0_executor is not None
                    if record.state is MigrationState.LOCAL:
                        m0_registry.observe_idle("tp4", record.migration_id)
                    elif shadow_active:
                        m0_registry.observe_active("tp4", record.migration_id)
                    start = None
                    proposed_rate = None
                    proposed_cancel = None
                    if record.state is MigrationState.LOCAL:
                        start = (
                            policy.evaluate(
                                request,
                                MigrationState.LOCAL,
                                pool1,
                                pool4,
                                risk_value,
                                rate.rate_bytes_s,
                                now_unix_s=now,
                                active_migrations=0,
                            ).action
                            is Action.START_SHADOW
                        )
                    elif shadow_active:
                        proposed_rate = copy.deepcopy(rate).step(
                            (
                                pool4.p99_tpot_s
                                if getattr(pool4, "tpot_samples", 0) > 0
                                else None
                            ),
                            policy.migration_bytes(request),
                            seconds_to_deadline=None,
                        )
                        proposed_cancel = (
                            False
                            if capacity_signal.active
                            else policy.should_abandon(
                                request,
                                pool1,
                                pool4,
                                rate.rate_bytes_s,
                                risk_value,
                            )[0]
                        )
                    progress: dict[str, Any] = {}
                    if shadow_active:
                        try:
                            manifest_path = run_dir / "session_manifest.json"
                            if (
                                m0_history_total_bytes is None
                                and manifest_path.is_file()
                            ):
                                manifest = load_json(manifest_path)
                                ranks = manifest.get("ranks") or []
                                if len(ranks) == 4 and all(
                                    "payload_bytes" in rank for rank in ranks
                                ):
                                    m0_history_total_bytes = sum(
                                        int(rank["payload_bytes"])
                                        for rank in ranks
                                    )
                            progress["history_total_bytes"] = (
                                m0_history_total_bytes
                            )
                            history_ready, _, _ = (
                                adapter.poll_initial_history_gpu_ready()
                            )
                            progress["all_ranks_history_resident"] = history_ready
                            if history_ready:
                                progress["history_resident_bytes"] = (
                                    m0_history_total_bytes
                                )
                                watermarks_ready, watermarks, _ = (
                                    adapter.poll_delta_gpu_resident_progress()
                                )
                                if watermarks_ready:
                                    progress["delta_lag_tokens"] = max(
                                        0,
                                        request.computed_tokens
                                        - min(watermarks.values()),
                                    )
                        except (OSError, ValueError, ActionError) as error:
                            audit.write(
                                {
                                    "kind": "manager_m0_observation_error",
                                    "tick": tick,
                                    "detail": str(error),
                                }
                            )
                    snapshot = m0_collector.collect(
                        telemetry_row,
                        migration_id=record.migration_id,
                        request_id=request.request_id,
                        channel_available=m0_registry.available(
                            "tp4", record.migration_id
                        ),
                        expected_remaining_tokens=(
                            table.expected_remaining(request.output_tokens)
                            if table.in_support(request.output_tokens)
                            else None
                        ),
                        current_context_tokens=(
                            request.computed_tokens + request.pending_tokens
                        ),
                        request_age_s=max(0.0, now - request.arrival_unix_s),
                        progress=progress,
                    )
                    if manager_m2 is not None and shadow_active:
                        m2_snapshot = snapshot
                    proposal = M0Proposal(
                        start=start,
                        rate_bytes_s=proposed_rate,
                        cancel=proposed_cancel,
                        origin="existing_controller_preview",
                    )
                    m0_executor.publish(
                        tick,
                        snapshot,
                        proposal,
                        manager_m0.decide(snapshot, proposal),
                    )

                if record.state is MigrationState.LOCAL:
                    step_local(
                        policy,
                        machine,
                        adapter,
                        audit,
                        record,
                        request,
                        pool1,
                        pool4,
                        risk_value,
                        rate,
                        now,
                        args.dry_run,
                        config,
                        recorder,
                        int(source_request["max_tokens"]),
                        diagnostic_trigger,
                        diagnostic_cutover,
                        args.diagnostic_earliest_ready_cutover,
                        capacity_signal,
                        args.stop_and_copy,
                        effective_m1_start_decision(
                            m1_start_decision, args.paired_stay,
                        ),
                    )
                elif shadow_active:
                    target_future = _start_target_if_ready(
                        run_dir=run_dir,
                        source_request=source_request,
                        requested_max_output_tokens=requested_max_output_tokens,
                        adapter=adapter,
                        recorder=recorder,
                        executor=executor,
                        target_url=config.target_url,
                        request_timeout_s=args.request_timeout_s,
                        target_future=target_future,
                        gpu_resident_shadow=args.gpu_resident_shadow,
                        cutover_output_tokens=(
                            record.candidate_cutover_output_tokens
                            if args.diagnostic_earliest_ready_cutover
                            else record.cutover_output_tokens
                        ),
                        stop_and_copy=args.stop_and_copy,
                        target_request_name=(
                            args.migration_id
                            if args.source_request_id
                            else None
                        ),
                        rolling_cutover=args.rolling_cutover,
                    )
                    if (
                        args.handoff_mode == "bridge"
                        and target_future is not None
                        and (
                            diagnostic_bridge is None
                            or request.output_tokens >= diagnostic_bridge
                        )
                    ):
                        machine.transition(
                            record.migration_id,
                            MigrationState.HANDOFF,
                            now,
                            "cutover manifest published and target admitted",
                        )
                        if args.gpu_resident_shadow:
                            atomic_json_dump(
                                {
                                    "format_version": 1,
                                    "status": "ACTIVE",
                                    "migration_id": record.migration_id,
                                    "started_unix_s": now,
                                    "scope": "TP1 suffix plus TP4 GPU prefix",
                                },
                                run_dir / "remote_attention_bridge.json",
                            )
                    elif (
                        args.handoff_mode == "shadow-only"
                        and target_future is not None
                        and (
                            not args.gpu_resident_shadow
                            or (run_dir / "cutover_manifest.json").exists()
                        )
                    ):
                        if (args.rolling_cutover
                                and record.cutover_output_tokens is None):
                            cutover = load_json(run_dir / 'cutover_manifest.json')
                            selection = load_json(
                                run_dir / 'rolling_freeze_selection.json'
                            )
                            boundary = int(cutover['cutover_num_output_tokens'])
                            if selection['cutover_output_tokens'] != boundary:
                                raise RuntimeError(
                                    'rolling freeze differs from manifest'
                                )
                            record.cutover_output_tokens = boundary
                            recorder.set_cutover(boundary, now)
                            machine.transition(
                                record.migration_id, MigrationState.READY_NOT_COMMITTED,
                                now, 'rolling boundary frozen with exact live delta',
                            )
                            audit.write({'kind': 'rolling_cutover_selected',
                                         'selection': selection})
                        if record.t_cutover is None:
                            record.t_cutover = now
                            audit.write(
                                {
                                    "kind": "shadow_only_final_sync",
                                    "unix_s": now,
                                    "reason": (
                                        "source freeze published; waiting for "
                                        "four-rank TP4 exact readback"
                                    ),
                                }
                            )
                        step_shadow_only_takeover(
                            machine,
                            adapter,
                            audit,
                            record,
                            recorder,
                            now,
                            args.dry_run,
                            (
                                config.tick_s
                                if args.ready_notification_mode == "UDP"
                                else 0.0
                            ),
                        )
                        ready_notification_waited = (
                            args.ready_notification_mode == "UDP"
                            and record.state is not MigrationState.TAKEOVER
                        )
                    else:
                        if manager_m4 is not None and step_m4_cancel(
                            manager_m4,
                            table,
                            machine,
                            adapter,
                            audit,
                            record,
                            request,
                            max_output_tokens=int(source_request["max_tokens"]),
                            ignore_eos=bool(source_request["ignore_eos"]),
                            source_free_kv_tokens=pool1.free_kv_tokens,
                            source_guard_free_kv_tokens=(
                                config.capacity_pilot.guard_free_kv_tokens
                            ),
                            source_capacity_pressure=capacity_signal.active,
                            now=now,
                            dry_run=args.dry_run,
                            recorder=recorder,
                        ):
                            continue
                        step_shadow(
                            policy,
                            machine,
                            adapter,
                            audit,
                            record,
                            request,
                            pool1,
                            pool4,
                            risk_value,
                            rate,
                            now,
                            args.dry_run,
                            recorder,
                            capacity_signal,
                            args.diagnostic_earliest_ready_cutover,
                            int(source_request["max_tokens"]),
                            manager_m2,
                            m2_snapshot,
                            manager_m3,
                            args.rolling_cutover,
                            args.rolling_reserve_tokens,
                        )
                        if (
                            args.diagnostic_earliest_ready_cutover
                            and record.state in {
                                MigrationState.SHADOW,
                                MigrationState.READY_NOT_COMMITTED,
                            }
                            and target_future is None
                            and record.candidate_cutover_output_tokens is not None
                        ):
                            target_future = _start_target_if_ready(
                                run_dir=run_dir,
                                source_request=source_request,
                                requested_max_output_tokens=(
                                    requested_max_output_tokens
                                ),
                                adapter=adapter,
                                recorder=recorder,
                                executor=executor,
                                target_url=config.target_url,
                                request_timeout_s=args.request_timeout_s,
                                target_future=target_future,
                                gpu_resident_shadow=args.gpu_resident_shadow,
                                cutover_output_tokens=(
                                    record.candidate_cutover_output_tokens
                                ),
                                rolling_cutover=args.rolling_cutover,
                                stop_and_copy=args.stop_and_copy,
                                target_request_name=(
                                    args.migration_id
                                    if args.source_request_id
                                    else None
                                ),
                            )
                elif record.state is MigrationState.HANDOFF:
                    step_handoff(
                        machine,
                        adapter,
                        audit,
                        record,
                        recorder,
                        now,
                        args.dry_run,
                    )
                if not ready_notification_waited:
                    time.sleep(config.tick_s)
        finally:
            audit.write(
                {
                    "kind": "run_end",
                    "final_state": record.state.value,
                    "ticks": tick,
                    "ranks_ready": sorted(record.ranks_ready),
                    "t_shadow_start": record.t_shadow_start,
                    "t_cutover": record.t_cutover,
                    "t_committed": record.t_committed,
                    "stopped_by_signal": _STOP,
                    "trigger_path": (
                        record.trigger_path.value
                        if record.trigger_path is not None
                        else None
                    ),
                }
            )
            audit.close()
            adapter.close()
            atomic_json_dump(
                recorder.stats(),
                run_dir / "response_proxy_stats.json",
            )

        if record.state is MigrationState.TAKEOVER:
            source_result = source_result or source_future.result()
            if target_future is None:
                raise RuntimeError("takeover committed without a target request")
            target_result = target_result or target_future.result()
        elif record.state in {
            MigrationState.CANCELLED,
            MigrationState.COMPLETED_ON_TP1,
        } or source_future.done():
            source_result = source_result or source_future.result()

    if source_result is not None:
        atomic_json_dump(source_result, run_dir / "source_response.json")
    if target_result is not None:
        atomic_json_dump(target_result, run_dir / "target_response.json")
    atomic_json_dump(recorder.stats(), run_dir / "response_proxy_stats.json")
    print(f"final state: {record.state.value}; audit: {run_dir / 'phase9_audit.jsonl'}")


if __name__ == "__main__":
    main()
