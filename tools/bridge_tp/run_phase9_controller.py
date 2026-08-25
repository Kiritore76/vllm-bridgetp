#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run one online BridgeTP Phase 9 controlled migration.

The runner owns both OpenAI-compatible streaming requests, the response seam,
and the fast controller loop. The TP1/TP4 servers and the Phase 8 stager must
already be running with the same run directory and migration ID.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vllm.bridge_tp.controller.action_adapter import (  # noqa: E402
    ActionAdapter,
    ActionError,
)
from vllm.bridge_tp.controller.audit import AuditLog  # noqa: E402
from vllm.bridge_tp.controller.config import ControllerConfig  # noqa: E402
from vllm.bridge_tp.controller.events import (  # noqa: E402
    Action,
    MigrationState,
    SourceRequestView,
)
from vllm.bridge_tp.controller.online_io import (  # noqa: E402
    ProxyRecorder,
    atomic_json_dump,
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
from vllm.bridge_tp.runtime_control import RuntimeControl  # noqa: E402

_STOP = False


def _handle_signal(_signum, _frame) -> None:
    global _STOP
    _STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--source-request", type=Path, required=True)
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
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _prepare_source_request(source: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    request = freeze_strict_greedy_sampling(source)
    request.update(
        {
            "request_id": f"bridgetp-phase9-{run_dir.name}",
            "stream": True,
            "return_token_ids": True,
        }
    )
    request.setdefault("ignore_eos", True)
    if int(request.get("max_tokens", 0)) <= 0:
        raise ValueError("source request max_tokens must be positive")
    return request


def _assert_fresh_run_dir(run_dir: Path) -> None:
    stale = [
        name
        for name in (
            "phase9_audit.jsonl",
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
    adapter: ActionAdapter,
    recorder: ProxyRecorder,
    executor: ThreadPoolExecutor,
    target_url: str,
    request_timeout_s: float,
    target_future: Future[dict[str, Any]] | None,
) -> Future[dict[str, Any]] | None:
    if target_future is not None:
        return target_future
    path = run_dir / "staging_manifest.json"
    if not path.exists():
        return None
    staging = load_json(path)
    target_request, cutover = build_target_request(
        source_request,
        staging,
        run_dir.name,
    )
    if recorder.proxy.cutover_index != cutover:
        raise RuntimeError(
            "stager cutover differs from controller cutover: "
            f"{cutover} != {recorder.proxy.cutover_index}"
        )
    atomic_json_dump(target_request, run_dir / "target_request.json")
    adapter.mark_target_request_admitted(note=f"target admitted at cutover {cutover}")
    return executor.submit(
        post_streaming_completion,
        target_url,
        target_request,
        request_timeout_s,
        recorder.on_target_token,
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
    if decision.action is not Action.START_SHADOW or dry_run:
        return
    trigger = request.output_tokens + 1
    cutover = trigger + config.handoff_output_tokens
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
    recorder.set_cutover(cutover, now)
    adapter.arm_shadow(
        trigger,
        rate.rate_gib_s,
        cutover_output_tokens=cutover,
        note=decision.reason,
    )
    record.trigger_output_tokens = trigger
    record.cutover_output_tokens = cutover
    record.t_decision = now
    machine.transition(
        record.migration_id,
        MigrationState.SHADOW,
        now,
        decision.reason,
    )


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
) -> None:
    remaining = policy.migration_bytes(request)
    new_rate = rate.step(
        pool4.p99_tpot_s,
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
        }
    )
    if not dry_run:
        adapter.set_rate(rate.rate_gib_s, note=rate.last_reason)

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
            binding = adapter.refresh_binding()
            if binding is None:
                adapter.disarm(reason)
            else:
                try:
                    adapter.cancel(reason)
                except ActionError as error:
                    audit.write({"kind": "action_error", "detail": str(error)})
        recorder.on_rollback(now, reason)
        machine.transition(
            record.migration_id,
            MigrationState.CANCELLED,
            now,
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
    if record.state is MigrationState.SHADOW:
        binding = adapter.refresh_binding()
        if binding is not None:
            try:
                adapter.cancel("source reached EOS before target ready")
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
    run_dir = args.run_dir.resolve()
    config.run_dir = str(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _assert_fresh_run_dir(run_dir)
    source_request = _prepare_source_request(
        load_json(args.source_request),
        run_dir,
    )
    atomic_json_dump(source_request, run_dir / "source_request.json")

    unified_response_path = run_dir / "unified_response.jsonl"

    def append_unified_token(token: dict[str, Any]) -> None:
        with unified_response_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(token, ensure_ascii=False) + "\n")
            handle.flush()

    recorder = ProxyRecorder(
        str(source_request["request_id"]),
        ProxyMode(config.proxy_mode),
        emission_sink=append_unified_token,
    )
    adapter = ActionAdapter(
        config.source_url,
        run_dir,
        expected_migration_id=args.migration_id or None,
    )
    probe = RuntimeControl(armed=False, note="phase 9 preflight").write(run_dir)

    target_future: Future[dict[str, Any]] | None = None
    audit: AuditLog | None = None
    record: MigrationRecord | None = None
    source_result: dict[str, Any] | None = None
    target_result: dict[str, Any] | None = None
    tick = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        source_future = executor.submit(
            post_streaming_completion,
            config.source_url,
            source_request,
            args.request_timeout_s,
            recorder.on_source_token,
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
        risk = RiskTracker(alpha=config.slow.ewma_alpha)
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
            },
        )
        machine = MigrationStateMachine(audit_sink=audit.write)
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
                if source_future.done():
                    source_result = source_future.result()
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
                request = read_source_progress(run_dir, source_request, now)
                if request is None:
                    time.sleep(config.tick_s)
                    continue
                audit.write(
                    {
                        "kind": "telemetry",
                        "tick": tick,
                        "state": record.state.value,
                        "output_tokens": request.output_tokens,
                        "risk_tp1": risk_value,
                        "tp1": pool1.__dict__,
                        "tp4": pool4.__dict__,
                        "rate_bytes_s": rate.rate_bytes_s,
                    }
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
                    )
                elif record.state is MigrationState.SHADOW:
                    previous_target = target_future
                    target_future = _start_target_if_ready(
                        run_dir=run_dir,
                        source_request=source_request,
                        adapter=adapter,
                        recorder=recorder,
                        executor=executor,
                        target_url=config.target_url,
                        request_timeout_s=args.request_timeout_s,
                        target_future=target_future,
                    )
                    if previous_target is None and target_future is not None:
                        machine.transition(
                            record.migration_id,
                            MigrationState.HANDOFF,
                            now,
                            "cutover manifest published and target admitted",
                        )
                    else:
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
                }
            )
            audit.close()
            atomic_json_dump(
                recorder.stats(),
                run_dir / "response_proxy_stats.json",
            )

        if record.state is MigrationState.TAKEOVER:
            source_result = source_result or source_future.result()
            if target_future is None:
                raise RuntimeError("takeover committed without a target request")
            target_result = target_result or target_future.result()
        elif source_future.done():
            source_result = source_result or source_future.result()

    if source_result is not None:
        atomic_json_dump(source_result, run_dir / "source_response.json")
    if target_result is not None:
        atomic_json_dump(target_result, run_dir / "target_response.json")
    atomic_json_dump(recorder.stats(), run_dir / "response_proxy_stats.json")
    print(f"final state: {record.state.value}; audit: {run_dir / 'phase9_audit.jsonl'}")


if __name__ == "__main__":
    main()
