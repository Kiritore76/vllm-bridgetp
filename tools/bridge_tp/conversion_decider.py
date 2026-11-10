# SPDX-License-Identifier: Apache-2.0
"""Rule-based TP layout conversion decider.

Six-card default (11-10): sustained observed deficits propose one fixed edge.
While the start check waits, a vanished deficit can withdraw the proposal;
after admission closes, demand changes cannot undo the conversion. No
pre-proposal tilt. The old service-budget estimate is recorded only. The
rules below describe the four-card and explicit legacy compatibility paths.

Does the current mix of requests suit the current layout? Two inputs, the
short-request rate and the final blocks of the long requests in flight, and
one neighbouring layout at a time (L0 and L2 are two steps apart):

- L1A -> L2: few short requests, many long ones, and long ones still arriving.
  A burst that stopped arriving is left to migration.
- L1A -> L0: strict, because L0 has no TP2 left for long requests. Short
  requests must exceed what L1A carries and few long ones may be in flight.
- L2 -> L1A: short requests exceed what L2 carries.
- L0 -> L1A: as soon as the TP1s cannot hold the long requests, or, more
  calmly, once L1A carries the short requests again.

Thresholds are initial values from one run per cell (static layouts, 11-03)
with a gap between opposite directions. Long-request blocks use the output cap,
not the unknown final length, so a shorter output makes them an overestimate.
With ``current_length`` the demand is what requests occupy now instead (user
11-04): prompt + generated of every request past short-request size, plus long
prompts held in the controller. Short traffic is left out because its occupancy
follows its rate, which has its own threshold. A bundle still counts as serving a
long request only from ``LONG_TOKENS``. The block thresholds came from final
lengths, so this signal reaches them later; they have not been retuned.

``trend_lookahead_s`` starts L1A -> L2 on the trend (user 11-04): the demand may
reach its threshold by extrapolating the last window's observed growth, while a
long request is still new within ``trend_recent_s``, for ``trend_hold_s``. A burst
stops being new before that hold ends, so it stays with migration.

``cancel_reason`` says when a proposal already in progress no longer fits.

A split never drains a long request: it has nowhere to migrate (no TP2 -> TP1),
so a split waits, without limit, until its bundle has none (user 11-03). L0 ->
L1A under long pressure is the exception: nothing else relieves L0, so it
drains. A candidate must hold for its hold time. Hard safety stays with the
conversion's start gate and valve. Inputs are the controller's
``layout_status.json`` and the registry's current layout. KV and queue readings
flip every second (decider GPU 11-01), so they use their recent maximum and
majority.

Prepare stage, ``tilt_lead_s`` before a proposal and only while the other
bundles are quiet (the start gate's service check, moved ahead of the load
the tilt itself adds): the driver may pre-start the replacement engines and
keep new long requests off the bundle. A candidate that stops holding ends it
at once. A bundle waiting for its long requests would otherwise never clear
while they keep arriving (replay 11-03), so that wait needs only KV room on the
other bundles, not their quiet, and keeps the long requests away for up to
``split_tilt_max_s``, longer than a deep long request runs. A long tilt moved
the pressure onto one TP-k (GPU 11-02), so it pauses while another bundle lacks
that room. Nothing is drained or stopped by a tilt. ``tilt_max_s`` limits a tilt
once a plan's start gate is waiting.
"""

from __future__ import annotations

import math

from tools.bridge_tp import six_conversion_budget as six_budget
from tools.bridge_tp.pressure_cpu import load

WINDOW_S = 60.0
# The short rate alone reads a shorter window. At 60s a short rate back at 8/s
# still read 5.6/s, and a 120s long blip converted to L2 (rotation GPU 11-05).
SHORT_WINDOW_S = 30.0
# A request is long when prompt + output cap reaches this.
LONG_TOKENS = 8192
BLOCK_TOKENS = 16
# Past this a request occupies more than a short one ever does (512 + 128 tokens
# is 40 blocks; the routing's short exemption is the same 64).
SHORT_BLOCKS = 64
# A deep long request (16384 + 8192 tokens) ends at 1536 blocks.
DEEP_BLOCKS = 1536
# A at TP1 and B at TP2 in L1A: B stays the migration target towards L2.
EDGES = {
    "L0": [["B", "TP2"]],
    "L1A": [["A", "TP2"], ["B", "TP1"]],
    "L2": [["A", "TP1"]],
}
RULES = dict(
    # L1A -> L2: 0.6 of the 16362 blocks of L1A.
    l2_max_short=6.0,
    l2_min_long_blocks=9817,
    # L2 -> L1A.
    l2_leave_short=7.0,
    # L1A -> L0: L1A queued at 12 req/s and did not at 10.
    l0_min_short=11.0,
    l0_max_long_blocks=3 * DEEP_BLOCKS,
    # L0 -> L1A: each of the four TP1s holds one deep long request.
    l0_leave_long_blocks=4 * DEEP_BLOCKS,
    l0_leave_short=9.0,
    hold_s=75.0,
    l0_hold_s=120.0,
    urgent_hold_s=30.0,
    urgent_capacity=1.0,
    cooldown_s=120.0,
    tilt_lead_s=60.0,
    tilt_max_s=90.0,
    split_tilt_max_s=360.0,
    prepare_capacity_max=0.8,
    smooth_capacity_s=15.0,
    smooth_waiting_s=10.0,
    current_length=False,
    trend_lookahead_s=0.0,
    trend_hold_s=45.0,
    trend_recent_s=30.0,
    # L2 -> L1A need not wait for the group's long requests: they migrate to
    # the other TP2 first (same-TP migration, user 11-05).
    evacuate_busy=False,
    cancel_hold_s=15.0,
)

# Six-card policy is explicit: existing four-card callers keep RULES unchanged.
# ``decision_budget``: 0 legacy thresholds, 1 the 300s service-budget estimate
# (kept for replays), 2 gap rules (default, user 11-10). Gap rules propose a
# middle move on an observed, sustained deficit instead of a predicted payback:
# the estimate needed how long demand lasts and how long the drain takes, and
# never passed in the 40-minute GPU run's long phase while L1AC failed 43 of
# its 61 long requests.
SIX_CARD_RULES = dict(
    RULES,
    decision_budget=2,
    # L1AC -> L2C: long demand close to all of L1AC's 21012 blocks (0.95), or a
    # long request held by the controller. 12600 also fired on a run where
    # static L1AC passed 35 of 36 long requests (replay 11-10).
    gap_merge_long_blocks=20000,
    gap_merge_hold_s=30.0,
    gap_merge_held_long_s=45.0,
    # L2C -> L1AC: 16/s is the highest rate L2C passed. At 17 a 17.5/s load
    # kept dipping below the line and the split came about 230s late.
    gap_split_short=16.0,
    gap_split_hold_s=30.0,
    # A proposal still waiting to start is withdrawn once its deficit has been
    # gone this long; after the group closed it is never undone.
    gap_withdraw_s=30.0,
    prepare_long_ratio=0.8,
    l2_max_short=15.0,
    l2_min_long_blocks=12600,
    l2_leave_short=17.0,
    l0_min_short=19.0,
    l0_max_long_blocks=1536,
    l0_leave_long_blocks=3072,
    l0_leave_short=17.0,
    l3_max_short=6.0,
    l3_min_long_blocks=16800,
    l3_leave_short=8.0,
    l3_leave_long_blocks=14500,
    l3_hold_s=120.0,
    extreme_exit_hold_s=30.0,
    current_length=True,
    trend_lookahead_s=60.0,
    evacuate_busy=False,
)


def layout_name(bundles):
    """Use the registry's naming for both two- and three-bundle layouts."""

    ids = sorted(bundles)
    return load("instance_pool").layout_label([bundles[b] > 1 for b in ids], ids)


def _layout_path(bundle_ids, merge_order):
    ids = tuple(bundle_ids)
    order = tuple(merge_order)
    if (
        len(ids) not in (2, 3)
        or len(set(ids)) != len(ids)
        or len(order) != len(ids)
        or set(order) != set(ids)
    ):
        raise ValueError("conversion needs two or three distinct ordered bundles")
    modes = dict.fromkeys(ids, 1)
    path = [dict(modes)]
    for bundle in order:
        modes[bundle] = 2
        path.append(dict(modes))
    return path


def adjacent_edges(bundle_ids=("A", "B"), merge_order=None):
    """Generate only the neighbouring edges along one selected layout path."""
    ids = tuple(bundle_ids)
    order = tuple(merge_order) if merge_order is not None else ids[1::-1] + ids[2:]
    path = _layout_path(ids, order)
    edges = {layout_name(modes): [] for modes in path}
    for bundle, before, after in zip(order, path, path[1:]):
        edges[layout_name(before)].append([bundle, "TP2"])
        edges[layout_name(after)].append([bundle, "TP1"])
    # Preserve the historical preference for merging over splitting at L1A.
    for values in edges.values():
        values.sort(key=lambda e: e[1] != "TP2")
    return {k: v for k, v in edges.items() if v}


def profile_rules(profile):
    """Return a separate policy copy; unknown profile names fail closed."""
    if profile == "four_card":
        return dict(RULES)
    if profile == "six_card":
        return dict(SIX_CARD_RULES)
    raise ValueError(f"unknown conversion profile: {profile}")


def profile_edges(profile):
    profile_rules(profile)
    return adjacent_edges(("A", "B", "C") if profile == "six_card" else ("A", "B"))


def registry_layout(registry):
    """{bundle: tp} of the ACTIVE instances, plus their ids per bundle."""
    bundles, members = {}, {}
    for row in registry.instances.values():
        if row.lifecycle != "ACTIVE":
            continue
        bundles[row.bundle_id] = row.tp_degree
        members.setdefault(row.bundle_id, []).append(row.ref.instance_id)
    return bundles, members


class ConversionDecider:
    """Evaluate the edges this run supports; propose one held candidate.

    ``edges``: {layout name: [(bundle, target mode), ...]}. ``execute``
    False (record only): a proposal is logged once per hold and the same
    edge is not proposed again within the cooldown; the layout never changes.
    """

    def __init__(self, edges=None, rules=None, execute=False, *, profile="four_card"):
        defaults = profile_rules(profile)
        if profile == "six_card":
            if not set(rules or {}) <= defaults.keys():
                raise ValueError("unknown six-card conversion rule")
            allowed = profile_edges(profile)
            if edges is not None and (
                not set(edges) <= allowed.keys()
                or any(
                    tuple(edge) not in map(tuple, allowed.get(layout, ()))
                    for layout, values in edges.items()
                    for edge in values
                )
            ):
                raise ValueError("six-card conversion edge is outside its path")
        if edges is None:
            edges = profile_edges(profile)
        self.edges = {k: [tuple(e) for e in v] for k, v in edges.items()}
        self.rules = dict(defaults, **(rules or {}))
        self.profile = profile
        if profile == "six_card" and (
            self.rules["current_length"] is not True
            or self.rules["evacuate_busy"] is not False
            or self.rules["decision_budget"] not in (0, 1, 2)
        ):
            raise ValueError(
                "six-card conversion uses current length and natural drain"
            )
        if profile == "six_card" and any(
            type(value) not in (int, float) or not math.isfinite(value) or value < 0
            for name, value in self.rules.items()
            if name not in {"current_length", "evacuate_busy"}
        ):
            raise ValueError("six-card numeric rules must be finite and nonnegative")
        self.execute = execute
        self.since = {}
        self.proposed_at = {}
        self.converted_at = None
        self.prepared = None  # dict(key, bundle, target_mode, at)
        self.history = []  # (now, tp1 cap, tp-k cap, tp-k waiting)
        self.no_room_at = -float("inf")
        self.long_seen = {}  # request id -> when first seen long
        self.long_last_seen = {}  # retain first-seen time while present/recent
        self.block_history = []  # (now, demand blocks)
        self.budget_evidence = six_budget.Evidence()
        self.committed_edge = None

    def smooth(self, now, tp1_cap, tpk_cap, waiting):
        r = self.rules
        self.history.append((now, tp1_cap, tpk_cap, waiting))
        horizon = max(r["smooth_capacity_s"], r["smooth_waiting_s"])
        self.history = [h for h in self.history if now - h[0] <= horizon]

        def peak(index):
            values = [
                h[index]
                for h in self.history
                if now - h[0] <= r["smooth_capacity_s"] and h[index] is not None
            ]
            return max(values, default=None)

        recent = [h[3] for h in self.history if now - h[0] <= r["smooth_waiting_s"]]
        return peak(1), peak(2), sum(recent) * 2 > len(recent)

    def signals(self, status, bundles, members, now):
        r = self.rules
        workload = status.get("workload") or {}
        arrivals = workload.get("arrivals", 0)
        long_arrivals = workload.get("long_arrivals", 0)
        short_rate = (arrivals - long_arrivals) / workload.get("window_s", WINDOW_S)
        if "short_window_s" in workload:
            short_rate = (
                max(
                    0,
                    workload["short_window_arrivals"] - workload["short_window_long"],
                )
                / workload["short_window_s"]
            )
        window = workload.get("window_s", WINDOW_S)
        instances = {
            i: v for i, v in status["instances"].items() if v["lifecycle"] == "ACTIVE"
        }
        tp1 = {i for b, ids in members.items() if bundles[b] == 1 for i in ids}

        def pressure(i):
            return (instances.get(i) or {}).get("pressure") or {}

        def long_blocks(ids):
            """Blocks by request: a migrating one is on two instances."""
            blocks = {}
            for i in ids:
                for q in (instances.get(i) or {}).get("requests", ()):
                    tokens = q["prompt_tokens"] + (
                        q["generated"]
                        if r["current_length"]
                        else q["max_output_tokens"]
                    )
                    if tokens >= LONG_TOKENS:
                        blocks[q["request_id"]] = -(-tokens // BLOCK_TOKENS)
            return sum(blocks.values())

        total_blocks = long_blocks(instances)
        if r["current_length"]:
            occupied = {}
            for i in instances:
                for q in instances[i].get("requests", ()):
                    blocks = -(-(q["prompt_tokens"] + q["generated"]) // BLOCK_TOKENS)
                    if blocks > SHORT_BLOCKS:
                        occupied[q["request_id"]] = blocks
            total_blocks = sum(occupied.values())
            # Long arrivals are not known from output caps; a request counts as
            # new when it first turns long, and a held long prompt is new too.
            held = workload.get("held_long_prompt_tokens", ())
            for i in instances:
                for q in instances[i].get("requests", ()):
                    if q["prompt_tokens"] + q["generated"] >= LONG_TOKENS:
                        self.long_seen.setdefault(q["request_id"], now)
                    if q["request_id"] in self.long_seen:
                        self.long_last_seen[q["request_id"]] = now
            # A request can outlive the arrival window. Keep its birth time
            # until it has been absent for a full window, including temporary
            # missing inventory or a generated-count reset during recompute.
            self.long_last_seen = {
                k: v for k, v in self.long_last_seen.items() if now - v <= window
            }
            self.long_seen = {
                k: v for k, v in self.long_seen.items() if k in self.long_last_seen
            }
            long_arrivals = sum(
                now - seen <= window for seen in self.long_seen.values()
            ) + len(held)
            total_blocks += sum(-(-n // BLOCK_TOKENS) for n in held)
        trend_blocks = total_blocks
        if r["trend_lookahead_s"]:
            self.block_history.append((now, total_blocks))
            self.block_history = [h for h in self.block_history if now - h[0] <= window]
            then, before = self.block_history[0]
            recent = workload.get("held_long_prompt_tokens") or any(
                now - seen <= r["trend_recent_s"] for seen in self.long_seen.values()
            )
            if recent and now - then >= r["trend_recent_s"]:
                rate = max(0.0, (total_blocks - before) / (now - then))
                trend_blocks = total_blocks + rate * r["trend_lookahead_s"]

        caps = {i: pressure(i).get("capacity") for i in instances}
        tp1_caps = [c for i, c in caps.items() if i in tp1 and c is not None]
        tpk_caps = [c for i, c in caps.items() if i not in tp1 and c is not None]
        waiting_now = any(
            ((instances.get(i) or {}).get("inventory") or {}).get("waiting")
            for b, ids in members.items()
            if bundles[b] > 1
            for i in ids
        )
        tp1_max, tpk_max, waiting = self.smooth(
            now, max(tp1_caps, default=None), max(tpk_caps, default=None), waiting_now
        )
        return dict(
            layout=layout_name(bundles),
            short_rate=short_rate,
            long_arrivals=long_arrivals,
            long_blocks=total_blocks,
            trend_blocks=round(trend_blocks),
            bundle_long_blocks={b: long_blocks(ids) for b, ids in members.items()},
            tp1_capacity_max=tp1_max,
            tpk_capacity_max=tpk_max,
            tpk_waiting=waiting,
            quiet={
                b: all(
                    load("unified_scheduler").route_gate_ready(
                        instances.get(i, {}).get("unified_fullness"), now
                    )
                    if self.profile == "six_card"
                    else pressure(i).get("zone") == "SPARE"
                    for i in ids
                )
                for b, ids in members.items()
            },
            kv_room={
                b: all(
                    pressure(i).get("capacity") is not None
                    and pressure(i)["capacity"]
                    < (1.0 if self.profile == "six_card" else r["prepare_capacity_max"])
                    for i in ids
                )
                for b, ids in members.items()
            },
        )

    def check(self, edge, s, bundles):
        """(failed reasons, waiting reasons, urgent) of one edge.

        ``failed`` stops the hold; ``waiting`` only delays the proposal.
        """
        r = self.rules
        bundle, mode = edge
        after = layout_name(dict(bundles, **{bundle: 1 if mode == "TP1" else 2}))
        if self.profile == "six_card":
            if self._six_after(s["layout"], edge) != after:
                return ["not a neighbouring six-card layout"], [], False
            return self._check_six(edge, s, after)
        short, blocks = s["short_rate"], s["long_blocks"]
        failed, waiting, urgent = [], [], False
        bundle_busy = bool(s["bundle_long_blocks"].get(bundle))
        if (s["layout"], after) == ("L1A", "L2"):
            if short > r["l2_max_short"]:
                failed.append("short rate too high for L2")
            if s.get("trend_blocks", blocks) < r["l2_min_long_blocks"]:
                failed.append("too few long-request blocks")
            if not s["long_arrivals"]:
                failed.append("no new long request: a burst is left to migration")
            tp1_cap, tpk_cap = s["tp1_capacity_max"], s["tpk_capacity_max"]
            urgent = (tp1_cap is not None and tp1_cap >= r["urgent_capacity"]) or (
                tpk_cap is not None and tpk_cap >= r["urgent_capacity"]
            )
        elif (s["layout"], after) == ("L1A", "L0"):
            if short < r["l0_min_short"]:
                failed.append("short rate within what L1A carries")
            if blocks > r["l0_max_long_blocks"]:
                failed.append("too many long-request blocks for L0")
            if bundle_busy:
                waiting.append("bundle serves a long request")
        elif (s["layout"], after) == ("L2", "L1A"):
            if short < r["l2_leave_short"]:
                failed.append("short rate within what L2 carries")
            urgent = s["tpk_waiting"]
            if bundle_busy and not r["evacuate_busy"]:
                waiting.append("bundle serves a long request")
        elif (s["layout"], after) == ("L0", "L1A"):
            if blocks >= r["l0_leave_long_blocks"]:
                urgent = True  # Nothing else relieves L0: this one drains.
            elif short > r["l0_leave_short"]:
                failed.append("L0 still suits the load")
            elif bundle_busy:
                waiting.append("bundle serves a long request")
        else:
            failed.append("not a neighbouring layout")
        return failed, waiting, urgent

    def _six_after(self, layout, edge):
        for before, values in profile_edges("six_card").items():
            if before != layout or list(edge) not in values:
                continue
            path = _layout_path(("A", "B", "C"), ("B", "A", "C"))
            modes = next(m for m in path if layout_name(m) == layout)
            modes[edge[0]] = 1 if edge[1] == "TP1" else 2
            return layout_name(modes)
        return None

    def _check_six(self, edge, s, after):
        r = self.rules
        short, blocks, new = s["short_rate"], s["long_blocks"], s["long_arrivals"]
        transition = s["layout"], after
        failed, waiting, urgent = [], [], False
        if transition == ("L1AC", "L0"):
            if short < r["l0_min_short"]:
                failed.append("short rate within what L1AC carries")
            if blocks > r["l0_max_long_blocks"]:
                failed.append("too many long-request blocks for L0")
            if new:
                failed.append("new or held long request prevents entering L0")
        elif transition == ("L0", "L1AC"):
            if not (
                short <= r["l0_leave_short"]
                or blocks >= r["l0_leave_long_blocks"]
                or new
            ):
                failed.append("L0 still suits the load")
        elif transition == ("L1AC", "L2C"):
            if short > r["l2_max_short"]:
                failed.append("short rate too high for L2C")
            if s.get("trend_blocks", blocks) < r["l2_min_long_blocks"]:
                failed.append("too few long-request blocks")
            if not new:
                failed.append("no new long request: a burst is left to migration")
            urgent = any(
                s.get(k) is not None and s[k] >= r["urgent_capacity"]
                for k in ("tp1_capacity_max", "tpk_capacity_max")
            )
        elif transition == ("L2C", "L1AC"):
            if short < r["l2_leave_short"]:
                failed.append("short rate within what L2C carries")
            urgent = s.get("tpk_waiting", False)
        elif transition == ("L2C", "L3"):
            if short > r["l3_max_short"]:
                failed.append("short rate too high for L3")
            if blocks < r["l3_min_long_blocks"]:
                failed.append("too few current long-request blocks for L3")
            if not new:
                failed.append("no new or held long request for L3")
        elif transition == ("L3", "L2C"):
            if not (
                short >= r["l3_leave_short"] or blocks <= r["l3_leave_long_blocks"]
            ):
                failed.append("L3 still suits the load")
        else:
            failed.append("not a neighbouring six-card layout")
        if edge[1] == "TP1" and s["bundle_long_blocks"].get(edge[0]):
            waiting.append("bundle serves a long request")
        return failed, waiting, urgent

    def _hold_seconds(self, edge, s, urgent):
        r = self.rules
        if self.profile == "six_card":
            after = self._six_after(s["layout"], edge)
            if after == "L0":
                return r["l0_hold_s"]
            if after == "L3":
                return r["l3_hold_s"]
            if s["layout"] in ("L0", "L3"):
                return r["extreme_exit_hold_s"]
        return (
            r["urgent_hold_s"]
            if urgent
            else r["l0_hold_s"]
            if edge[1] == "TP1" and s["layout"] == "L1A"
            else r["trend_hold_s"]
            if r["trend_lookahead_s"] and tuple(edge) == ("A", "TP2")
            else r["hold_s"]
        )

    def observe(self, status, bundles, members, now):
        """One decision record; ``proposal`` is the edge to convert, if any."""
        r = self.rules
        s = self.signals(status, bundles, members, now)
        if self.profile == "six_card" and r["decision_budget"] == 2:
            return self.observe_gap(status, bundles, members, s, now)
        if self.profile == "six_card" and r["decision_budget"]:
            return self.observe_budget(status, bundles, members, s, now)
        if self.profile == "six_card" and (
            set(bundles) != {"A", "B", "C"} or s["layout"] not in self.edges
        ):
            self.since.clear()
            self.prepared = None
            # A partial ACTIVE registry during conversion is not a new layout.
            s["layout"] = None
        candidates, proposal, approaching = {}, None, None
        cooling = (
            self.execute
            and self.converted_at is not None
            and now - self.converted_at < r["cooldown_s"]
        )
        for edge in self.edges.get(s["layout"], ()):
            key = f"{edge[0]}->{edge[1]}"
            failed, waiting, urgent = self.check(edge, s, bundles)
            if failed:
                self.since.pop(key, None)
            else:
                self.since.setdefault(key, now)
            held = now - self.since[key] if key in self.since else 0.0
            need = self._hold_seconds(edge, s, urgent)
            repeated = (
                not self.execute
                and now - self.proposed_at.get(key, -float("inf")) < r["cooldown_s"]
            )
            usable = not failed and not cooling and not repeated
            ready = usable and held >= need and not waiting
            candidates[key] = dict(
                failed=failed,
                waiting=waiting,
                urgent=urgent,
                held_s=round(held, 3),
                ready=ready,
            )
            predicted_prepare = (
                self.profile == "six_card"
                and s["layout"] == "L1AC"
                and tuple(edge) == ("A", "TP2")
                and not cooling
                and failed == ["too few long-request blocks"]
                and s.get("trend_blocks", 0)
                >= r["prepare_long_ratio"] * r["l2_min_long_blocks"]
            )
            if (
                usable and held >= need - r["tilt_lead_s"] or predicted_prepare
            ) and approaching is None:
                approaching = (key, edge, bool(waiting))
            if ready and proposal is None:
                proposal = dict(
                    bundle=edge[0],
                    target_mode=edge[1],
                    held_s=held,
                    layout=s["layout"],
                )
                self.proposed_at[key] = now
                self.since.pop(key, None)
        self.update_prepared(approaching, s, bundles, now)
        prepared = self.prepared
        if prepared and not all(
            room for b, room in s["kv_room"].items() if b != prepared["bundle"]
        ):
            self.no_room_at = now
        tilt = (
            [prepared["bundle"]]
            if prepared
            and now - prepared["at"] <= r["split_tilt_max_s"]
            and now - self.no_room_at > r["smooth_capacity_s"]
            else []
        )
        record = dict(
            evaluated_unix_s=now,
            mode="EXECUTE" if self.execute else "RECORD_ONLY",
            signals=s,
            candidates=candidates,
            cooling=cooling,
            proposal=proposal,
            prepared=dict(prepared) if prepared else None,
            tilt=tilt,
            preparation=dict(
                phase="PREPARING" if prepared else "OBSERVING",
                approaching=approaching[0] if approaching else None,
                retained_no_room=[
                    b
                    for b in bundles
                    if approaching
                    and b != approaching[1][0]
                    and not s["kv_room"].get(b)
                ],
                retained_not_quiet=[
                    b
                    for b in bundles
                    if approaching and b != approaching[1][0] and not s["quiet"].get(b)
                ],
                tilt_active=bool(tilt),
                tilt_expired=bool(
                    prepared and now - prepared["at"] > r["split_tilt_max_s"]
                ),
                capacity_recovery_wait_s=max(
                    0.0, r["smooth_capacity_s"] - (now - self.no_room_at)
                )
                if prepared
                else None,
            ),
        )
        if proposal is not None and not self.execute:
            self.prepared = None  # record only: this hold is over
        return record

    def _check_gap(self, edge, s, after, demand):
        """(failed, waiting, immediate) of one edge under the gap rules.

        Middle moves need a deficit the current layout already shows; the two
        extreme layouts keep their strict entry and quick exit conditions.
        """
        r = self.rules
        transition = s["layout"], after
        if transition == ("L1AC", "L2C"):
            failed = []
            held_s = demand.get("held_long_wait_s", 0.0)
            if s["short_rate"] > r["l2_max_short"]:
                failed.append("short rate too high for L2C")
            if not s["long_arrivals"]:
                failed.append("no new long request: a burst is left to migration")
            queued = held_s >= r["gap_merge_held_long_s"]
            if s["long_blocks"] < r["gap_merge_long_blocks"] and not queued:
                failed.append("long demand within what L1AC holds")
            return failed, [], queued and not failed
        if transition == ("L2C", "L1AC"):
            failed, waiting = [], []
            if s["short_rate"] < r["gap_split_short"]:
                failed.append("short rate within what L2C carries")
            # Shorts are also held when long requests fill the KV; that is a
            # long deficit and must not read as a reason to split.
            if demand.get("held_long_count", 0):
                failed.append("a long request is held: not a short deficit")
            if s["bundle_long_blocks"].get(edge[0]):
                waiting.append("bundle serves a long request")
            return failed, waiting, False
        failed, waiting, _ = self._check_six(edge, s, after)
        return failed, waiting, False

    def _gap_demand(self, status, signals):
        """Fold the controller's held requests into the signals; the demand."""
        demand = (status.get("workload") or {}).get("conversion_demand")
        if status.get("workload_origin_unix_s") is not None and (
            status.get("service_shape") is None or demand is None
        ):
            # GPU 11-10: these were silently absent for a whole 40-minute run.
            raise RuntimeError(
                "six-card decision signals missing: service_shape/conversion_demand"
            )
        demand = demand or {}
        # Unified runs hold every request in one queue: its long prompts and
        # count are published with the demand, not as held_long_prompt_tokens.
        held = demand.get("held_long_prompt_tokens", ())
        signals["long_blocks"] += sum(-(-n // BLOCK_TOKENS) for n in held)
        signals["long_arrivals"] += demand.get("held_long_count", 0)
        signals["held_long_wait_s"] = demand.get("held_long_wait_s", 0.0)
        signals["held_short_wait_s"] = demand.get("held_short_wait_s", 0.0)
        signals["held_short_count"] = demand.get("held_short_count", 0)
        return demand

    def gap_pending_failed(self, status, bundles, members, now, layout, edge):
        """Why a proposed, not yet started move no longer has its deficit."""
        signals = self.signals(status, bundles, members, now)
        demand = self._gap_demand(status, signals)
        signals["layout"] = layout
        after = self._six_after(layout, edge)
        if after is None:
            return ["not a neighbouring six-card layout"]
        return self._check_gap(edge, signals, after, demand)[0]

    def observe_gap(self, status, bundles, members, signals, now):
        """Latch one observed deficit; the driver may withdraw before closure."""
        r = self.rules
        demand = self._gap_demand(status, signals)
        layout = signals["layout"]
        complete = set(bundles) == {"A", "B", "C"} and layout in self.edges
        cooling = (
            self.execute
            and self.converted_at is not None
            and now - self.converted_at < r["cooldown_s"]
        )
        proposal, candidates = None, {}
        if not complete:
            # A partial ACTIVE registry during a conversion is not a layout.
            self.since.clear()
            signals["layout"] = None
        for edge in self.edges.get(layout, ()) if complete else ():
            key = f"{edge[0]}->{edge[1]}"
            after = self._six_after(layout, edge)
            failed, waiting, immediate = self._check_gap(edge, signals, after, demand)
            if failed:
                self.since.pop(key, None)
            else:
                self.since.setdefault(key, now)
            held_s = now - self.since[key] if key in self.since else 0.0
            hold_s = (
                r["gap_merge_hold_s"]
                if (layout, after) == ("L1AC", "L2C")
                else r["gap_split_hold_s"]
                if (layout, after) == ("L2C", "L1AC")
                else self._hold_seconds(edge, signals, False)
            )
            repeated = (
                not self.execute
                and now - self.proposed_at.get(key, -float("inf")) < r["cooldown_s"]
            )
            ready = (
                not failed
                and not waiting
                and (immediate or held_s >= hold_s)
                and not cooling
                and not repeated
                and self.committed_edge is None
            )
            # Record only: the retired estimate, for comparison with this rule.
            budget = six_budget.estimate(status, bundles, members, edge, now, signals)
            candidates[key] = dict(
                failed=failed,
                waiting=waiting,
                urgent=immediate,
                held_s=held_s,
                hold_s=hold_s,
                ready=ready,
                budget=budget,
            )
            if ready and proposal is None:
                proposal = dict(
                    bundle=edge[0],
                    target_mode=edge[1],
                    layout=layout,
                    held_s=held_s,
                    rule="held long request" if immediate else "sustained deficit",
                )
        if proposal is not None:
            if self.execute:
                self.committed_edge = (proposal["bundle"], proposal["target_mode"])
            else:
                key = f"{proposal['bundle']}->{proposal['target_mode']}"
                self.proposed_at[key] = now
            self.since.clear()
        self.prepared = None
        return dict(
            evaluated_unix_s=now,
            mode="EXECUTE" if self.execute else "RECORD_ONLY",
            signals=signals,
            candidates=candidates,
            cooling=cooling,
            proposal=proposal,
            prepared=None,
            tilt=[],
            decision="SUBMIT"
            if proposal
            else "COMMITTED"
            if self.committed_edge
            else "KEEP",
            policy="SIX_GAP_RULES_V1",
            preparation=dict(
                phase="COMMITTED" if self.committed_edge else "OBSERVING",
                approaching=None,
                retained_no_room=[],
                retained_not_quiet=[],
                tilt_active=False,
                tilt_expired=False,
                capacity_recovery_wait_s=None,
            ),
        )

    def observe_budget(self, status, bundles, members, signals, now):
        """Strict, common entry policy; no business-affecting half-prepared state."""
        layout = signals["layout"]
        complete = set(bundles) == {"A", "B", "C"} and layout in self.edges
        cooling = (
            self.execute
            and self.converted_at is not None
            and now - self.converted_at < self.rules["cooldown_s"]
        )
        proposal, candidates = None, {}
        if not complete:
            self.budget_evidence.rows.clear()
            signals["layout"] = None
        for edge in self.edges.get(layout, ()) if complete else ():
            key = f"{edge[0]}->{edge[1]}"
            after = self._six_after(layout, edge)
            budget = six_budget.estimate(status, bundles, members, edge, now, signals)
            if budget["supported"]:
                demand = status["workload"]["conversion_demand"]
                signals["short_rate"] = budget["short_rate"]
                signals["long_arrivals"] = demand["long_recent_count"] + demand.get(
                    "held_long_count", 0
                )
            failed, _, urgent = self._check_six(edge, signals, after)
            extreme = after in {"L0", "L3"}
            # Preserve strict extreme-layout eligibility. Middle moves use both
            # traffic classes and payback, rather than a single rate/KV line.
            hard = not failed if extreme else True
            span = (
                self.rules["l0_hold_s"]
                if after == "L0"
                else self.rules["l3_hold_s"]
                if after == "L3"
                else self.rules["urgent_hold_s"]
            )
            evidence = self.budget_evidence.observe(
                (layout, key),
                now,
                budget["supported"],
                budget.get("demand_supports_move", False),
                hard,
                span,
            )
            repeated = (
                not self.execute
                and now - self.proposed_at.get(key, -float("inf"))
                < self.rules["cooldown_s"]
            )
            ready = (
                evidence["ready"]
                and budget["worthwhile"]
                and not cooling
                and not repeated
                and self.committed_edge is None
            )
            # Extreme entries require uninterrupted eligibility, even though
            # ordinary service-budget noise uses a rolling evidence fraction.
            if extreme:
                ready = ready and all(
                    row[2] for row in self.budget_evidence.rows[(layout, key)]
                )
            candidates[key] = dict(
                failed=failed if extreme else [],
                waiting=[],
                urgent=urgent,
                held_s=evidence["coverage_s"],
                ready=ready,
                budget=budget,
                evidence=evidence,
            )
            if ready and (
                proposal is None
                or budget["net_tokens"] > proposal["budget"]["net_tokens"]
            ):
                proposal = dict(
                    bundle=edge[0],
                    target_mode=edge[1],
                    layout=layout,
                    held_s=evidence["coverage_s"],
                    budget=budget,
                )
        if proposal is not None:
            if self.execute:
                self.committed_edge = (proposal["bundle"], proposal["target_mode"])
            else:
                self.proposed_at[f"{proposal['bundle']}->{proposal['target_mode']}"] = (
                    now
                )
        self.prepared = None
        self.since.clear()
        return dict(
            evaluated_unix_s=now,
            mode="EXECUTE" if self.execute else "RECORD_ONLY",
            signals=signals,
            candidates=candidates,
            cooling=cooling,
            proposal=proposal,
            prepared=None,
            tilt=[],
            decision="SUBMIT"
            if proposal
            else "COMMITTED"
            if self.committed_edge
            else "KEEP",
            policy="SIX_SERVICE_BUDGET_V1",
            preparation=dict(
                phase="COMMITTED" if self.committed_edge else "OBSERVING",
                approaching=None,
                retained_no_room=[],
                retained_not_quiet=[],
                tilt_active=False,
                tilt_expired=False,
                capacity_recovery_wait_s=None,
            ),
        )

    def update_prepared(self, approaching, s, bundles, now):
        """Enter once the others allow it (see the module doc); leave at once."""
        prepared = self.prepared
        if prepared is not None and (
            approaching is None or approaching[0] != prepared["key"]
        ):
            self.prepared = prepared = None
        if prepared is None and approaching is not None:
            key, edge, waiting = approaching
            others = [b for b in bundles if b != edge[0]]
            if others and all(
                s["kv_room"].get(b)
                and (self.profile == "six_card" or waiting or s["quiet"].get(b))
                for b in others
            ):
                self.prepared = dict(
                    key=key, bundle=edge[0], target_mode=edge[1], at=now
                )

    def cancel_reason(
        self, edge, layout, short, blocks, *, long_arrivals=None, trend_blocks=None
    ):
        """Why a conversion in progress no longer fits the load, or None."""
        r = self.rules
        if self.profile == "six_card":
            after = self._six_after(layout, edge)
            if after is None:
                return "not a neighbouring six-card layout"
            # Preserve the middle edge's existing cancellation semantics.
            if (layout, after) == ("L1AC", "L2C"):
                return (
                    "short rate too high for L2C" if short > r["l2_max_short"] else None
                )
            if (layout, after) == ("L2C", "L1AC"):
                return (
                    "short rate within what L2C carries"
                    if short < r["l2_leave_short"]
                    else None
                )
            if long_arrivals is None and (layout, after) != ("L3", "L2C"):
                return "new-long signal unavailable"
            failed, _, _ = self._check_six(
                edge,
                dict(
                    layout=layout,
                    short_rate=short,
                    long_blocks=blocks,
                    long_arrivals=long_arrivals,
                    trend_blocks=trend_blocks,
                    bundle_long_blocks={},
                ),
                after,
            )
            return "; ".join(failed) if failed else None
        bundle, mode = edge
        after = {
            ("L1A", "A", "TP2"): "L2",
            ("L1A", "B", "TP1"): "L0",
            ("L2", "A", "TP1"): "L1A",
            ("L0", "B", "TP2"): "L1A",
        }.get((layout, bundle, mode))
        if after == "L2" and short > r["l2_max_short"]:
            return "short rate too high for L2"
        if after == "L0" and (
            short < r["l0_min_short"] or blocks > r["l0_max_long_blocks"]
        ):
            return "L0 no longer suits the load"
        if layout == "L2" and short < r["l2_leave_short"]:
            return "short rate within what L2 carries"
        if (
            layout == "L0"
            and blocks < r["l0_leave_long_blocks"]
            and short > r["l0_leave_short"]
        ):
            return "L0 suits the load again"
        return None

    def cancelled(self, now=None):
        """A cancelled conversion started nothing; four-card needs no cooldown.

        The six-card gap policy cools down after a withdrawal or abort as well:
        without it the next second proposed the same move again (GPU 11-10).
        """
        if (
            now is not None
            and self.profile == "six_card"
            and self.rules["decision_budget"] == 2
        ):
            self.converted_at = now
        self.since.clear()
        self.prepared = None
        self.committed_edge = None
        self.budget_evidence.rows.clear()

    def converted(self, now):
        self.converted_at = now
        self.since.clear()
        self.prepared = None
        self.committed_edge = None
        self.budget_evidence.rows.clear()
