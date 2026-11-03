# SPDX-License-Identifier: Apache-2.0
"""Rule-based TP layout conversion decider (user 11-03; record-only first).

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

WINDOW_S = 60.0
# A request is long when prompt + output cap reaches this.
LONG_TOKENS = 8192
BLOCK_TOKENS = 16
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
)


def layout_name(bundles):
    """L0 / L1A / L2 from {bundle: tp}; four-card names (A, B)."""
    degrees = set(bundles.values())
    if degrees == {1}:
        return "L0"
    if 1 not in degrees:
        return "L2"
    return "L1A"


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

    def __init__(self, edges, rules=None, execute=False):
        self.edges = {k: [tuple(e) for e in v] for k, v in edges.items()}
        self.rules = dict(RULES, **(rules or {}))
        self.execute = execute
        self.since = {}
        self.proposed_at = {}
        self.converted_at = None
        self.prepared = None  # dict(key, bundle, target_mode, at)
        self.history = []  # (now, tp1 cap, tp-k cap, tp-k waiting)
        self.no_room_at = -float("inf")

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
        window = workload.get("window_s", WINDOW_S)
        instances = {
            i: v for i, v in status["instances"].items() if v["lifecycle"] == "ACTIVE"
        }
        tp1 = {i for b, ids in members.items() if bundles[b] == 1 for i in ids}

        def pressure(i):
            return (instances.get(i) or {}).get("pressure") or {}

        def long_blocks(ids):
            """Final blocks by request: a migrating one is on two instances."""
            blocks = {}
            for i in ids:
                for q in (instances.get(i) or {}).get("requests", ()):
                    tokens = q["prompt_tokens"] + q["max_output_tokens"]
                    if tokens >= LONG_TOKENS:
                        blocks[q["request_id"]] = -(-tokens // BLOCK_TOKENS)
            return sum(blocks.values())

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
            short_rate=(arrivals - long_arrivals) / window,
            long_arrivals=long_arrivals,
            long_blocks=long_blocks(instances),
            bundle_long_blocks={b: long_blocks(ids) for b, ids in members.items()},
            tp1_capacity_max=tp1_max,
            tpk_capacity_max=tpk_max,
            tpk_waiting=waiting,
            quiet={
                b: all(pressure(i).get("zone") == "SPARE" for i in ids)
                for b, ids in members.items()
            },
            kv_room={
                b: all(
                    pressure(i).get("capacity") is not None
                    and pressure(i)["capacity"] < r["prepare_capacity_max"]
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
        short, blocks = s["short_rate"], s["long_blocks"]
        failed, waiting, urgent = [], [], False
        bundle_busy = bool(s["bundle_long_blocks"].get(bundle))
        if (s["layout"], after) == ("L1A", "L2"):
            if short > r["l2_max_short"]:
                failed.append("short rate too high for L2")
            if blocks < r["l2_min_long_blocks"]:
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
            if bundle_busy:
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

    def observe(self, status, bundles, members, now):
        """One decision record; ``proposal`` is the edge to convert, if any."""
        r = self.rules
        s = self.signals(status, bundles, members, now)
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
            need = (
                r["urgent_hold_s"]
                if urgent
                else r["l0_hold_s"]
                if edge[1] == "TP1" and s["layout"] == "L1A"
                else r["hold_s"]
            )
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
            if usable and held >= need - r["tilt_lead_s"] and approaching is None:
                approaching = (key, edge, bool(waiting))
            if ready and proposal is None:
                proposal = dict(bundle=edge[0], target_mode=edge[1], held_s=held)
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
        )
        if proposal is not None and not self.execute:
            self.prepared = None  # record only: this hold is over
        return record

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
                s["kv_room"].get(b) and (waiting or s["quiet"].get(b)) for b in others
            ):
                self.prepared = dict(
                    key=key, bundle=edge[0], target_mode=edge[1], at=now
                )

    def converted(self, now):
        self.converted_at = now
        self.since.clear()
        self.prepared = None
