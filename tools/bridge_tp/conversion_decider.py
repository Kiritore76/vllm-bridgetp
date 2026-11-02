# SPDX-License-Identifier: Apache-2.0
"""Rule-based TP layout conversion decider (user 11-01; record-only first).

Two rules from the measured four-card behaviour, kept simple on purpose and
tuned later from real runs:

- MERGE a TP1 bundle into a TP-k when long requests take over: their share
  of recent arrivals is high, TP1s migrate often, or a TP1's KV runs near its
  guard. Only if the merged layout still carries the short-request rate.
- SPLIT a TP-k bundle into TP1s when short requests take over while nothing
  is under KV pressure and the short load approaches the current layout's
  capacity (or the TP-k queues): convert while it is still calm, before the
  peak. Never a bundle that serves a long request.

A candidate must hold for ``hold_s`` (``urgent_hold_s`` when an instance's KV
is at its guard: liveness under pressure), and a capacity-motivated split must
gain ``benefit_factor`` times the capacity a conversion removes. Hard safety
stays with the conversion's start gate and valve. Inputs are the
controller's ``layout_status.json`` and the registry's current layout. KV and
queue readings flip every second (decider GPU 11-01), so conditions use their
recent maximum and majority.

Prepare stage, ``tilt_lead_s`` before a proposal and only while the other
bundles are quiet (the start gate's service check, moved ahead of the load
the tilt itself adds): the driver may pre-start the replacement engines and
keep new long requests off the bundle, for at most ``tilt_max_s``. A candidate
that stops holding ends it at once. A long tilt only moved the pressure onto
one TP-k (GPU 11-02), and 53s were enough to clear a bundle.
"""

from __future__ import annotations

WINDOW_S = 120.0
# A request is long when prompt + output cap reaches this (one TP1 holds ~4).
LONG_TOKENS = 8192
# Four-card short-request (512/128) capacity, req/s (rate sweep, 11-01).
SHORT_CAPACITY_REQ_S = {"L0": 16.5, "L1A": 12.5, "L2": 9.5}
RULES = dict(
    min_arrivals=10,
    merge_long_share=0.30,
    merge_tp1_migrations_per_min=2.0,
    merge_tp1_capacity=0.9,
    merge_short_margin=1.25,
    split_long_share=0.10,
    split_max_capacity=0.5,
    split_short_load=0.6,
    hold_s=180.0,
    urgent_hold_s=60.0,
    urgent_capacity=1.0,
    benefit_factor=3.0,
    conversion_s=60.0,
    cooldown_s=600.0,
    tilt_lead_s=60.0,
    tilt_max_s=90.0,
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
        self.history = []  # (now, tp1 cap, any cap, tp-k cap, tp-k waiting)

    def smooth(self, now, tp1_cap, any_cap, tpk_cap, waiting):
        r = self.rules
        self.history.append((now, tp1_cap, any_cap, tpk_cap, waiting))
        horizon = max(r["smooth_capacity_s"], r["smooth_waiting_s"])
        self.history = [h for h in self.history if now - h[0] <= horizon]

        def peak(index):
            values = [
                h[index]
                for h in self.history
                if now - h[0] <= r["smooth_capacity_s"] and h[index] is not None
            ]
            return max(values, default=None)

        recent = [h[4] for h in self.history if now - h[0] <= r["smooth_waiting_s"]]
        return peak(1), peak(2), peak(3), sum(recent) * 2 > len(recent)

    def signals(self, status, bundles, members, now):
        r = self.rules
        workload = status.get("workload") or {}
        arrivals = workload.get("arrivals", 0)
        window = workload.get("window_s", WINDOW_S)
        instances = {
            i: v for i, v in status["instances"].items() if v["lifecycle"] == "ACTIVE"
        }
        tp1 = {i for b, ids in members.items() if bundles[b] == 1 for i in ids}

        def pressure(i):
            return (instances.get(i) or {}).get("pressure") or {}

        caps = {i: pressure(i).get("capacity") for i in instances}
        tp1_caps = [c for i, c in caps.items() if i in tp1 and c is not None]
        tpk_caps = [c for i, c in caps.items() if i not in tp1 and c is not None]
        starts = [s for _, s in workload.get("migration_starts", ())]
        long_demand = {
            b: sum(
                q["prompt_tokens"] + q["max_output_tokens"]
                for i in ids
                for q in (instances.get(i) or {}).get("requests", ())
                if q.get("prompt_tokens", 0) + q["max_output_tokens"] >= LONG_TOKENS
            )
            for b, ids in members.items()
        }
        waiting_now = any(
            ((instances.get(i) or {}).get("inventory") or {}).get("waiting")
            for b, ids in members.items()
            if bundles[b] > 1
            for i in ids
        )
        tp1_max, any_max, tpk_max, waiting = self.smooth(
            now,
            max(tp1_caps, default=None),
            max((c for c in caps.values() if c is not None), default=None),
            max(tpk_caps, default=None),
            waiting_now,
        )
        return dict(
            layout=layout_name(bundles),
            arrivals=arrivals,
            long_share=workload.get("long_arrivals", 0) / arrivals
            if arrivals >= r["min_arrivals"]
            else None,
            short_rate=(arrivals - workload.get("long_arrivals", 0)) / window,
            tp1_migrations_per_min=sum(s in tp1 for s in starts) / (window / 60),
            tp1_capacity_max=tp1_max,
            tpk_capacity_max=tpk_max,
            capacity_max=any_max,
            tpk_waiting=waiting,
            bundle_long_tokens=long_demand,
            tpk_long_tokens=sum(
                tokens for b, tokens in long_demand.items() if bundles[b] > 1
            ),
            quiet={
                b: all(
                    pressure(i).get("zone") == "SPARE"
                    and pressure(i).get("capacity") is not None
                    and pressure(i)["capacity"] < r["prepare_capacity_max"]
                    for i in ids
                )
                for b, ids in members.items()
            },
        )

    def check(self, edge, s, bundles):
        """(failed reasons, waiting reasons, urgent) of one edge.

        ``failed`` stops the hold; ``waiting`` only delays the proposal: a
        split waits for its bundle to finish long requests, for at most
        ``tilt_max_s`` past the hold (liveness: then it simply drains).
        """
        r = self.rules
        bundle, mode = edge
        after = dict(bundles, **{bundle: 1 if mode == "TP1" else 2})
        capacity_now = SHORT_CAPACITY_REQ_S[s["layout"]]
        capacity_after = SHORT_CAPACITY_REQ_S[layout_name(after)]
        failed, waiting, urgent = [], [], False
        if mode != "TP1":  # MERGE
            if bundles.get(bundle) != 1:
                return ["bundle is not TP1"], [], False
            share = s["long_share"]
            tp1_cap, tpk_cap = s["tp1_capacity_max"], s["tpk_capacity_max"]
            # The TP-k counts too: keeping long requests off the TP1s moves
            # their pressure there (GPU 11-02).
            tpk_long_pressure = bool(s["tpk_long_tokens"]) and tpk_cap is not None
            need = [
                share is not None and share >= r["merge_long_share"],
                s["tp1_migrations_per_min"] >= r["merge_tp1_migrations_per_min"],
                tp1_cap is not None and tp1_cap >= r["merge_tp1_capacity"],
                tpk_long_pressure and tpk_cap >= r["merge_tp1_capacity"],
            ]
            if not any(need):
                failed.append("no long-request pressure")
            if s["short_rate"] * r["merge_short_margin"] > capacity_after:
                failed.append("merged layout cannot carry the short rate")
            urgent = (tp1_cap is not None and tp1_cap >= r["urgent_capacity"]) or (
                tpk_long_pressure and tpk_cap >= r["urgent_capacity"]
            )
        else:  # SPLIT
            if bundles.get(bundle, 1) == 1:
                return ["bundle is not TP-k"], [], False
            share = s["long_share"]
            if share is None or share > r["split_long_share"]:
                failed.append("long share not low (or too few arrivals)")
            if (
                s["capacity_max"] is None
                or s["capacity_max"] >= r["split_max_capacity"]
            ):
                failed.append("KV pressure somewhere")
            if not (
                s["short_rate"] >= r["split_short_load"] * capacity_now
                or s["tpk_waiting"]
            ):
                failed.append("short load not approaching capacity")
            if s["bundle_long_tokens"].get(bundle):
                waiting.append("bundle serves a long request")
            gain = (capacity_after - capacity_now) * r["cooldown_s"]
            cost = capacity_now / 2 * r["conversion_s"]
            if gain < r["benefit_factor"] * cost:
                failed.append("capacity gain below benefit factor")
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
            need = r["urgent_hold_s"] if urgent else r["hold_s"]
            repeated = (
                not self.execute
                and now - self.proposed_at.get(key, -float("inf")) < r["cooldown_s"]
            )
            usable = not failed and not cooling and not repeated
            patient = not waiting or held >= need + r["tilt_max_s"]
            ready = usable and held >= need and patient
            candidates[key] = dict(
                failed=failed,
                waiting=waiting,
                urgent=urgent,
                held_s=round(held, 3),
                ready=ready,
            )
            if usable and held >= need - r["tilt_lead_s"] and approaching is None:
                approaching = (key, edge)
            if ready and proposal is None:
                proposal = dict(bundle=edge[0], target_mode=edge[1], held_s=held)
                self.proposed_at[key] = now
                self.since.pop(key, None)
        self.update_prepared(approaching, s, bundles, now)
        prepared = self.prepared
        tilt = (
            [prepared["bundle"]]
            if prepared and now - prepared["at"] <= r["tilt_max_s"]
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
        """Enter the prepare stage once the others are quiet; leave at once."""
        prepared = self.prepared
        if prepared is not None and (
            approaching is None or approaching[0] != prepared["key"]
        ):
            self.prepared = prepared = None
        if prepared is None and approaching is not None:
            key, edge = approaching
            others = [b for b in bundles if b != edge[0]]
            if others and all(s["quiet"].get(b) for b in others):
                self.prepared = dict(
                    key=key, bundle=edge[0], target_mode=edge[1], at=now
                )

    def converted(self, now):
        self.converted_at = now
        self.since.clear()
        self.prepared = None
