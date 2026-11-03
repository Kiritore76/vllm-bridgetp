# SPDX-License-Identifier: Apache-2.0
"""Phase-shift workload for the conversion decider (user 11-01).

From an L1A start the load changes character twice, so each decider rule
has a phase in which it should fire, and one in which nothing should:

- P1 short (0-360s): 512/128 at 4 req/s for 60s, then 9 req/s (~70% of the
  L1A short capacity, no KV pressure): SPLIT of the TP-k bundle expected.
- P2 long (360-720s): 0.5 req/s short plus 0.4 req/s 6144/2048 long (~44%
  of arrivals, final length 8192): MERGE of the TP1 bundle expected.
- P3 calm (720-840s): 512/128 at 3 req/s: no conversion expected.

Arrivals stop at 840s; the 2048-token outputs then drain. With the decider's
120s window and 180s hold, the two proposals fall inside their phases.
"""

from tools.bridge_tp import qwen3_long_burst_pair as long_burst
from tools.bridge_tp import qwen3_pressure_four_mode as pressure
from tools.bridge_tp import qwen3_rate_sweep as rate_sweep
from tools.bridge_tp import qwen3_stay_migrate as matched

# (start, end, short req/s, long req/s, phase label) and the long request.
SCENARIOS = {
    # Explicit cycle mechanism check, not the later mixed performance trial.
    "conversion_cycle": dict(
        phases=(
            (0, 300, 1.0, 0.0, "cycle_background"),
            (300, 312, 1.0, 0.5, "first_l1a_burst"),
            (312, 840, 1.0, 0.0, "cycle_background"),
            (840, 852, 1.0, 0.5, "second_l1a_burst"),
            (852, 1080, 1.0, 0.0, "cycle_tail"),
        ),
        long=(16384, 8192),
    ),
    "phase_shift": dict(
        phases=(
            (0, 60, 4.0, 0.0, "P1_short"),
            (60, 360, 9.0, 0.0, "P1_short"),
            (360, 720, 0.5, 0.4, "P2_long"),
            (720, 840, 3.0, 0.0, "P3_calm"),
        ),
        long=(6144, 2048),
    ),
    # Executed conversions (user 11-02): each scenario gives one decider rule
    # its case, with time left to see the new layout after the conversion.
    # Many mid-length long requests: more than assist's cap on a TP1 bundle.
    "merge_heavy": dict(
        phases=((0, 420, 0.5, 0.4, "long"), (420, 480, 1.0, 0.0, "tail")),
        long=(6144, 2048),
    ),
    # Fewer of them: the TP-k still has room during a tilt.
    "merge_moderate": dict(
        phases=((0, 420, 0.5, 0.25, "long"), (420, 480, 1.0, 0.0, "tail")),
        long=(6144, 2048),
    ),
    # Few, very long: KV pressure rather than their share; assist matters most.
    "merge_deep": dict(
        phases=((0, 420, 0.5, 0.05, "long"), (420, 480, 1.0, 0.0, "tail")),
        long=(16384, 8192),
    ),
    # Layout boundary (user 11-02): where a static L2 starts to beat L1A. The
    # grid is these two, merge_moderate/merge_heavy, and deep_*/merge_deep.
    "merge_light": dict(
        phases=((0, 420, 0.5, 0.1, "long"), (420, 480, 1.0, 0.0, "tail")),
        long=(6144, 2048),
    ),
    "deep_light": dict(
        phases=((0, 420, 0.5, 0.02, "long"), (420, 480, 1.0, 0.0, "tail")),
        long=(16384, 8192),
    ),
    "deep_mid": dict(
        phases=((0, 420, 0.5, 0.035, "long"), (420, 480, 1.0, 0.0, "tail")),
        long=(16384, 8192),
    ),
    # Below split_trickle's peak: the short side of the boundary.
    "short_mid": dict(
        phases=(
            (0, 60, 4.0, 0.0, "warm"),
            (60, 420, 6.0, 0.1, "peak"),
            (420, 480, 1.0, 0.0, "tail"),
        ),
        long=(6144, 2048),
    ),
    # The deep long burst (user 11-02) migrated at 0.5 req/s of short requests
    # and did not at 6: the rungs between, on the same six long requests.
    "burst_short2": dict(
        phases=(
            (0, 120, 2.0, 0.0, "warm"),
            (120, 132, 2.0, 0.5, "burst"),
            (132, 420, 2.0, 0.0, "after"),
            (420, 480, 1.0, 0.0, "tail"),
        ),
        long=(16384, 8192),
    ),
    "burst_short3": dict(
        phases=(
            (0, 120, 3.0, 0.0, "warm"),
            (120, 132, 3.0, 0.5, "burst"),
            (132, 420, 3.0, 0.0, "after"),
            (420, 480, 1.0, 0.0, "tail"),
        ),
        long=(16384, 8192),
    ),
    # Migration's own case (user 11-02): a burst of six very long requests over
    # steady short load near the L2 limit. L0 cannot hold them, L2 queues the
    # short requests, and no layout has time to convert.
    "burst_short6": dict(
        phases=(
            (0, 120, 6.0, 0.0, "warm"),
            (120, 132, 6.0, 0.5, "burst"),
            (132, 420, 6.0, 0.0, "after"),
            (420, 480, 1.0, 0.0, "tail"),
        ),
        long=(16384, 8192),
    ),
    "burst_short7": dict(
        phases=(
            (0, 120, 7.0, 0.0, "warm"),
            (120, 132, 7.0, 0.5, "burst"),
            (132, 420, 7.0, 0.0, "after"),
            (420, 480, 1.0, 0.0, "tail"),
        ),
        long=(16384, 8192),
    ),
    # Where L0 starts to beat L1A (user 11-02): sustained short load with a few
    # very long requests, spaced so that they rarely share a TP1.
    "steady_short10": dict(
        phases=(
            (0, 60, 4.0, 0.0, "warm"),
            (60, 420, 10.0, 0.01, "peak"),
            (420, 480, 1.0, 0.0, "tail"),
        ),
        long=(16384, 8192),
    ),
    "steady_short12": dict(
        phases=(
            (0, 60, 4.0, 0.0, "warm"),
            (60, 420, 12.0, 0.01, "peak"),
            (420, 480, 1.0, 0.0, "tail"),
        ),
        long=(16384, 8192),
    ),
    "burst_short8": dict(
        phases=(
            (0, 120, 8.0, 0.0, "warm"),
            (120, 132, 8.0, 0.5, "burst"),
            (132, 420, 8.0, 0.0, "after"),
            (420, 480, 1.0, 0.0, "tail"),
        ),
        long=(16384, 8192),
    ),
    # Six fit without moving anything (GPU 11-02: no migration). Nine is the
    # most L1A can hold at final length (1536 blocks; 7 on the TP2, 1 per TP1),
    # so it fits only if placement ends up right, whatever arrival did.
    "burst9_short6": dict(
        phases=(
            (0, 120, 6.0, 0.0, "warm"),
            (120, 138, 6.0, 0.5, "burst"),
            (138, 420, 6.0, 0.0, "after"),
            (420, 480, 1.0, 0.0, "tail"),
        ),
        long=(16384, 8192),
    ),
    # A short peak with a trickle of long requests, which keep the TP-k from
    # ever being free of them unless routed away.
    # Each layout's own load in one run (user 11-03): L0's (steady_short12), L2's
    # (deep_mid's upper rate), then L1A's (burst_short8, twice). A burst waits for
    # the long requests of the phase before, and the next for it: together they
    # would exceed every layout's KV, which no layout could show anything about.
    # P1 is long enough to outlast entering L0 (replay 11-03: about 340s).
    "mixed": dict(
        phases=(
            (0, 60, 4.0, 0.0, "warm"),
            (60, 660, 12.0, 0.01, "p1_short"),
            (660, 1080, 1.0, 0.05, "p2_long"),
            (1080, 1350, 8.0, 0.0, "p3_mixed"),
            (1350, 1362, 8.0, 0.5, "p3_mixed"),
            (1362, 1650, 8.0, 0.0, "p3_mixed"),
            (1650, 1662, 8.0, 0.5, "p3_mixed"),
            (1662, 1920, 8.0, 0.0, "p3_mixed"),
            (1920, 1980, 1.0, 0.0, "tail"),
        ),
        long=(16384, 8192),
    ),
    "split_trickle": dict(
        phases=(
            (0, 60, 4.0, 0.0, "warm"),
            (60, 420, 9.0, 0.1, "peak"),
            (420, 480, 1.0, 0.0, "tail"),
        ),
        long=(6144, 2048),
    ),
}
# 12 req/s is past what static L1A serves (GPU 11-03: about 11.25), so its p1 only
# measured how long the phase was; 10 keeps L1A steady.
SCENARIOS["mixed_p1_10"] = dict(
    SCENARIOS["mixed"],
    phases=tuple(
        (start, end, 10.0 if phase == "p1_short" else short, long_rate, phase)
        for start, end, short, long_rate, phase in SCENARIOS["mixed"]["phases"]
    ),
)
PHASES = SCENARIOS["phase_shift"]["phases"]
LONG_PROMPT, LONG_OUTPUT = SCENARIOS["phase_shift"]["long"]


def jobs(inputs, scenario="phase_shift"):
    value = SCENARIOS[scenario]
    short = rate_sweep.request(inputs)
    long = long_burst.request(inputs, *value["long"])
    rows = []
    for start, end, short_rate, long_rate, phase in value["phases"]:
        for rate, body, role in (
            (short_rate, short, "short_probe"),
            (long_rate, long, "long_input_output"),
        ):
            if not rate:
                continue
            for n in range(int((end - start) * rate)):
                rows.append(
                    dict(
                        at_s=round(start + n / rate, 6),
                        route=None,
                        request=dict(body),
                        workload_role=role,
                        workload_phase=phase,
                    )
                )
    rows.sort(key=lambda j: (j["at_s"], j["workload_role"]))
    for index, row in enumerate(rows):
        row["id"] = f"phase-{index}"
    return rows


def config(inputs, scenario="phase_shift"):
    cfg = pressure.updated_config(inputs)
    values = jobs(inputs, scenario)
    cfg.update(
        schema="QWEN3_PHASE_SHIFT_V1",
        recipes=[
            dict(
                name=scenario.replace("_", "-") + "-r1",
                scenario=scenario,
                repetition=1,
                arm_order_offset=0,
                fixed_output=True,
                jobs=values,
                input_sha256=matched.digest(values),
                planned_output_tokens=sum(j["request"]["max_tokens"] for j in values),
                routing_scope="all arrivals use the layout's router; no forced pool",
                phases=[list(p) for p in SCENARIOS[scenario]["phases"]],
            )
        ],
        note="phases for the conversion decider; one run per scenario, a "
        "functional check, not a comparison",
    )
    return cfg
