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

import random

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
# Where converting starts to pay (user 11-04): long-request demand above L1A's room
# for 900s instead of 420s. P1 and p3 only need to outlast their transients; p3
# keeps its two bursts at the same offsets.
SCENARIOS["mixed_long_p2"] = dict(
    phases=(
        (0, 60, 4.0, 0.0, "warm"),
        (60, 360, 10.0, 0.01, "p1_short"),
        (360, 1260, 1.0, 0.05, "p2_long"),
        (1260, 1530, 8.0, 0.0, "p3_mixed"),
        (1530, 1542, 8.0, 0.5, "p3_mixed"),
        (1542, 1830, 8.0, 0.0, "p3_mixed"),
        (1830, 1842, 8.0, 0.5, "p3_mixed"),
        (1842, 1860, 8.0, 0.0, "p3_mixed"),
        (1860, 1920, 1.0, 0.0, "tail"),
    ),
    long=(16384, 8192),
)
# A short prompt that only grows long: nothing about it is visible on arrival, so
# TP1 fills during output and only migration can help (user 11-04).
SCENARIOS["long_output"] = dict(
    phases=(
        (0, 60, 4.0, 0.0, "warm"),
        (60, 420, 4.0, 0.1, "peak"),
        (420, 480, 1.0, 0.0, "tail"),
    ),
    long=(512, 8192),
)
# Rotation with long segments of three lengths (design 11-05): one clearly worth
# converting for (900s), one that must not trigger it (120s, under the 167s the
# decider took on the first long phase), one at the edge (480s). Short to long
# took about 242s and long to short about 178s, so one run shows where it pays.
# Seed: the first from 1 whose long arrivals are near nominal (L1 41-49, blip 5-7,
# L2 21-27, S1 2-5), fixed before any GPU run and not chosen by any result.
SCENARIOS["mixed_rotation"] = dict(
    phases=(
        (0, 60, 4.0, 0.0, "warm"),
        (60, 420, 10.0, 0.01, "s1_short"),
        (420, 1320, 1.0, 0.05, "l1_long"),
        (1320, 1800, 8.0, 0.0, "s2_short"),
        (1800, 1920, 4.0, 0.05, "b_blip"),
        (1920, 2220, 8.0, 0.0, "s3_short"),
        (2220, 2700, 1.0, 0.05, "l2_long"),
        (2700, 3120, 8.0, 0.0, "s4_short"),
        (3120, 3180, 1.0, 0.0, "tail"),
    ),
    long=(16384, 8192),
    arrival_seed=6,
)
# Long-request capacity (user 11-06): short 8/s, deep long requests stepping up
# every 240s, so each layout and TP2 slot count shows when long requests stop
# fitting. Lives of about 200-300s put about 5, 10, 15 and 20 in flight.
# Seed: the first from 1 whose long arrivals fall in r1 3-7, r2 7-12, r3 11-18,
# r4 15-23 (3/10/18/17), fixed before any GPU run, not chosen by results.
SCENARIOS["mixed_long_ramp"] = dict(
    phases=(
        (0, 60, 4.0, 0.0, "warm"),
        (60, 300, 8.0, 0.02, "r1"),
        (300, 540, 8.0, 0.04, "r2"),
        (540, 780, 8.0, 0.06, "r3"),
        (780, 1020, 8.0, 0.08, "r4"),
        (1020, 1080, 1.0, 0.0, "tail"),
    ),
    long=(16384, 8192),
    arrival_seed=6,
)
# Its first three steps only (user 11-06: shorter runs for quicker answers). Each
# phase keeps its start and the seed, so these arrivals equal the full ramp's.
SCENARIOS["mixed_long_ramp_r3"] = dict(
    SCENARIOS["mixed_long_ramp"],
    phases=SCENARIOS["mixed_long_ramp"]["phases"][:4] + ((780, 840, 1.0, 0.0, "tail"),),
)
# Multi-target stage 4 (user 11-07): the ramp without its overloaded 0.08/s step,
# two longer steps instead so TP1 sources trigger KV moves while TP2 targets can
# still serve. Seed 6 is kept (chosen before any run of this load): r1 equals the
# ramp's r1 and the first 240s of s2 equal its r2; long counts are 3/14/19.
SCENARIOS["mixed_long_steps"] = dict(
    phases=(
        (0, 60, 4.0, 0.0, "warm"),
        (60, 300, 8.0, 0.02, "r1"),
        (300, 660, 8.0, 0.04, "s2"),
        (660, 1020, 8.0, 0.05, "s3"),
        (1020, 1080, 1.0, 0.0, "tail"),
    ),
    long=(16384, 8192),
    arrival_seed=6,
)
PHASES = SCENARIOS["phase_shift"]["phases"]
LONG_PROMPT, LONG_OUTPUT = SCENARIOS["phase_shift"]["long"]


def arrival_times(start, end, rate, seed, role):
    """Evenly spaced without a seed; a seeded Poisson process with one."""
    if seed is None:
        return [start + n / rate for n in range(int((end - start) * rate))]
    stream = random.Random(f"{seed}:{start}:{role}")
    times = []
    at = start + stream.expovariate(rate)
    while at < end:
        times.append(at)
        at += stream.expovariate(rate)
    return times


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
            seed = value.get("arrival_seed")
            for at in arrival_times(start, end, rate, seed, role):
                rows.append(
                    dict(
                        at_s=round(at, 6),
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
                arrival_seed=SCENARIOS[scenario].get("arrival_seed"),
            )
        ],
        note="phases for the conversion decider; one run per scenario, a "
        "functional check, not a comparison",
    )
    return cfg
