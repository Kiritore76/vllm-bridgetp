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

PHASES = (
    # (start, end, short req/s, long req/s)
    (0, 60, 4.0, 0.0, "P1_short"),
    (60, 360, 9.0, 0.0, "P1_short"),
    (360, 720, 0.5, 0.4, "P2_long"),
    (720, 840, 3.0, 0.0, "P3_calm"),
)
LONG_PROMPT, LONG_OUTPUT = 6144, 2048


def jobs(inputs):
    short = rate_sweep.request(inputs)
    long = long_burst.request(inputs, LONG_PROMPT, LONG_OUTPUT)
    rows = []
    for start, end, short_rate, long_rate, phase in PHASES:
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


def config(inputs):
    cfg = pressure.updated_config(inputs)
    values = jobs(inputs)
    cfg.update(
        schema="QWEN3_PHASE_SHIFT_V1",
        recipes=[
            dict(
                name="phase-shift-r1",
                scenario="phase_shift",
                repetition=1,
                arm_order_offset=0,
                fixed_output=True,
                jobs=values,
                input_sha256=matched.digest(values),
                planned_output_tokens=sum(j["request"]["max_tokens"] for j in values),
                routing_scope="all arrivals use the layout's router; no forced pool",
                phases=[list(p) for p in PHASES],
            )
        ],
        note="short peak, long-request phase, calm tail; for the conversion "
        "decider; one run, a functional check, not a comparison",
    )
    return cfg
