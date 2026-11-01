# SPDX-License-Identifier: Apache-2.0
"""Short-request arrival-rate sweep: where does the layout's capacity end?

One fixed request shape (512 prompt / 128 output, fixed output) at a steady
rate for ``duration_s``, then the backlog drains. Rates bracket the expected
saturation of a TP2 pair (~10 req/s) and of four TP1s (~18 req/s), so a
crossover between layouts, if any, falls inside. Arrivals stop after
``duration_s`` so an overloaded layout cannot build an unbounded queue.
"""

import copy

from tools.bridge_tp import qwen3_pressure_four_mode as pressure
from tools.bridge_tp import qwen3_stay_migrate as matched

RATES = (4, 8, 12, 16)
DURATION_S = 60
PROMPT, OUTPUT = 512, 128
SEED = 20261101


def request(inputs, prompt=PROMPT, cap=OUTPUT):
    value = copy.deepcopy(inputs["requests"]["2048"])
    tokens = value["prompt"]
    value["prompt"] = tokens[: prompt - 32] + tokens[-32:]
    value.pop("request_id", None)
    value.update(
        max_tokens=cap,
        ignore_eos=True,
        seed=SEED,
        stream=True,
        return_token_ids=True,
    )
    if len(value["prompt"]) != prompt:
        raise ValueError("rate sweep prompt length differs")
    return value


def config(inputs, rates=RATES, duration_s=DURATION_S):
    cfg = pressure.updated_config(inputs)
    recipes = []
    body = request(inputs)
    for rate in rates:
        jobs = [
            dict(
                id=f"rate_{rate}-{n}",
                at_s=round(n / rate, 6),
                route=None,
                request=copy.deepcopy(body),
                workload_role="short_probe",
                workload_phase="sustained",
            )
            for n in range(int(rate * duration_s))
        ]
        recipes.append(
            dict(
                name=f"rate-sweep-rate_{rate}-r1",
                scenario=f"rate_{rate}",
                repetition=1,
                arm_order_offset=0,
                fixed_output=True,
                jobs=jobs,
                input_sha256=matched.digest(jobs),
                planned_output_tokens=sum(j["request"]["max_tokens"] for j in jobs),
                routing_scope="all arrivals use the layout's router; no forced pool",
                rate_req_s=rate,
                duration_s=duration_s,
            )
        )
    cfg.update(
        schema="QWEN3_RATE_SWEEP_V1",
        recipes=recipes,
        note=f"short {PROMPT}/{OUTPUT} fixed-output requests at {list(rates)} req/s "
        f"for {duration_s}s each, then drain; throughput and latency per rate; "
        "one run per cell, a screen not a formal comparison",
    )
    return cfg
