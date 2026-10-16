# Offline health reference and KV relief

The completed operating-region data exports 22 repeated pure-decode cells in
`offline_health_reference.json`. Twenty are available for exact Nd/context
lookup. The two TP4 Nd32 cells remain diagnostic only because their delivery
anomaly is unresolved. Unknown Nd/context returns UNKNOWN; there is no
interpolation, fixed batch limit, new request admission gate or serving switch.
The file records its input hashes and five source archive identities. The
interrupted tenth-scene prefix is excluded; its finalized replacement is used
once. Lookup medians retain run-level ranges and bad-window diagnostics.

The offline helpers live in `tools/bridge_tp/pressure_health_reference.py`.
`health_efficiency` reports observed closed-window ITL goodoutput divided by
the same measured Nd/context reference. It does not estimate migration returns
or physical KV capacity. High goodoutput does not suppress a KV warning, and
missing health lookup support does not imply that the pool is spare.

## Independent KV warning

Keep the existing 2000-token guard. Let F be native physical free blocks, G be
ceil(2000 / block_size), U be visible prompt blocks not already owned, r be
observed decode growth in blocks/s, A be inventory age and T be START through
source-KV release latency. Warn when either:

```
F <= G
F <= G + U + ceil(r * (A + T))
```

The first condition works even when growth is flat or unknown. The second
starts candidate evaluation before the guard is crossed, using fresh physical
facts and a measured release budget. A stale reading is UNKNOWN. The observed
F is never rewritten and no capacity is reserved for ordinary new requests.

U is summed per request as max(0, ceil(prompt_tokens / block_size) - owned_blocks).
Allocated prefill blocks are already reflected in F and ownership; subtracting
all unfinished prompt tokens again would double count them. Unknown ownership
keeps the projection unknown. This single full-attention KV-group observation
does not invent ownership for unsupported groups or prefix reuse.

Do not extend a brief prefill-allocation spike across the entire release
window. The offline replay uses the past two-second native generated-token
growth for r, and accounts for visible unallocated prompt blocks separately.
It neither assumes future request arrivals nor credits future request endings.
This extrapolation is an engineering warning, not a guaranteed demand bound.

The current T is the historical maximum observed START-to-source-release,
16.864101 seconds. Freeze/handoff can take only hundreds of milliseconds;
substituting that interval would omit most of the allocation growth before
source blocks are actually freed. The empirical maximum is not a statistical
upper bound and needs later confirmation under the new conditions.

## Candidate and START requirements

A KV warning requests priority capacity-candidate evaluation even if service
goodoutput looks healthy or health support is UNKNOWN. `capacity_candidate`
checks only physical opportunities and always returns NO_START_IN_OFFLINE_REPLAY.

- Required unique release must cover the projected deficit back to the guard.
- Release may occur after crossing the warning guard; it still needs evidence
  of completion before physical exhaustion. Guard crossing alone is not an
  exhaustion rejection.
- Target native free blocks must cover existing migration promises plus the
  proposed request's target demand, with fresh evidence.
- Actual START still requires a bound source stream, current ownership,
  context/stop feasibility, target service budget, available migration channel,
  and enough remaining work to survive the transfer.

The normal service path can prefer longer remaining work. Capacity relief
prioritizes sufficient timely physical release among feasible candidates;
it cannot wait for bad ITL or an observed preemption. A warning is not evidence
that every remaining request is worth migrating.

## Past-only replay result

Replay used actual observer capture/generation records from the nine finalized
scenes plus the replacement, never the interrupted prefix. Out of 2981 active
source inventory evaluations, only the two Nd6 mid-context scenes warned:

| Scene | First warning free blocks | Free tokens | Warning snapshots | Physically feasible snapshots |
|---|---:|---:|---:|---:|
| region-mid-n6-n32-r1 | 330 | 5280 | 69 | 45 |
| region-mid-n6-n32-r2 | 332 | 5312 | 68 | 44 |

The old below-125-block condition produced no trigger in these recordings.
The revised warning creates advance capacity opportunities. The other eight
scenes did not warn. Physical opportunities are repeated snapshots, not
independent migrations or approvals of the target service budget.

These forced-output scenes all ended without KV preemption. At the first
warning the candidates had only about 513-530 tokens left to their stop caps;
remaining-life and target-prefill checks can therefore still reject START.
This replay does not prove that 137 warnings should become 137 migrations or
that the rule prevents preemption in a sustained growth workload. It establishes
that low KV can independently request action before the guard is reached.

## Reproduction

Export with the environment Python from the repository root:

```
python -B -m tools.bridge_tp.update_pressure_health_reference \
  --cells /path/to/combined_working_cells.json \
  --windows /path/to/combined_window_observations.json \
  --provenance /path/to/sources.json \
  --service-evidence experiments/phase9/pressure_balanced_v2/service_evidence.json \
  --diagnostic-shape target:32:2 --diagnostic-shape target:32:3 \
  --output experiments/phase9/pressure_balanced_v2/offline_health_reference.json
```

The local past-only replay is preserved at
`D:/work/bridgetp/control/replay_pressure_health_kv_20261016.py`; its receipts and
report are under `control/pressure_health_update_20261016`. The serving policy,
its runtime config and the existing GPU collection modes remain unchanged.
