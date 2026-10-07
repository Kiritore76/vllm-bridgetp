# Rolling Shadow-only cutover

Mechanism: `ROLLING_NO_HISTORY_WAIT_V1`, opt-in with `--rolling-cutover`
and `--rolling-reserve-tokens 512` on the probability pilot collector.
START still comes from the frozen probability collector. Predictor training,
guard=2000, SLO slow-interval allowance=2%, and H=300 s stay frozen.

Historical KV injection and live delta injection remain sequential on each
rank. The full-rank exact GPU readback publishes one compact initial receipt
per rank, including every logical block and its allocated physical block ID.
Legacy per-block receipts remain readable by offline acceptance. This removes
hundreds of redundant file publications between history writeback and the
first delta receive, without removing the exact readback.

The controller admits a dormant target using a capacity reservation R. R is
not the executable source freeze boundary. The source first plans B only
after four initial exact resident receipts and a four-rank APPLIED live delta
ACK, with at most 16 tokens of lag. It reserves at least 64 future output tokens
of publication lead, increased by measured generation speed. When lag grows
near B, it moves B forward before freezing. Source decode continues throughout
history and catch-up. All plans remain inside the original allocation.

At the exact selected B, source freeze and the final delta use the existing
request freeze/atomic takeover protocol. END_SESSION closes the receiver at
the actual computed prefix, rather than at R. Before scheduler promotion, the
connector replaces placeholder tokens, resets the computed prefix to B, and
restores the target output budget to total_budget-B. Surplus allocated blocks
remain owned by the target request and are never marked computed or cached
beyond the actual prefix.

If the reservation is exhausted before a safe freeze, Shadow is cleaned and
the source remains owner. Acceptance requires matching exhaustion, complete
source output, no freeze/commit, and source/target/stager cleanup evidence.
This is a START cost outcome; it is not a successful takeover. Missing or
corrupt evidence remains a technical exclusion. A cancelled zero-token target
does not replace a fully proved completed source in SLO or horizon scoring.

CPU checks exercise the production source hook and scheduler connector
callback with controlled request/clock fixtures, planner deferral/exhaustion,
plan evidence tampering, allocation remapping, cleanup, active owner scoring,
and the existing stream/state/probability protocols. CUDA writes, NCCL timing,
and numerical continuation still require the server pilot.

Pilot validation must report actual START positions separately from actual B,
plan count, reservation R, four-rank history readiness before first plan,
first APPLIED ACK, pre-freeze lag, target prefix/budget remap, contiguous visible
stream, handoff interval, cleanup, and whole-pool H-window GoodOutput. Short,
medium and long generated-token bands are coverage checks after probability
selection, never fixed-token START gates. Engineering pilot data is kept
separate from future formal function-fitting data.
