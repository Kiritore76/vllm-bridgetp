# Source guard is a warning, not an OOM deadline

The probability collector's source-guard contract is now version 2. The source
guard reserves KV capacity; missing its projected crossing time does not prove
OOM and does not prohibit starting Shadow. This implements the user's explicit
2026-10-06 correction and applies to the F1/F2 probability collection path.

- Keep the bucket-based `p_guard_est` calculation and separate urgency U/S.
- A nonpositive conservative S bound records `guard_deadline_warning` and
  `source_release_may_miss_guard` in `guard_warnings`, not physical rejections.
- At H<=0, record `GUARD_REACHED`, T_guard=0 and guard-event probability bounds
  [1,1] if the prediction is valid. This is an already-reached guard event,
  not an OOM probability of one. U has no finite value and remains null;
  negative S and the explicit guard state preserve the urgency evidence.
- Starting remains possible with positive actual source KV headroom after
  pending prefill reservations. Exhausting that headroom still rejects ordinary
  admission and records a capacity protection requirement.
- Fresh telemetry, valid prediction/output budget, target KV reservations,
  channel availability, delta feasibility, and subsequent four-rank exact
  readiness and safe handoff gates remain enforced. Probability threshold and
  assigned action still select START; a warning does not force all STAY arms
  to migrate.

Version 2 gate records separate guard warning ticks from actual source capacity
requirements. Warnings alone do not exclude a new effect sample. Genuine
capacity requirements and safety overrides keep their existing exclusion.
Legacy snapshots retain their previous gate replay semantics. Do not overwrite
the returned version 1 archives or relabel their effect eligibility in place.

CPU validation: 166 targeted unit/regression tests pass. Read-only replay of the
returned coverage STAY trajectories admits 37/33/33 previously guard-rejected
ticks for long/target8/short profiles without changing their bucket probabilities.
The first p_lower>=0.8 candidates have U about 0.809/1.466/0.981, respectively.
This does not prove that actual late migration catches up; a new GPU pilot is
required before claiming that runtime result. Predictor training and SLO remain
frozen. Existing launchers pinned to the prior HEAD do not exercise this change.
