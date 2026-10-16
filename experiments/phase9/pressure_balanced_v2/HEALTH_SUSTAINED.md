# Health policy sustained paired pilot

Cloud: `bash tools/bridge_tp/run_pressure_pilot.sh --health-sustained`.
This reuses the existing wrapper, tracked token prompts and single evidence
archive. Old `--sustained` remains the historical KV-only workload; this entry
explicitly uses health_service_policy.json and the updated health_reference_v2.

Four fresh runs: repetition1 STAY then migrate; repetition2 migrate then STAY.
Each repetition uses identical prompt tokens, seed20261015 and arrival pattern.
151 requests per run / 53388 full-output tokens. Keep one migration maximum,
0.90 efficiency,0.80 target load,10s persistence,125-block guard and the measured
16.86410117149353s KV lookahead. No forced START and no new ingress reservation.
Background metrics, async observation logging and host/timing diagnostics stay
enabled in both arms.

The opening service-pressure phase matches the validated spare-target screen:
source3 requests prompt2048/output4096 at5,5.02,5.04s; target1 request
prompt2048/output6144 at0s. Continue the workload with:

- TP4 prompt2048/output6144 at90,180,270s.
- TP1 prompt2048/output4096 at110,165,220,275s.
- TP1 competition prompt2048/output1 every2s from20 through298s (140 requests).

This is a measured trial workload, not a fitted optimal arrival rate. Initial
post-warmup headroom check is unchanged: source1405 blocks, target1021 blocks.
Later requests enter vLLM normally and may queue or preempt; no claim is made
that every future full stop cap fits simultaneously. Exact unsupported
Nd/context shapes remain UNKNOWN; all such intervals remain in evidence.

Score the first300s using existing full-request ITL100ms/slow1%/maxgap1s weight.
Fully drain all requests; drain gaps affect quality but drain tokens stay out
of the300s numerator. TTFT has no new penalty. Require actual arrival lag<=0.5s,
in-flight coverage>=95% and output in all30 ten-second bins in both arms for
a usable paired measurement. Intended coverage is never treated as measured
coverage. Inspect actual START path, remaining life, commit/source release,
pool-level goodoutput, queue/preemption/native shape evidence and diagnostics.

The workload tests the benefit of the current single-session policy under
continued arrivals. It does not validate multishot migration or automatic
ingress routing, and two pairs do not establish a universal optimal threshold.
Missing migrations/coverage are reported, never discarded or retried by seed.

Allow roughly35–45 minutes for four server startups,300s windows and full drain;
the existing900s per-run timeout remains. Progress and packing use the existing
entry. Return `pressure-balanced-git-v2-health-sustained-*.tar.gz` from
`/root/autodl-tmp/bridgetp/results`.
