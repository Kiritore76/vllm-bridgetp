# Continuous fixed-window probability pilot

Use `--window-token-goodoutput --arrival-window-s 300
--evaluation-horizon-s 300 --arrival-wave-period-s 24` with constructed source
count 3, target count 0, and probability threshold 0.8. Collect one START/STAY
pair before expanding. The original B2 pressure-long recipe remains frozen:
7168 anchor tokens, 4480 source prompt tokens, natural EOS, and 2048 background
output cap. The first burst is three source peers; later distinct recipe inputs
arrive every 8 seconds, beginning at 28 seconds. There are 37 peers in total;
the last planned arrival is at 292 seconds relative to the anchor first output.
Client origin remains the earliest request arrival; schedule lag is audited.

This is a workload pilot, not a fitted function or independent risk calibration.
All arrivals are pre-registered and identical in the paired arms. Completion
times and EOS lengths may vary normally. Predictor, greedy sampling, SLO v6,
guard warning policy, four-rank restore and watermark constraints remain frozen.

The opt-in scoring policy is `WINDOW_TOKENS_FULL_REQUEST_SLO_DRAIN_V1`.
It counts visible tokens in [origin, origin + H) from eventually completed,
full-request SLO-valid requests. Requests completing after H may contribute
their in-window tokens. Drain tokens never enter the numerator or denominator.
The full-request SLO audit includes drain; this is retrospective request SLO,
not an assertion that SLO is known at H or assessed only on its prefix. Service
failure or SLO failure remains zero. Missing evidence is technical exclusion.
Legacy completed-by-H scoring remains the default and old raw labels are
unchanged. Do not mix scoring versions in a fitted dataset without replaying
them under the same explicitly chosen contract.

The pilot records inflight coverage, 10-second output-bin coverage and last
in-window output time. Inflight coverage includes queued requests; output-bin
coverage is additional evidence, not GPU utilization. Check both paired arms
have at least 90% coverage and output in the last 10 seconds, with at least half
the requests passing frozen SLO. These workload checks are not effect-label
filters: preserve all raw data and negative outcomes even if the load is not
ready to expand. Require an actual START takeover and a non-starting STAY for
mechanism validation. No positive net-benefit requirement.

Collection IDs derive from the frozen protocol and manifest identities,
including their unique output paths. Sample IDs use that collection identity;
the shared seed/request-group dependence label is retained. This prevents
different recipe/run samples from overwriting each other without claiming
independent repetitions.
