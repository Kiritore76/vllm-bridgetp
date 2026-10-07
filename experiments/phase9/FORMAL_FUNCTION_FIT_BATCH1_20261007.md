# First formal conditional GoodOutput collection

Keep guard2000, slow2pct, H=arrival window=300s, full-request SLO drain
settlement, frozen predictor and unchanged M2/M3/M4. Old engineering pilots
are excluded from formal fitting. Constructed requests estimate conditional
benefit; they do not calibrate natural OOM probabilities.

First run one held-out engineering validation pair with theta0.8 and U>=1.
U is an experimental stratum independent of probability. Both assignments
select the first physically feasible observation satisfying both conditions.
Guard reached counts as urgent, even though U has no finite value at zero
T_guard. The physical source/target/channel gate remains authoritative.

The formal training batch has 24 arms: three load strata, two fresh seeds per
stratum, two probability thresholds, START/STAY per threshold.

| Stratum | Seeds | Thresholds | Source prompt | Target |
|---|---|---|---|---|
| Existing source pressure | 20261041,20261042 | 0,0.05 | 4480 | idle |
| Higher source pressure | 20261043,20261044 | 0,0.8 | 6656 | idle |
| Sustained target background | 20261045,20261046 | 0,0.01 | 4480 | 2 jobs/wave |

All anchors have 7168 prompt tokens and 24 task sections. Source jobs have six
sections, max output2048, natural EOS. Source arrivals are every8s, or12s in
the higher-pressure stratum. Target-assigned rows only receive 6144-token
prompts,24 sections,max output4096, with12s subsequent spacing. Log actual
concurrency at decision and stream overlap during copying.

Seed20261040 and engineering_validation content belong only to the urgent
validation pair. Formal training uses new engineering_train content. Future
controller validation/test requests, seeds and source trees remain disjoint;
reserve20261101..20261106 for validation and20261201..20261206 for final test.
Do not use predictor training or retune predictor weights.

All outcomes, including negative benefit, no start, service failures and
handoff SLO violations remain. Technical collection errors are diagnosed
separately. Policy labels and state matching quality are distinct: unequal
START/STAY candidate states are not exact counterfactual evidence.
Group by seed/request tree/load block; do not split ticks as independent
samples. This first batch is six independent training blocks, not twelve
independent threshold pairs. Use grouped uncertainty and collect more blocks
before judging a threshold or fitting a sufficiently supported function.

The urgent validation pair does not enter training. It may need further
coverage if no physically feasible urgent candidate is observed; that does
not erase collected formal outcomes in supported strata. Do not claim the
U>=1 domain is supported without actual START evidence.
