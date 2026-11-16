# e61: Episode-start-conditioned whole-day PIT sampling

## Prospectively fixed hypothesis

Completed e59 shows a strong context/prevalence difference between calibration
and validation; completed e60 finds no demonstrated benefit from merely
permuting within-day PIT order. Test conditioning which daily noise trajectory
is sampled on known initial context, while retaining its complete joint path.
This tests the distribution, not a searched temporal permutation or retuning
of failed e58 tails. No new quantile fitting or agent training is planned.

The primary candidate is e51's unchanged five-quantile model at its previously
selected alpha.25, with a five-nearest-day PIT sampler. Use e58's unchanged
upper-tail model at alpha.25 only as a secondary mechanism contrast. Report
the full two-parent by global/context sampler comparison; the secondary is
not an adoption fallback if the primary fails. Preserve both original model
hashes and all original global-reference outcomes.

## Fixed causal context and controls

Use exactly six episode-start features: NH4, DO1, DO2, DO3, temperature and flow.
Compute feature mean/standard deviation from the158 fitting-day initial
features only, floor standard deviations at1e-6 and report any floor usage.
Use squared Euclidean distance after this scaling. Select exactly five nearest
eligible calibration days, breaking distance ties by original calibration-day
index. Sort the selected five indices into their original chronological/index
order before uniform sampling. Do not search feature sets, weights or k.

For each five-member/eight-draw/target-date key retain the original e38 SHA256
seed and uniform integer day draw, replacing the global30-day pool by the
selected five-day pool. Copy the chosen member's entire95-step five-channel
PIT path without reordering, scaling or changing NO3. Conditioning is fixed
at episode start and independent of future states, actions or concentrations.
Common-noise comparative air rollouts must therefore use the same chosen path.
Future exogenous values do not enter the neighbor metric.

On calibration, exclude the query date from eligible donors for both the
context-five and global29-day diagnostic. No calibration donor shares the
query's target sequence. On validation, use all30 eligible calibration days.
Alpha, model and k are already fixed; there is no validation-driven selection.
Report leave-one-day-out calibration proper scores as diagnostics, not new
training or held-out independent confirmation. The same calibration labels
still estimated the PIT banks, a residual limitation to disclose.

Before candidate validation, require full-pool/no-exclusion noise paths to
reproduce original q.paths bitwise, including original RNG consumption and
day ordering. Require selected donors to be eligible and exact copied tuples,
deterministic tie handling and reproducibility, fitting/calibration/test split
separation, and unchanged initial-context extraction under future-state/air/
exogenous perturbation. No future target is used by the sampler. Preserve
fixed physical clipping and underlying monotonic transition constraints.

Original global validation forecasts must reconstruct both parent summaries
before interpreting the context variants. Independently reconstruct donor
indices and full-day forecast with sklearn/scalar interpolation, check common
noise air ordering and report the selected primary16-draw sensitivity.

## Unchanged and stronger outcome requirements

Apply the complete original e16 point gate, member frequencies and recursive
air-order tolerance. Also apply all e58 predeclared probability/timing rules:
NH4 and DO paired-date Brier difference upper endpoints at most zero versus
e40; DO upper endpoint at most zero versus persistence; DO AP greater than
observed prevalence, DO AUC greater than.5 and at least one new DO onset matched
within four steps at probability threshold.5. Retain all NH4/DO onset0/1/4,
MAE, frequency, Brier, AP/AUC, native/global/secondary outcomes and10000 paired
circular five-day date-bootstrap intervals. No criterion or parameter is
changed after outcomes. Model members/noise draws/dates are not policy seeds.

Reused validation remains exploratory, not independent confirmation or real
plant counterfactual validation. Never use final41 test days. Source data,
donor-level rows, trajectories and PIT banks remain local. If the primary
fails, preserve that failure and do not train policies on it or switch to
the favorable secondary as a rescue. A new freeze/agent comparison requires
the full declared environment criteria and a separate recorded decision.

The specification was committed at5d6a0e1 before implementation or outcomes.
context_paths.py implements only the fixed start-context, pool and whole-day
copy rule. preflight.json now passes full-pool/original path bitwise identity
for both parent banks, future-state/exogenous/air invariance, self exclusion,
chronological pool order and deterministic ties. Independent scalar RNG/donor
reconstruction reproduces every checked tuple. Both selected parent PIT banks
also reconstruct bitwise from authenticated model weights and original
calibration data. Fitting-only initial-context scales use all158 fitting dates
with no standard-deviation floor needed. No candidate validation is scored.

Run from the workspace into a new output destination:

```powershell
python Threemool/Lee/experiments/e61/preflight.py --data kaggle_data/darrow_15min.pkl --e51 results/e51_blended_pit --e58 results/e58_conditional_upper --out results/e61_context_preflight_replay.json
```

Use pinned threemool Python, CPU, DARROW_FAST_DYNAMICS=1 and numerical thread
counts1 for OMP/OPENBLAS/MKL/NUMBA. The preflight rejects an existing output.
An initial path-only preflight also passed before the bank-reconstruction
extension; the published preflight.json is the completed stronger check.
No e61 process remains live. Next implement leave-one-day-out calibration,
both global-reference validation replays, all fixed context variants and the
independent forecast/event/adoption audits without modifying historical code.
