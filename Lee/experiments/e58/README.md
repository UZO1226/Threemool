# Lee/e58: Conditional upper DO tail with the original quantile backbone fixed

## Prospectively fixed hypothesis

The e40/e51 observed-state support audit finds9 of26 inner-calibration new DO
events outside every member's attainable support at alpha.5. e52's unconditional
paired tail magnitudes instead overpredict validation DO frequency. Test a
conditional fitted upper tail, retaining the original central dynamics.
This is an environment-model hypothesis, not an agent gain or a proposed
retuning of failed e57 smoothing. Unsupported observed-state events do not
prove that an entire free-running day is impossible to predict.

Keep all five e40 quantile files and their exact hashes, original training-only
state clip bounds, features, physical costs/thresholds,158 fit/30 calibration/
40 validation/41 reserved-test split and five environment model members.
Fit only three additional next-state DO quantile.99 heads per member,15 new
regressors total. Use the original80-percent date subsets without replacement,
seeds100*m+j,200 HGB iterations, learning rate.05,15 leaves, minimum40 samples
per leaf,L2=1, no early stopping and the same safety monotonic constraints.
Mask NO3 for the added DO heads and verify that no tree splits on it. Use the
same scoped e40 quantile-leaf bound repair with its version/source guard;
installed library files and historical runtime files must stay unchanged.

## Fixed tail and calibration

Compute the original sorted/clipped five conditional quantiles. Preserve all
five. For each DO channel append max(original upper.95, clipped learned.99).
NH4 and NO3 append a duplicate of their original upper endpoint. Original
NH4/NO3 inverse sampling is unchanged at a fixed current state/noise. For DO,
use the original inverse below.95 and extend linearly to the added upper
endpoint between.95 and.99, capping above.99. Implement it as the original
five-level sample plus a nonnegative DO-only extension so an identically zero
extension reconstructs the old sampler bitwise. Do not sort new heads into
the central backbone or expand physical clip bounds. Quantile crossing is
reported and bounded by the max construction, not rescued with a new setting.
All state trajectories can still change through subsequent DO/NH4 feedback.

Estimate PIT after actual alpha mixing and final physical clipping, one bank
for alpha0,.25,.5,.75, as in e51. Use the six-level CDF/atom intervals, preserving
NO3 PIT.5 and the original random uniforms, row pairing and block/noise keys.
The zero-extension control must exactly reproduce e51's four banks and
selected forecasts. Preserve original1/4/16/95-step blocks and the original20
uniform/PIT/alpha conditions. Select only by the unchanged inner proper score;
validation is not used to choose a quantile, tail strength, alpha or block.
There is one new tail level.99 and no multiplier or quantile search.

## Verification before outcomes

Require synthetic endpoint, atom/tie, zero-extension inverse/CDF and numerical
np.interp checks. On original calibration data, verify all zero-extension banks
against e51 and a complete95-step zero-extension rollout against the original
backbone with identical noise. Check interpolation after actual clipped mixing,
including observed NO3 inputs beyond the original clip bounds. Preserve these
as implementation checks, not evidence of generalization or extra policy seeds.

Before primary validation, verify unchanged backbone hashes, fit/calibration/
validation/test separation, finite fixed clipping, deterministic NO3, added
head source constraints, fast/sklearn inference and common-noise coordinate/
recursive air ordering. An independent full-day audit must rebuild the
backbone and upper prediction with sklearn and a separate inverse calculation.
Zero-control failures are implementation issues to diagnose, not permission
to lower the outcome criterion. Source data and stochastic banks stay local.

## Outcome criterion and scope

Keep the complete original e16 point gate, its model-member count, all member
frequencies and selected16-draw sensitivity. Separately report NH4/DO bias,
MAE, Brier, AP/AUC, onset0/1/4-step matching and uncertainty against old,
persistence, e40 and e51. Use the existing10000 paired five-day date-block
resamples for Brier differences. Model members/noise draws/dates are not policy
training seeds, and a point frequency pass alone is insufficient adoption.

For this candidate, additionally require NH4 and DO paired-date Brier difference
upper endpoints at most zero versus frozen e40, DO upper endpoint at most
zero versus persistence, DO AP above observed prevalence, DO AUC above.5
and at least one matched new DO onset within four steps. These stronger
pre-outcome conditions do not replace or lower the original gate. Report
uncertainty and every failure; do not claim independent confirmation from
reused validation or attribute a new onset to a proven plant counterfactual.

Do not replace e42's frozen environment or train policies on this candidate
unless its declared environment checks pass. No final test, team protocol,
Paper or other-member change. Preserve failed families and their archives.
If successful, subsequent agent comparisons must isolate new environment
effects from agent changes under a newly documented freeze and equal policy
training budgets. This experiment itself is not an FCSRL extension win.

The prospective specification was committed at35b8da3 before implementation,
fitting or validation outcomes. upper_tail.py implements the fixed six-level
kernel without changing historical sources or global backbone levels.
preflight.json passes synthetic endpoints/atoms, the independent np.interp
inverse reference, exact reproduction of all four e51 calibration banks and
40 complete95-step calibration-only zero-extension forecasts. No new head
has been fitted, and neither validation nor final test is evaluated here.

From the workspace, reproduce the preflight into a new destination:

```powershell
python Threemool/Lee/experiments/e58/preflight.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --parent-freeze Threemool/Lee/experiments/e42/environment_freeze.json --out results/e58_zero_preflight_replay.json
```

The next step is the fixed15-head DO upper-quantile fit, then source/constraint
verification and the declared calibration/validation/audit pipeline. The
preflight establishes kernel compatibility, not a safety-prediction gain.

The fixed fit implementation is committed at69ad9a8 before fitting. Launch:

```powershell
python Threemool/Lee/experiments/e58/fit_upper.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --parent-freeze Threemool/Lee/experiments/e42/environment_freeze.json --preflight results/e58_zero_preflight.json --out results/e58_conditional_upper
```

The job runs in the threemool Python environment with OMP, OPENBLAS, MKL and
NUMBA thread counts fixed to1, four fitting threads, CPU only. It authenticates
the preflight inputs, split separation and copied backbone hashes before
fitting. fit_summary.json is written only after all15 heads and numeric/order
checks complete. An existing output directory is never reused. Inspect actual
process liveness before resuming; a partial output folder is not a live job.

All15 heads finish and fit_summary.json records zero fast/sklearn discrepancy,
coordinate-order discrepancy2.22e-16, exact central backbone predictions and
unchanged non-DO endpoints. This is implementation validation only.

The evaluator launches under95ac42b with all numerical library threads1:

```powershell
python Threemool/Lee/experiments/e58/evaluate_upper.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --runs results/e58_conditional_upper
```

It writes progress for all20 inner conditions and fixes selection.json before
validation. Existing evaluation manifests prohibit accidental duplicate runs.
It does not decide adoption without the independent audit and additional event
criteria. After it completes, run the independent five-member full-day check:

```powershell
python Threemool/Lee/experiments/e58/audit_upper.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --runs results/e58_conditional_upper --out results/e58_conditional_upper/audit.json
```

After the independent audit passes, reconstruct candidate and all four fixed
references and apply every predeclared point/Brier/ranking/onset condition:

```powershell
python Threemool/Lee/experiments/e58/audit_events.py --data kaggle_data/darrow_15min.pkl --runs results/e58_conditional_upper --freeze Threemool/Lee/experiments/e42/environment_freeze.json --audit results/e58_conditional_upper/audit.json --e51 results/e51_blended_pit --out results/e58_conditional_upper/event_audit.json
```

The event audit authenticates all candidate/reference weights and splits,
reproduces old/persistence/e40 reference metrics and candidate/e51 Brier values,
and uses the original10000 circular five-day bootstrap resamples. Initial
ongoing events are excluded from onset matching. Every condition is retained
in the decision, with no validation-driven parameter change. Evaluation is
still running; neither independent/event audit has executed yet. No agent is
trained on e58 and no final test is evaluated.

The stochastic bank remains local and must be excluded from shared archives.
After restoring the completed summary and six authenticated model files, it
can be reconstructed from calibration data only:

```powershell
python Threemool/Lee/experiments/e58/rebuild_bank.py --data kaggle_data/darrow_15min.pkl --runs results/e58_conditional_upper --out results/e58_conditional_upper/pit_bank_rebuilt.npz
```

The helper checks model/data hashes and split separation, reproduces all four
fixed banks, selects the already recorded alpha and compares arrays exactly
if the original bank is available. It reports the array checksum and refuses
to overwrite its output. Its complete-run execution check remains pending
until selection and summary.json exist; syntax checking alone is not proof
that bank reconstruction has passed.
