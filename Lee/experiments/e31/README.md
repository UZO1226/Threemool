# Lee/e31: Joint one-step and full-day training

## Hypothesis

Every recent neural environment is trained only on recursively simulated
full-day trajectories. Short-horizon DO errors remain high and event timing
is poor even when aggregate frequency is closer. e30's rare-event weighting
also fails the adoption gate. Test whether local transition supervision is
missing from the full-day objective, rather than increasing the same loss's
epoch count or rescaling predicted event frequencies.

Use .5 original full95-step loss plus .5 original one-step loss evaluated at
every observed training transition. Initialize each one-step sample's hidden
state from its strictly past four-hour observed context. Full-day training
still carries simulated latent states throughout, so both open-loop consistency
and observed-state local response have explicit gradients. No future observed
states are injected into development rollouts or policy environments.

Keep e24's architecture, .995 matrix cap, source/date split158/30, five fitting
seeds and80-percent date subsets, Adam .001, batch16days, 80epochs and shared
20/40/80 inner checkpoint selection. Keep the original four-step innovations,
16-scale search, bias calibration and adoption gate. Do not combine e30 event
weights, e27 cap, e28 partition or e29 continuation. The .5/.5 weight is fixed
prospectively; no development tuning of loss weights. Development40 remains
exploratory reused audit; test41 remains untouched.

Before training, independently check that teacher-forced step0 agrees with
the full-day step0 for the same physical initial state, exogenous inputs and
recorded current/previous air. Later predictions should differ when observed
states are reintroduced. Check a finite nonzero teacher-loss gradient. Verify
date and shape alignment through the original sequence builder and Data.past.

Units, horizon, reward/cost/action limits, source/data hashes, original FCSRL
commit and CPU packages follow e24. No policy training on a failed environment.
Report NH4 and DO frequency, timing, probability and uncertainty separately.
Failure leads to further justified investigation, not goal completion.

```powershell
python Threemool/Lee/experiments/e31/check_auxiliary.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out Threemool/Lee/experiments/e31/auxiliary_check.json
python Threemool/Lee/experiments/e31/joint_horizon.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e31_joint_horizon
```
