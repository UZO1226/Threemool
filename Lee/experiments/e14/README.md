# Lee/e14: DARROW environment audit and direct state prediction

## Preregistered procedure

The existing delta GBM is compared with persistence and two direct next-state
GBM ensembles (7 and 15 leaves, 200 boosting iterations, learning rate 0.05,
minimum leaf size 40, L2 1). Each ensemble has five members trained on the
same reproducible 80 percent subsets of training days' transitions as e07.
Direct models constrain their own current concentration to be nondecreasing
and retain e07 air monotonicity assumptions. These are assumptions, not
identified causal effects. Existing data, state, action, reward and cost remain.

Selection uses validation only. Test dates remain sealed until selection and
controller settings are frozen. Report next-state and delta metrics separately.
Primary metric is mean of per-state daily rollout MAE divided by training
state standard deviation. Adopt a candidate only if its primary metric improves
at least 10 percent over old GBM, NH4 MAE does not worsen, and no other state's
MAE worsens by more than 10 percent. Persistence is a diagnostic, not an
eligible control environment because it has no action response.

Report single-step accuracy, recursive 95-step error, constraint frequency
calibration and action sensitivity. One-step monotonicity does not establish
daily trajectory monotonicity or real process validity. Start from observed
daily initial states; all subsequent states are recursively predicted.

## Planned agent comparison

Freeze the selected environment, calibrate cost limit using training replay
only, and retrain baseline FCSRL plus PPO-Lag and an extension using matched
seeds and budgets. Select checkpoints on validation. Evaluate the final frozen
controllers on test once, without tuning afterwards. An environment change
and an agent extension must have separate comparisons.

Artifacts remain outside Git in results/e14_audit and kaggle_data. Small JSON
reports, source hashes, commands and conclusions are versioned here.
