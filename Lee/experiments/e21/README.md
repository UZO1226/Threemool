# Lee/e21: Forecast horizon and logged-action support diagnostic

This is a diagnostic on the already selected e18/e19 forecast, not a new
policy evaluation or model selection. No fitting to development labels.

Replay measured air and reset model concentrations to observations every
1, 4, 16, 48 or 95 transitions. Keep the same train-derived innovation paths.
Report trajectory MAE and NH4/DO event frequencies and Brier scores. The
95-step case must reproduce e18. Frequent resets deliberately supply later
observations; they cannot evaluate policies that would change those states.
They are only a diagnostic of one-step error versus compounding drift.

Separately fit 5/95 percent logged-air quantiles on the first 158 training
dates from current state, exogenous variables and previous-zone commands.
Use the last 30 training dates for a global 90-percent interval adjustment.
Audit empirical coverage on development observations and fraction of fixed
0.75/1.25 recorded-air probes outside this band. The intervals are descriptive,
not a proof of action identifiability or a conditional coverage guarantee.
Actual policy-generated states are not assessed by this logged-state check.

Original test dates remain unused. Development is reused exploratory data.
Do not convert diagnostic reset results into an RL improvement claim or
silently shorten the common evaluation episode. This cannot replace an
independently validated intervention-response model.
