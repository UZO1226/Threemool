# Lee/e19 results

The 100-combination inner comparison selects original:n1:d1, the same mean
and innovation settings as e18. None of the four linear-temperature variants
is chosen. The expansion to NH4 innovation scales 1.5 and 2 also does not
displace scale 1 on inner proper score. The unchanged safety gate fails.
No alternative is selected by looking at its development performance.

Temperature coefficients were estimated only on fitting dates. The full
multiplier-1 slopes are between -0.01045 and -0.01270 mg N/L per 15-minute
update per degree, before the stable autoregression and subsequent bias
calibration. These are fitted predictive coefficients, not measured causal
nitrification constants. Fast inference including temperature extrapolation
matches sklearn in every tested member. Reused component trajectories match
the selected full joint simulation, as required by the diagonal structure.

The selected development results therefore reproduce e18: NH4 frequency
5.64 percent versus 17.32 percent observed, DO 1.77 versus 1.16, and NH4
trajectory MAE 1.751. Sixteen-draw sensitivity remains in summary.json.
No new agent training is warranted. Test observations remain unused.

The lab-file header inspection found influent COD/BOD5/TKN/Ntot/TSS/Q and
effluent COD/BOD5/TKN/NH4/NOx/Ntot/TSS/Q, not reactor biomass or actuator
response measurements. Those files are not added as model inputs. Their
measurement availability times were not established; contemporaneous lab
targets must not be treated as online observations without that check.

## Next question

The diagonal model removed state interactions to ensure contraction, while
the earlier nonlinear cooperative model kept interactions without contraction.
e20 will test coupled but globally contractive state feedback before declaring
these two requirements incompatible in this dataset. This is not a claim that
temperature has no effect, nor proof that the current data cannot support
any useful predictive model.

## Reproduction

Execution commit eaf2d4b. Same data, dates, packages, CPU threads and action
bounds as e16-e18. Full model and residual banks are local under
results/e19_temperature. Source data and residual banks are not committed.

```powershell
python Threemool/Lee/experiments/e19/temperature.py --data kaggle_data/darrow_15min.pkl --contractive-summary results/e17_contractive/summary.json --out results/e19_temperature
```

Use a fresh output path on rerun. Models, exact hashes, inner scores,
calibration traces and verification outcomes are in summary.json.
