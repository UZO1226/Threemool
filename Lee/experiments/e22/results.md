# Lee/e22 results

Inner proper scoring selects history 16 and innovation scales NH4 1, DO 1.
The unchanged gate still fails. No agent is trained on this model and test
observations remain unused.

| Development metric | Original delta GBM | Selected latent model |
| --- | --- | --- |
| Normalized trajectory MAE | 1.0261 | 0.9775 |
| NH4 trajectory MAE | 2.4823 | 1.4359 |
| NO3 trajectory MAE | 1.6458 | 2.5587 |
| NH4 predicted frequency (%) | 28.48 | 8.50 |
| DO predicted frequency (%) | 1.80 | 1.69 |

Actual NH4 and DO frequencies are 17.32 and 1.16 percent. NH4 MAE improves
42.2 percent but its event-frequency error is still 8.82 percentage points.
The NO3 error prevents the aggregate-error adoption criterion as well.
Only two structural members pass both frequency tolerances. Member NH4
predictions range 3.85 to 15.23 percent. No member is chosen from these
development results, and model seeds are not policy training seeds.

The past-context causality, positive bounded state matrices and independent
NumPy/Torch rollout agreement checks pass. More air has the required
conditional ordering under common innovations. Learned latent coordinates
are not identified reactor biomass or internal nitrogen measurements.

Fitting loss still decreases at the 20-epoch budget. Training on 16-step
sequences also differs from the 95-step audit horizon. e23 therefore checks
full-day fitting and fixed longer checkpoint budgets before interpreting
this failure as evidence against latent dynamics generally. This changes
environment training, not the original FCSRL algorithm or adoption gate.

## Reproduction

Execution commit e27d459. Same observed source data and first-158/last-30
training split as e16. Strictly earlier observations may initialize causal
context; no future concentration or action enters that initializer. Checkpoints,
full fitting and calibration logs and residuals are under results/e22_latent.
Exact checkpoint hashes and normalization are recorded in summary.json.

```powershell
python Threemool/Lee/experiments/e22/latent.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e22_latent
```

Use a fresh output path. Source observations and residual banks stay outside
Git. Data concentration unit is mg/L; air source documentation is m3/h.
