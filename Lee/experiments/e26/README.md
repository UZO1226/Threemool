# Lee/e26: Long-correlated NH4 innovations and an expanded scale search

## Hypothesis

Resume the user goal after e25. The e24 inner proper score decreases as NH4
innovation scale reaches its maximum tested value 1. That is an unresolved
search boundary, not an external blocker. Its four-step residual blocks also
cannot preserve longer NH4 residual episodes. Examine larger scales and
within-day blocks before concluding that stochastic calibration cannot work.

Keep e24's corrected 40-epoch weights, five fitting seeds, normalization, bias
and DO innovation scale 1 fixed. Vary only NH4 noise: scales 0,.5,1,1.5,2,3,4
and block lengths 1,4,16,48,95. Keep DO residuals from the original four-step
paths. Thus this diagnostic deliberately relaxes cross-output noise timing
while keeping the slow NH4 and fast DO hypotheses separate. NO3 noise remains
zero. Blocks never cross a calibration date. Reusing inner residual data for
selection is exploratory, not out-of-sample noise calibration.

Select by the existing normalized CRPS plus NH4 and DO Brier score on the last
30 training dates only. Fitted weights use only the first158 training dates.
The official development40 is reused exploratory audit, not fresh confirmation.
Final test41 is unused. Check the 4-step/scale1 reconstruction against e24,
including identical random draws, before scoring the selected candidate.
Retain the e16 adoption gate without relaxing any threshold. Report the
selected result, the e24 control, 16-draw sensitivity, separate event Brier,
precision/recall at probability .5, event runs and probability reliability.
Do not use any of these extra diagnostics to select on development labels.

Full-day horizon is 95 transitions, air bound0 to9000 m3/h, concentrations
mg/L, NH4>4 and maxDO>2, reward negative normalized air and summed indicator
cost remain unchanged. Monotone ordering uses common noise and .75/1.25 air
probes. A gate failure leads to another justified hypothesis, not goal completion.

Data hashes, CPU packages, original FCSRL source commit and preparation follow
e24 and Compare/PROTOCOL v1.1, except the documented environment experiment.
No new policy-training seeds are represented by the five model members.

```powershell
python Threemool/Lee/experiments/e26/correlated.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --runs results/e24_fixed_initialization --out results/e26_correlated
```

Use CPU and OMP_NUM_THREADS=1, OPENBLAS_NUM_THREADS=1, MKL_NUM_THREADS=1.
The runner records the source commit at start. Large residual paths remain
outside Git; compact selected metrics and reproduction scripts are versioned.
