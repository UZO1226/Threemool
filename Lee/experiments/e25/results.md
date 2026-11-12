# Lee/e25 initial-anchor result

Execution commit 356c267. The inner proper score selects weight zero, retaining
e24 unchanged. The score is .67460 at zero, against the remaining candidate
values preserved in summary.json. No development labels choose the weight.
The zero endpoint exactly reconstructs e24, including the 11.45-percentage-point
NH4 underprediction and zero calibrated members. The unchanged gate fails.

The full anchor endpoint is a persistence control for NH4 only. Its development
frequency is 30 percent, but that does not authorize choosing a nonzero weight
after seeing development outcomes. In the inner period, days starting above
the threshold are only 3.33 percent versus 30 percent in development. This
observed shift helps explain why an apparently appealing development correction
is not selected by the available training-only criterion. It is not proof of
the unique cause of drift or that every history model must fail.

No new policy is trained, and no same-air agent improvement is claimed. Original
test observations remain unused. Reproduce with the same runtime and source
hashes as e24:

```powershell
python Threemool/Lee/experiments/e25/anchor.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --runs results/e24_fixed_initialization --out results/e25_anchor
```
