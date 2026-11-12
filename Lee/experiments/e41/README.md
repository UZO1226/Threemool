# Lee/e41: Longer causal operating history

## Hypothesis and paired control

e34's local/full-day joint model still underpredicts NH4 violations. Its initial
context covers only4 hours. Unobserved slow operating conditions could require
longer history. This tests an operating-history proxy, not an observed biomass
measurement or recovery of the Tilburg mechanistic model.

Use e34 unchanged as the control. Add strictly past24h,72h and7d summary
statistics for its19 normalized context channels: mean, standard deviation,
last-minus-first value and fraction of the window actually available. Fit
normalization only on the original158 fitting dates. A window ends before
episode start. Keep context fixed throughout all95 predicted steps. Windows
can use preceding recorded dates, including preceding validation observations;
this assumes those observations are available at the episode start. It does
not permit observations from later in that episode. No laboratory series or
final test outcomes enter these features.

The original short context, initializer, transition backbone, loss, five member
seeds, date subsets,80 epochs,20/40/80 checkpoint selection, shared bias bounds,
noise search and e16 adoption gate are preserved. A parallel174-to64-to7
encoder adds an episode forcing offset. Its final layer starts at zero, and
its construction restores the Torch RNG, preserving initial predictions and
common training shuffle draws. Extra capacity is a limitation: a passing result
would require a capacity-matched short-history control before attributing it
specifically to longer history.

Before fitting, independently compare extracted statistics with direct slices,
verify that scrambling all future validation rows does not alter the initial
context, check zero-head equivalence to e34, active new-head gradients and
full-day cached NumPy versus Torch predictions after a nonzero update. Do not
launch until the current e40 experiment's result justifies this next alternative.

Record all failures and date uncertainty. No policy training before full
environment adoption. This is exploratory reused validation, not independent
confirmation. Original FCSRL commit, runtime, reward, cost, thresholds, action
range, concentration interpretation and dataset hashes follow e34/e24 and the
source review. No official protocol changes. Keep raw data and checkpoints out
of Git.

```powershell
python Threemool/Lee/experiments/e41/check_context.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out Threemool/Lee/experiments/e41/context_check.json
python Threemool/Lee/experiments/e41/long_context.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e41_long_context
```
