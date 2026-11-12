# Lee/e29: Continue optimization past the selected training boundary

## Hypothesis and design

e27 selects its final80-epoch checkpoint and fitting losses are still declining.
Failure at that boundary does not establish optimization adequacy. Continue
each of its five epoch80 members for240 additional full-day epochs on exactly
the same seeded80-percent158-date subset. Use a fresh Adam optimizer at .0003
because the original optimizer state was not saved. This is continued fitting
with a lower learning rate, not an exact resumption of the original trajectory.
Do not label320 epochs convergence without evidence from the traces and inner
selection. No e28 partition change or e26 correlation change is combined here.

Save additional-epoch0/80/240 weights, corresponding to total80/160/320 epochs.
Select a single checkpoint by the original inner proper score after unchanged
bias calibration. Include the unchanged epoch0 control. Re-select the original
16 innovation scales on the same last30 training dates. First158 dates alone
fit weights; development40 audits the chosen candidate; test41 stays unused.
Retain model cap1.05, .995-independent monotone response signs, original four-
step noise, architecture/loss/batch16 and physical clipping. The higher cap
was not exercised in e27's development coefficient maxima, which stayed below1;
do not describe it as demonstrated amplifying dynamics.

The runner verifies warm-start checkpoint SHA256 against e27 before training.
The generic full-day runner keeps its original defaults. New optional arguments
control continuation only. At a common checkpoint, the same five model seeds
are compared; they remain distinct from policy-training seeds. Keep the e16
adoption gate and report NH4/DO separately with 16-draw sensitivity. If the
best checkpoint is still the last, document that unresolved boundary instead
of ending the goal. If the control wins, retain the failed continuation.

Units, data hashes, preparation,95-step horizon, reward, cost, action limits,
CPU runtime and original FCSRL commit follow e27/e24. No official protocol
files change. Development reuse makes this exploratory, not fresh confirmation.

```powershell
python Threemool/Lee/experiments/e29/continue_fit.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e29_continued
```
