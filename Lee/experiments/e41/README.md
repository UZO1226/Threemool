# Lee/e41: Longer causal operating history

Status: all five model members, unchanged gate, independent reconstruction and
fixed event audit are complete. The adoption gate fails. See results.md and
the compact JSON reports. No controller is trained on this candidate.

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

## Execution rationale after frozen-policy studies

The earlier e40 point-gate pass does not settle its event-timing limitations:
DO AUC.48475 and AP.01149 are near chance, and no DO onset is matched within
an hour at probability threshold.5. These forecast limits were measured
independently of policy outcomes. e43/e45/e46/e48 do not establish a complete
policy-extension improvement, but those failures must not be used to select a
more favorable simulator. e40 remains the frozen historical policy environment.

Now execute the previously prepared e41 forecasting hypothesis as a separate
model experiment. Keep the e34 control,158 fitting/30 inner-calibration dates,
five model-member seeds,80 epochs,20/40/80 shared-checkpoint selection and
16 inner noise settings unchanged. The existing e16 gate is not relaxed.
Compare reconstruction, errors, frequency, Brier and uncertainty with e34,
e40 and persistence. Event diagnostics retain threshold.5 and onset tolerances
0/1/4 steps from the existing audit. Point-frequency gates alone cannot justify
replacing e40. No policy training or new environment freeze is automated.

context_recheck.json reproduces the earlier causal/gradient/independent rollout
checks. A further preflight also verifies every fit/calibration/validation
episode has finite strictly past initial context and that the history table
ends at validation, before fitting. The operating-history assumption and
extra-capacity caveat above remain. Model-member seeds are not policy seeds;
date-block intervals describe exploratory reused validation only.

The launched training source commit is982124f. Its output folder is
results/e41_long_context and console log is results/e41_long_context_run.log.
The same shell runs e24/audit.py with model-kind long_context after training
success, reconstructing selected forecasts independently. Training model seeds
0..4 are sequential and distinct from policy seeds. Do not duplicate this run
based on a quiet log. No checkpoints or raw inputs enter Git.

After reconstruction, execute the fixed event audit:

```powershell
python Threemool/Lee/analysis/audit_long_history_events.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --runs results/e41_long_context --control results/e34_joint_teacher --freeze Threemool/Lee/experiments/e42/environment_freeze.json --out Threemool/Lee/experiments/e41/event_reference_audit.json
```

It reconstructs selected e41 and e34 checkpoints with hash checks, verifies
date/target identity and reproduces e40/persistence Brier, AP and AUC before
reporting e41 comparisons. Threshold.5, onset tolerances0/1/4 and10000 circular
five-day paired Brier intervals match the earlier event audit. Onset/ranking
diagnostics do not select epochs or tune probabilities on validation.
