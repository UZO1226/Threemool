# Lee/e54: Causal previous-observation FCSRL and repeated-current control

## Prospective full-policy hypothesis

e53 finds a small training-only history benefit beyond current information,
with stronger NH4 coverage under fixed lower-air actions. This does not prove
an RL effect. Test one previous public observation in FCSRL, rather than another
auxiliary cost definition. The frozen environment has an unobserved member
and temporally correlated noise; recent changes may help infer conditions.
Any benefit can be simulator-specific and is not proof of plant observability.

Keep the existing e42 e40 freeze for this comparison. It passes the original
joint point gate, independent reconstruction and action-order checks. e51's
frequency/onset partial benefit does not dominate its DO probability error;
e52 fails the joint gate and worsens both Briers. Keeping e40 follows forecast
evidence and preserves separation from agent effects, not favorable controller
outcomes. Its unresolved DO ranking/onset and reused-validation limitations
remain explicit. Do not declare the model perfect or the full goal complete.

## Two variants and unchanged control

Use the same11 current observations and append the previous observed NH4,
NO3,DO1,DO2,DO3 and previous observed prior-air feature, giving17 inputs.
At episode start repeat current values. Do not read earlier factual rows,
hidden member/noise, extra exogenous fields or future concentrations.
Maintain history from actual observations after executed clipped actions.
Terminal next-observation history uses the last current observation; auto-reset
next-observation history repeats its new initial observation. Never cross days.

The repeated-current control has the exact same17-input networks but appends
the corresponding current six values, adding no information. Keep both this
control and the original11-input FCSRL baseline. Preserve actual environment
costs/rewards, action limits, PID, feasibility recurrence/discount, projection,
EMA, bucket support, optimizer settings, budgets and reset sampling.

Construct the original11-input agent under each policy seed, then expand only
the observation-input columns in actor/critics and state encoders. Insert six
zero columns, copy all old weights/biases and preserve Torch RNG. Shift the
actor/critic latent columns after the appended observation block. Rebuild
optimizer references to include new parameters; keep fixed-encoder gradients
disabled. Report numerical initialization agreement rather than assuming
bitwise GEMM identity after a width change. Both variants share this procedure.
First11 observation-normalizer updates remain the original computations.

Compute the original-width linear contribution and added-column contribution
separately in the expanded input layers. With zero added weights this retains
the original arithmetic path while allowing nonzero new-column gradients.
Both history and repeated-current variants use the same implementation and
parameter count. Verify output/RNG agreement explicitly and retain any measured
floating-point differences in the preflight report.

Use process-scoped trainer/environment adapters under Lee/e54. Preserve
historical e42 and common source files because their hashes guard prior
checkpoints. Training buffers receive17 features through a scoped dimension
override; the underlying11-input environment is unchanged. Restore patched
globals on exit. Serialize feature mode/dimension and verify it on loading.

## Checks before full training

Require paired real-environment steps to preserve all physical observations,
costs, rewards, air, days and reset/terminal outputs. Check strict history
indices for multiple episodes in both modes. Check common initial parameters,
actor/value/encoder outputs, RNG, new-column gradients, frozen encoder and
optimizer membership. Save/load replay must agree. Run separate6080-step
smokes and independent checkpoint replay; smoke seeds are not full results.
Complete the comparison/evaluation adapters before launching pinned batches.

## Full comparison fixed before outcomes

Train history and repeated-current policy seeds0..4,401280 actual transitions
each, using the existing400000-step request,8 environments,760-step rollouts
and five diagnostic evaluations. All final policies are primary. Reuse the
already complete five original FCSRL baselines only after source/freeze,
budget, seed/config and independent native-replay guards pass. Do not omit
an unfavorable seed or choose an intermediate checkpoint.

Evaluate every final policy on the same five-member/eight-draw/40-date cases.
Reinitialize history for every replay and air query. The existing air-only
offset matcher chooses candidate actions without reading water quality, using
the same brackets, tie rules, query cap and per-seed1-percent tolerance.
History must reflect executed offset/clipped actions, not unexecuted means.
Retain native and adjusted results and all generated query outputs.

Use the unchanged five-seed bootstrap criterion: all air matches pass, cost
and NH4 difference interval upper endpoints are below zero, DO upper endpoint
is at most zero. Report both variants against original FCSRL. A secondary
common-reference history-minus-repeated comparison checks history-specific
attribution with the same paired metrics. No architecture-only or unique
history claim follows from beating the original baseline alone.

Model members/noise draws/dates are not policy seeds. Reused validation and
multiple tested hypotheses remain exploratory. Final test41 stays untouched.
No official team protocol, Paper or other member files change. Preserve raw
data/banks outside Git; save code/compact results under Lee/e54 and checkpoint/
log archives in the existing verified draft. Record original FCSRL commit
d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache license and all modified paths.
The active goal continues through full comparison and diagnosis if it fails.

Status: implementation, preflight and all three6080-step smokes pass. preflight.json
records exact initial actor/value/encoder outputs and RNG, both new-column
gradients, optimizer references, frozen encoder, first11 normalizer statistics,
serialization, scoped-global restoration and190 paired environment steps per
mode. Both variants have406755 trainable parameters. Historical files stay
unchanged. pipeline_check.json confirms all saved-policy native replays,
learned new columns and both air-only matching smokes. Initial collector mean
cost is7.15625 for all three methods. History/repeated air errors are.05116
and.00608 percent, with9/10 callbacks. These are pipeline results, not method
performance. baseline_reuse_check.json validates all five existing full-budget
baseline loads; each final comparison still independently checks native replay.
The comparison uses e44's existing matcher and five-seed report;
history_contrast.py adds the secondary same-reference contrast.

```powershell
python Threemool/Lee/experiments/e54/check_history.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --out Threemool/Lee/experiments/e54/preflight.json
python Threemool/Lee/experiments/e42/run_policies.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --trainer Threemool/Lee/experiments/e54/train_history.py --out results/e54_smoke --jobs new:ppolag_fcsrl:0 new:ppolag_fcsrl_history:0 new:ppolag_fcsrl_repeated:0 --parallel 3 --steps 6080 --evaluations 1
```

All output folders are fresh. Before full training, independently replay each
smoke with native_replay.py and allow-smoke, then compare both variants with
compare.py at the same6080-step original smoke baseline. Do not modify pinned
Python sources while those batches or future full policy batches run. Keep
smoke results separate from the401280-step five-seed performance reports.

The smoke implementation commit is915fe38. Full training runs both variants
from fresh initialization, without warm-starting smoke checkpoints:

```powershell
python Threemool/Lee/experiments/e42/run_policies.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --trainer Threemool/Lee/experiments/e54/train_history.py --out results/e54_full --jobs new:ppolag_fcsrl_history:0 new:ppolag_fcsrl_history:1 new:ppolag_fcsrl_history:2 new:ppolag_fcsrl_history:3 new:ppolag_fcsrl_history:4 new:ppolag_fcsrl_repeated:0 new:ppolag_fcsrl_repeated:1 new:ppolag_fcsrl_repeated:2 new:ppolag_fcsrl_repeated:3 new:ppolag_fcsrl_repeated:4 --parallel 5 --steps 400000 --evaluations 5
python Threemool/Lee/experiments/e54/run_comparisons.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baseline-root results/e42_baselines/new_ppolag_fcsrl --candidate-root results/e54_full/new_ppolag_fcsrl_history --seeds 0 1 2 3 4 --out results/e54_history_comparison --parallel 5
python Threemool/Lee/experiments/e54/run_comparisons.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --baseline-root results/e42_baselines/new_ppolag_fcsrl --candidate-root results/e54_full/new_ppolag_fcsrl_repeated --seeds 0 1 2 3 4 --out results/e54_repeated_comparison --parallel 5
```

After all completed pairs, use e44/report.py for each variant and
history_contrast.py for the secondary comparison. Do not summarize smokes as
five full seeds or stop a variant because its first seed looks unfavorable.

Execution snapshot: source5f901a2 launches all ten full jobs in
results/e54_full, with five history seeds first and five repeated-current
seeds pending. The same live shell queues both five-seed comparison batches,
then both e44 reports and history_contrast.json after successful training.
Do not start duplicate comparison folders while this pipeline is active.
Read actual process/tool state before waiting or restarting. All pinned
Python sources stay unchanged; documentation snapshots are not live monitors.
