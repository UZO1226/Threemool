# Lee/e34: Joint recursive and teacher-forced fitting with persistent context

## Prospective interaction test

e32's persistent context improves inner proper score and DO frequency fidelity,
while e31's local supervision improves NH4 frequency but still fails. Those
partial effects are not additive evidence. Test a defined interaction rather
than declaring a combined success. The e33 kinetic prior also fails the original
gate; its actual diagnostics are recorded separately, not used to tune weights.

Use the e32 architecture and original158/30 training split. Change only its
training loss to .5 original fully recursive day loss plus .5 original loss
on teacher-forced transitions. Unlike e31's independently initialized local
samples, teacher forcing here carries the two simulated hidden coordinates
through the entire day, replaces only the five observed physical states with
training observations before each step, and holds the pre-episode context
fixed throughout. This keeps the persistent forcing context consistent between
the two losses. Future observed states are training supervision, never supplied
to development rollouts or policy environments.

Keep the original .995 cap, five fitting seeds,80-percent subsets, batch16days,
Adam .001,80epochs, common20/40/80 inner selection, bias calibration, four-step
innovations,16-scale search and e16 adoption gate. Do not combine kinetic
scaling, tail weights, alternate partition or continuation. The .5/.5 coefficient
is fixed before execution, with no development tuning. Development40 remains
reused exploratory audit and test41 remains unevaluated.

Before fitting verify full-day and teacher-forced predictions agree at step0,
targets align exactly, later predictions differ, the held context is exactly
the original initial context and the teacher-loss gradient is finite/nonzero.
Independent cached/Torch audit and separate event/probability/uncertainty checks
remain required. Matching average frequency alone does not justify adoption.

The architecture is identical to e32, so audit loads --model-kind context.
These remain model-fitting seeds, not policy seeds. Data/source hashes,
original FCSRL commit, CPU packages, physical units, horizon, reward, cost and
action limits follow e24/e32. No new agents until an environment passes.

```powershell
python Threemool/Lee/experiments/e34/check_teacher.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out Threemool/Lee/experiments/e34/teacher_check.json
python Threemool/Lee/experiments/e34/joint_teacher.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --out results/e34_joint_teacher
```
