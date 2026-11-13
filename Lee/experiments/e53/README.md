# Lee/e53: Training-only probe of recent observations for FCSRL

## Prospective diagnostic

The existing policy observes11 current features, whereas its frozen stochastic
environment retains a member and a95-step noise block within each episode.
Recent concentration/action history might help infer unobserved conditions.
This is an agent-observation hypothesis, distinct from e41's environment-history
fit. It is not proof of real-plant memory or that a new policy will improve.

Before a policy implementation, probe the value of one previous observed vector
on the already frozen e40 environment. Generate one95-step episode for each
of188 training dates and five environment members, one noise draw per case,
with uniform random clipped actions using a fixed RNG seed5301. No validation
or final test episodes are generated. Use corrected terminal air/reward and
current public11 observations only. No hidden member/noise/exogenous fields
enter features. Append the previous observed NH4/NO3/DO1/DO2/DO3 and previous
observed prior-air feature. At episode start repeat the current observation;
history never crosses a reset. Target the next NH4 and DO costs separately.

Fit fixed HGB event probes on158 training dates and report the remaining30
training dates. Both subsets belong to the policy training period; the frozen
environment's calibration bank was estimated from that same period. These
are feature diagnostics within a known simulated law, not independent plant
generalization. Keep generated arrays/checkpoints local or in the artifact
draft, excluding source data and banks from Git.

Compare three feature sets: current11 plus action, current11 plus previous6
plus action, and an18-input repeated-current control with current6 replacing
previous6. Fit five date-subsampled model seeds0..4,80 percent of the158 dates,
all five environment members on selected dates,200 HGB iterations, learning
rate.05,15 leaves, leaf40,L2=1, no early stop, weighting or calibration.
No hyperparameter or threshold search. Report both event Brier/AP/AUC and
onsets among currently safe observations, counts and10000 five-day paired
Brier intervals on the30 held-out training dates. Do not count model members,
probe seeds or generated transitions as policy training seeds.

The repeated-current probe adds no new information. Any tree fit differences
can include split-tie effects and do not alone identify policy-network capacity
effects. A future policy experiment would require its own reset/terminal,
initialization/RNG, gradient, serialization and independent replay checks,
same-architecture control and all five final policy seeds under the original
matched-air criterion. It must preserve actual costs, PPO/PID and FCSRL targets.

This probe does not select or replace the model being evaluated in e52.
Finish e52 and choose any subsequent environment solely by environment
evidence, never by favorable controller outcomes. Historical e40 policies and
all failures remain fixed. No full policy training or adoption is automatic.

Status: specified and implemented before generated outcomes. No controller
is trained. Generated input/action/label arrays remain local for independent
reconstruction and are excluded from Git and model/log archives.

```powershell
python Threemool/Lee/experiments/e53/probe_history.py --data kaggle_data/darrow_15min.pkl --reference-summary results/e17_contractive/summary.json --freeze Threemool/Lee/experiments/e42/environment_freeze.json --out results/e53_history_probe
```
