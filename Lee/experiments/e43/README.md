# Lee/e43: Constraint-wise feasibility supervision

## Hypothesis and scope

Within the frozen e40 environment, does separately supervising NH4 and DO
feasibility improve the original FCSRL representation at matched air?
The design follows the team's [extension review](../../../Yoo/analysis/FCSRL_EXTENSION_REVIEW_20261111.md).
This is an implementation and controlled test of that candidate, not a novelty
claim or a change to the team's protocol. Our baseline uses summed cost and
limit12, rather than the review's separate6/6 cost setup.

Preserve shared Encoder, action-conditioned latent transition, one-step
dynamics consistency, distributional discounted-max feasibility targets,
EMA, actor, reward/cost critics, PPO, PID, observations and daily cost limit.
Only representation supervision changes. Each head predicts its own binary
NH4 or DO discounted-max target. Average the two distribution losses so their
total coefficient remains1; keep dynamics coefficient.5. Original63 buckets,
bounds[-2,4], discount.9, EMA.05 and all optimizer settings remain fixed.

Compare three methods: original one-head FCSRL from e42, a two-head control
where both heads receive summed-cost targets, and two-head constraint-wise
supervision. The second head is independently initialized using43000 plus the
policy seed in a restored RNG context. Both two-head variants have identical
initial weights, initial actor and RNG. Their common parent parameters match
the original baseline exactly. An identical cloned head would never exercise
extra capacity under identical labels, so the capacity control uses this
independent initialization. No state-history or actor correction is added.

Each method uses policy seeds0..4 and401280 transitions under exactly the
e42 freeze. Primary comparison uses final policies, native and air-matched
evaluation, separate NH4/DO, and paired seed uncertainty as specified in e42.
Comparisons use the same common eight noise paths per member/date. Extra
heads never change the immediate cost or give the actor a new observation.
The capacity comparison is required before attributing a result to constraint
identity. Negative results are retained and do not become positive evidence
through checkpoint selection or another model's scores.

## Verification and provenance

Before full training, verify reset/termination target boundaries against a
hand-computed example, component labels versus duplicated summed labels,
identical split/control initial parameters and parent RNG, nonzero finite
head/encoder gradients, original-objective equivalence in the duplicated-head
limit, EMA behavior and checkpoint round trips. Then run a single-update
pipeline smoke for both variants. These checks are not policy seeds or proof
of performance. Source attribution is in split_feasibility.py; original FCSRL
commit d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache-2.0, remains unmodified.
The original [license](../../code/darrow/FCSRL_LICENSE) is included in the repository.

e42's terminal accounting correction applies to every method. Frozen model
quality limits, reused validation and final test reservation also remain.
No hyperparameter search is authorized by this experiment's definition; a
different candidate requires a separately recorded hypothesis.

```powershell
python Threemool/Lee/experiments/e43/check_split.py --out Threemool/Lee/experiments/e43/split_check.json --checkpoint results/e43_unit_checkpoint
python Threemool/Lee/experiments/e42/run_policies.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --trainer Threemool/Lee/experiments/e43/train_split.py --out results/e43_smoke --steps 6080 --evaluations 1 --parallel 1 --jobs new:ppolag_fcsrl_split:0 new:ppolag_fcsrl_capacity:0
```

Use fresh output/checkpoint paths when reproducing checks. The full experiment
uses the same runner with400000 requested steps,401280 actual, five evaluations
and all five seeds for each of the two variants. Keep the full frozen original
FCSRL comparison from e42 as the primary baseline.

The first full pair is executed with one additional worker while the five
e42 baseline workers run:

```powershell
python Threemool/Lee/experiments/e42/run_policies.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --trainer Threemool/Lee/experiments/e43/train_split.py --out results/e43_first_pair --parallel 1 --jobs new:ppolag_fcsrl_split:0 new:ppolag_fcsrl_capacity:0
```

The remaining eight jobs are still required. The following is the planned
continuation when baseline workers free CPU resources, not a completed run:

```powershell
python Threemool/Lee/experiments/e42/run_policies.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --trainer Threemool/Lee/experiments/e43/train_split.py --out results/e43_remaining --parallel 5 --jobs new:ppolag_fcsrl_split:1 new:ppolag_fcsrl_split:2 new:ppolag_fcsrl_split:3 new:ppolag_fcsrl_split:4 new:ppolag_fcsrl_capacity:1 new:ppolag_fcsrl_capacity:2 new:ppolag_fcsrl_capacity:3 new:ppolag_fcsrl_capacity:4
```
