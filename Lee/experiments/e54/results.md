# Complete history/repeated-current policy comparison

All ten full policies finish under source5f901a2: history and repeated-current
policy seeds0..4,401280 transitions each. Both matching batches and reports
finish without interruption. All native checkpoint replays, training-source
hashes, seed/config/budget checks and original freeze identity pass. No test
date is used and no intermediate checkpoint replaces a final policy.

| Matched-air mean | Original FCSRL | History | Repeated current |
| --- | --- | --- | --- |
| Air, m3 per23.75h | 52020.577481 | 52035.617433 | 52037.797158 |
| Summed violation cost | 22.464125 | 23.627375 | 22.150750 |
| NH4 violation, percent | 23.208684 | 24.378289 | 22.857368 |
| DO violation, percent | .437763 | .492632 | .459211 |

Both variants match the reference air for every policy seed within1 percent.
History-minus-original cost difference+1.163250 has paired-seed95 interval
[.174750,2.239000]. NH4 difference+1.169605 percentage points has interval
[.099342,2.490395]. DO difference+.054868 has interval[-.133947,.250395].
Four of five history seeds have larger cost and NH4 violation; one improves.
These exploratory reused-validation comparisons do not establish population
effects after many trials, but contradict a claimed complete improvement.

Repeated-current cost difference-.313375 has interval[-4.941375,4.892500],
NH4-.351316 points has interval[-5.312237,5.402763], and DO+.021447 has
interval[-.455789,.578816]. Every quality interval crosses zero. Do not adopt
the control on its slightly lower mean or omit its unfavorable seeds.

Both variants fail all three predefined quality conditions. The secondary
history-minus-repeated cost interval[-3.683000,5.518500], NH4 interval
[-4.490000,5.788947] and DO interval[-.532895,.613158] also cross zero.
Common-reference checkpoint/air identity and mutual air matching pass, but
no history-specific benefit is supported.

History's native policy uses62397.01m3 versus52020.58 and reduces mean cost
to15.955125, while DO increases to.897763 percent. That native result uses
about20 percent more air and is an energy/safety tradeoff, not the objective's
same-air improvement. Both native and adjusted results stay in the reports.
Offset matching changes the executed actions and their observed history.

The preflight established identical initial actor/value/encoder outputs and
RNG, active new-column gradients, correct optimizer references, serialization
and reset-safe physical equivalence. Smokes and independent native replays
passed. Both variants have406755 trainable parameters, with the repeated-current
variant controlling the added input capacity. No implementation fault was
identified by these checks. That does not prove there can be no other fault.

The training-only e53 feature-prediction gain did not translate into a better
learned controller in this trial. Extra history changes policy/value and
representation learning together; data predictability, auxiliary objectives,
optimization and action allocation are different questions. Do not attribute
the failure solely to one loss or declare past observations universally useless.
No new target, threshold, air matcher, seed selection or validation-tuned
hyperparameter rescues this failed hypothesis.

history_comparison.json, repeated_comparison.json and history_contrast.json
retain all final-policy seed differences, sample SDs and10000 paired-seed
bootstrap intervals. The four artifact indexes preserve all ten checkpoints,
full logs and generated evaluation/air-query outputs in the existing draft,
with remote size/SHA256 checks. Raw inputs and stochastic banks are excluded.
The original FCSRL Apache license is included in policy archives.

Known e40 DO ranking/onset and model-dependence limitations remain. These are
five policy seeds, not1600 independent policy seeds per evaluation. Historical
models and failures remain untouched, and no external blocker prevents further
work. A following hypothesis can apply learned one-step cost predictions
directly to local actions, using only the same collected PPO transitions and
retaining an unchanged base policy. Specify data/sample-budget controls before
implementation. This result does not complete the active goal.
