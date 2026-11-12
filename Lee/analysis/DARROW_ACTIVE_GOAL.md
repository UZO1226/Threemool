# Active DARROW research goal

The user explicitly resumed the full objective. Prior e16 to e25 validation
failures are evidence for further experiments, not sufficient goal completion.
No agent improvement over FCSRL has yet been verified. The exploratory
environment point-gate result and its remaining limits are recorded below.
No genuine external condition currently prevents useful work.

## Latest execution status, 2026-11-13

The controller previously reported usageLimited and the user subsequently
reactivated the goal. That interruption was not a successful research outcome.
The user explicitly requested that work resume.

Current policy checkpoint: both original-model and frozen-model FCSRL have
completed all five401280-transition seeds. Frozen-model PPO is still running.
Split-supervision seed0 is complete and capacity-control seed0 is running in
results/e43_first_pair. The other eight extension jobs remain required.
The first actual matched-air split/FCSRL seed0 comparison is running in
results/e44_split_first. Do not infer improvement from a single seed.
All ten cross-model evaluations of the five original-model policies complete
in results/e44_old_policy_cross, and their own-environment replay reproduces
stored final metrics. Report and capacity-contrast checks reject missing seeds,
smokes, mixed references and excessive actual air differences.

e40 has now completed and passes all original point-estimate adoption gates.
Its independent sklearn/full-day interpolation reconstruction passes. Selected
pit_b95_a0.5 forecasts NH4/DO violation frequencies13.4993/.7211 percent versus
observed17.3158/1.1579. Normalized MAE.775476 and NH4 MAE1.977722 both improve
over the old model. Four members pass and sixteen-draw sensitivity retains
ensemble frequency tolerances. Rare DO event timing remains weak and date-bias
intervals extend beyond the point tolerances. Treat this as a frozen exploratory
research simulator, not independent or real-plant validation.

The additional frozen event-reference audit improves NH4 AP.30003 to.42675
and both Brier errors versus the old model, but NH4 onset recall within one
hour drops10/89 to3/89. DO AP.01149 and AUC.48475 are near chance and no new
DO onset is matched. DO Brier is worse than persistence, with date-block
difference interval[.000046,.004223]. Do not equate passing average-frequency
gates with validated onset prediction. Keep these negative results in any
policy conclusion and do not change the model midway through a paired study.

e42 freezes exact checkpoints, PIT bank and inputs. Its20-episode full-day
adapter check agrees with the forecast runner; noise, terminal observations,
reward/cost/air and test rejection checks pass. A distinct historical terminal
reward/air alias is independently reproduced and corrected in both new-model
and original-model wrappers. Historical code/results remain preserved. Paired
old/new FCSRL baselines must both use this correction and cost limit12.
Both one-update original FCSRL smokes complete, with6080 transitions each,
checkpoint saving and validation in both simulators. Full paired baselines
are in results/e42_baselines: five original-model FCSRL seeds have completed,
followed by five frozen-model FCSRL seeds and five frozen-model PPO seeds, each
401280 transitions and cost limit12. Poll the live runner or child processes,
not this record, before deciding whether they are still running.

The five completed original-model policies were archived with full logs and
checkpoints, no raw inputs or stochastic banks. Draft release id443333191,
tag lee-e42-e44-policy-comparison-20261113, asset740540715 has verified remote
byte count and SHA256. See e42/old_baseline_artifacts.json. e44 native checkpoint
replay and source-hash checks pass on the completed short smokes, which the
report rejects as performance evidence. Cross-model evaluation of all five
completed original policies is in results/e44_old_policy_cross. The actual
matched-air final-policy comparisons and five-seed reports remain outstanding.

e43 implements constraint-wise feasibility supervision plus an independently
initialized two-head summed-cost capacity control. Its hand-computed boundary,
common initialization/RNG, original-loss limit, gradient, EMA and serialization
checks pass. Both one-update smokes complete in results/e43_smoke. Full split
seed0 is running in results/e43_first_pair, with full capacity-control seed0
queued in the same one-worker runner. This uses the sixth CPU core while the
five baseline workers run. Seeds1..4 for each extension variant are still
required; launch them into a fresh result folder when baseline workers free
the available cores, and do not duplicate the seed0 jobs. e44's air-only
selector checks pass, but actual matched-policy evaluation and paired seed
analysis are still required. No agent improvement has been established.
Do not modify the e42/darrow Python
sources while the baseline runner is active: their hashes are locked and later
job launches deliberately fail if the implementation changes.

e41 remains a prepared alternative and is not fitted while the
validated environment is frozen for agent comparisons. Earlier statuses below
are historical and must not trigger duplicate runs.
Ordinary tool execution resumed e40 from committed source d31b068. Its actual
Python process started and reported completed median fits. Do not infer ongoing
execution from this historical entry; inspect the live process or tool session.

e34 to e37 completed without passing the original adoption gate. The joint
teacher/recursive model predicts NH4 violations9.11 percent, proper calibration
7.66 percent, and the two causal innovation variants5.86 percent, against actual
17.32 percent. Retain the independent audits and failed results.

e38 conditional quantiles predict NH4 frequency16.43 percent but DO2.69 percent
against actual1.16 percent. Recursive air ordering fails, despite exact agreement
between fast and sklearn inference. An independent synthetic example reproduces
monotonicity violations in quantile leaf updates. e40 restores ancestral split
bounds during fitting only and repeats e38 with unchanged selection and gates.
The synthetic correction removes the observed reversals; this does not establish
that the full wastewater model is suitable. Its full experiment is pending.

e39's implicit saturating NH4 reaction predicts violations4.57 percent and also
fails adoption. Its independent fitted-model audit passes reconstruction and
reports NH4 bias95 percent interval[-15.69,-9.41] percentage points. No new policy
training has occurred, no final test outcomes were used, and no improvement over
FCSRL has been established. Earlier in-progress descriptions below are historical.

e41 is prepared as a distinct causal-history hypothesis if e40 fails. It adds
strictly past24h/72h/7d summaries to the e34 joint model, preserving its initial
prediction through a zero output head. It requires direct-slice, future-row
scrambling, common initialization/RNG and independent rollout checks before
execution. Preparation is not a fitted result, and its added capacity needs a
matched control before any successful effect is attributed to longer history.

![Recorded-air safety frequency audit](figures/darrow_environment_progress.png)

The plot shows ensemble frequency criteria only. Member ranges are not confidence
intervals. All independent forecast-error, member and air-order checks remain
required; an apparently closer average is not environment adoption.

## Completed since resumption

- e26 checks larger NH4 noise scales and longer within-day correlations. Inner
  selection chooses block95,scale1; NH4 frequency6.86 percent remains too low.
- e27 changes only the matrix gate cap with equivalent initial coefficients.
  Inner selection chooses80 epochs; NH4 frequency8.64 percent remains too low.
  Its observed rows stay below1, so amplification was not demonstrated.
- e28 changes the inner date partition only and freezes scoring normalization.
  NH4 frequency7.97 percent remains too low and DO worsens. Not adopted.
- Independent full-day forward/reconstruction, date-block uncertainty and
  separate event-probability diagnostics were run. DO frequency matching alone
  is not an event-accuracy claim. See each experiment's summary and audit.
- Public-source review distinguishes the actual Tilburg process model, the
  researchers' BSM1 paper and a different-plant LSTM methodology. The author-
  hosted ROM presentation was downloaded successfully after browser timeouts.
  No compatible executable Tilburg model has yet been recovered.

## In progress and next decisions

e29 finished all five total320-epoch fits. The total80-epoch control wins inner
selection because additional fitting worsens the proper score despite lower
training error. It is a retained failed continuation, not a reason to end the goal.

e30 completed the factor4 tail-event regression comparison. It predicts NH4
violations6.75 percent and DO2.40 percent against actual17.32 and1.16, failing
both calibration checks. Its gradient and ordinary-loss-limit checks pass, so
this is a failed weighting hypothesis, not the former dead-head defect.

e31 completed joint original one-step/full-day training at fixed .5/.5 weight.
NH4 violation prediction9.75 percent and DO1.975 percent still fail the gate.
One member meets both tolerances; four are required. Step0/target alignment
and finite-gradient checks pass. It is retained as a failed joint objective.

e32's persistent-context model completes, with better inner proper score but
NH4 frequency8.37 percent and just one calibrated member. Independent initial,
full-day reconstruction and uncertainty checks pass; the safety gate fails.

e33 completes the explicit kinetic prior comparison: NH4 frequency8.58 percent
and zero calibrated members still fail. Initial and independent full-day checks
pass. Its result does not establish a calibrated Tilburg process model.

e34 tests joint recursive and teacher-forced losses in the e32 context model,
holding the episode context fixed and carrying hidden coordinates through the
teacher-forced sequence. It changes training only. Independent step0, target,
fixed-context and gradient checks pass before fitting. Poll
its actual live tool session, not this file or the presence of checkpoints.
Output is results/e34_joint_teacher outside Git. Do not start a duplicate run
because a status observation times out.

After completion, run the general e24 audit with context model kind and the original
e17 reference. Retain gate/16-draw sensitivity and separate event metrics, and
archive e31 to e33 separately from completed e26 to e30. e29's audit uses the slow-cap
class and its epoch0 control must reconstruct e27.

A distinct executable next hypothesis if e34 fails is causal pre-episode
innovation assimilation: replay only the last16 already-observed transitions,
using recorded past air and observed past states to estimate a member-specific
mean transition residual. Add a training-calibrated shrinkage of that residual
as a fixed future bias, without reading any observations after episode start.
Use a frozen parent, e24 as the simple first control, and choose NH4/DO gains
on inner30 dates only, preserving all gates and common-noise action ordering.
Check that scrambling future concentration targets leaves the estimated bias
unchanged, and that zero gain exactly reconstructs the parent. The parent
initializer's past-window/time-of-day support and residual persistence need
explicit diagnostics. This is a new hypothesis, not yet a demonstrated fix.

Completed e26 to e33 checkpoints/logs are in a [draft release](https://github.com/UZO1226/Threemool/releases/tag/untagged-a6e857f0f8bb43d06f44).
Remote byte count and SHA256 match the local manifest. Source data, residual
banks and downloaded publications are excluded. The first attachment contains
e26 to e30; the second separately contains e31 to e33. Neither claims to contain
ongoing e34. Draft release id442901392 is stable; GitHub may regenerate its
temporary untagged URL after title/description edits.
If a candidate passes, freeze the exact model/noise/budget and implement the
neural environment adapter before any paired-seed agent comparison. If it
fails, examine the failure and pursue another justified model hypothesis.

Open executable alternatives include a training objective that emphasizes
rare NH4/DO tail events without selecting on development, shape/time-aware
multi-step loss from the reviewed paper, and a temperature-dependent biological
mass-balance model. These are untested hypotheses, not promises of improvement.
Current surrogate failures and unavailable public code do not exhaust them.
Do not terminate the goal while they or other justified steps remain viable.

Preserve official test41, original adoption thresholds and team rules. Changes
remain under Lee/ on lee/darrow-safety-calibration. The environment-model seed
count does not satisfy the eventual five policy-training-seed requirement.
