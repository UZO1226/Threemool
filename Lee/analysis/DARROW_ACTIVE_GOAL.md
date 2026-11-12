# Active DARROW research goal

The user explicitly resumed the full objective. Prior e16 to e25 validation
failures are evidence for further experiments, not sufficient goal completion.
No environment or agent improvement has yet met the requested verification.
No genuine external condition currently prevents useful work.

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

e33 tests explicit NH4 kinetic temperature scaling informed by the primary
BSM2-Python ASM1 implementation. It is not the calibrated Tilburg process model.
Its15C identity, cold-depletion direction, air signs, observed-head gradients
and full-day Torch/cache agreement pass before fitting. Poll
its actual live tool session, not this file or the presence of checkpoints.
Output is results/e33_kinetic_scaling outside Git. Do not start a duplicate run
because a status observation times out.

After completion, run the general e24 audit with kinetic model kind and the original
e17 reference. Retain gate/16-draw sensitivity and separate event metrics, and
archive e31 to e33 separately from completed e26 to e30. e29's audit uses the slow-cap
class and its epoch0 control must reconstruct e27.

Completed e26 to e30 checkpoints/logs are in a [draft release](https://github.com/UZO1226/Threemool/releases/tag/untagged-5322d2890e2cffbff442).
Remote byte count and SHA256 match the local manifest. Source data, residual
banks and downloaded publications are excluded. e31 and e32 are not claimed
as present in that already-frozen attachment.
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
