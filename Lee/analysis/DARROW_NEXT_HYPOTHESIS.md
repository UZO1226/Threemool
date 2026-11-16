# Prospective follow-up, not an executed result

Do not change the ongoing e43/e44 comparisons or omit seed0. Complete all five
split and capacity-control seeds first. Both variants fail the first matched
seed, which does not establish their five-seed effect. No new policy is trained
by this note and no validation threshold is relaxed.

## Measured clues

The failed split pair does not have more zero-air actions or sharper average
changes than its baseline. Logged encoder-sync KL also does not demonstrate a
large single-update EMA shock. Training-only observation probes show mixed
NH4 sensitivities across policies. These are descriptive observations, not
causal identification or a theorem about optimal action direction.

The paper focuses on binary costs. Its probability interpretation requires
additional limiting assumptions, and the bootstrap shortcut is discussed with
that binary premise. Our summed cost can be2, so that theoretical probability
claim does not directly follow for this application. This is not a claim that
the installed code is erroneous for its original tasks. Primary sources:
[ICML publication](https://proceedings.mlr.press/v235/cen24b.html) and
[author manuscript, section4.2](https://arxiv.org/html/2405.11718v2#S4.SS2).

## Candidate if the complete comparison fails

One feasible next hypothesis is severity-aware feasibility supervision in the
original single-head representation. The current binary indicators assign
the same NH4 label to4.1 and12, although the recovery conditions differ.
Raw observations and dynamics already contain the concentrations, so this is
an auxiliary-supervision hypothesis, not a claim that all severity information
is absent from FCSRL.

Preserve actual environment reward, cost c=1[NH4>4]+1[maxDO>2], PID limit12,
observations, network size, fixed environment, five seeds and401280 transitions.
Use a training-only auxiliary target

    c_aux = c + max(NH4/4 - 1, 0) + max(maxDO/2 - 1, 0)

with the installed discounted-max feasibility recurrence, unchanged dynamics
loss and total representation coefficients. The denominators are the existing
thresholds, not validation-fitted coefficients. This retains the same zero-cost
state set and adds violation magnitude outside it. It predicts a different
auxiliary quantity and does not inherit the paper's binary probability theorem.
Do not silently substitute a signed-margin label or a different Bellman operator.

Before fitting, check safe/boundary/violating cases, reset and termination
boundaries, unchanged actual costs and actor observations, identical initial
parameters/RNG, original-label limit, nonzero gradients and serialization.
Compare final policies at the same reference air with the existing five-seed
criterion. Any effect includes the changed target magnitude; do not attribute
it exclusively to information content without another appropriate control.
This candidate remains prospective until the current complete results justify
executing a separate recorded experiment.

## Training-only feasibility preflight

audit_severity_labels.py evaluates17860 observed transitions from188 training
days, excluding initial states. No validation/test observations or fitted
policy are used. There are16005 zero-cost,1708 single-violation and147
double-violation transitions. Within cost1, auxiliary quartiles are1.06036,
1.30480 and1.84214; its range is1.00025 to3.00434. Thus the proposed target
actually distinguishes some severities currently sharing the same cost label.
This does not establish that the extra supervision helps the actor.

The factual maximum is4.32667. Reconstructing the frozen environment's existing
clipping rule from its verified fit-day list gives a conservative new-state
label bound6.04761. Existing distribution buckets are in symlog space, so their
upper endpoint4 corresponds to53.59815 on the original scale. With discount.9,
the maximum recurrence target is bounded by48.23834, including the largest
possible predicted bootstrap. The proposed labels therefore fit the existing
distribution support without changing bucket count or range. This checks
numeric compatibility, not representational accuracy or policy improvement.
Compact evidence: ../experiments/e42/severity_label_audit.json.
