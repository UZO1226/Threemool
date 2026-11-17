# e72: Diagnose component-risk trades after failed e71 replication

Prospective diagnostic after the complete failed e71 fresh block,2026-11-17.
e71 reduces DO on all fresh seeds but fails cost and NH4. Its eligibility
uses summed predicted NH4/DO rates, and its budgeted interpolation can select
an action with higher predicted NH4 even when some expanded action improves
the sum. Test this mechanism before specifying another controller.

Replay no environment and fit no new policy or risk head. On all85120 held
captured states of each actual policy seed0 through9, load the exact archived
e71 calibration, original NN and horizon4 regressors. Compare the e71 native
action to the same-state e63 native action. Separately report predicted NH4
and DO increases, crossed trades, joint nonincrease, and current regime.
Reconstruct e71's existing held changed-action and summed-risk-worse counts.
Check1024 deterministic rows with original sklearn predictions.

Also count whether any of the unchanged nine candidates improves the summed
rate while neither component exceeds its e63 predicted rate by more than
the existing numerical tolerance1e-12. Report lower/higher-air opportunities.
This feasibility screen does not implement, fit, select or evaluate a new
controller, and its favorable counts cannot count as measured water-quality
improvement. A later candidate needs its own prospective implementation,
controls, all-seed actual-air comparison and independent trajectory audits.

Use both original and exposed fresh blocks without pooling outcome statistics.
The fresh5 through9 outcomes are now known and cannot be described as an
unseen quality confirmation for a later method. Any new confirmation seeds
must be declared separately. Final test41 remains unused. Preserve all e71
outcomes and unchanged e63 primary references, world, inputs and budgets.

```powershell
python Threemool/Lee/experiments/e72/diagnose_components.py --workspace C:/Users/user/Downloads/IoT_project2 --out results/e72_component_diagnostic
```
