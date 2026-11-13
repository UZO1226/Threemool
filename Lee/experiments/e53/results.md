# Full-range training-only feature probe

Source9fcf818 generates89300 training-only transitions across188 dates and
five frozen e40 members, with one fixed-noise episode per member/date and
uniform[-1,1] actions. Thirty fixed HGB fits compare three feature sets.
No validation or final test episode and no policy learning is performed.
audit.json reconstructs strict lags, initialization, previous air, saved
probabilities, event metrics and date intervals. Maximum probability error is0.

| Held-training metric | Current only | Previous observation | Repeated current |
| --- | --- | --- | --- |
| NH4 Brier | .000466773 | .000424048 | .000466773 |
| DO Brier | .011735516 | .011522763 | .011735516 |
| NH4 AP | .988300 | .990731 | .988300 |
| DO AP | .967366 | .969168 | .967366 |

History-minus-current NH4 Brier difference is-.000042725,
five-day training-date interval[-.000094347,-.000002753]. DO difference
is-.000212753, interval[-.000392134,-.000023556]. Repeated-current predictions
are exactly current-only predictions for these fixed trees; both history
contrasts therefore coincide. This is a small feature-prediction gain inside
a known simulated law, not a causal estimate of policy-network capacity or a
controller improvement. Model/probe seeds are not policy seeds.

There are only4 new NH4 events among14166 currently-safe held-training rows.
Both current and history classifiers recall2/4 at.5; new-event AP rises.557986
to.577652. DO has168 new events among13089 eligible rows: AP.674579 to.695472,
and recall74/168 to77/168. Do not infer general NH4 onset skill from four cases.

The full-range mean air4500m3/h is high relative to the existing baseline's
roughly2190m3/h. Before generating further outcomes, the README specifies a
single lower-air coverage diagnostic with uniform[-1,0] actions and all other
settings unchanged. Both reports must remain visible. This follow-up addresses
coverage, not a validation search for favorable policy performance.

The frozen model was itself calibrated using the policy training period.
Holding out30 dates here only separates probe fitting, not model-law estimation
or real-plant generalization. Existing DO forecast limitations remain. No
environment replacement, action selector or FCSRL extension has been verified
by this supervised result. A future policy trial needs its own controls,
five final policy seeds and the unchanged matched-air criterion.

summary.json and audit.json retain counts, members, dates and diagnostics.
artifacts.json identifies the draft model/log archive with verified size/SHA256.
Generated arrays and original data are excluded. Re-running probe_history.py
regenerates local arrays for the independent audit.

## Completed fixed lower-air follow-up

Source2af8bfc also completes the prospectively fixed uniform[-1,0] range.
low_air_audit.json independently reconstructs all probabilities and intervals
with zero error. Both ranges retain the same188 training dates, five members,
fixed draw, seed5301, three feature sets and30 model fits each. No validation,
test or controller training is performed.

| Lower-air held-training metric | Current only | History | Repeated current |
| --- | --- | --- | --- |
| NH4 Brier | .010611026 | .009339846 | .010611026 |
| DO Brier | .001270189 | .001183774 | .001270189 |
| NH4 new-event AP | .518012 | .625155 | .518012 |
| DO new-event AP | .039810 | .051576 | .039810 |

History-minus-current NH4 Brier difference-.001271181 has interval
[-.001678447,-.000868027]. DO difference-.000086416 has interval
[-.000242481,.000002882], crossing zero. Repeated-current again predicts
exactly the current-only outputs. The result supports testing history as an
additional input, without establishing a controller effect or real-plant skill.

There are183 new NH4 events among10075 eligible rows; recall rises86/183
to96/183. Only3 new DO events occur among14202 eligible rows, with recall0
for both methods. Thus the two fixed action ranges expose different risk
coverage. Keep their reports separate and do not pool them as independent
dates or select an action range by policy outcomes.

low_air_summary.json and low_air_artifacts.json preserve the complete second
report and verified model/log archive. The main simulation remains frozen.
A full FCSRL trial, with a same-size repeated-current control and unchanged
five-seed matched-air criterion, is specified separately in e54 before training.
