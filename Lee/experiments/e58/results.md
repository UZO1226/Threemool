# Conditional DO upper-tail outcome

The candidate fails adoption. No environment freeze or agent training is
changed. The prospective specification is35b8da3, fit source69ad9a8 and
evaluation source95ac42b. All20 inner conditions selectpit_b95_a0.25 using
calibration only. The final41 test days remain unused.

| Validation quantity | e58 | Observed or reference |
| --- | ---: | ---: |
| Normalized MAE | .817763 | Original point-error condition passes |
| NH4 MAE | 1.997386 | Original point-error condition passes |
| NH4 violation frequency, percent | 13.170395 | 17.315789 observed |
| DO violation frequency, percent | 6.766447 | 1.157895 observed |
| Members passing both frequency tolerances | 0/5 | At least4/5 required |
| NH4 Brier | .129472 | .126486 frozen e40 |
| DO Brier | .020414 | .013327 frozen e40 |
| DO average precision | .010811 | .011579 event prevalence |
| DO ROC AUC | .425680 | Greater than.5 required |
| New DO onsets matched within four steps | 0/25 | At least one required |

DO Brier difference versus e40 is+.007086 with paired circular five-day
bootstrap95-percent interval[.004065,.010512]. Versus persistence it is
+.008835 with interval[.004349,.014511]. NH4 difference versus e40 is+.002986
with interval[-.005815,.013369], failing the predeclared non-inferiority
condition. All seven declared adoption conditions fail. NH4 new-onset matches
are1/89 at tolerance0/1 and5/89 at tolerance4; these do not rescue DO failure.
All model-member frequencies,16-draw sensitivity and comparisons against old,
persistence, e40 and e51 are retained in summary.json and event_audit.json.

The independent audit reconstructs full aggregate scores and separately
rebuilds all95 steps for the first validation date/draw of every member using
sklearn and scalar np.interp. Maximum discrepancy is8.88e-16. Numeric inference,
coordinate order, zero-extension controls and clipped-mixing identity pass.
These checks do not prove absence of every possible implementation defect.
The independently rebuilt calibration bank exactly matches the original;
bank_rebuild_check.json records its array checksum. Source observations and
both stochastic banks are excluded from artifacts.

The post-outcome support diagnostic changes no setting and uses calibration
only at the already selected alpha.25. Of306 observed one-step DO event rows,
167 lie outside every original member's upper support, versus2 after the
conditional upper extension. For26 new events, the counts are4 versus2.
Non-event rows with some upper support rise103 to278. This confirms broader
observed-state support, not correct event probability, timing or a free-running
counterfactual. The prior alpha.5 support audit used a different fixed alpha
and is not numerically interchangeable with this alpha.25 diagnostic.

The broader tail alone does not solve calibrated event prediction under
recursive deployment. This is a failed environment hypothesis, not an agent
comparison. Reused validation, season differences, five model members rather
than policy seeds and unknown plant counterfactuals remain limitations. No
tail strength, upper quantile, criterion or favorable seed is selected after
these outcomes. Further diagnosis should distinguish conditioning of temporal
noise from simple attainable support before another prospectively fixed test.

The complete models and generated reports are archived in
lee-e58-conditional-do-upper.zip. artifacts.json records archive/file checksums
and the existing environment draft release location after remote verification.
The reconstruction commands remain in README.md.
