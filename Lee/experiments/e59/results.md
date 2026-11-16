# Context and observed-state diagnosis

This completed diagnostic does not fit or adopt a model. The initial command
under4313cd0 failed on the stored e51 report path after scoring. The corrected
run reads the preserved repository report, authenticates its setting/Brier
values before scoring and completes all fixed comparisons. The initial
failure is retained in initial_execution_failure.json. Final test rows remain
unused and no per-observation data or probability arrays are published.

| DO metric on validation | e51 observed-state | e51 recursive | e58 observed-state | e58 recursive |
| --- | ---: | ---: | ---: | ---: |
| Brier | .007737 | .014217 | .007537 | .020414 |
| Average precision | .448782 | .011710 | .495948 | .010811 |
| ROC AUC | .747077 | .480338 | .818324 | .425680 |
| New onsets matched within four steps | 2/25 | 0/25 | 2/25 | 0/25 |

The exact recursive metrics, rather than rounded table entries, are preserved
in diagnosis.json. Resetting to observed states improves several descriptive
metrics, but is not a deployable world or policy improvement. Only2/25 new DO
onsets are matched even with observed-state inputs. This does not prove that
long-horizon feedback is the sole cause or that noise conditioning will solve
the problem. No statistical significance is claimed from these mean contrasts.
NH4 observed-state Brier is identically.026043 for e51/e58, consistent with
their unchanged NH4 backbone and fixed noise at the same actual current state.

Actual DO next-step prevalence is4.257162 percent in fit158,10.736842 percent
in calibration30 and1.157895 percent in validation40. Temperature median falls
from20.663081 to18.052465 to16.072383 degrees in the same order. Validation
temperature is outside the fit range on12.473684 percent of rows and outside
the calibration range on15.736842 percent. These are descriptive context
differences, not proof that temperature causes the errors. Current air remains
inside both historical feature ranges; range inclusion is not causal support.
All other predeclared marginals and outside-range fractions are retained.

Calibration observed-state DO AP is.321841 for e51 and.657904 for e58, with
Brier.092971/.074999. Calibration reuses PIT-estimation dates and is not
held-out evidence. Do not treat the favorable one-step e58 contrast as rescuing
its failed recursive adoption. No subgroup or parameter is selected here.

The next diagnosis should separate state-error accumulation from temporal
noise organization, using fixed reset horizons and common-noise controls on
calibration before specifying another environment candidate. The existing
e42 freeze and all agent comparisons remain unchanged. Reasonable follow-up
work remains and the research goal is still active.
