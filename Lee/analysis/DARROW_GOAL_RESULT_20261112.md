# DARROW environment and extension investigation

## Decision

An improvement over base FCSRL has not been verified. The current stopping
condition is a demonstrated validation limit: no selected environment passes
the safety calibration gate needed to interpret a same-air controller comparison.
Corrected models can improve average factual prediction error while seriously
underpredicting NH4 safety events over a full day. Training more policies in
those environments cannot distinguish a genuine control gain from exploiting
that modeling error. This is a limit of the present evidence and evaluation
setup, not a proof that DARROW or every possible model cannot work.

The prerequisite phase was completed through e16 to e25. The dependent phase
of freezing an adopted environment and training a new multi-seed agent extension
was not reached. It is explicitly not reported as completed or successful.
Existing e15 results already include five policy seeds per method; the new
five-member environment fits are different and cannot substitute for that test.

## What changed and what failed

First158 training days fit models and last30 calibrate/select. The 40 development
days are exploratory and have been reused, including for hypotheses chosen
between experiments. Official 41-day test outcomes remain unevaluated. The
95-transition episode, two separate NH4/DO thresholds, action bound and team
protocol are preserved. Air is measured in m3/h and concentrations in mg/L.

| Experiment | Change | Result |
| --- | --- | --- |
| e16 | Signed cooperative state and air response, output bias | Recursive order repaired, safety calibration failed |
| e17 | Contractive autoregression with exogenous tree forcing | Lower forecast error, NH4/DO violations too rare |
| e18 | Cross-output, within-day residual innovations | Better probabilistic score, persistent NH4 underprediction |
| e19 | Linear cold-temperature continuation | Inner score selected original model |
| e20 | Learned coupled contractive state response | NH4 underprediction remained |
| e21 | Reset-horizon and logged-air support diagnostics | Accumulating drift and weak support for large air probes |
| e22/e23 | History-conditioned latent model and full-day training | Historical defect found, conclusions about model family invalidated |
| e24 | Repair NO3 initialization and repeat identical full-day design | DO frequency close, NH4 unsafe calibration still failed |
| e25 | Initial-NH4 persistence anchor | Inner score selected zero anchor, no adoption |

The NO3 defect is our implementation error, not a dataset property. A deliberately
broken control reproduces its zero prediction/gradient, and a regression check
verifies the repair. Old checkpoints are retained with the defect label. The
corrected run also passes independent full-day Torch/NumPy agreement and saved
score reconstruction.

## Quantified current limit

| Quantity | Persistence | Original delta GBM | Corrected e24 | Actual |
| --- | --- | --- | --- | --- |
| Normalized recursive MAE | .9639 | 1.0261 | .8725 | |
| NH4 MAE (mg/L) | 2.3361 | 2.4823 | 1.5135 | |
| NH4 violation (%) | 30.00 | 28.48 | 5.87 | 17.32 |
| DO violation (%) | .00 | 1.80 | 1.27 | 1.16 |

NH4 underprediction is 11.45 percentage points, with exploratory five-day
block bootstrap 95-percent interval [8.02,14.49] points. All five e24 members
predict 4.57 to 6.56 percent and fail the NH4 tolerance. Sixteen innovation
draws still predict only 6.07 percent. Resetting from measured states every
step gives 15.14 percent, but chaining the same model for 95 steps gives 5.87
percent. Later measured resets cannot be supplied as if they were responses
to a new control policy.

DO aggregate frequency improves substantially, but DO Brier .01272 is still
worse than the .01158 no-event persistence reference. Frequency alone does
not establish that the right events are predicted. Both original and simple
references, proper scores and date uncertainty appear in e24/audit.json.

Twelve and a half percent of development timestamps are colder than the full
training range. Initial NH4 exceeds 4 on 3.33 percent of inner calibration days
and 30 percent of development days. These are measured distribution changes,
not proof of the unique physical cause. More epochs were not proved converged,
and no claim is made that further optimization or another model is impossible.

The training-only logged-air interval diagnostic covers 85.32 percent of
development recorded air. Probes at .75 and 1.25 times recorded air fall outside
its estimated interval in 94.71 and 69.32 percent of cases. This is descriptive,
not a proof of zero action overlap. Crucially, logged outcomes do not provide
independent quality labels for the unobserved alternative air actions. A
monotonic architectural prior fixes response direction, not response magnitude.

Source review additionally found that DO1/2/3 correspond to air3/4/5, that
dynamic influent is generated ASM1 input rather than online measurement, and
that NH4 is measured before the final clarifier. Its real-time availability
and mapping must be explicit before claiming a deployable controller. The
historical official baseline is preserved with its mapping limitation recorded.
See DARROW_SOURCE_REVIEW_20261112.md and the referenced team source manifest.

## Agent effects are separate

e15's risk-sampling extension reduced air by roughly 12.7 percent relative to
base FCSRL but increased NH4 violation from 8.81 to 12.21 percent in its old
surrogate. It was not a same-air improvement, and intervals included no effect.
Those are conditional simulator results, not observed treatment quality. No
e16 to e25 environmental gain is attributed to that agent or used to relabel
its failure. The previously approved e14/e15 draft-release archive remains
unchanged.

## What would reopen a credible comparison

The material next input is either a treatment-process simulator with a checked
DARROW sensor/action mapping and independent aeration-response validation, or
additional provenance-checked cold-period training and separate development
observations with known available inputs and useful action variation. A newly
specified model could also be investigated, but repeated fitting to the same
development period would remain exploratory and would not create independent
response validation. The final test must not be repurposed to solve that gap.

After an environment actually meets the recorded gate, freeze its weights,
noise process and training-only budget; preregister an agent extension; use
paired policy-training seeds and matched-air comparisons on the same environment;
report NH4 and DO separately with uncertainty. Revisit environment and policy
changes as separate factors. Until then a claim of beating FCSRL is unsupported.

Code, configurations, failed summaries and checks are in the personal work
branch lee/darrow-safety-calibration. Model artifacts and optimization logs
are separate draft-release attachments with SHA256 manifests; raw DARROW files
and residual banks are excluded. Source data must be obtained through the
team's authorized route to regenerate residual banks.
