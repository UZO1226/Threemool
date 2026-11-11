# Lee/e15: Risk-stratified FCSRL training after environment audit

This is an exploratory personal experiment, not a change to Compare/PROTOCOL v1.
It uses the e14 selected model. No test dates are evaluated: PROTOCOL reserves
them for the team's one final comparison. The selection statement in the initial
e14 plan is superseded by that existing team rule.

## Frozen hypotheses and settings

Compare PPO-Lag, FCSRL + PPO-Lag, and FCSRL with risk-stratified training resets.
The extension changes only training-day sampling. Rank training days by observed
NH4 violation frequency and sample the highest-risk quarter with probability
one half, the remaining days with probability one half. Ties use date order.
The hypothesis is that exposure to difficult days reduces validation NH4
violations under the seasonal shift. This is a sampling extension, not a new
FCSRL representation loss, and no originality claim is made.

All methods use seeds 0, 1, 2, 3, 4, 401280 transitions, 8 environments,
760-step rollouts, 66 updates and five validation checkpoints. Reward, action,
state and cost remain e07 definitions. Calibrate the daily training cost limit
using training replay in the frozen environment. Compare final policies as
required by v1, with selected checkpoints as secondary results. Always report
NH4 and DO violations separately and air volume over 23.75 hours, not measured
electricity or a full 24-hour total.

Primary extension comparison is seed-paired final cost and NH4 violation,
with air consumption, DO violation, standard deviations and all seed results.
Report an improvement only if mean cost and NH4 violation decrease while air
increases at most 5 percent and DO violation does not increase more than one
percentage point. Report paired 95 percent bootstrap intervals (10000 draws,
fixed analysis seed 2026); intervals spanning zero do not establish improvement.
The 5 percent threshold is an experimental preference, not a universal rule.
Failure and energy-safety trade-offs are preserved. Five seeds are exploratory.

Environment selection is based on prediction metrics in e14. Controller results
only compare methods within that frozen environment, and cannot establish
real-plant performance or isolate gains due to changing the environment.

## Execution notes

The first run at results/e15_runs was interrupted by Windows refusing an
atomic progress.json replacement while the runner was reading progress.
No completed run metrics were produced. All partial artifacts are preserved
and excluded. Bounded retries for transient PermissionError were added to
training and runner JSON writes. The full comparison restarts from scratch
at results/e15_runs_retry, with unchanged learning settings and random seeds.

Frozen final policies are also evaluated without retuning on the old delta
GBM and direct 15-leaf validation simulators. These are model sensitivity
checks, not independent real-world validation. No test dates are used.
