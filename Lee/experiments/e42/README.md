# Lee/e42: Freeze the validated quantile environment and train paired baselines

e40 passes the unchanged environment point gates and independent reconstruction.
Freeze its five quantile checkpoints, PIT bank, selected alpha.5/block95,
normalization inputs and model summary before controller comparisons. The
manifest records exact SHA256 values. It does not claim independent validation,
accurate rare DO timing or real-plant causal identification. See e40/results.md.

## Environment adapter

Preserve11 observations, total air[0,9000]m3/h, recorded zone fractions, reward
-air/9000, cost1[NH4>4]+1[maxDO>2], cost limit12 and95 steps per daily episode.
Derive fallback fractions only from official188 training dates, correcting
unused legacy full-table metadata. Do not rewrite the canonical prepared file.
The unchanged train/validation records never exercise that fallback.

Adapter checks also reproduced a historical terminal-step alias in VecAerationEnv:
reset overwrites self.prev_air, which shares storage with the local executed
air. The returned last-step reward and info air then reflect the next day's
initial air. A constant4500m3/h training-day probe returns4416.41 at the final
step in the historical wrapper. The new adapter and a separate original-model
wrapper both preserve the executed action before reset. State predictions,
costs and terminal observations are unchanged in the independent regression.
Apply this correction to every newly trained baseline and extension. Preserve
old code and historical results; those old policies are sensitivity references,
not a fully paired control for this corrected runner. New old-model baselines
therefore use the same correction to separate transition-model effects.

Each training episode samples a full95-step inner-calibration PIT block for
its structural member. Keep the NO3 quantile index.5 and the original state
blend.5. Episode noise randomness is independent of policy RNG and day sampling.
Hold member fixed for each760-step training update as in the original runner.
For deterministic-action validation, use common eight-noise-draw paths for all
40 dates and five members,1600 episodes; reproduce the e40 factual replay.
Eight stochastic draws are an explicitly required difference from v1.1's
deterministic200-episode surrogate, not a change to the team protocol.

Check independent rollout identity, reward/cost arithmetic, final-step terminal
observation before reset, noise reproducibility, common-noise air order and
train-only fallback before any new training. Reject test split construction.

## Paired baselines and separation

Run original PPO-Lag and original FCSRL on this frozen environment, seeds0..4,
401280 transitions each,8 environments,760 steps and66 updates. Keep all PPO,
representation, PID and cost-limit settings fixed to the original DARROW
application. Use final policies as primary. Intermediate checkpoints are
diagnostic and must not replace a failing final result.

Also train the same original FCSRL seeds and budget on the original delta GBM.
Evaluate old-trained and new-trained final policies on both simulators without
retuning. This separates changing the environment from changing the agent.
No direct claim that rewards/costs from different environments are comparable.
Native evaluations and matched-air evaluations are both required. A separate
e43 extension changes representation supervision only within the frozen e40
environment, with a capacity control and the same five policy seeds.

Before seeing extension outcomes, specify matching by an additive deterministic
policy action offset in[-2,2], clipped to the existing action bounds. Target
each paired FCSRL baseline's native validation air using common stochastic
paths. The matching routine may inspect only mean air, never reward, cost or
NH4/DO outcomes. Find a bracket with an air-only grid and bisect for at most25
queries, retaining the nearest air value with a fixed smallest-offset tie
rule. Endpoints supply zero/maximum air; report any discontinuity or failure
to reach the target. This adjusts the policy being evaluated and must be
reported alongside native results. It does not establish a better native
controller. Only call it matched if residual air mismatch is within1 percent.
Include the feasible matching range and do not extrapolate. A training-air
calibrated fixed offset is an optional transfer sensitivity, not permission
to select offsets using water-quality outcomes.
Report NH4 and DO separately, total cost, air, per-seed values and paired
10000-resample seed bootstrap intervals. Five seeds are exploratory. Improvement
requires lower total cost and NH4 at matched air without higher DO, supported
by seed uncertainty; preserve trade-offs and failures.

Runtime and source hashes follow the freeze manifest. Original FCSRL commit
d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache-2.0. Existing wrappers are
the DARROW application, not a reproduction of every paper hyperparameter.
No original source or official protocol modification. Raw data, checkpoints
and full logs remain outside Git and are archived with checksums.
