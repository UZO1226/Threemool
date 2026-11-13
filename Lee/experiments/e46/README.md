# Colder-day exposure with unchanged FCSRL

This separate sampling hypothesis follows e43 and e45, which fail their
complete five-seed matched-air criteria. No e46 policy result exists yet.
The existing risk-day sampler failed in e15 on a different model. We do not
erase that failure, repeat it under another name or claim a new RL algorithm.

Observed training temperature averages19.714C versus15.962C in reused
validation. Initial NH4 violations occur in7.45 versus30 percent of days.
Baseline final training rollout costs are below validation costs, but these
changing-policy summaries do not identify a causal failure mechanism.
See ../e45/training_shift_audit.json. Validation is diagnostic, not independent
confirmation, and final test41 is unused.

The hypothesis is that more exposure to colder observed training days can
improve generalization of the original FCSRL policy at matched air. Rank only
training days by mean recorded temperature and give the coldest quarter half
the reset probability, and all remaining days the other half. Stable ties use
original date order. The quarter/half allocation follows the pre-existing
e15 exposure strength; no cutoff or probability is searched on validation.
All188 training days retain positive probability. Weighted observed mean
temperature is18.801C, still above validation, so this does not remove the
distribution gap or assert coverage of unseen conditions.

Networks, feasibility targets, reward, actual cost, limit12, frozen model,
401280 transitions, five policy seeds and final-policy evaluation remain
unchanged. Only reset-day sampling changes. The PID therefore receives a
different training-day cost distribution; this is part of the intervention,
not a guarantee on uniform training or validation safety. Uniform validation
evaluation uses the original frozen environment and the existing air-only
matching algorithm and three-quality-condition criterion. Report native and
adjusted policies, both constraints and all five paired seed intervals.

Before any fit, check stable ties, finite/positive normalized weights, no
validation/test sampling, actual reset frequencies and exact outputs through
a full95-transition day at common fixed cases. Then run a6080-step smoke,
load its final checkpoint and independently replay validation. Full training
must wait for those checks. No original FCSRL or frozen source is modified;
the standalone trainer only scopes its training class reference substitution.
Sampling metadata is written to sampling.json alongside every run config.

Original FCSRL d882b91054d52c7a63a18a21f7db67bbbed82b63, Apache-2.0, is
preserved. This remains exploratory personal work, not a team protocol change.
