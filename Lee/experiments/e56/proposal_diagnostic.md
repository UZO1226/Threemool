# Held-state proposal geometry

This exploratory diagnostic uses the same85120 held captured training states for each original policy seed. It runs no environment query, changes no controller parameter, and does not select the horizon or penalty. These are logged PPO states, not validation or candidate native trajectories.

| Policy seed | Horizon | Flat predicted risk, percent | Risk-only changed action, percent | Fixed-penalty changed action, percent | Changes suppressed by penalty, percent | Mean best predicted rate gain |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 1 | 66.6800 | 22.3191 | 0.6555 | 21.6635 | 0.00076706 |
| 0 | 4 | 60.3360 | 24.0954 | 1.0009 | 23.0945 | 0.00097504 |
| 1 | 1 | 69.0742 | 22.6903 | 0.0458 | 22.6445 | 0.00020147 |
| 1 | 4 | 55.0188 | 31.5919 | 0.0611 | 31.5308 | 0.00028498 |
| 2 | 1 | 66.4098 | 23.9603 | 2.0430 | 21.9173 | 0.00213491 |
| 2 | 4 | 64.3750 | 26.4203 | 3.1614 | 23.2589 | 0.00328843 |
| 3 | 1 | 67.3484 | 24.3092 | 2.0336 | 22.2756 | 0.00205954 |
| 3 | 4 | 58.6607 | 33.4821 | 2.4401 | 31.0421 | 0.00231430 |
| 4 | 1 | 66.1407 | 23.7253 | 0.9445 | 22.7808 | 0.00127786 |
| 4 | 4 | 53.2072 | 36.7881 | 1.6283 | 35.1598 | 0.00188713 |

Flatness uses max-minus-min predicted summed rate at most1e-12 over the fixed five candidates. Every percentage uses all held rows as denominator. Risk-only choices minimize the model prediction before the fixed squared-action penalty; they are diagnostic proposals, not evaluated alternative controllers. Counts use executed clipped action identity, so duplicate clipped candidates do not falsely count as changes.

The fixed penalty leaves only about.046 to3.161 percent of logged-state actions changed, although unpenalized predicted-risk choices differ on about22.319 to36.788 percent. Many predicted curves are flat, and most nonzero predicted gains are small relative to the declared penalty. This describes the algorithm, not the true consequences of those actions. It neither proves that a smaller penalty helps nor isolates the cause of any final controller result.

Horizon4 and horizon1 predict different target variables. Their numerical prediction gaps are not an accuracy comparison or independent policy performance evidence. There are no new counterfactual labels here. Stochastic logged actions, changing training policies, surrogate limitations and season/state shift still matter. The primary/control matched-air reports and independent action replays remain required.

proposal_diagnostic.json retains all counts, quantiles and fitted-model hashes. The generator is Lee/analysis/diagnose_rate_proposals.py. One output-field spelling was clarified from1e12 to1e_minus12 without changing any computed number; the JSON preserves the actual generation source digest, published generator digest and reversible label-only revision. Original fitted models and pinned runtime sources are unchanged.
