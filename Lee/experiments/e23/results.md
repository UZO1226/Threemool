# Lee/e23 historical results and initialization correction

Inner selection chose epoch 80 and innovation scales NH4 1, DO 1. The
historical gate failed. However, e24 identifies an implementation defect:
the free-sign NO3 air coefficients were initialized to -5, even though the
equilibrium forcing initialization assumed zero for that channel. The NO3
output was clamped to zero and received zero gradient. Both e22 and e23
NO3 MAE equal the mean observed NO3, 2.5586867786, which exposed the bug.

These historical numbers are preserved, not silently replaced:

| Development metric | Historical e23 |
| --- | --- |
| Normalized trajectory MAE | 0.9931 |
| NH4 trajectory MAE | 1.4040 |
| NO3 trajectory MAE | 2.5587 |
| NH4 predicted frequency (%) | 7.61 |
| DO predicted frequency (%) | 2.64 |

Observed frequencies are 17.32 and 1.16 percent. No agent was trained on
this environment and no test was evaluated. The failing implementation is
not evidence that all full-day latent dynamics models cannot fit these data.
Shared network training also changes when the dead head is repaired, so the
other channels cannot be assumed unaffected. e24 repeats the same full-day
design with a minimal initialization fix and an independent equilibrium and
gradient regression check. Fitting loss was still decreasing; convergence
was not proved by the 80-epoch budget.

Execution commit b9fb668. Exact checkpoint hashes, fitting losses, inner
scores and all selected metrics are in summary.json. Runtime files remain
under results/e23_full_day. Original observations and residuals are not in Git.
