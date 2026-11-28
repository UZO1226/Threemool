# e71: Gate projection on expanded-search risk improvement

Prospective hypothesis, 2026-11-17. e69's air-budget correction fails the
primary e63 criterion; e70 finds same-state action changes increase native
air and occupancy partly offsets this. Four of five e69 calibrations bracket
lambda0, permitting budget balancing by breaking predicted-risk ties.
Its held changes exceed e67's unpenalized wider-search changes by29976,
19478,32972,7403,21681 states out of85120 per seed. These are lower bounds
on changes where the wider unpenalized search would retain e63, not physical
counterfactual-quality labels. Test whether preserving the reference in
states without an estimated expanded-search benefit improves robustness.

Keep e69's same NNs, horizon4 rate models, nine-point radius0.5 search,
fitting/held split and64-step mean-only calibration. At each state compute
the e63 action and predicted summed rate over its original first five
candidates. A state is eligible only if the best of all nine candidates has
predicted summed rate at least1e-12 lower. The tolerance is a fixed numerical
signal test, not a tuned safety or adoption threshold. At ineligible states
return the original e63 action exactly, including non-NN original corrections.

Fit the same two shadow prices and global interpolation weight on eligible
fitting states from updates1~52 only, matching their mean e63 native action.
Ineligible fitting actions remain unchanged, so also require the overall
fitting mean constraint within1e-10. If there are no eligible fitting states,
use reference-only mode for that entire policy seed. No seed is discarded.
Retain the original bounds[-2^26,2^26],64 bisections, first-candidate tie rule,
lambda0 identity rule and deterministic convex action interpolation. This
adds a fixed eligibility rule, not more fitted parameters, environment data,
NN/rate fits or policy training steps.

Eligibility only means some expanded proposal has lower estimated risk.
The budgeted/interpolated selected action need not itself have lower predicted
risk. Disclose its predicted-risk changes; do not claim a risk bound, optimal
projection, causal support or plant safety. Mixture nonlinearity and state
shift remain unresolved, and the frozen e42 world is unchanged.

Before candidate evaluation require all five held-state checks: exact e63
fallback outside eligibility, radius/bounds, strict serialization and fixed
1024-row independent sklearn decisions. Verify the old global e69 control
on identical held states without the gate. Full seed0 controls must reproduce
the NN under constant risk with fitted parameters and e63 under reference-only
mode. Then run every original policy seed's native/matched comparison and
independent action/trajectory audit. Primary remains authenticated matched
e63 including its fixed offsets. e69 is a secondary mechanism comparator.

All actual-air matches and the unchanged10000 paired-seed bootstrap rule,
RNG2026, are mandatory: cost and NH4 upper endpoints below zero and DO at most
zero versus e63. Do not change the predicate, tolerance, radius, weights or
gate after validation outcomes. Use every seed0~4 and preserve failures.
Fresh seeds5~9 stay a separate required confirmation block after any initial
success; no fresh candidate quality is inspected. Each NN trained401280
steps and no additional environment-training transitions are added.
Reused validation40 is exploratory and final test41 remains excluded.

## Initial result and fresh confirmation

Latest outcome: both blocks and every independent audit are complete.
The original block passes, but the actual fresh seed5 through9 block fails
cost and NH4. See [fresh_results.md](fresh_results.md). The overall gain
does not replicate. Keep the prospective plan below and both full outcomes.

[results.md](results.md) records the completed original block and immutable
archive. The initial block passes, but fresh confirmation remains required.
Fresh adapters preserve signal_policy.py and e69/budget_policy.py unchanged,
retain actual seed IDs5 through9, and use e68's authenticated matched e63
references including their fixed offsets. Calibration uses training only.
All five original/captured training pairs and h4 models are prepared by e68.

From the original workspace after restoring dependencies by their hashes,
using the threemool Python environment with numerical threads1:

```powershell
python Threemool/Lee/experiments/e71/fresh_fit.py --workspace C:/Users/user/Downloads/IoT_project2 --certificate results/e68_fresh_references/baseline_authentication.json --baselines results/e68_fresh_baselines --rates results/e68_fresh_rates --out results/e71_fresh_fit
python Threemool/Lee/experiments/e71/fresh_compare.py --data kaggle_data/darrow_15min.pkl --freeze Threemool/Lee/experiments/e42/environment_freeze.json --candidate results/e68_fresh_baselines/captured/new_ppolag_fcsrl/seed_5 --baseline results/e68_fresh_baselines/original/new_ppolag_fcsrl/seed_5 --correction results/e68_fresh_rates/models/h4/seed_5 --certificate results/e68_fresh_references/baseline_authentication.json --preflight results/e71_fresh_fit/preflight.json --projection results/e71_fresh_fit/seed_5 --previous results/e68_fresh_references --out results/e71_fresh_smoke --native-only
python Threemool/Lee/experiments/e71/run_fresh.py --workspace C:/Users/user/Downloads/IoT_project2
python Threemool/Lee/analysis/verify_e71_blocks.py --workspace C:/Users/user/Downloads/IoT_project2 --out results/e71_blocks_verified.json --require-fresh
```

All stages and both block verifications have now completed. Existing
study directories are immutable; reproduce in a new workspace with fresh
output paths, adjusting the runner's workspace layout consistently. Final
weights and runtime manifests retain original401280-step policy identity.
The archive scripts reject incomplete runs or failed independent audits.
