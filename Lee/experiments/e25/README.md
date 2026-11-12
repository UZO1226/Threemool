# Lee/e25: Initial-condition anchoring as a persistence control

## Hypothesis and prospective comparison

The corrected e24 model still misses NH4 violation frequency during full-day
rollouts. A known initial NH4 value above 4 occurs on 3.33 percent of inner
calibration days but 30 percent of development days. These values are available
at episode start, unlike future observed reset states. Retaining that observed
initial condition may be more robust than autonomous mean reversion when the
season changes. This hypothesis was chosen after exploratory development
inspection, so this is not independent confirmation.

Keep e24's chosen 40-epoch model, calibrated bias and innovation scales fixed.
For NH4 only, output (1-w) times its forecast plus w times the initial observed
NH4. Other outputs and internal latent states are unchanged. This represents a
forecast model with a persistent hidden episode anchor, not a claim about the
actual plant's response. It never reads future measured states. It weakens
NH4 action sensitivity but preserves the paired-air ordering. It does not
repair or independently identify counterfactual aeration effects.

Choose w from 0,.1,...,1 by the original inner proper score only, using the
same draws for each candidate. Select once before development scoring. Audit
the selected choice and both endpoint controls, retain the original e16 gate
and 16-draw sensitivity. No development-label optimization, no test evaluation,
no policy training unless an environment passes. The w=0 endpoint must exactly
reproduce e24. Five model-fitting members are not five policy-training seeds.

Use the same data hashes, first158/last30 training split, original FCSRL
commit and CPU runtime as e24. Model files remain outside Git.
