# Self-improvement trials versus e63

This block uses authenticated matched e63 as its primary comparator.
The earlier DARROW_AGENT_TRIALS.md table retains original FCSRL as its comparator.
Keep the two blocks separate. Five original policy seeds0 through4 are used here.

| Trial | Matched cost | NH4 percent | DO percent | Cost difference95% interval | NH4 interval, pp | DO interval, pp | Joint result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [e65](../experiments/e65/results.md) | 21.025750 | 21.810395 | 0.321974 | [-0.076250,-0.000875] | [-0.076579,0.001579] | [-0.005526,0.001842] | Fail |
| [e67](../experiments/e67/results.md) | 21.730500 | 22.741316 | 0.132895 | [-0.506625,1.910125] | [-0.427105,2.416316] | [-0.448289,-0.018289] | Fail |
| [e69](../experiments/e69/results.md) | 21.167375 | 22.102632 | 0.178816 | [-0.766125,0.926250] | [-0.645000,1.160789] | [-0.312368,-0.018421] | Fail |
| [e71](../experiments/e71/results.md) | 20.089750 | 20.900132 | 0.246974 | [-1.430500,-0.486750] | [-1.382368,-0.472368] | [-0.135263,-0.018421] | Pass |

Differences are candidate minus e63. Cost sums the two violation indicators over95 steps,
not money or electricity. Air is actual m3 per23.75-hour episode. All air matches, cost and
NH4 upper endpoints below zero, and DO upper endpoint at most zero are required.
Intervals use10000 paired policy-seed resamples, RNG2026. All native values remain in the JSON.

Pending complete committed reports: e73.
Pending does not prove a process is live or a partial outcome is a completed trial.

Environment e66 independently reconstructs but fails all seven adoption conditions.
It is a separate forecast experiment and does not replace e42 or enter the policy table.

e71 passes this original seed block, but fails its separate fresh seed5 through9
confirmation on cost and NH4. See e71/fresh_results.md and block_verification.json.
The fresh DO gain cannot rescue the joint failure. Do not pool the blocks.
This table contains only original seed0 through4 results, not overall adoption.
Reused validation and multiple hypotheses remain exploratory; this is not independent
observational confirmation or physical plant safety. Final test41 is preserved.

Regenerate with summarize_self_improvement.py using committed compact reports only.
