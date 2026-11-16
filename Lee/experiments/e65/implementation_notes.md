# Implementation checks

The first held-state preflight completed for all five seeds. Changed actions
were1481,3870,2637,3650,4885 out of85120 per seed. Its independent sklearn
subsets passed. Before any simulator query, importing the comparison failed
because the historical e63 directory shadowed a same-named preflight module.
Replace that utility import with a local SHA256 helper; no controller or
statistical rule changes. Preserve results/e65_preflight locally and rerun
to results/e65_preflight_v2 so the corrected implementation has new pinned
source hashes. No unsuccessful process is treated as still running.

The e65 runner audits the native smoke before starting all five complete
comparisons. Each comparison replays the authenticated e63 matched baseline,
reloads and replays the new native controller, then uses the unchanged e44
air-only matcher. Every full candidate trajectory is independently checked
against original sklearn model predictions before reporting the five seeds.
