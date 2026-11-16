# Air-constraint transport diagnostic

All five seeds complete both unchanged native replays. All ten complete
episode summaries reproduce the prior e63/e69 outputs exactly, action means
independently reconstruct executed air, and both decomposition identities
hold. This is failure analysis, not a new controller or improvement claim.

| Air component, m3 per23.75-hour episode | Five-seed mean | Paired-seed95% interval |
| --- | --- | --- |
| Actual e69 native minus e63 native | 3046.989225 | [962.678696,5131.299754] |
| Symmetric action-mapping component | 7612.759656 | [4918.318471,10667.132586] |
| Symmetric visited-state component | -4565.770431 | [-6256.428200,-2455.885606] |

The two symmetric components sum to the actual native-air difference.
Mapping is positive for every seed, and occupancy is negative for every
seed. Thus these observations do not support a story in which changed
occupancy amplifies the native air increase. In this decomposition it
partially offsets a larger same-state action change. Inputs include previous
executed air as well as concentrations and exogenous variables, so occupancy
does not mean chemical concentrations alone. This is algebra inside a frozen
model, not physical causal identification.

Both occupancies record four fixed current NH4/DO regimes, their counts and
both policies' conditional action means. No state array is saved. The same
validation40 days, model members and noise paths are used, and uncertainty
uses10000 paired policy-training-seed resamples, RNG2026. Transitions and
model/noise draws are not treated as independent policy seeds.

## Posthoc descriptive exposure check

check_exposure.py separately authenticates all330 capture files and counts
current-state regimes without fitting or environment queries. The following
are pre-action DO>2 exposure percentages, not next-step violation frequencies
or onset recall:

| Policy seed | Fit updates1~52 | Held53~66 | e63 native | e69 native |
| --- | --- | --- | --- | --- |
| 0 | 2.656566 | 1.523731 | .013158 | .004605 |
| 1 | 1.126961 | .203242 | .069737 | .008553 |
| 2 | 2.995951 | 3.160244 | .464474 | .455263 |
| 3 | 2.714448 | 1.960761 | 1.141447 | .939474 |
| 4 | 2.579390 | 1.419173 | .025658 | .004605 |

State mixes differ, but this alone cannot identify the cause of the mapping
change or prove that conditioning on these regimes will fix it. Preserve
within-regime uncertainty and do not promote exposure differences into an
adoption criterion. The original e69 joint failure remains unchanged.

The diagnostic uses1520000 evaluation transitions and zero environment
training steps. Final test41 and fresh candidate quality are unused.
artifacts.json records the size/digest-verified immutable diagnostic archive,
including the separate e69 mechanism contrast and air-drift record. Its
metadata links the prior e69 weight archive; no new weights, source
observations, captures or PIT arrays are included. A first live-directory
partial-archive check raced with completion; the fixed incomplete fixture
with missing per-seed records is correctly rejected before archival. The
complete archive and every stored entry are verified.
