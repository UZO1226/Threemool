# Lee/e35 raw pre-episode innovation results

Execution commit42d11f5. Zero-gain inner/development parent reconstruction and
future-target-scrambling invariance pass. Inner selection chooses NH4 gain0
and DO gain .5, score .6683719 against parent .6746007. No new weights are fit.

Development normalized MAE .87267, NH4 frequency5.8579 percent and DO1.6099
percent fail the unchanged NH4 calibration and member criteria. NH4 remains
essentially unchanged from e24; small changes with zero direct NH4 gain arise
from DO-to-NH4 feedback. Zero members meet both tolerances. Air order passes.
Separate event diagnostics,16-draw sensitivity and date intervals are in summary.

Mean past NH4 innovation is .08946mg/L on inner dates versus .26867 on
development dates, but the gain is not set from the development difference.
Raw residuals can include a persistent offset because the parent global bias
was optimized for recursive forecasts rather than teacher-forced mean residual.
e37 tests centering that offset using inner dates only. This is a new hypothesis,
not a claim that e35's implemented raw-innovation algorithm was incorrect.
No future observed states enter adaptation, no policy is trained and final
test outcomes remain unused.
