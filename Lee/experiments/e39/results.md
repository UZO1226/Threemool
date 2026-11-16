# Result

Source commit: 4599833. Inner selection chooses epoch80 and NH4 noise scale.5,
DO scale1. The implicit saturating reaction model fails the original adoption
gate. No final test was evaluated and no policy was trained.

Normalized MAE is.773974 and NH4 MAE1.45106. NH4 violation frequency is4.57434
percent versus17.31579 observed. DO is1.07697 percent versus1.15789 observed.
All five structural members underestimate NH4 beyond the allowed tolerance.
Common-noise air ordering passes with zero observed reversals.

Independent full-day Torch and cached NumPy reconstruction passes. The circular
five-day date bootstrap NH4 frequency bias interval is[-15.68890,-9.41043]
percentage points, with mean-12.74145. The interval uses repeatedly reused
validation dates, not independent confirmation or policy seeds. The model
improves average error relative to the old predictor while still missing NH4
violations. It is unsuitable for adopting the control environment.

This is a tested hybrid saturation hypothesis, not a reproduction of the
Tilburg process model or a calibrated nitrogen mass balance. See audit.json,
summary.json and base_compatibility_check.json for reconstruction, event and
uncertainty diagnostics. The state matrix diagnostic describes the backbone,
not a contraction proof for the nonlinear reaction.
