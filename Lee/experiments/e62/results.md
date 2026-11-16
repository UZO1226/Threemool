# Calibration channel decomposition outcome

All four conditions and both e61 calibration identity controls complete under
8f41b0e. No new fitting, validation, final test or policy training occurs.
The data are the30 calibration dates, which also estimated the PIT banks;
these are descriptive internal-model results, not independent confirmation.

| NH4 path | DO paths | NH4 frequency, percent | DO frequency, percent | NH4 Brier | DO Brier |
| --- | --- | ---: | ---: | ---: | ---: |
| Global | Global | 12.523684 | 3.016667 | .083618 | .106755 |
| Global | Context | 14.304386 | 2.170175 | .083075 | .108154 |
| Context | Global | 12.681579 | 3.184211 | .084335 | .106958 |
| Context | Context | 10.869298 | 2.174561 | .077615 | .108922 |

Actual calibration frequencies are5.508772 percent NH4 and10.736842 percent DO.
Every NH4/DO Brier contrast versus global/global has a circular five-day
bootstrap95-percent interval containing zero. For global NH4/context DO, NH4
change is-.000543 with interval[-.012123,.009910] and DO change is+.001400 with
interval[-.003897,.007252]. Full context changes NH4 by-.006003 with interval
[-.020582,.009153] and DO by+.002168 with interval[-.003129,.008184].

Changing either channel also changes later state feedback, and mixed conditions
break the original NH4/DO noise pairing. The pattern is not a simple additive
channel effect and does not identify plant causality. It does not provide a
demonstrated channel-only repair or justify selecting the favorable mean as a
new validation candidate. Full metrics/onsets/uncertainty are in summary.json.

e61's primary still fails its complete validation criterion. No alternate
copula, neighbor count or mixture is selected from this diagnostic. Both
completed studies share the verified aggregate archive indexed at
../e61/artifacts.json, with observations/banks/trajectories excluded. No e62
process remains live. The overall research goal remains active.
