# Lee e14 and e15 reproduction artifacts

Draft attachment for team reproduction of the DARROW model audit and the
five-seed controller comparison. These are exploratory personal experiments,
not adopted team results or a change to Compare/PROTOCOL v1.

The archive includes the environment models, all 15 final checkpoints and
validation-selected checkpoints, full successful-run logs, configurations,
metrics, source and input hashes, and the upstream FCSRL license notice.
Original DARROW observations and prepared observation tables are excluded.

Archive layout:

- environment/: old delta GBM, two direct-state candidates and audit reports.
- agent_runs/: run manifest, full training logs, final and selected policies.
- licenses/: upstream FCSRL attribution.
- artifact_manifest.json: file sizes and SHA256 checksums.

Code and conclusions are on branch lee/darrow-model-audit. Model training
used b9fefe5; controller training used c1c50f923ee495f78b7c71e868096fbc79738bd2.
Use the matching project Python environment and add Lee/code/darrow to the
Python import path when loading direct-state model pickles.

All controller evaluation uses validation dates. The final team test split
was not used for policy selection or reported controller evaluation.
Safety calibration is still inadequate, and the risk-sampling extension
did not meet its preregistered improvement criteria. See conclusions.md.
