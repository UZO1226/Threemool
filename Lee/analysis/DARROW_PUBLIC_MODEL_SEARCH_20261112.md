# Public-model search during the renewed goal

## Authoritative sources checked

The official [Tilburg modeling interview](https://www.wastewater.ai/holistic-optimisation-of-wastewater-treatment-plants/)
states that the plant process model trained RL and reduced-order models.
This establishes existence, not public weights, code, parameters or a usable
Kaggle simulator. No such download was established from that interview.

The DARROW researchers' [AL-SAC paper metadata at imec](https://imec-publications.be/entities/publication/cd299e2b-92af-4f3c-a789-8f1a3290880a/full)
identifies BSM1 as the evaluated environment. DOI10.1016/j.eswa.2026.132705.
It is relevant control methodology but not a recovered Tilburg environment.
Publisher direct retrieval returned403, so no inaccessible supplementary-code
statement is inferred. Secondary search listings are discovery aids only.

An author-hosted [DARROW reduced-order model presentation](https://www.unav.edu/documents/d/instituto-de-ciencia-de-los-datos-e-inteligencia-artificial/4-2-reduced_order_models_2026-pdf)
describes synthetic water-line data from a calibrated plant and designed
variation, with additional biological states and control inputs. Those inputs
are not simply our five concentrations and recorded air. This supports checking
input compatibility before any reuse, rather than treating the presentation as
a downloadable pretrained environment. Download/code availability remains open.

[Mohammadi et al.,2024](https://arxiv.org/html/2403.15091v1) studies multi-step
simulation drift using model-generated inputs and a shape/time loss. It concerns
a different plant/task and is methodology, not a Tilburg model. Our existing
full-day training already uses recursive predicted states; the shape-aware
loss remains a distinct executable future hypothesis. No authors' code was
found by a text search of that paper.

Repeated official resources-page fetches timed out. A timeout is not proof of
non-public availability. Public search for the exact plant/project and the
researchers' control-paper title has not yet established a compatible code
repository. No contact message was sent. No access denial has been used to
terminate the goal while model-development alternatives remain available.

## Current route

Continue existing surrogate work in parallel with source discovery: e26 checks
unresolved stochastic-search boundaries; e27 checks a state-memory prior; e28
checks seasonal coverage within official training. Original adoption criteria
and the final test remain fixed. BSM1 is available as a distinct benchmark
already studied in this repo, but cannot substitute for the user's DARROW
objective or be relabeled as validation of Tilburg counterfactual responses.
