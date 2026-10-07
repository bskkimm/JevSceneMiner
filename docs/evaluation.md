# Evaluation and limitations

Development has used curated driving examples and local camera review.
This repository does not yet publish representative accuracy results.
Synthetic examples check adapters and contracts; deterministic demo labels do not measure Jev.

## Reproducible experiments

Keep together: source/map identifiers and versions; Git commit; schema and exact questions; model; past/future and both cadences; geometry/signal availability; merge settings; manual annotations; display corrections; raw answers; locked independent GT; timing and fresh tokens.

`score` compares lateral intervals with one-to-one matching at minimum 0.5 s overlap.
It reports per-label precision/recall, boundary timing, time confusion and agreement, plus longitudinal behavior and probability calibration where GT is available.
nuPlan scenario tags are contextual, not exhaustive maneuver GT.

Use whole held-out sessions and separate raw results from merged scenes.
A maneuver tail is boundary postprocessing, not improved raw probability.
Compare the rules baseline and map-free ablation against unchanged GT.

## Timing and cost

`runtime/<session>.json` records the latest invocation's inference seconds, selected samples, fresh answers, fresh input tokens and estimated cost.
Preparation, viewer building and manual review are separate.
Cached answers contribute no fresh tokens.
The estimate uses a published-rate constant, not a billing receipt; verify [current pricing](https://docs.typesafe.ai/models).
Compare 1 Hz and 2 Hz in independent folders with identical model/questions/window and unchanged GT.

## Evidence limits

Maps, localization and painted boundaries can disagree.
A successor transition or footprint overlap alone does not establish a physical lane change.
The continuous reference is a mapped proxy, not a persistent camera-verified original lane.
Avoidance needs temporal interaction and ego response, not a nearby snapshot.
Selection and occlusion can omit actor history.

Future context makes this an offline labeling tool.
Recording ends and gaps limit context; speed smoothing can soften short braking peaks.
Projection uses poses and approximate height references rather than measured road surface.
Disclose manual corrections and never derive evidence annotations from evaluation target labels.
