# Evaluation and limitations

Development has used curated driving examples and local camera review.
This repository does not yet publish representative accuracy results.
Synthetic examples check adapters and contracts; deterministic demo labels do not measure Jev.

## Reproducible experiments

Run the included **synthetic scoring contract** without an API key:

```bash
uv run jevsceneminer benchmark benchmarks/synthetic/manifest.json --out out/benchmark
```

`report.json` and `report.md` compare an exact interval sequence with one whose
lane change starts 1 s late. Both detect the maneuver; the delayed version has
1 s start error and 0.917 lateral agreement. These are hand-written labels,
**not Jev predictions or measured real-data accuracy**.

The benchmark command reads saved nested scene documents only. A
[`jsm-benchmark-1` manifest](../benchmarks/synthetic/manifest.json) declares:

- `name`, `kind` (`synthetic`, `development` or `held_out`) and GT `provenance`.
- `run_configuration`: settings for every named candidate; `gt` is reserved.
- `sessions`: unique IDs, assessed `start_ns`/`end_ns`, one `gt` and every candidate in `runs`.
- Each file reference: `path` relative to the manifest (absolute paths also work), and lowercase `sha256` of its exact bytes.

GT and candidates must cover **exactly the declared interval**, in order,
without gaps or overlaps. Include explicit `keep_lane` scenes and complete
longitudinal phases. Unknown labels, duplicate IDs, reused session files,
missing runs/files and changed hashes are errors. Partial annotations from
individual viewer edits need a separate, completely labeled assessment interval;
they cannot establish accuracy for an entire session.

Reports record the manifest/input hashes, tool/scorer hashes, declared settings,
one-to-one maneuver precision/recall, boundary errors and time agreement.
Agreement uses the existing 0.1 s grid; very short fragments can be rounded away.
There are no API calls, inferred gap labels or silently omitted sessions.
The output cannot overwrite a manifest or listed input. Same bytes and scorer
produce identical reports; update hashes deliberately after a reviewed input change.

To evaluate real runs, freeze independently reviewed GT and saved candidate scene
files. Use whole held-out sessions, or declared intervals selected before inspecting
candidate errors, and include label/schema definitions in `run_configuration`.
Record the Git commit and source/map version. `kind: held_out` is a declaration,
not a check that your data was independent. Keep private recordings and evaluation
files outside Git; publish aggregate results only after checking their provenance.

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
