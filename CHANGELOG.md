# Changelog

## Unreleased

- Add an API-free, hash-locked saved-result benchmark with complete coverage checks and a clearly synthetic boundary example.
- Keep time-agreement samples inside their assessed interval; identical 0.55 s/1.55 s intervals no longer incur a false missing-label penalty at the end.
- Organize implementation under `src/jevsceneminer` with source, evidence, inference, scene and viewer groups; preserve former public imports and CLI behavior.
- Refresh the README with a real Boston highlight preview and an inline click-to-play full video.

## 0.2.0 — 2026-10-07

- Publish shared rosbag/nuPlan evidence with explicit context intervals, mapped intersection facts, ego-footprint shares and source-ID object histories.
- Save canonical `jsm-sample-1.1` facts beside Jev's readable input; provide opt-in reviewed inputs and a separate compact experiment.
- Define 11 lateral decisions and 5 longitudinal decisions in questions `jsm-1.4`.
- Store lateral scenes with consecutive longitudinal phases; retain legacy flat-scene reading.
- Add opt-in maneuver tails without changing raw probabilities.
- Add camera trajectory projection, integrated fixed-range BEV, probability panels and nested GT editing.
- Package default labels, viewer resources and the explicit `nuplan-1hz` preset.
- Support bounded classification reruns with unchanged outside raw rows and recorded latest timing/fresh-token cost.
- Add pre-commit, CI on Python 3.10/3.11, clean-wheel checks and API-free synthetic demos.

Compatibility: plain CLI defaults remain past10/future10 with 2 Hz inference/table. Presets are explicit.
Questions and script changes invalidate cached answers. Camera height corrections are approximate and display-only.
The compact input is experimental and has different questions; this release makes no representative accuracy claim.

## 0.1.0

Initial proof-of-concept pipeline and local review.
