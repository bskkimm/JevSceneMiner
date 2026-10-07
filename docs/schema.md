# Sample and scene contracts

## Target and cadence

Jev classifies the lateral maneuver active at NOW, with the whole past/future window as context.
The longitudinal question uses up to 2 s either side of NOW to avoid assigning later acceleration to earlier braking.

Inference cadence (`--step`) and table cadence (`--table-step`) are independent.
A complete -10 to +15 s table has 26 rows at 1 Hz or 51 at 2 Hz, including NOW.
Underlying motion and map/event measurements use finer observations.
At recording edges or gaps, missing context remains missing.

## Processed JSONL

Every `steps/<session>.jsonl` row has:

| Field | Meaning |
| --- | --- |
| `t_ns` | NOW: integer Unix nanoseconds |
| `script` | Readable evidence sent to Jev |
| `facts.schema_version` | `jsm-sample-1.1` for default input |
| `facts.target` | Classification targets |
| `facts.ego_samples` | Regular motion/map/signal table |
| `facts.lanes` | Aliases, source IDs, neighbors and mapped successors |
| `facts.window`, `local_speed` | Numerical summaries with explicit intervals |
| `facts.intersections` | Mapped occupancy intervals, separate from turn motion |
| `facts.lane_evidence` | Footprint shares, connected segments and window reference |
| `facts.object_tracks` | Stable source IDs, footprints, gaps and event anchors |
| `facts.object_selection` | Retained/qualifying counts and policy |
| `facts.map_free_object_script` | Independently selected map-free evidence |
| `lateral`, `longitudinal` | Raw winners, after classification |
| `lateral_probs`, `longitudinal_probs` | Raw distributions |
| `model`, `input_tokens`, `cache_key` | Answer provenance |

Facts and readable sections share measurements. Only the script goes to Jev.
The [complete synthetic sample](../examples/sample.json) includes both representations and illustrative answers.

## Geometry

Footprint share is oriented body area inside a mapped polygon divided by total body area.
Virtual intersection polygons can each cover 100% of a body; nonexclusive shares need not sum to 100%.
The mapped total uses a union, with ambiguous overlap and unmapped area reported separately.

Connected observed segments and unique end connections are unioned to reduce segment-end artifacts.
The continuous reference starts at the first observed chain in the window: a map proxy, not a verified maneuver-origin lane.
Native successors do not identify one unique physical keep-lane path.
Map alignment, lane order and polygon overlap alone cannot establish lane changes.

Body clearance is the shortest 2D edge-to-edge gap between ego and object rectangles.
Center distance and pose-reference-to-footprint distance are separate.
Hypothetical lane-centered-body clearance is a proxy, not observed ego motion.
Missing dimensions/identity leave measurements unavailable; an unmatched object is not declared off-road.

Facts retain available observations for selected stable object IDs.
The readable script emphasizes NOW, closest passage, events and endpoints.
Up to eight qualifying tracks are retained by default; disclosed omissions and occlusions prevent treating them as a complete world state.
Moving crossing actors may need explicit supplementation; see [review.md](review.md).

## Final scenes

Each scene has one lateral decision and consecutive longitudinal phases:

```json
{
  "start_ns": "1767000020000000000",
  "end_ns": "1767000024000000000",
  "driving_decision": "lane_change_right",
  "longitudinal_phases": [
    {"end_ns": "1767000021500000000", "decision": "decelerating"},
    {"end_ns": "1767000023000000000", "decision": "cruising"},
    {"end_ns": "1767000024000000000", "decision": "accelerating"}
  ]
}
```

This explanatory example is separate from the demo's actual cruising-only output.
The first phase starts at the scene start; each next starts at the previous end.
Ends strictly increase, with the final end equal to scene end.
Scene documents use decimal strings for nanoseconds to preserve browser precision.
See [generated scenes](../examples/scenes.json) for the session envelope and configuration.

Raw answers cover half an inference step either side of NOW, clipped to recording limits.
Merging smooths short lateral runs, demotes insufficient mean support and independently smooths speed phases.
At 1 Hz with a 2 s minimum, a two-step keep-lane interruption can survive; a one-step interruption can be smoothed.
Gaps split sequences.

`--maneuver-tail 2` extends into contiguous following keep-lane by up to 2 s, without replacing another maneuver or bridging gaps.
Speed phases come from the absorbed interval.
Raw probabilities remain unchanged; support statistics are summaries, not probabilities for extended boundaries.

Legacy flat scenes remain readable; saving GT writes nested scenes.
Reviewed verbose inputs use `jsm-sample-1.1-reviewed`.
The separate compact experiment uses `jsm-sample-1.2` / `compact-3`.
