# Adapt your driving data

Use this guide with a coding agent or as a manual checklist when your recordings
differ from the built-in nuPlan and Autoware-topic MCAP inputs. A source adapter
converts your raw format into the shared data used by preprocessing. Correct
conversion lets you reuse evidence generation, Jev classification and scene
merging; it does not guarantee classification accuracy.

Start with a small representative recording and its format documentation. Include
an ordinary lane-following interval and, where available, a maneuver, an object
interaction and a recording gap. Keep private recordings and generated runs out
of Git. Choose a new output directory so existing results are not overwritten.
If you use a hosted coding agent, file contents it inspects may be sent to that
service; choose samples you are authorized to share, or use synthetic examples.

```text
Raw recording + source documentation
                |
                v
Field mapping + unresolved questions
                |
                v
Source reader / map adapter
                |
                v
Shared Session + LaneMap -> facts + readable script
                |
                v
Numerical checks + representative review
                |
                v
Conversion report + tested changes for review
```

## Choose the smallest integration

| Your input | Starting point |
| --- | --- |
| Supported Autoware messages on different topic names | Configure `--topics`; confirm the decoded field layout still matches the reader. |
| Different messages, files or databases | Add a reader under `src/jevsceneminer/inputs/` that produces the shared `Session`. |
| Different vector-map format or coordinates | Add a map loader compatible with `LaneMap`; verify its frame against ego and objects. |
| Already prepared samples | Check the [sample contract](schema.md) and session metadata before reusing classification. |

Topic overrides change discovery, not message decoding. The current CLI explicitly
supports the two built-in source families; there is no generic private-data import
flag or adapter plugin registry. A new reader also needs CLI integration and tests.

The current MCAP reader requires ego and indicator messages, and `run` requires a
map. The nuPlan reader explicitly marks its unrecorded indicator unavailable.
For another source, define missing-signal behavior in its adapter and metadata;
do not fabricate indicator reports or map geometry to satisfy a built-in reader.
A recording without a usable map needs an explicit integration design before it
can provide equivalent lane evidence through this pipeline.

## 1. Describe the source before changing code

Record the clock origin, timestamp unit, coordinate frames, units, pose reference
point, vehicle dimensions and actor identity rules. Inspect actual samples as
well as documentation. If either is unavailable or contradictory, list the
specific unresolved question before implementing the affected conversion.

Produce a mapping table like this; source names below are illustrative:

| Source field | Shared field | Conversion to verify |
| --- | --- | --- |
| `timestamp_us` | `EgoTrack.t` | Integer Unix microseconds × 1,000 → integer Unix nanoseconds; a relative clock also needs a known origin. |
| `position` | `EgoTrack.x`, `y` | Transform to meters in the map frame. |
| `orientation` | `EgoTrack.yaw` | Convert conventions to radians and unwrap across ±π. |
| `velocity` | `EgoTrack.v` | Resolve body/world frame, longitudinal direction and units. |
| `angular_velocity` | `EgoTrack.yaw_rate` | Radians/s; positive must mean turning left. |
| `object_id`, dimensions | `DetectedObject` | Preserve stable source IDs and known length/width; otherwise leave unavailable. |

Document whether each value is recorded, derived or unavailable, including any
derivation interval and assumptions. Do not silently fill required motion arrays
with zeros. If essential motion is missing, stop preparation for that source or
implement and test a defensible derivation.

## 2. Use the existing conversion boundaries

Read the [repository layout](architecture.md), [sample contract](schema.md) and
[contribution contracts](CONTRIBUTING.md#contracts-to-preserve) first. Use grouped
modules for new code; keep legacy import bridges working.

| Code | Responsibility |
| --- | --- |
| [`inputs/bag.py`](../src/jevsceneminer/inputs/bag.py) | Shared `EgoTrack`, `DetectedObject` and `Session` types; MCAP reader |
| [`inputs/nuplan.py`](../src/jevsceneminer/inputs/nuplan.py) | Example reader converting another source into `Session` |
| [`evidence/lanes.py`](../src/jevsceneminer/evidence/lanes.py) | `Lane`, `LaneMap`, projection and matching |
| [`inputs/nuplan_map.py`](../src/jevsceneminer/inputs/nuplan_map.py) | Example native-map loader |
| [`evidence/facts.py`](../src/jevsceneminer/evidence/facts.py) | `build_timeline`: common numerical signals |
| [`evidence/script.py`](../src/jevsceneminer/evidence/script.py) | `render_sample`: facts and readable Jev input |
| [`cli.py`](../src/jevsceneminer/cli.py) | Source selection, preparation and session metadata |

Prefer source-specific conversions over changing shared classification, labels or
merge behavior to accommodate raw-format differences. These internal interfaces
are implementation reference points, not a versioned plugin API.

The reader must provide aligned motion arrays and time-ordered object/signal
observations. Handle duplicate timestamps and recording gaps explicitly. Do not
invent a stable object ID from its list position. Missing dimensions cannot support
body-footprint clearance. Record whether the ego pose is the rear axle, body
center or another point before configuring `ego_geometry`.

The map adapter must align lanes, centerlines and available polygons with the
motion frame. Preserve known neighbors, successors and intersection attributes.
A segment-ID transition is evidence, not a lane-change label. Multiple mapped
successors do not identify a unique physical keep-lane successor. Independent
polygon shares can overlap; do not normalize them to an invented exclusive lane
ownership. An unmatched object has an unknown map relationship.

Camera integration is separate: footage needs capture timestamps, poses and
calibration supported by the viewer. Reusing classification does not automatically
make a new source's camera overlay work.

## 3. Validate evidence before inference

Prepare a short recording without API requests. For built-in inputs, use
`run --dry-run` as shown in [usage](usage.md). For a new source, first implement and test
its preparation path; do not assume an existing flag accepts your format.

| Check | Evidence to inspect |
| --- | --- |
| Time | Integer nanoseconds, known origin, strictly increasing ego times, aligned signals and gap behavior |
| Motion | Finite values, correct units/signs, yaw wrap handling and displacement/speed consistency within source tolerances |
| Frames and body | Ego, objects and map use the intended frame; body-center offsets and dimensions match source documentation |
| Lanes | Ordinary following, boundaries and successor transitions remain interpretable; missing matches stay unknown |
| Objects | IDs persist across observations; missing detections, dimensions and selected/omitted tracks remain visible |
| Sample | NOW is `t=0`; configured cadence/window are respected; gaps and recording edges do not gain fabricated observations |
| Script | Units, aliases, events and summary intervals agree with numerical facts; important evidence actually appears in the script |

Only the readable script goes to Jev. Adding a numerical fact without rendering
the relevant evidence cannot improve the classifier's input. See the full
[synthetic sample](../examples/sample.json) and [script](../examples/sample.txt).

Inference and table cadence are separate. A complete -10 to +15 s table has 26
rows at 1 Hz or 51 at 2 Hz, including NOW. Motion processing uses finer observations;
check whether the source sampling and gaps support those measurements rather than
assuming a denser table creates new evidence.

Review representative camera/BEV frames where available. A valid JSON structure
does not establish that a map is aligned to painted lanes or that a maneuver has
enough evidence. Without visual or other independent reference, report physical
alignment as unverified. Do not move the map or override lane evidence just to
force an expected label. Use documented, opt-in [evidence annotations](review.md)
for reviewed corrections.

Add small original synthetic fixtures that exercise the new conversions and
meaningful failures: timestamp units, frame/sign conventions, missing signals,
unstable IDs and gaps as applicable. If shared preprocessing changes, cover both
built-in adapters. Use the checks in [Contributing](CONTRIBUTING.md#development);
tests must block unexpected API calls. The existing synthetic demo exercises
conversion and cached illustrative answers, not Jev accuracy.

## 4. Report what is ready and what is missing

Before proposing a merge, provide:

- The source-field mapping, transformation assumptions and unresolved questions.
- One generated sample and its readable script, with important changes identified.
- Validation commands/results, reviewed intervals and observed conversion failures.
- Missing evidence and the labels it may limit, such as map-supported turns or
  avoidance without actor history and usable geometry.
- Changed files, CLI integration and any metadata/cache compatibility effects.

Keep private reports and samples outside tracked files. Public PRs should use
original synthetic reproductions. A schema-valid sample is not evidence of label
accuracy; evaluate new labels with independent ground truth. Changed scripts,
questions or model settings must not reuse incompatible cached answers.

## Copyable coding-agent task

Replace the bracketed paths and source description before using this prompt:

```text
Adapt JevSceneMiner to [describe my recording format].
Representative data: [path I authorize you to inspect]
Source documentation: [path or reference]
Fresh output directory: [path outside tracked files]

Read AGENTS.md, docs/adapting-private-data.md, docs/schema.md and
docs/CONTRIBUTING.md. Inspect the source and existing readers first.

1. Produce a source-to-shared-field mapping. State clock, units, frames,
   pose reference, dimensions, actor identity and unavailable evidence.
2. Ask about unresolved source semantics before implementing dependent
   conversions. Continue work that does not depend on those answers.
3. Implement the smallest source reader/map adapter and CLI integration
   needed. Preserve the shared sample contract and existing adapters.
4. Add meaningful synthetic tests and prepare representative samples
   without paid inference. Verify facts and the exact readable Jev script.
5. Show one sample, validation results, assumptions, limitations and the
   diff. Identify evidence that is insufficient for particular labels.

Do not invent missing signals, actor IDs, physical lane ownership or GT.
Keep private logs, maps, images, credentials and generated runs out of Git.
Do not send private recordings to external services or run paid inference
without my explicit authorization. If essentials are unavailable, explain
the limitation instead of claiming successful adaptation.
```

## Agent instruction discovery

The root [AGENTS.md](../AGENTS.md) points compatible coding agents here when a task
involves a new source or conversion problem. Automatic discovery depends on the
agent and its configuration; an arbitrary Markdown file is not automatically
loaded. If your tool does not read `AGENTS.md`, explicitly ask it to read this
guide, or configure its instruction file to reference it.

See the [AGENTS.md format](https://agents.md/) and
[GitHub's repository instruction documentation](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions/add-repository-instructions)
for tool support. The instructions guide development; they do not launch an agent
or turn the CLI into an automatic data-conversion service.
