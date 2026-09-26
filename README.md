# JevSceneMiner

Find interesting scenes in driving logs with [Jev](https://docs.typesafe.ai/introduction), a fast general-purpose classifier from [TypeSafe AI](https://typesafe.ai/).

> **Status:** proof of concept. Reads Autoware rosbags (MCAP) with a Lanelet2 map, or [nuPlan](https://www.nuscenes.org/nuplan) logs with their maps, and labels the ego vehicle's driving decisions. No driving data is included: bring your own logs, or download nuPlan.

## Goal

Driving logs run for hours, but the moments worth studying, such as a car cutting in, a pedestrian crossing, or a hard brake, last only seconds. Finding them by hand is slow, and hand-writing a rule for every scenario doesn't scale.

JevSceneMiner describes each moment of a log as a short text **script** and lets Jev decide which scenario it belongs to. Scenarios are defined in plain language, so adding a new one doesn't require training a new model.

## Pipeline

```
Autoware rosbag + Lanelet2 map, or nuPlan log + map
    │  every 0.5 s: a script of the 5 s before and 10 s after NOW
    ▼
script  ──────►  Jev  ──────►  scenes
 • ego state                    • lateral / longitudinal label
 • lanes and route              • probability
 • traffic light, indicator     • start / end time
 • surrounding objects
```

1. **Script:** facts computed from the full-rate data (Jev is not asked to do arithmetic): lane and lane changes, speed, acceleration, yaw rate, heading change, indicator, traffic light, the intersections and stops along the ego's path, and nearby objects. Only relative facts: no map IDs, positions or clock times.
2. **Jev:** two Choice questions per script (9 lateral labels such as `turn_left`, `u_turn` and `lane_change_right`; 5 longitudinal labels such as `cruising` and `hard_braking`), with the definitions in [`labels.yaml`](labels.yaml). Jev returns a probability for every label.
3. **Scenes:** consecutive steps with the same labels become one scene with start/end times; short flickers and low-confidence maneuvers are smoothed away.

## Labels

Each script gets two questions. Definitions are plain language in [`labels.yaml`](labels.yaml) (version `jsm-1.0`); edit them or add labels without retraining anything.

| Lateral (what the ego does sideways) | Needs indicator |
|---|---|
| `keep_lane`: staying in its lane, including curves and stops | |
| `turn_left` / `turn_right`: a turn at an intersection (heading change about 45–150°) | LEFT / RIGHT |
| `u_turn`: turning back the way it came (about 180°) | far side |
| `lane_change_left` / `lane_change_right`: moving into the neighbor lane | LEFT / RIGHT |
| `avoidance`: moving sideways to pass a stopped vehicle or obstacle, then returning | |
| `pull_over`: moving to the road edge and stopping | curb side |
| `pull_away`: leaving the road edge back into traffic | far side |

| Longitudinal (what the speed does) |
|---|
| `stopped`, `accelerating`, `decelerating`, `hard_braking`, `cruising` |

"Curb side" is the side traffic drives on (`traffic_side: left` by default; `--traffic-side right` for e.g. the US), and "far side" is the other one. With `--require-indicator`, a maneuver without its indicator becomes `keep_lane`.

## Example

One step's script (shortened, illustrative values), then the kind of scene it ends up in:

```
Driving log of the ego vehicle around NOW (left-hand traffic). Times are relative to NOW. ...

LANES
A: road; left neighbor: B (dashed line, can be crossed); right neighbor: none
B: road; left neighbor: none; right neighbor: A (dashed line, can be crossed)

TIME   LANE     OFFSET   SPEED     ACCEL       YAW RATE  HEADING  INDICATOR  LIGHT
t=-2s  B        -0.0 m   14 km/h   +0.0 m/s²   +0 °/s    +0°      off        green
t=-1s  B        -0.9 m   14 km/h   +0.0 m/s²   -6 °/s    -2°      RIGHT      green
t=+0s  A        +1.7 m   14 km/h   +0.0 m/s²   +0 °/s    +0°      RIGHT      green   <- NOW
t=+1s  A        +0.3 m   14 km/h   +0.0 m/s²   +6 °/s    +2°      RIGHT      green

EVENTS
t=-1.5s: indicator RIGHT on
t=+0.0s: moves from B into its right neighbor A

OBJECTS AT NOW (up to 8 within 30 m)
car: 13 m ahead, 3 m left, left neighbor lane, 14 km/h
```

```json
{"scene_id": "scene_004", "start_ns": "1767000002750000000", "end_ns": "1767000006250000000",
 "lateral": "lane_change_right", "longitudinal": "cruising",
 "lateral_prob": 0.91, "longitudinal_prob": 0.97}
```

## Usage

```bash
uv sync
echo "TYPESAFE_API_KEY=..." > .env           # gitignored

# Scripts + cost estimate only, no Jev calls:
uv run jevsceneminer run <session_dir>... --map lanelet2_map.osm --out out/run1 --dry-run

# Full run (answers are cached, so re-runs are free):
uv run jevsceneminer run <session_dir>... --map lanelet2_map.osm --out out/run1

# Jev + scenes for folders prepared with --dry-run; re-stitch saved answers without Jev:
uv run jevsceneminer classify out/run1
uv run jevsceneminer restitch out/run1 --out out/run1b --min-prob 0.5

uv run pytest
```

Useful options of `run` (see `--help`):

| Option | Default | Meaning |
|---|---|---|
| `--step` | 0.5 | seconds between steps (one script and one Jev call each) |
| `--past` / `--future` | 5 / 10 | seconds before / after NOW in each script's table |
| `--min-scene` | 2.0 | shorter label runs are merged into a neighbor |
| `--min-prob` | 0.4 | maneuvers with a lower mean probability become `keep_lane` |
| `--require-indicator` | off | maneuvers need their indicator (see Labels) |
| `--traffic-side` | left | side traffic drives on |
| `--lateral-only` | off | ask only the lateral question (half the cost) |
| `--topics` | Autoware defaults | YAML with other topic names (see below) |

`classify --strip-map` removes every map-derived fact from saved scripts, to measure how much the map helps.

Jev is billed per input token; `--dry-run` prints an estimate (roughly 2,200 tokens per call: 7,200 calls and about $0.70 per hour of driving at 2 Hz).

If ROS is sourced in your shell, run with `env -u PYTHONPATH` so the venv's packages are used.
No ROS installation is needed: messages are decoded with the definitions stored in each MCAP file.

### Input topics

A session is a folder of MCAP chunks recorded with [Autoware](https://github.com/autowarefoundation/autoware).
By default these topics are read (the first one present per role):

| Role | Topic | Type |
|---|---|---|
| ego | `/localization/kinematic_state`, or `/api/vehicle/kinematics` | `nav_msgs/Odometry`, `autoware_adapi_v1_msgs/VehicleKinematics` |
| objects | `/perception/object_recognition/tracking/objects`, or `/perception/object_recognition/objects` | `autoware_perception_msgs/TrackedObjects`, `PredictedObjects` |
| traffic lights | `/perception/traffic_light_recognition/traffic_signals` | `autoware_perception_msgs/TrafficLightGroupArray` |
| turn indicator | `/vehicle/status/turn_indicators_status` | `autoware_vehicle_msgs/TurnIndicatorsReport` |

The light the ego is approaching is found by matching the recognized group ids to the
`traffic_light` regulatory elements of the lanes ahead on the ego's path. Other recordings
can name their own topics with `--topics topics.yaml`, e.g. `lights: [/my/nearest_light]`
(a topic with a single `TrafficLightGroup` is taken as the light ahead).

### Output

```
out/run1/
├── steps/<session>.jsonl      one line per step: the script and Jev's answers with probabilities
├── scenes/<session>.json      scenes: start_ns / end_ns, lateral / longitudinal label, probability
├── rules/scenes/<session>.json  the rule baseline, same format
├── tags/, camera/             nuPlan only: scenario tags and front-camera frame times
├── report/                    from `score`: report.md and report.json
├── indicator/<session>.json   periods with the turn indicator on: [start_s, end_s, 2=LEFT | 3=RIGHT]
├── meta/<session>.json        session span and the settings used
└── cache/                     Jev answers keyed by script + questions
```

## nuPlan

[nuPlan](https://www.nuscenes.org/nuplan) has public driving logs and HD maps from Boston, Pittsburgh,
Las Vegas and Singapore ([CC BY-NC 4.0](https://www.nuscenes.org/terms-of-use): non-commercial use).
Download the maps, some log databases (e.g. the mini split) and, for the viewer, the camera images, into
nuPlan's usual layout:

```
nuplan/
├── maps/<city>/<version>/map.gpkg
├── data/cache/<split>/<log>.db
└── sensor_blobs/<log>/CAM_F0/*.jpg
```

```bash
# Scripts for two logs; consecutive slices of one recording can be joined with a comma:
uv run jevsceneminer run --out out/nuplan --dry-run \
    --nuplan nuplan/data/cache/mini/<log>.db \
    --nuplan nuplan/data/cache/mini/<log_part1>.db,nuplan/data/cache/mini/<log_part2>.db

uv run jevsceneminer classify out/nuplan        # Jev
uv run jevsceneminer score out/nuplan           # vs nuPlan's scenario tags (and --gt, once labeled)
uv run jevsceneminer view out/nuplan --gt gt/ --sensor-root nuplan/sensor_blobs   # http://127.0.0.1:8650
```

What differs from rosbags:

- **No turn indicator:** nuPlan does not record it. Scripts say "not recorded", and the label definitions
  drop their indicator requirement (the `[[with||without]]` parts of `labels.yaml`).
- **Traffic side** comes from the city (Singapore drives on the left, the US cities on the right).
- **Traffic lights** are given per intersection lane, so the light ahead is the one of the lane the ego takes.
- **Scenario tags** (e.g. `starting_left_turn`, `stationary`) are nuPlan's automatic labels. `score` uses them
  as a rough check; they mark moments, not maneuver spans, so they are not GT.

### Rule baseline, scoring and the viewer

Every run also writes `rules/scenes/`: what plain geometry and speed thresholds find (turns through
intersection lanes with a real heading change, lane switches, speed thresholds), to see whether Jev beats
simple arithmetic.

`score` compares runs with GT scene files (same format as `scenes/`): per-label precision / recall
(a maneuver counts when it overlaps a GT maneuver of the same label by at least 0.5 s, one-to-one),
start / end timing, confusion in seconds, agreement over time and Jev's probability calibration.

`view` is a local page: front camera, a bird's-eye view of lanes, ego and objects, timeline bars for GT,
Jev and the rule baseline, Jev's probability, the scenario tags, and the script Jev saw. It is also the
labeling tool: pick a label, drag on the GT row, save (or start from a run and fix it).

## Roadmap

- Hand-labeled GT for about an hour of nuPlan driving, and the first Jev vs rules comparison on it.

## License

[Apache-2.0](LICENSE)

This is an independent project and is not affiliated with TypeSafe AI.
