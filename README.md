# JevSceneMiner

Find interesting scenes in driving logs with [Jev](https://docs.typesafe.ai/introduction), a fast general-purpose classifier from [TypeSafe AI](https://typesafe.ai/).

> **Status:** proof of concept. v1 reads Autoware rosbags (MCAP) and a Lanelet2 map, and labels the ego vehicle's driving decisions.

## Goal

Driving logs run for hours, but the moments worth studying, such as a car cutting in, a pedestrian crossing, or a hard brake, last only seconds. Finding them by hand is slow, and hand-writing a rule for every scenario doesn't scale.

JevSceneMiner describes each moment of a log as a short text **script** and lets Jev decide which scenario it belongs to. Scenarios are defined in plain language, so adding a new one doesn't require training a new model.

## Pipeline

```
rosbag + Lanelet2 map
    │  every 0.5 s: a script of the 5 s before and 10 s after NOW
    ▼
script  ──────►  Jev  ──────►  scenes
 • ego state                    • lateral / longitudinal label
 • lanes and route              • probability
 • traffic light, indicator     • start / end time
 • surrounding objects
```

1. **Script:** facts computed from the full-rate data (Jev is not asked to do arithmetic): lane and lane changes, speed, acceleration, yaw rate, heading change, indicator, traffic light, the intersections and stops along the ego's path, and nearby objects. Only relative facts: no map IDs, positions or clock times.
2. **Jev:** two Choice questions per script (10 lateral labels such as `turn_left`, `u_turn` and `lane_change_right`; 5 longitudinal labels such as `cruising` and `hard_braking`), with the definitions in [`labels.yaml`](labels.yaml). Jev returns a probability for every label.
3. **Scenes:** consecutive steps with the same labels become one scene with start/end times; short flickers and low-confidence maneuvers are smoothed away.

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

# Right-hand traffic (e.g. the US): flips which indicator a U-turn, pull-over and pull-away need
uv run jevsceneminer run ... --traffic-side right

uv run pytest
```

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
├── indicator/<session>.json   periods with the turn indicator on: [start_s, end_s, 2=LEFT | 3=RIGHT]
├── meta/<session>.json        session span and the settings used
└── cache/                     Jev answers keyed by script + questions
```

## License

[Apache-2.0](LICENSE)

This is an independent project and is not affiliated with TypeSafe AI.
