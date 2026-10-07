# Run and review your logs

Set `TYPESAFE_API_KEY` in your environment or an ignored `.env` file.
`--dry-run` prepares inputs without inference; remove it or run `classify` when ready to call the API.
Download logs, maps and optional images separately under their source terms. No driving dataset is included.

### nuPlan

```bash
uv run jevsceneminer run \
  --nuplan /path/to/log.db --nuplan-maps /path/to/maps \
  --preset nuplan-1hz --out out/nuplan --dry-run
uv run jevsceneminer classify out/nuplan --maneuver-tail 2
uv run jevsceneminer view out/nuplan --gt out/nuplan-gt \
  --sensor-root /path/to/sensor_blobs --port 8650
```

| Explicit `nuplan-1hz` preset | Value |
| --- | --- |
| Context | 10 s past, 15 s future |
| Inference / table | 1 Hz / 1 Hz (26 rows in a complete window) |
| Model / concurrent workers | `jev-1.13.0` / 8 |
| Minimum lateral scene / speed phase | 2 s / 1 s |
| Minimum mean lateral support | 0.4 |
| Maneuver tail | Up to 2 s into following keep-lane |
| Camera height reference | Stationary landmarks; approximate 0.24 m offset |

Explicit flags override preset values. Add `--table-step 0.5` for a 2 Hz table with 1 Hz inference.
Without a preset, defaults remain 10 s past / 10 s future, 2 Hz inference / 2 Hz table, no tail, and raw camera height.
Height correction is display-only and approximate.
Separate `classify` commands inherit the prepared model; pass merging flags explicitly.

### Autoware rosbag

```bash
uv run jevsceneminer run /path/to/session \
  --map /path/to/lanelet2_map.osm --traffic-side left \
  --past 10 --future 15 --step 1 --table-step 1 \
  --model jev-1.13.0 --out out/autoware --dry-run
```

The map must contain `local_x` / `local_y` in the logged pose frame.
Autoware topics are discovered automatically; use `--topics` for overrides.
Missing fields stay unknown. For footprint evidence, configure a known ego model:
`--ego-length 4.8 --ego-width 2.0 --ego-center-offset 0` when the pose is at body center.
nuPlan uses its Pacifica dimensions and rear-axle reference.

### Open a remote viewer on your laptop

Run `view` on the machine with the data. On your laptop:

```bash
ssh -N -o ExitOnForwardFailure=yes -L 18656:127.0.0.1:8650 your-machine
```

Keep SSH open and visit **http://127.0.0.1:18656**. Replace the alias and remote port for your machine.

## Outputs and review

```text
out/run/
├── meta/        sources, geometry, cadence, context and display settings
├── steps/       JSONL: script, facts, raw answers/probabilities at each NOW
├── cache/       answers keyed by model + exact questions + exact script
├── scenes/      lateral intervals with consecutive speed phases
├── runtime/     latest inference timing, fresh tokens and estimated cost
├── indicator/   recorded indicator periods
├── rules/       simple geometry/speed baseline
└── camera/, tags/  optional camera index and scenario tags
```

The camera has BEV at top left, **Lateral Decision** and **Longitudinal Decision** probabilities at top right, and the merged decision at the bottom.
Panels update at 1 Hz during playback.
Lateral colors agree across camera, BEV and timeline; longitudinal winners share one color.
The BEV spans 50 m across and 60 m vertically, with ego 95% down.
Projection uses image-capture pose and calibration: a recorded future path, not a forecast or measured road surface.

**Edit GT** supports adding, splitting, merging and relabeling scenes and editing consecutive speed phases.
Changes save only when you select **Save**. Legacy flat scenes remain readable.
The `?mock=1` preview uses fictional browser-only answers.

```bash
# Rebuild scenes without calling Jev.
uv run jevsceneminer restitch out/nuplan --out out/restitch --maneuver-tail 2

# Rerun selected NOWs: elapsed seconds, inclusive start / exclusive end.
uv run jevsceneminer classify out/nuplan --from-s 100 --to-s 120 --maneuver-tail 2

# Compare with independently labeled scenes.
uv run jevsceneminer score out/nuplan --gt out/nuplan-gt
```

Ranged inference preserves outside raw rows and rebuilds scenes from all answers.
It requires compatible outside answers and applies to each session in the folder.
Read [the evidence review guide](review.md) before correcting evidence or using the optional compact experiment.

## Labels

[labels.yaml](../labels.yaml) defines the two Jev questions in plain language.

| Lateral decision | Longitudinal phase |
| --- | --- |
| `keep_lane`, `turn_left`, `turn_right`, `u_turn` | `stopped`, `accelerating`, `decelerating` |
| `lane_change_left`, `lane_change_right` | `hard_braking`, `cruising` |
| `branch_left`, `branch_right` | |
| `avoidance`, `pull_over`, `pull_away` | |

Changing definitions changes cache keys. Indicator-dependent text is omitted when indicators were not recorded.
The optional `--require-indicator` merge gate is disabled by default.


For the shared sample and scene contracts, see [schema.md](schema.md). For development checks, see [CONTRIBUTING.md](../CONTRIBUTING.md).
