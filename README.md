<div align="center">

# JevSceneMiner

**From driving logs to reviewable scenes.**

[![Quality checks](https://github.com/bskkimm/JevSceneMiner/actions/workflows/ci.yml/badge.svg)](https://github.com/bskkimm/JevSceneMiner/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

[Demo](#boston-demo) · [Quick start](#quick-start) · [Usage](docs/usage.md) · [Schema](docs/schema.md) · [Contributing](docs/CONTRIBUTING.md)

</div>

Mine driving maneuvers from recorded vehicle data with [Jev](https://docs.typesafe.ai/introduction).
Convert motion, lane geometry and object interactions into shared readable evidence, then classify each moment and merge it into timestamped scenes.

Read the [project write-up on Medium](https://medium.com/@bskkim2022/turning-driving-logs-into-reviewable-scenes-with-jev-77b8c6977e6c) for the motivation, approach and lessons learned.

<p align="center">
  <img src="https://github.com/user-attachments/assets/377da8f2-5185-469d-9325-9f0da8dea1b5" width="960" alt="Boston highlights: turn right, lane change and lane following">
</p>

<p align="center"><em>Turn right → lane change → keep lane</em></p>

## Boston demo

Press Play below. If playback stalls after this page has been open for a while, refresh the page and press Play again.

https://github.com/user-attachments/assets/4b97b6ed-595d-4b18-beac-87b1e636cca4

*Real nuPlan footage and Jev results. 1080p · 1 min 22 s · 3.5× playback · long stops shortened.*
[Demo details and data attribution](docs/demo.md).

## What you can do

- **Shared evidence.** Input adapters produce the same [sample schema](docs/schema.md); missing evidence stays unknown.
- **Editable labels.** Define [lateral and longitudinal decisions](configs/labels.yaml) in plain language, with probabilities for each answer.
- **Complete maneuvers.** One lateral scene contains consecutive speed phases, such as decelerating, cruising and accelerating.
- **Evidence review.** Synchronized camera, bird's-eye view (BEV), timeline, raw probabilities and ground-truth editing.

## Why Jev

| Benefit | Detail |
| --- | --- |
| Fast batch inference | **933 fresh samples in 25.9 s**, using 8 workers. |
| Low inference cost | **Estimated $0.32** for that run, about **$0.00034/sample**. |
| Cloud-based inference | Local preprocessing and review; readable evidence goes to TypeSafe's API. No local inference GPU. |

One recorded Boston run with Jev 1.13; lateral and longitudinal questions share one request.
See [measurement conditions](docs/evaluation.md#recorded-boston-run) and [pricing](https://docs.typesafe.ai/models). Cost varies with context and questions.

## What scene mining enables

Use reviewed scene labels for:

- **Data curation:** measure scene coverage and guide further collection.
- **Scenario-based training:** balance datasets and target weak driving behaviors.
- **Scenario-based validation:** organize driving-performance metrics by scene.
- **Failure analysis:** find maneuver patterns around recorded incidents.
- **Regression testing:** reuse reviewed scene windows across model/software versions.
- **Dataset search and comparison:** find maneuvers and compare scenario coverage.

## Roadmap

**Hybrid Jev + rule-based reasoning (planned):** combine model probabilities with motion, map and temporal checks to improve accuracy and consistency. Measure gains against independent, held-out ground truth.

## Pipeline

```text
Autoware MCAP + Lanelet2 ─┐
                          ├─► preprocess ─► structured sample ─► Jev
nuPlan SQLite + map ──────┘                   facts + script      │
                                                                  ▼
                                                          raw probabilities
                                                                  │
                                                  merge lateral scenes + speed phases
                                                                  │
                                                     camera / BEV / timeline / GT
```

Jev receives the **readable script**, with past and future observations around NOW.
Numerical facts remain alongside it for inspection. Camera images are used for review.

```text
One lateral scene:       lane_change_right
Consecutive speed phases: decelerating │ cruising │ accelerating
Output:                  start_ns / end_ns / driving_decision / longitudinal_phases
```

See a complete [sample](examples/sample.json), its [Jev input](examples/sample.txt) and [merged scenes](examples/scenes.json).

### Input adapters

Built-in adapters support **nuPlan logs + maps** and **Autoware-topic MCAP rosbags + Lanelet2 maps**.

Labs and companies can adapt their own recordings by developing preprocessing that produces the shared [sample contract](docs/schema.md) and [session metadata](src/jevsceneminer/cli.py#L132); classification and scene merging can then be reused.
Raw-format loading, coordinate alignment, map matching and camera review may need source-specific integration.

## Quick start

Linux · Python 3.10+ · [uv](https://docs.astral.sh/uv/getting-started/installation/).
Try the complete synthetic pipeline **without an API key**:

```bash
git clone https://github.com/bskkimm/JevSceneMiner.git
cd JevSceneMiner
uv sync --frozen
uv run python examples/pipeline_demo.py --source rosbag --out out/demo
uv run jevsceneminer view out/demo --gt out/demo-gt --port 8650
```

Open **http://127.0.0.1:8650**. The demo generates an MCAP, a map, processed samples and illustrative cached answers; API requests are blocked.
It includes timeline and script data. Camera review requires your own nuPlan sources and images.
Use `--source nuplan` with a new output directory to try the other adapter.

### Run on supported inputs

```bash
# Prepare inputs without calling the API.
uv run jevsceneminer run \
  --nuplan /path/to/log.db --nuplan-maps /path/to/maps \
  --preset nuplan-1hz --out out/nuplan --dry-run

# Set TYPESAFE_API_KEY in your environment or an ignored .env file first.
uv run jevsceneminer classify out/nuplan --maneuver-tail 2
uv run jevsceneminer view out/nuplan --gt out/nuplan-gt \
  --sensor-root /path/to/sensor_blobs --port 8650
```

The explicit `nuplan-1hz` preset uses **10 s past + 15 s future**, **1 Hz inference** and a **1 Hz table**.
Without a preset, defaults are 10 s past + 10 s future at 2 Hz.
See [usage](docs/usage.md) for rosbag inputs, options, outputs and remote viewing.

## Documentation

| Guide | Contents |
| --- | --- |
| [Usage](docs/usage.md) | Source setup, configuration, outputs and viewer |
| [Sample and scene schema](docs/schema.md) | Context, lane/body evidence and consecutive phases |
| [Evidence review](docs/review.md) | Opt-in annotations, missing actor history and selected reruns |
| [Evaluation](docs/evaluation.md) | Offline benchmark, independent GT and limitations |
| [Repository layout](docs/architecture.md) | Source groups and import compatibility |
| [Contributing](docs/CONTRIBUTING.md) | Development checks and contribution contracts |
| [Changelog](docs/CHANGELOG.md) | Release history |

## Status and license

**Experimental, offline scene mining:** future observations provide context.
Map matching and polygon overlap are evidence, with physical lane ownership subject to source quality.
The demo is a reviewed example; representative accuracy requires [held-out evaluation](docs/evaluation.md).

Code and original synthetic examples: [Apache-2.0](LICENSE).
External recordings, maps and images retain their source terms. No raw driving dataset is included.
This independent project is not affiliated with TypeSafe AI or the nuPlan authors.
