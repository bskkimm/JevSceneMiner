# JevSceneMiner

Find interesting scenes in driving logs with [Jev](https://docs.typesafe.ai/introduction), a fast general-purpose classifier from [TypeSafe AI](https://typesafe.ai/).

> **Status:** early proof of concept. The input data format (e.g. rosbag or OSM) is not decided yet.

## Goal

Driving logs run for hours, but the moments worth studying, such as a car cutting in, a pedestrian crossing, or a hard brake, last only seconds. Finding them by hand is slow, and hand-writing a rule for every scenario doesn't scale.

JevSceneMiner describes each part of a log as short text and lets Jev decide which scenarios it contains. Scenarios are defined in plain language, so adding a new one doesn't require training a new model.

## Pipeline

```
driving log
    │  split into short time windows
    ▼
scene text  ──────►  Jev  ──────►  scenes
 • ego state                        • scenario
 • surrounding objects              • probability
 • map (optional)                   • start / end time
```

1. **Scene text:** describe each time window in text.
   - **Ego state:** speed, acceleration, and lane of the ego vehicle (the car that recorded the log)
   - **Surrounding objects:** nearby vehicles, pedestrians, and cyclists, and where they are relative to the ego vehicle
   - **Map (optional):** nearby lanes, intersections, and crosswalks
2. **Jev:** ask Jev about each scenario, e.g. "Is a vehicle cutting in front of the ego vehicle?" Jev returns a probability for each one.
3. **Scenes:** keep the windows where a scenario's probability is high, and report them with the window's start and end time.

Example output (made-up numbers):

| Scenario | Probability | Start | End |
|---|---|---|---|
| cut_in | 0.91 | 12.0 s | 22.0 s |
| pedestrian_crossing | 0.84 | 95.0 s | 105.0 s |

## License

[Apache-2.0](LICENSE)

This is an independent project and is not affiliated with TypeSafe AI.
