"""A simple rule baseline (no Jev): what plain geometry and speed thresholds find.

    turn_left / turn_right / u_turn   the ego drives through an intersection lane marked as a
                                      turn and its heading changes by 40 degrees or more from
                                      20 m before to 20 m after it (more than 135: u_turn);
                                      from 30 m before it enters until 2 s after it leaves
    lane_change_left / _right         the matched lane switches to its left / right neighbor:
                                      2.5 s on either side of the switch
    stopped / hard_braking /          speed and acceleration thresholds that follow the
    accelerating / decelerating /     definitions in labels.yaml
    cruising

Everything else is keep_lane. Avoidance and pull-over / pull-away have no rule.
"""

from __future__ import annotations

import math

import numpy as np

from jevsceneminer.evidence.facts import DT_S, LANE_EVERY, Timeline
from jevsceneminer.inference.jev import Answer
from jevsceneminer.evidence.lanes import LaneMap
from jevsceneminer.scenes.merge import Scene, Step, stitch

TURN_LEAD_M = 30.0
TURN_MEASURE_M = 20.0          # heading change is measured this far before and after the lane
MIN_TURN_DEG = 40.0
U_TURN_DEG = 135.0
TURN_TAIL_S = 2.0
LANE_CHANGE_HALF_S = 2.5
STOPPED_BELOW_MS = 0.3
SPEED_CHANGE_MS = 1.0          # over the +-2 s window, as in labels.yaml
HARD_BRAKING_MS2 = -3.0


def lateral_spans(tl: Timeline, lanemap: LaneMap) -> list[tuple[float, float, str]]:
    """(start_s, end_s, label) in seconds since the timeline start, turns first."""
    lanes = tl.lanes
    x, y, yaw = tl.x[::LANE_EVERY], tl.y[::LANE_EVERY], tl.yaw[::LANE_EVERY]
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])[:len(lanes)]
    dt = DT_S * LANE_EVERY
    spans = []
    k = 0
    while k < len(lanes):
        p = lanes[k]
        direction = lanemap.lanes[p.lane_id].turn_direction if p is not None else None
        if direction in ("left", "right", "u-turn"):
            j = k
            while j + 1 < len(lanes) and lanes[j + 1] is not None and lanes[j + 1].lane_id == p.lane_id:
                j += 1
            a = int(np.searchsorted(s, s[k] - TURN_MEASURE_M))
            b = min(len(s) - 1, int(np.searchsorted(s, s[j] + TURN_MEASURE_M)))
            change = math.degrees(yaw[b] - yaw[a])
            if abs(change) >= MIN_TURN_DEG:
                label = "u_turn" if abs(change) > U_TURN_DEG else ("turn_left" if change > 0 else "turn_right")
                start = int(np.searchsorted(s, s[k] - TURN_LEAD_M))
                spans.append((start * dt, j * dt + TURN_TAIL_S, label))
            k = j + 1
            continue
        k += 1
    for k in range(1, len(lanes)):
        a, b = lanes[k - 1], lanes[k]
        if a is None or b is None or a.lane_id == b.lane_id:
            continue
        rel = lanemap.relation(a.lane_id, b.lane_id)
        if rel in ("left", "right"):
            spans.append((k * dt - LANE_CHANGE_HALF_S, k * dt + LANE_CHANGE_HALF_S, f"lane_change_{rel}"))
    return spans


def longitudinal_at(tl: Timeline, i: int) -> str:
    if tl.v[i] < STOPPED_BELOW_MS:
        return "stopped"
    if tl.accel[i] < HARD_BRAKING_MS2:
        return "hard_braking"
    w = int(round(2.0 / DT_S))
    change = tl.v[min(tl.n - 1, i + w)] - tl.v[max(0, i - w)]
    if change > SPEED_CHANGE_MS:
        return "accelerating"
    if change < -SPEED_CHANGE_MS:
        return "decelerating"
    return "cruising"


def rule_scenes(tl: Timeline, lanemap: LaneMap, times: list[int], step_s: float, min_scene_s: float,
                start_ns: int, end_ns: int) -> list[Scene]:
    spans = lateral_spans(tl, lanemap)
    steps = []
    for t in times:
        i = tl.index(t)
        if i is None:
            continue
        rel_s = (t - tl.t0_ns) / 1e9
        # Turns were added first, so a turn wins over a lane change inside it.
        lat = next((label for a, b, label in spans if a <= rel_s <= b), "keep_lane")
        lon = longitudinal_at(tl, i)
        steps.append(Step(t, Answer(lat, {lat: 1.0}, lon, {lon: 1.0})))
    return stitch(steps, step_s, min_scene_s, start_ns, end_ns)
