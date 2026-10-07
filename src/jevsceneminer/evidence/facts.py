"""Per-session signals derived from the raw messages, in plain units.

Everything the script states is computed here, from the full-rate data: Jev is not asked
to do arithmetic.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np

from jevsceneminer.inputs.bag import INDICATOR_OFF, DetectedObject, Session
from jevsceneminer.evidence.lanes import LaneMap, Projection

DT_S = 0.02            # ego signals are resampled to 50 Hz
LANE_EVERY = 5         # lanes are matched every 5th sample (10 Hz)
MAX_EGO_GAP_S = 0.2    # a sample further than this from an ego message has no data
SMOOTH_S = 0.5         # moving-average window for speed, acceleration, jerk, yaw rate
LIGHT_AHEAD_M = 150.0  # a traffic light further along the ego's path than this is not "the" light


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < 2:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, window - 1 - pad), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


@dataclass
class Timeline:
    t0_ns: int
    valid: np.ndarray       # bool per 50 Hz sample
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray         # rad, unwrapped
    v: np.ndarray           # m/s, smoothed
    accel: np.ndarray       # m/s^2
    jerk: np.ndarray        # m/s^3
    yaw_rate: np.ndarray    # rad/s, smoothed
    lanes: list             # Projection | None per 10 Hz sample
    session: Session
    lights: list[tuple[int, str]]   # the light ahead of the ego at each light message
    object_times: list[int]

    @property
    def n(self) -> int:
        return len(self.x)

    def time_ns(self, i: int) -> int:
        return self.t0_ns + int(round(i * DT_S * 1e9))

    def index(self, t_ns: int) -> int | None:
        i = int(round((t_ns - self.t0_ns) / 1e9 / DT_S))
        if 0 <= i < self.n and self.valid[i]:
            return i
        return None

    def lane(self, i: int) -> Projection | None:
        return self.lanes[min(len(self.lanes) - 1, int(round(i / LANE_EVERY)))]

    def indicator_at(self, t_ns: int) -> int:
        s = self.session
        k = bisect.bisect_right(s.indicator_t, t_ns) - 1
        return int(s.indicator[k]) if k >= 0 else INDICATOR_OFF

    def light_at(self, t_ns: int, max_age_s: float = 1.5) -> str:
        k = bisect.bisect_right(self.lights, (t_ns, "\uffff")) - 1
        if k < 0 or t_ns - self.lights[k][0] > max_age_s * 1e9:
            return "unknown"
        return self.lights[k][1]

    def objects_at(self, t_ns: int, max_dt_s: float = 0.3) -> list[DetectedObject] | None:
        times = self.object_times
        k = bisect.bisect_left(times, t_ns)
        near = [j for j in (k - 1, k) if 0 <= j < len(times) and abs(times[j] - t_ns) <= max_dt_s * 1e9]
        if not near:
            return None
        return self.session.objects[min(near, key=lambda j: abs(times[j] - t_ns))][1]


def lights_ahead(lanes: list, x: np.ndarray, y: np.ndarray, lanemap: LaneMap) -> list[tuple[int, ...]]:
    """Per lane sample: the traffic light ids of the next lane with lights on the ego's own path.

    The path is the one the ego actually drives (the logs are offline), so the light is
    the one it is approaching, not a guess from the map's branches.
    """
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    out: list[tuple[int, ...]] = [()] * len(lanes)
    ahead, ahead_s = (), None
    for k in range(len(lanes) - 1, -1, -1):
        p = lanes[k]
        ids = lanemap.lanes[p.lane_id].traffic_lights if p is not None and p.lane_id in lanemap.lanes else ()
        if ids:
            ahead, ahead_s = ids, s[k]
        out[k] = ahead if ahead_s is not None and ahead_s - s[k] <= LIGHT_AHEAD_M else ()
    return out


def resolve_lights(session: Session, t0_ns: int, ahead: list[tuple[int, ...]]) -> list[tuple[int, str]]:
    """Each light message reduced to the text of the light ahead ("none" when there is none)."""
    out = []
    for t, groups in session.lights:
        if None in groups:                       # the topic already gives the light ahead
            out.append((t, groups[None]))
            continue
        k = min(len(ahead) - 1, max(0, int(round((t - t0_ns) / 1e9 / (DT_S * LANE_EVERY)))))
        ids = ahead[k] if ahead else ()
        if not ids:
            out.append((t, "none"))
        else:
            out.append((t, next((groups[i] for i in ids if i in groups), "unknown")))
    return out


def build_timeline(session: Session, lanemap: LaneMap) -> Timeline:
    e = session.ego
    ts = (e.t - e.t[0]) / 1e9
    grid = np.arange(0.0, ts[-1] + 1e-9, DT_S)
    right = np.clip(np.searchsorted(ts, grid), 0, len(ts) - 1)
    left = np.clip(right - 1, 0, len(ts) - 1)
    gap = np.minimum(np.abs(grid - ts[left]), np.abs(ts[right] - grid))
    valid = gap <= MAX_EGO_GAP_S

    w = int(round(SMOOTH_S / DT_S))
    x, y = np.interp(grid, ts, e.x), np.interp(grid, ts, e.y)
    yaw = np.interp(grid, ts, e.yaw)
    v = moving_average(np.interp(grid, ts, e.v), w)
    accel = moving_average(np.gradient(v, DT_S), w)
    # Jerk is smoothed over a longer window: it is the noisiest signal.
    jerk = moving_average(np.gradient(accel, DT_S), 2 * w)
    yaw_rate = moving_average(np.interp(grid, ts, e.yaw_rate), w)

    sub = slice(None, None, LANE_EVERY)
    lanes = lanemap.match_track(x[sub], y[sub], yaw[sub], valid[sub])
    ahead = lights_ahead(lanes, x[sub], y[sub], lanemap)
    return Timeline(int(e.t[0]), valid, x, y, yaw, v, accel, jerk, yaw_rate, lanes, session,
                    lights=resolve_lights(session, int(e.t[0]), ahead),
                    object_times=[t for t, _ in session.objects])
