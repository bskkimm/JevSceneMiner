"""Render the script Jev reads: what the ego vehicle does around one NOW moment.

The script only holds relative, derived facts (no map IDs, absolute positions or clock
times). Lanes get short names (A = the lane at NOW) and their relations are spelled out.
"""

from __future__ import annotations

import math
import re
from string import ascii_uppercase

import numpy as np

from .bag import INDICATOR_LEFT, INDICATOR_RIGHT
from .facts import DT_S, LANE_EVERY, Timeline
from .lanes import LaneMap, wrap

HEADER = ("Driving log of the ego vehicle around NOW (Japan, left-hand traffic). Times are relative "
          "to NOW. OFFSET = distance from the lane center (+ = left). HEADING = change since NOW "
          "(+ = left). YAW RATE + = turning left.")
INDICATOR_TEXT = {INDICATOR_LEFT: "LEFT", INDICATOR_RIGHT: "RIGHT"}
OBJECT_RADIUS_M = 30.0
MAX_OBJECTS = 8
MIN_EXISTENCE = 0.5
STOP_BELOW_MS, MOVE_ABOVE_MS = 0.1, 0.3
ROUTE_LOOK_S = 30.0    # route context looks this far before and after NOW (a turn can start 30 s before the intersection)


class Aliases:
    def __init__(self):
        self._names: dict[int, str] = {}

    def __call__(self, lane_id: int) -> str:
        if lane_id not in self._names:
            k = len(self._names)
            self._names[lane_id] = ascii_uppercase[k % 26] * (k // 26 + 1)
        return self._names[lane_id]

    def __contains__(self, lane_id: int) -> bool:
        return lane_id in self._names

    def items(self):
        return list(self._names.items())


def _t(seconds: float) -> str:
    return f"t={seconds:+.1f}s"


def _lane_kind(lanemap: LaneMap, lane_id: int) -> str:
    lane = lanemap.lanes[lane_id]
    if lane.is_intersection:
        return f"intersection lane, turn direction {lane.turn_direction.upper()}"
    return lane.kind


def _describe_lane(lanemap: LaneMap, names: Aliases, lane_id: int) -> str:
    lane = lanemap.lanes[lane_id]
    parts = [_lane_kind(lanemap, lane_id)]
    for side, nb, crossable in (("left", lane.left, lane.left_crossable), ("right", lane.right, lane.right_crossable)):
        if nb is None:
            parts.append(f"{side} neighbor: none")
        else:
            who = names(nb) if nb in names else f"a {lanemap.lanes[nb].kind} lane"
            line = "dashed line, can be crossed" if crossable else "solid line"
            parts.append(f"{side} neighbor: {who} ({line})")
    if lane.following:
        named = [names(f) for f in lane.following if f in names]
        others = [f for f in lane.following if f not in names]
        text = ", ".join(named)
        if others:
            kinds = ", ".join(sorted(_lane_kind(lanemap, f) for f in others))
            text += (" and " if named else "") + f"{len(others)} other lane(s) ({kinds})"
        parts.append(f"leads into: {text}")
        if len(lane.following) >= 2 and lane.kind != "shoulder":
            if any(not lanemap.lanes[f].is_intersection for f in lane.following):
                parts.append("the road forks here")
            else:
                parts.append("an intersection starts here")
    return f"{names(lane_id)}: " + "; ".join(parts)


def _transition_text(lanemap: LaneMap, names: Aliases, a: int | None, b: int | None) -> str:
    if a is None:
        return f"enters mapped lane {names(b)}"
    if b is None:
        return f"leaves the mapped lanes (from {names(a)})"
    rel = lanemap.relation(a, b)
    if rel == "left":
        return f"moves from {names(a)} into its left neighbor {names(b)}"
    if rel == "right":
        return f"moves from {names(a)} into its right neighbor {names(b)}"
    if rel in ("following", "near"):
        if lanemap.lanes[b].is_intersection:
            return f"continues from {names(a)} into {names(b)} ({_lane_kind(lanemap, b)})"
        return f"continues from {names(a)} into {names(b)}"
    if rel == "previous":
        return f"moves back from {names(a)} into {names(b)}"
    return f"moves from {names(a)} into {names(b)} (not directly connected)"


def render(tl: Timeline, lanemap: LaneMap, now_ns: int, past_s: int = 5, future_s: int = 10) -> str | None:
    """Script for one NOW moment, or None when there is no ego data at NOW."""
    i_now = tl.index(now_ns)
    if i_now is None:
        return None
    names = Aliases()
    now_lane = tl.lane(i_now)
    if now_lane is not None:
        names(now_lane.lane_id)
    yaw_now = tl.yaw[i_now]
    rel_s = lambda i: (i - i_now) * DT_S  # noqa: E731
    lo = max(0, i_now - int(round(past_s / DT_S)))
    hi = min(tl.n - 1, i_now + int(round(future_s / DT_S)))

    # Events at full resolution ------------------------------------------------------
    events: list[tuple[float, str]] = []
    prev = None
    first = True
    for i in range(lo - lo % LANE_EVERY, hi + 1, LANE_EVERY):
        if i < lo or not tl.valid[i]:
            continue
        p = tl.lane(i)
        cur = p.lane_id if p is not None else None
        if cur is not None:
            names(cur)
        if not first and cur != prev:
            events.append((rel_s(i), _transition_text(lanemap, names, prev, cur)))
        prev, first = cur, False

    t_lo, t_hi = tl.time_ns(lo), tl.time_ns(hi)
    s = tl.session
    ind_prev = tl.indicator_at(t_lo)
    for t, state in zip(s.indicator_t, s.indicator):
        if t_lo < t <= t_hi and state != ind_prev:
            what = f"indicator {INDICATOR_TEXT[state]} on" if state in INDICATOR_TEXT else "indicator off"
            events.append(((t - now_ns) / 1e9, what))
            ind_prev = state

    moving = tl.v[lo] > STOP_BELOW_MS
    for i in range(lo + 1, hi + 1):
        if moving and tl.v[i] < STOP_BELOW_MS:
            moving = False
            events.append((rel_s(i), "comes to a stop"))
        elif not moving and tl.v[i] > MOVE_ABOVE_MS:
            moving = True
            events.append((rel_s(i), "starts moving"))

    light_prev = tl.light_at(t_lo)
    for t, text in tl.lights:
        # "unknown" readings are dropouts, not changes of the light.
        if t_lo < t <= t_hi and text != light_prev and text != "unknown":
            events.append(((t - now_ns) / 1e9, f"traffic light changes from {light_prev} to {text}"))
            light_prev = text
    events.sort(key=lambda e: e[0])

    # Timeline, one line per second ---------------------------------------------------
    rows = []
    for k in range(-past_s, future_s + 1):
        t_ns = now_ns + int(k * 1e9)
        i = tl.index(t_ns)
        if i is None:
            rows.append(f"t={k:+d}s   (no data)")
            continue
        p = tl.lane(i)
        lane = names(p.lane_id) if p is not None else "off map"
        offset = f"{p.offset:+.1f} m" if p is not None else "-"
        ind = INDICATOR_TEXT.get(tl.indicator_at(t_ns), "off")
        row = (f"t={k:+d}s".ljust(7) + lane.ljust(9) + offset.ljust(9)
               + f"{tl.v[i] * 3.6:.0f} km/h".ljust(10) + f"{tl.accel[i]:+.1f} m/s²".ljust(12)
               + f"{math.degrees(tl.yaw_rate[i]):+.0f} °/s".ljust(10)
               + f"{math.degrees(tl.yaw[i] - yaw_now):+.0f}°".ljust(9) + ind.ljust(11) + tl.light_at(t_ns))
        rows.append(row + ("   <- NOW" if k == 0 else ""))

    # Window summary (numbers Jev should not have to compute) -------------------------
    summary = []
    if tl.valid[lo] and tl.valid[hi]:
        turn = math.degrees(tl.yaw[hi] - tl.yaw[lo])
        summary.append(f"heading change over the window: {turn:+.0f}°")
    k_brake = lo + int(tl.accel[lo:hi + 1].argmin())
    if tl.accel[k_brake] < -0.3:
        summary.append(f"strongest braking: {tl.accel[k_brake]:.1f} m/s² at {_t(rel_s(k_brake))}")
    k_jerk = lo + int(tl.jerk[lo:hi + 1].argmin())
    if tl.jerk[k_jerk] < -0.3:
        summary.append(f"sharpest braking onset (jerk): {tl.jerk[k_jerk]:.1f} m/s³ at {_t(rel_s(k_jerk))}")
    if now_lane is not None:
        offs = [tl.lane(i).offset for i in range(lo, hi + 1, LANE_EVERY)
                if tl.valid[i] and tl.lane(i) is not None and tl.lane(i).lane_id == now_lane.lane_id]
        if offs:
            summary.append(f"offset while in lane {names(now_lane.lane_id)}: {min(offs):+.1f} m to {max(offs):+.1f} m")
    entry = next((e for e in events if e[0] > 0 and "intersection lane" in e[1]), None)
    if entry is not None:
        i_entry = i_now + int(round(entry[0] / DT_S))
        dist = float(np.hypot(np.diff(tl.x[i_now:i_entry + 1]), np.diff(tl.y[i_now:i_entry + 1])).sum())
        summary.append(f"enters the intersection lane {dist:.0f} m after NOW ({_t(entry[0])})")
    elif now_lane is not None and lanemap.lanes[now_lane.lane_id].is_intersection:
        summary.append(f"NOW is inside intersection lane {names(now_lane.lane_id)}")

    # Objects at NOW --------------------------------------------------------------------
    objects = []
    objs = tl.objects_at(now_ns)
    if objs is None:
        objects.append("(no object data)")
    else:
        c, s_ = math.cos(-yaw_now), math.sin(-yaw_now)
        ex, ey = tl.x[i_now], tl.y[i_now]
        near = []
        for o in objs:
            if o.existence < MIN_EXISTENCE:
                continue
            dx, dy = o.x - ex, o.y - ey
            fx, fy = c * dx - s_ * dy, s_ * dx + c * dy
            dist = math.hypot(fx, fy)
            if dist <= OBJECT_RADIUS_M:
                near.append((dist, fx, fy, o))
        near.sort(key=lambda r: r[0])
        for dist, fx, fy, o in near[:MAX_OBJECTS]:
            where = f"{abs(fx):.0f} m {'ahead' if fx >= 0 else 'behind'}, {abs(fy):.0f} m {'left' if fy >= 0 else 'right'}"
            objects.append(f"{o.kind}: {where}, {_object_lane(lanemap, names, now_lane, o)}, "
                           + ("stopped" if o.speed < 0.5 else f"{o.speed * 3.6:.0f} km/h")
                           + (", oncoming" if o.speed >= 0.5 and abs(wrap(o.yaw - yaw_now)) > math.radians(135) else ""))
        if not near:
            objects.append("none within 30 m")

    route = _route_context(tl, lanemap, i_now)
    lanes_block = [_describe_lane(lanemap, names, lane_id) for lane_id, _ in names.items()]
    out = [HEADER, "", "LANES"] + (lanes_block or ["(ego is off the mapped lanes)"])
    out += ["", "TIME   LANE     OFFSET   SPEED     ACCEL       YAW RATE  HEADING  INDICATOR  LIGHT"] + rows
    out += ["", "EVENTS"] + ([f"{_t(t)}: {text}" for t, text in events] or ["none"])
    summary.insert(0, "indicator during the table: " + _indicator_text(tl, t_lo, t_hi, now_ns))
    out += ["", "SUMMARY"] + (summary or ["none"])
    out += ["", f"ROUTE CONTEXT (along the ego's actual path, up to {ROUTE_LOOK_S:.0f} s before/after NOW)"] + route
    out += ["", f"OBJECTS AT NOW (up to {MAX_OBJECTS} within {OBJECT_RADIUS_M:.0f} m)"] + objects
    return "\n".join(out)


def _object_lane(lanemap: LaneMap, names: Aliases, now_lane, o) -> str:
    heading_free = o.kind in ("pedestrian", "bicycle", "animal", "unknown")
    cands = lanemap.candidates(o.x, o.y, o.yaw, max_heading_diff=math.pi if heading_free else math.radians(60))
    if not cands:
        return "off the road"
    lane_id = min(cands, key=lanemap.emission_cost).lane_id
    if now_lane is None:
        return "in a lane"
    rel = lanemap.relation(now_lane.lane_id, lane_id)
    return {"same": "same lane", "left": "left neighbor lane", "right": "right neighbor lane",
            "following": "lane ahead", "near": "lane ahead", "previous": "lane behind"}.get(rel, "other lane")


def _route_context(tl: Timeline, lanemap: LaneMap, i_now: int) -> list[str]:
    """Where the ego's path meets intersections and stops, beyond the +-5 s table."""
    look = int(round(ROUTE_LOOK_S / DT_S))
    lo, hi = max(0, i_now - look), min(tl.n - 1, i_now + look)
    settle = int(round(2.0 / DT_S))   # heading change is measured until 2 s after leaving

    def turn_of(i):
        p = tl.lane(i) if tl.valid[i] else None
        return lanemap.lanes[p.lane_id].turn_direction if p is not None else None

    def path_m(i0, i1):
        a, b = sorted((i0, i1))
        return float(np.hypot(np.diff(tl.x[a:b + 1]), np.diff(tl.y[a:b + 1])).sum())

    rel = lambda i: _t((i - i_now) * DT_S)  # noqa: E731

    # Contiguous stretches of intersection lanes on the path, with their turn direction.
    spans, cur = [], None
    for i in range(lo - lo % LANE_EVERY, hi + 1, LANE_EVERY):
        if i < lo:
            continue
        d = turn_of(i)
        if d is not None and cur is not None and i - cur[1] <= LANE_EVERY:
            cur[1] = i
            cur[2].append(d)
        elif d is not None:
            cur = [i, i, [d]]
            spans.append(cur)
    lines = []
    for a, b, dirs in spans:
        direction = max(set(dirs), key=dirs.count).upper()
        turn = math.degrees(tl.yaw[min(b + settle, tl.n - 1)] - tl.yaw[a])
        what = f"turn direction {direction}, heading change {turn:+.0f}°"
        if a <= i_now <= b:
            lines.append(f"NOW is inside an intersection lane ({what}): entered at {rel(a)}, leaves at {rel(b)}")
        elif a > i_now:
            lines.append(f"intersection ahead: enters at {rel(a)}, {path_m(i_now, a):.0f} m ahead ({what})")
        else:
            lines.append(f"intersection behind: left at {rel(b)}, {path_m(b, i_now):.0f} m ago ({what})")

    stopped = lambda i: tl.v[i] < STOP_BELOW_MS  # noqa: E731

    def where(i):
        p = tl.lane(i) if tl.valid[i] else None
        return f"in a {lanemap.lanes[p.lane_id].kind} lane, offset {p.offset:+.1f} m" if p is not None else "off the mapped lanes"

    if stopped(i_now):
        a = i_now
        while a > lo and stopped(a - 1):
            a -= 1
        b = i_now
        while b < hi and stopped(b + 1):
            b += 1
        lines.append(f"NOW the ego is stopped ({where(i_now)}): stopped from {rel(a)} to {rel(b)}")
    else:
        nxt = next((i for i in range(i_now + 1, hi + 1) if stopped(i)), None)
        if nxt is not None:
            lines.append(f"next stop: the ego stops at {rel(nxt)}, {path_m(i_now, nxt):.0f} m ahead, {where(nxt)}")
        prv = next((i for i in range(i_now - 1, lo - 1, -1) if stopped(i)), None)
        if prv is not None:
            lines.append(f"last stop: the ego started moving at {rel(prv)}, {path_m(prv, i_now):.0f} m ago, {where(prv)}")
    lines.insert(0, "indicator within this range: "
                 + _indicator_text(tl, tl.time_ns(lo), tl.time_ns(hi), tl.time_ns(i_now)))
    return lines


def _indicator_text(tl: Timeline, t_lo: int, t_hi: int, now_ns: int) -> str:
    """LEFT/RIGHT periods between t_lo and t_hi, e.g. "LEFT from t=-3.2s to t=+4.0s", or "never on"."""
    s = tl.session
    periods, state, start = [], tl.indicator_at(t_lo), t_lo
    for t, st in zip(s.indicator_t, s.indicator):
        if t_lo < t <= t_hi and st != state:
            if state in INDICATOR_TEXT:
                periods.append((state, start, int(t)))
            state, start = int(st), int(t)
    if state in INDICATOR_TEXT:
        periods.append((state, start, t_hi))
    if not periods:
        return "never on (no LEFT or RIGHT)"
    return "; ".join(f"{INDICATOR_TEXT[st]} from {_t((a - now_ns) / 1e9)} to {_t((b - now_ns) / 1e9)}"
                     for st, a, b in periods)


_MAP_EVENT = ("moves from ", "continues from ", "enters mapped lane", "leaves the mapped lanes", "moves back from ")
_MAP_SUMMARY = ("offset while in lane", "enters the intersection lane", "NOW is inside intersection lane")
_MAP_ROUTE = ("intersection ahead", "intersection behind", "NOW is inside an intersection lane")
_LANE_OF_OBJECT = re.compile(r", (same lane|left neighbor lane|right neighbor lane|lane ahead|lane behind|"
                             r"other lane|off the road|in a lane)(?=, )")
_WHERE_STOP = re.compile(r"(, in a [a-z ]+ lane, offset [+-][0-9.]+ m|, off the mapped lanes)")
_WHERE_STOPPED = re.compile(r" \((in a [a-z ]+ lane, offset [+-][0-9.]+ m|off the mapped lanes)\)")
_LANE_COLUMNS = slice(7, 25)   # LANE (9 chars) + OFFSET (9 chars) after the 7-char TIME column


def strip_map(text: str) -> str:
    """The same script with everything that comes from the map removed (for a no-map ablation).

    Lanes, lane offsets, lane-change events, intersections and the lane relation of objects
    go; speed, acceleration, yaw rate, heading, indicator, light and object positions stay.
    """
    out, section = [], None
    for line in text.split("\n"):
        if line == "LANES":
            section = "LANES"
            continue
        if section == "LANES":
            if line == "":
                section = None
            continue
        if line.startswith("TIME "):
            section = "TABLE"
        elif line in ("EVENTS", "SUMMARY") or line.startswith(("ROUTE CONTEXT", "OBJECTS AT NOW")):
            section = line.split(" ")[0]
            out.append(line)
            continue
        elif line == "" and section == "TABLE":
            section = None
        if section == "TABLE" and "(no data)" not in line:
            line = line[:_LANE_COLUMNS.start] + line[_LANE_COLUMNS.stop:]
        elif section == "EVENTS" and any(k in line for k in _MAP_EVENT):
            continue
        elif section == "SUMMARY" and line.startswith(_MAP_SUMMARY):
            continue
        elif section == "ROUTE" and line.startswith(_MAP_ROUTE):
            continue
        elif section == "ROUTE":
            line = _WHERE_STOPPED.sub("", _WHERE_STOP.sub("", line))
        elif section == "OBJECTS":
            line = _LANE_OF_OBJECT.sub("", line)
        out.append(line)
    text = "\n".join(out).replace(" OFFSET = distance from the lane center (+ = left).", "")
    return text.replace("EVENTS\n\n", "EVENTS\nnone\n\n")
