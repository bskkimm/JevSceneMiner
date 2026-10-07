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
from . import sample_context as context
from .lane_evidence import LaneEvidence

SAMPLE_SCHEMA_VERSION = "jsm-sample-1.1"

CONTEXT_NOTE = " ROAD CONTEXT is a lane-map fact; map match unavailable means missing evidence, not a non-intersection. "
HEADER = ("Driving log of the ego vehicle around NOW ({side}-hand traffic). "
          "Classify the observed lateral maneuver active at t=0; past and future are context. "
          "Classify longitudinal behavior separately using LOCAL SPEED CONTEXT. "
          "Times are relative to NOW. OFFSET = distance from the lane center (+ = left). "
          "RELATIVE HEADING = unwrapped yaw(t) - yaw(NOW), positive left; "
          "its sign alone does not indicate turning direction. YAW RATE + = turning left. "
          "Map occupancy boundaries do not define maneuver start/end." + CONTEXT_NOTE)
INDICATOR_TEXT = {INDICATOR_LEFT: "LEFT", INDICATOR_RIGHT: "RIGHT"}
OBJECT_RADIUS_M = 30.0
MAX_OBJECTS = 8
MIN_EXISTENCE = 0.5
STOP_BELOW_MS, MOVE_ABOVE_MS = 0.1, 0.3
TABLE_STEP_S = 0.5   # table cadence; independent of the interval between inference samples


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
    precision = 1 if math.isclose(seconds * 10, round(seconds * 10), abs_tol=1e-8) else 2
    return f"t={seconds:+.{precision}f}s"


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


def _z(value: float, digits: int) -> float:
    """``value`` rounded, with -0 turned into 0 (a "-0 km/h" reads like reversing)."""
    return round(value, digits) + 0.0


def _path_heading(points: np.ndarray, distance_m: float, from_end: bool = False) -> float | None:
    """Map heading over a short arc-length span, independent of point spacing."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        return None
    if from_end:
        points = points[::-1]
    lengths = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    if lengths[-1] < 1e-3:
        return None
    target = min(distance_m, lengths[-1])
    end = np.array([np.interp(target, lengths, points[:, k]) for k in (0, 1)])
    delta = end - points[0]
    if from_end:
        delta = -delta
    if np.linalg.norm(delta) < 1e-3:
        return None
    return math.atan2(delta[1], delta[0])


def _fork_context(tl: Timeline, lanemap: LaneMap, names: Aliases, i_now: int, lo: int, hi: int) -> list[str]:
    """Only observed, direct successor transitions at non-intersection forks."""
    out, previous = [], None
    for i in range(lo - lo % LANE_EVERY, hi + 1, LANE_EVERY):
        if i < lo:
            continue
        # Do not infer a transition across missing ego or lane matches.
        if not tl.valid[i] or (previous is not None and not tl.valid[previous[0]:i + 1].all()):
            previous = None
            continue
        projection = tl.lane(i)
        if projection is None:
            previous = None
            continue
        current = projection.lane_id
        if previous is not None and current != previous[1]:
            source = lanemap.lanes[previous[1]]
            successors = tuple(dict.fromkeys(source.following))
            branches = [lanemap.lanes.get(k) for k in successors]
            if (current in successors and len(successors) >= 2 and not source.is_intersection
                    and source.kind != "shoulder" and all(b is not None and not b.is_intersection
                                                         and b.kind != "shoulder" for b in branches)):
                heading = _path_heading(source.centerline, 10.0, from_end=True)
                headings = [_path_heading(b.centerline, 20.0) for b in branches]
                if all(h is not None for h in headings):
                    # Compare the same supported distance: a short straight prefix must
                    # not become a different branch merely because its sibling extends.
                    common_span = min(20.0, min(float(np.linalg.norm(np.diff(b.centerline, axis=0),
                                                                     axis=1).sum()) for b in branches))
                    headings = ([_path_heading(b.centerline, common_span) for b in branches]
                                if common_span >= 5.0 else [None] * len(branches))
                if heading is not None and all(h is not None for h in headings):
                    angles = [math.degrees(wrap(h - heading)) for h in headings]
                    low, high = min(angles), max(angles)
                    # Parallel continuations and backwards connections are not clear fork evidence.
                    if high - low >= 5.0 and all(abs(a) < 90 for a in angles):
                        def side(angle):
                            if angle == high:
                                return "LEFT" if sum(high - a < 5 for a in angles) == 1 else "UNKNOWN"
                            if angle == low:
                                return "RIGHT" if sum(a - low < 5 for a in angles) == 1 else "UNKNOWN"
                            return "MIDDLE"
                        options = "; ".join(f"{names(k)}: {side(a)} ({a:+.0f}° relative to approach)"
                                            for k, a in zip(successors, angles))
                        chosen = side(angles[successors.index(current)])
                        out.append(f"t={(i - i_now) * DT_S:+.1f}s: mapped non-intersection fork from "
                                   f"{names(source.id)} into {names(current)}; options {options}; "
                                   f"chosen branch: {chosen} (direct successor, not a neighbor-lane crossing)")
        previous = (i, current)
    return out or ["no clear mapped non-intersection fork transition observed within this window; "
                   "missing map matches or ambiguous geometry are insufficient branch evidence"]


def _local_speed_context(tl: Timeline, i_now: int, lo: int, hi: int, facts=None) -> list[str]:
    facts = facts if facts is not None else context.speed_context(tl,i_now,lo,hi)
    status = "complete window" if facts['complete_window'] else "partial window"
    if facts['missing_ego_data']:
        status += "; missing ego data"
    speed = facts['speed_now_m_s']
    out = [f"observed interval: {_t(facts['from_s'])} to {_t(facts['to_s'])} ({status})",
           f"speed at NOW: {speed:.2f} m/s ({speed*3.6:.2f} km/h)"]
    if facts['speed_change_m_s'] is not None:
        out.append(f"speed endpoints: {facts['start_speed_m_s']:.2f} -> {facts['end_speed_m_s']:.2f} m/s; "
                   f"speed change: {facts['speed_change_m_s']:+.2f} m/s over the observed interval")
    else:
        out.append("speed change: unknown (missing ego data)")
    for label,key,unit in (("minimum acceleration","minimum_acceleration_m_s2","m/s²"),
                           ("minimum jerk","minimum_jerk_m_s3","m/s³")):
        measurement=facts[key]
        out.append(f"{label}: {measurement['value']:+.2f} {unit} at {_t(measurement['time_s'])}"
                   if measurement else f"{label}: unknown")
    out.append("acceleration and jerk are derived from smoothed speed; short peaks can be softened")
    return out


def _lane_body_context(evidence, names, sample_indices):
    g=evidence['ego_geometry']
    out=['Map-derived estimates; painted-boundary alignment unvalidated. Virtual intersection polygons may overlap.',
         'REFERENCE is the first observed connected lane chain in this window, not an inferred maneuver origin.',
         'Connected segment end caps are unioned; segment shares can overlap and must not be summed as exclusive lanes.']
    ids=evidence['reference_lane_ids']
    out.append('REFERENCE: '+(' -> '.join(names(lid) for lid in ids) if ids else 'unavailable'))
    out.append(f"Continuous reference uses midpoint-blended centerline joins; maximum native join gap {evidence['max_centerline_join_gap_m']:.2f} m.")
    if g:
        out.append(f"Ego body: {g['length_m']:.3f} x {g['width_m']:.3f} m; pose reference {g['pose_reference']}; "
                   f"body center {g['center_forward_m']:+.3f} m forward of pose ({g['source']}).")
    else:out.append('Ego body geometry unavailable; body occupancy and body clearance remain unknown.')
    out.append('Margins are local cross-section envelope estimates; negative alone does not prove body boundary crossing.')
    out.append('BODY TIME    REF(m) IN/OUT(%) CROSS-SECTION MARGINS(m) SEGMENT SHARES(%) U=unmapped X=ambiguous C=lateral boundary evidence')
    for row in sample_indices:
        t=_t(row['time_s']);off=row['reference_offset_m'];inside=row['reference_lane_fraction'];outside=row['outside_reference_fraction']
        offset=f'{off:+.2f}' if off is not None else 'unknown'
        shares=', '.join(f"{names(p['lane_id'])}:{p['fraction']*100:.0f}"+(f"({p['direction_relation']})" if p['direction_relation']!="same-direction" else "") for p in row['lane_occupancy']) or 'unknown'
        body=f'{inside*100:.0f}/{outside*100:.0f}' if inside is not None else 'unknown'
        margins='/'.join(f'{row[k]:+.2f}' if row[k] is not None else 'unknown'
                         for k in ('left_boundary_margin_m','right_boundary_margin_m'))
        extra=f" U:{row['unmapped_fraction']*100:.0f} X:{row['ambiguous_overlap_fraction']*100:.0f}" if row['mapped_fraction'] is not None else ''
        cross=','.join(row['crossed_boundary_sides']) or ('outside' if row['lateral_departure'] else 'none' if row['lateral_departure'] is not None else 'unknown')
        out.append(f'body {t} {offset} {body} {margins} {shares}{extra} C:{cross}')
    for span in evidence['departure_intervals']:
        begin='first boundary departure' if span['entry_observed'] else 'first outside observation (onset unknown)'
        end=f"first return within reporting tolerance at {_t(span['return_s'])}" if span['return_s'] is not None else 'return unobserved within this continuous interval'
        out.append(f"{begin}: {_t(span['first_outside_s'])}; last outside {_t(span['last_outside_s'])}; {end}.")
    out.append(f"Departure reporting tolerance: {evidence['departure_reporting_fraction']:.0%} body area outside; not a scene-label threshold.")
    return out


def _interaction_context(tracks, missing_ids, selection=None):
    out=['Geometry supports interaction evidence, not a causal claim of avoidance.',
         'Body clearance is the shortest 2D edge-to-edge gap between oriented ego and object footprints; unknown without dimensions.',
         "lane corridor: ego's matched centerline buffered 1.15 m each side, a legacy proxy; reference-lane footprint intrusion is reported separately.",
         'Selected observations preserve NOW and closest passage. Additional geometry-event anchors are used when available.']
    if selection:
        out.append(f"Tracks retained {selection['retained_tracks']} / {selection['qualifying_tracks']}; "
                   f"omitted {selection['omitted_tracks']}; priority: {selection['policy']}.")
    if missing_ids:out.append('stable object IDs unavailable for some nearby observations; no associations invented')
    if not tracks:out.append('no nearby identified object tracks available')
    for rank,track in enumerate(tracks):
        obs=track['observations'];points=[]
        selected=track['key_observations']
        if rank>=3:
            selected=[p for p in selected if any(r in p['anchor_reasons'] for r in ('NOW','closest body passage','closest center passage'))]
        for p in selected:
            dimensions=f"{p['length_m']:.2f} x {p['width_m']:.2f} m" if p['length_m'] is not None and p['width_m'] is not None else 'unknown'
            gap=f"{p['ego_body_clearance_m']:.2f} m" if p['ego_body_clearance_m'] is not None else 'unknown'
            points.append(f"{_t(p['time_s'])} [{', '.join(p['anchor_reasons'])}]: ahead {p['ahead_m']:+.1f} m, "
                f"left {p['left_m']:+.1f} m, speed {p['speed_m_s']:.2f} m/s, body clearance {gap}")
        p=min(obs,key=lambda p:abs(p['time_s']))
        dimensions=f"{p['length_m']:.2f} x {p['width_m']:.2f} m" if p['length_m'] is not None and p['width_m'] is not None else 'unknown'
        out.append(f"track {track['alias']} ({track['kind']}, size near NOW {dimensions}), observed {_t(obs[0]['time_s'])} to {_t(obs[-1]['time_s'])}: "+'; '.join(points))
        clear=[p for p in obs if p['ego_reference_to_object_footprint_m'] is not None] if rank<3 else []
        if clear:
            p=min(clear,key=lambda p:p['ego_reference_to_object_footprint_m'])
            out.append(f"track {track['alias']}: minimum ego-reference-to-object-footprint distance {p['ego_reference_to_object_footprint_m']:.2f} m at {_t(p['time_s'])}; this point distance excludes ego body.")
        elif rank<3:out.append(f"track {track['alias']}: footprint distance unknown (object dimensions unavailable)")
        clear=[p for p in obs if p['ego_body_clearance_m'] is not None]
        if clear:
            p=min(clear,key=lambda p:p['ego_body_clearance_m'])
            out.append(f"track {track['alias']}: minimum body clearance {p['ego_body_clearance_m']:.2f} m at {_t(p['time_s'])}.")
        if rank>=3:continue
        intrusions=[p for p in obs if p['center_distance_m']<=30 and p['ahead_m']>0 and (p['reference_lane_overlap_m2'] or 0)>.01]
        out.append(f"track {track['alias']}: reference-lane intrusion while ahead observed at "+
            (', '.join(_t(p['time_s']) for p in (intrusions[0],intrusions[-1])) if intrusions else 'none')+
            '; missing geometry is not evidence of a clear lane.')
        proxies=[p for p in obs if p['lane_centered_body_clearance_proxy_m'] is not None and abs(p['time_s'])<=3]
        if proxies:
            p=min(proxies,key=lambda p:p['lane_centered_body_clearance_proxy_m'])
            out.append(f"track {track['alias']}: lane-centered body clearance proxy {p['lane_centered_body_clearance_proxy_m']:.2f} m at {_t(p['time_s'])}; this hypothetical placement does not prove intent or collision.")
    return out


def render(tl: Timeline, lanemap: LaneMap, now_ns: int, past_s: int = 10, future_s: int = 10,
           traffic_side: str = "left", table_step_s: float = TABLE_STEP_S) -> str | None:
    sample=render_sample(tl,lanemap,now_ns,past_s,future_s,traffic_side,table_step_s)
    return sample['script'] if sample is not None else None


def render_sample(tl: Timeline, lanemap: LaneMap, now_ns: int, past_s: int = 10, future_s: int = 10,
           traffic_side: str = "left", table_step_s: float = TABLE_STEP_S) -> dict | None:
    """Script for one NOW moment, or None when there is no ego data at NOW."""
    if not math.isfinite(table_step_s) or table_step_s < DT_S:
        raise ValueError(f"table_step_s must be finite and at least {DT_S} seconds")
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
    trace = context.lane_trace(tl,lo,hi)
    events: list[tuple[float, str]] = []
    prev = None
    first = True
    previous_i = None
    for i, cur in trace:
        if not tl.valid[i] or (previous_i is not None and not tl.valid[previous_i:i+1].all()):
            first,prev,previous_i = True,None,None
            continue
        if cur is not None:
            names(cur)
        if not first and cur != prev:
            events.append((rel_s(i), _transition_text(lanemap, names, prev, cur)))
        prev, first, previous_i = cur, False, i

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

    # Regular ticks around NOW, plus exact window edges if cadence does not divide them.
    step_ns = int(round(table_step_s * 1e9))
    past_ns, future_ns = int(round(past_s * 1e9)), int(round(future_s * 1e9))
    offsets = sorted({-past_ns, future_ns, *range(-(past_ns // step_ns) * step_ns,
                                               (future_ns // step_ns) * step_ns + 1, step_ns)})
    time_texts = [f"t={offset / 1e9:+g}s" for offset in offsets]
    time_width = max(8, max(len(text) for text in time_texts) + 1)
    rows, ego_samples = [], []
    for offset_ns, time_text in zip(offsets, time_texts):
        k = offset_ns / 1e9
        t_ns = now_ns + offset_ns
        i = tl.index(t_ns)
        if i is None:
            ego_samples.append(dict(time_s=k,ego_data_available=False))
            rows.append(f"{time_text}   (no data)")
            continue
        p = tl.lane(i)
        lane = names(p.lane_id) if p is not None else "off map"
        mapped_lane = lanemap.lanes[p.lane_id] if p is not None else None
        road_context = (f"intersection ({mapped_lane.turn_direction.upper()})" if mapped_lane.is_intersection
                   else mapped_lane.kind) if mapped_lane else "map match unavailable"
        offset = f"{_z(p.offset, 1):+.1f} m" if p is not None else "-"
        ind = INDICATOR_TEXT.get(tl.indicator_at(t_ns), "off") if s.has_indicator else "n/a"
        ego_samples.append(dict(time_s=k,ego_data_available=True,lane=lane,road_context=road_context,
                                intersection=bool(mapped_lane.is_intersection) if mapped_lane else None,
                                map_turn_direction=mapped_lane.turn_direction.upper() if mapped_lane and mapped_lane.is_intersection else None,
                                offset_m=float(p.offset) if p else None,speed_m_s=float(tl.v[i]),
                                acceleration_m_s2=float(tl.accel[i]),yaw_rate_rad_s=float(tl.yaw_rate[i]),
                                relative_heading_deg=math.degrees(tl.yaw[i]-yaw_now),indicator=ind,light=tl.light_at(t_ns)))
        row = (time_text.ljust(time_width) + lane.ljust(9) + road_context.ljust(26) + offset.ljust(9)
               + f"{_z(tl.v[i] * 3.6, 0):.0f} km/h".ljust(10) + f"{_z(tl.accel[i], 1):+.1f} m/s²".ljust(12)
               + f"{_z(math.degrees(tl.yaw_rate[i]), 0):+.0f} °/s".ljust(10)
               + f"{_z(math.degrees(tl.yaw[i] - yaw_now), 0):+.0f}°".ljust(18) + ind.ljust(11) + tl.light_at(t_ns))
        rows.append(row + ("   <- NOW" if k == 0 else ""))

    # Summaries and rendered intervals consume the same canonical numeric facts.
    window=context.window_context(tl,i_now,lo,hi,now_lane,names)
    map_spans=context.intersections(tl,lanemap,trace,i_now,lo,hi)
    local=context.speed_context(tl,i_now,lo,hi)
    geometry=LaneEvidence(tl,lanemap,lo,hi,i_now)
    body_evidence=geometry.summary()
    selection={};map_free_tracks=[]
    tracks,missing_ids=context.object_tracks(tl,lanemap,i_now,lo,hi,geometry=geometry,selection_info=selection,map_free_tracks=map_free_tracks)
    body_rows=[]
    for row in ego_samples:
        index=tl.index(now_ns+round(row["time_s"]*1e9))
        if index is not None:
            body=geometry.at(index)
            row["matched_lane_id"]=body["matched_lane_id"]
            row["body_occupancy"]=body
            body_rows.append(body)
    interval=f"over {_t(window['from_s'])} to {_t(window['to_s'])}"
    summary=[]
    if window['heading_change_deg'] is not None:
        summary.append(f"heading change {interval}: {window['heading_change_deg']:+.0f}°")
    for label,key,unit in (("minimum longitudinal acceleration","minimum_acceleration_m_s2","m/s²"),
                           ("minimum longitudinal jerk","minimum_jerk_m_s3","m/s³")):
        measurement=window[key]
        if measurement:
            summary.append(f"{label}: {measurement['value']:.1f} {unit} "
                           f"at {_t(measurement['time_s'])} ({interval})")
    peak=window['peak_absolute_yaw_rate_rad_s']
    if peak:
        summary.append(f"peak absolute yaw rate: {math.degrees(peak['value']):+.1f} °/s "
                       f"at {_t(peak['time_s'])} ({interval})")
    onset_text=", ".join(_t(t) for t in window['braking_onsets_s']) or "none observed"
    summary.append(f"braking onset (observed crossing from acceleration >= -0.3 to < -0.3 m/s²): "
                   f"{onset_text} ({interval}); jerk alone is not braking onset")
    offset=window['lane_offset']
    if offset:
        summary.append(f"offset while in lane {offset['lane']}: {offset['min_m']:+.1f} m to "
                       f"{offset['max_m']:+.1f} m over {_t(offset['from_s'])} to {_t(offset['to_s'])}")
    entry = next((e for e in events if e[0] > 0 and "intersection lane" in e[1]), None)
    if entry is not None:
        i_entry = i_now + int(round(entry[0] / DT_S))
        dist = float(np.hypot(np.diff(tl.x[i_now:i_entry + 1]), np.diff(tl.y[i_now:i_entry + 1])).sum())
        summary.append(f"enters the intersection lane {dist:.0f} m after NOW ({_t(entry[0])})")
    elif now_lane is not None and lanemap.lanes[now_lane.lane_id].is_intersection:
        summary.append(f"NOW is inside intersection lane {names(now_lane.lane_id)}")

    # Objects at NOW --------------------------------------------------------------------
    objects = [];map_free_objects=[]
    objs = tl.objects_at(now_ns)
    if objs is None:
        objects.append("(no object data)")
        map_free_objects.append("(no object data)")
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
        free_aliases={t["track_id"]:t["alias"] for t in map_free_tracks}
        free_near=sorted(near,key=lambda r:(r[3].track_id not in free_aliases,r[0]))
        for dist,fx,fy,o in free_near[:MAX_OBJECTS]:
            prefix=free_aliases.get(o.track_id,"")
            where=f"{abs(fx):.0f} m {'ahead' if fx>=0 else 'behind'}, {abs(fy):.0f} m {'left' if fy>=0 else 'right'}"
            speed="stopped" if o.speed<.5 else f"{o.speed*3.6:.0f} km/h"
            speed+=(", oncoming" if o.speed>=.5 and abs(wrap(o.yaw-yaw_now))>math.radians(135) else "")
            map_free_objects.append(f"{prefix+' ' if prefix else ''}{o.kind}: {where}, {speed}")
        track_aliases={t["track_id"]:t["alias"] for t in tracks}
        # Prefer the same tracked actors that the temporal context discusses.
        near.sort(key=lambda r:(r[3].track_id not in track_aliases,r[0]))
        for dist, fx, fy, o in near[:MAX_OBJECTS]:
            where = f"{abs(fx):.0f} m {'ahead' if fx >= 0 else 'behind'}, {abs(fy):.0f} m {'left' if fy >= 0 else 'right'}"
            prefix=(track_aliases[o.track_id]+" ") if o.track_id in track_aliases else ""
            objects.append(f"{prefix}{o.kind}: {where}, {_object_lane(lanemap, names, now_lane, o)}, "
                           + ("stopped" if o.speed < 0.5 else f"{o.speed * 3.6:.0f} km/h")
                           + (", oncoming" if o.speed >= 0.5 and abs(wrap(o.yaw - yaw_now)) > math.radians(135) else ""))
        if not near:
            objects.append("none within 30 m")
            map_free_objects.append("none within 30 m")

    route = _route_context(tl, lanemap, i_now, lo, hi, map_spans)
    forks = _fork_context(tl, lanemap, names, i_now, lo, hi)
    local_speed = _local_speed_context(tl, i_now, lo, hi, local)
    interactions = _interaction_context(tracks, missing_ids,selection)
    body_text=_lane_body_context(body_evidence,names,body_rows)
    lanes_block = [_describe_lane(lanemap, names, lane_id) for lane_id, _ in names.items()]
    out = [HEADER.format(side=traffic_side), "", "LANES"] + (lanes_block or ["(ego is off the mapped lanes)"])
    table_header = ("TIME".ljust(time_width) + "LANE".ljust(9) + "ROAD CONTEXT".ljust(26)
                    + "OFFSET".ljust(9) + "SPEED".ljust(10) + "ACCEL".ljust(12)
                    + "YAW RATE".ljust(10) + "RELATIVE HEADING".ljust(18) + "INDICATOR".ljust(11) + "LIGHT")
    out += ["", table_header] + rows
    out += ["", "EVENTS"] + ([f"{_t(t)}: {text}" for t, text in events] or ["none"])
    summary.insert(0, "indicator during the table: " + _indicator_text(tl, t_lo, t_hi, now_ns))
    out += ["", "LOCAL SPEED CONTEXT (around NOW, at most 2 s each side)"] + local_speed
    out += ["", "SUMMARY"] + (summary or ["none"])
    out += ["", "FORK CONTEXT (map alternatives and observed path within the sample window)"] + forks
    out += ["", f"ROUTE CONTEXT (along the ego's actual path, past {past_s} s / future {future_s} s)"] + route
    out += ["", "EGO BODY AND LANE OCCUPANCY"] + body_text
    out += ["", f"OBJECTS AT NOW (up to {MAX_OBJECTS} within {OBJECT_RADIUS_M:.0f} m)"] + objects
    out += ["", "OBJECT INTERACTION CONTEXT (same sample window, source IDs only)"] + interactions
    lane_facts=[dict(alias=alias,id=lane_id,kind=lanemap.lanes[lane_id].kind,
                     intersection=lanemap.lanes[lane_id].is_intersection,
                     map_turn_direction=lanemap.lanes[lane_id].turn_direction,
                     following=[dict(alias=names(f) if f in names else None,
                                     intersection=lanemap.lanes[f].is_intersection,
                                     map_turn_direction=lanemap.lanes[f].turn_direction)
                                for f in lanemap.lanes[lane_id].following])
                for lane_id,alias in names.items()]
    map_free_text=strip_map("\n".join([f"OBJECTS AT NOW (up to {MAX_OBJECTS} within {OBJECT_RADIUS_M:.0f} m)"]+map_free_objects
                  +["", "OBJECT INTERACTION CONTEXT (same sample window, source IDs only)"]
                  +_interaction_context(map_free_tracks,missing_ids)))
    facts=dict(schema_version=SAMPLE_SCHEMA_VERSION,lanes=lane_facts,target=dict(lateral="active at NOW",longitudinal="around NOW within 2 s"),
               ego_samples=ego_samples,window=window,local_speed=local,intersections=map_spans,
               lane_evidence=body_evidence,object_selection=selection,map_free_object_script=map_free_text,
               object_tracks=tracks,objects_missing_ids=missing_ids,
               lane_corridor_half_width_m=context.LANE_CORRIDOR_HALF_WIDTH_M)
    return dict(t_ns=now_ns,script="\n".join(out),facts=facts)


def _object_lane(lanemap: LaneMap, names: Aliases, now_lane, o) -> str:
    heading_free = o.kind in ("pedestrian", "bicycle", "animal", "unknown")
    cands = lanemap.candidates(o.x, o.y, o.yaw, max_heading_diff=math.pi if heading_free else math.radians(60))
    if not cands:
        return "map relationship unknown"
    lane_id = min(cands, key=lanemap.emission_cost).lane_id
    if now_lane is None:
        return "in a lane"
    rel = lanemap.relation(now_lane.lane_id, lane_id)
    return {"same": "same lane", "left": "left neighbor lane", "right": "right neighbor lane",
            "following": "lane ahead", "near": "lane ahead", "previous": "lane behind"}.get(rel, "other lane")


def _route_context(tl: Timeline, lanemap: LaneMap, i_now: int, lo: int, hi: int, spans=None) -> list[str]:
    """Map occupancy has explicit last-inside/first-outside boundaries, not maneuver bounds."""
    def path_m(i0,i1):
        a,b=sorted((i0,i1))
        return float(np.hypot(np.diff(tl.x[a:b+1]),np.diff(tl.y[a:b+1])).sum())
    rel=lambda i:_t((i-i_now)*DT_S)
    spans=spans if spans is not None else context.intersections(tl,lanemap,context.lane_trace(tl,lo,hi),i_now,lo,hi)
    lines=[]
    for span in spans:
        a=i_now+round(span['first_inside_s']/DT_S)
        b=i_now+round(span['last_inside_s']/DT_S)
        outside=span['first_outside_s']
        turn=span['heading_change_deg']
        measurement=(f"heading change {turn:+.0f}°" if turn is not None else "heading change unknown")
        what=(f"map_turn_direction {span['map_turn_direction']}; {measurement} "
              f"over {rel(a)} to {rel(b)} (mapped occupancy only)")
        if span['starts_at_window_edge'] or span['ends_at_window_edge']:
            what += " (observed within sample)"
        if span['entry_observed']:
            entry=f"entered at {rel(a)}"
        elif span['starts_at_window_edge']:
            entry=f"already inside at window start ({rel(lo)})"
        else:
            entry=f"first inside observed at {rel(a)}; entry unknown (missing map/ego evidence)"
        if outside is not None:
            exit_=f"last inside at {rel(b)}, first outside at {_t(outside)}"
        elif span['ends_at_window_edge']:
            exit_=f"last inside at {rel(b)}; still inside at window end ({rel(hi)})"
        else:
            exit_=f"last inside at {rel(b)}; exit unknown (missing map/ego evidence)"
        if span['first_inside_s']<=0 and (outside is None and span['last_inside_s']>=0 or outside is not None and outside>0):
            lines.append(f"NOW is inside an intersection lane ({what}): {entry}, {exit_}")
        elif a>i_now:
            lines.append(f"intersection ahead: {entry}, {path_m(i_now,a):.0f} m ahead ({what}); {exit_}")
        else:
            exit_i=i_now+round(outside/DT_S) if outside is not None else b
            leave=f"left at {_t(outside)}, " if outside is not None else "exit time unknown, "
            lines.append(f"intersection behind: {leave}{path_m(exit_i,i_now):.0f} m ago ({what}); {entry}, {exit_}")

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
    if not tl.session.has_indicator:
        return "not recorded in this log"
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
                             r"other lane|off the road|map relationship unknown|in a lane)(?=, )")
_WHERE_STOP = re.compile(r"(, in a [a-z ]+ lane, offset [+-][0-9.]+ m|, off the mapped lanes)")
_WHERE_STOPPED = re.compile(r" \((in a [a-z ]+ lane, offset [+-][0-9.]+ m|off the mapped lanes)\)")


def strip_map(text: str, map_free_object_script: str | None = None) -> str:
    """The same script with everything that comes from the map removed (for a no-map ablation).

    Lanes, lane offsets, lane-change events, intersections and the lane relation of objects
    go; speed, acceleration, yaw rate, heading, indicator, light and object positions stay.
    """
    if map_free_object_script is not None:
        text=text.split("OBJECTS AT NOW",1)[0]+map_free_object_script
    out, section = [], None
    lane_columns = slice(7, 25)   # legacy scripts used a 7-character time column
    for line in text.split("\n"):
        if line == "EGO BODY AND LANE OCCUPANCY":
            section="BODY"
            continue
        if section == "BODY":
            if line == "":section=None
            continue
        if line.startswith("FORK CONTEXT"):
            section = "FORK"
            continue
        if section == "FORK":
            if line == "":
                section = None
            continue
        if line == "LANES":
            section = "LANES"
            continue
        if section == "LANES":
            if line == "":
                section = None
            continue
        if line.startswith("TIME "):
            section = "TABLE"
            lane_columns = slice(line.index("LANE"), line.index("SPEED"))
        elif line in ("EVENTS", "SUMMARY") or line.startswith(("LOCAL SPEED CONTEXT", "ROUTE CONTEXT", "OBJECTS AT NOW", "OBJECT INTERACTION CONTEXT")):
            section = "INTERACTION" if line.startswith("OBJECT INTERACTION") else line.split(" ")[0]
            out.append(line)
            continue
        elif line == "" and section == "TABLE":
            section = None
        if section == "TABLE" and "(no data)" not in line:
            line = line[:lane_columns.start] + line[lane_columns.stop:]
        elif section == "EVENTS" and any(k in line for k in _MAP_EVENT):
            continue
        elif section == "SUMMARY" and line.startswith(_MAP_SUMMARY):
            continue
        elif section == "ROUTE" and line.startswith(_MAP_ROUTE):
            continue
        elif section == "ROUTE":
            line = _WHERE_STOPPED.sub("", _WHERE_STOP.sub("", line))
        elif section == "INTERACTION" and any(k in line for k in ("lane corridor", "reference-lane intrusion", "lane-centered body", "Tracks retained")):
            continue
        elif section == "INTERACTION":
            def without_map_anchors(match):
                reasons=[r for r in match.group(1).split(", ") if r not in ("body departure","body return","lateral excursion peak","first observed ahead intrusion","last observed ahead intrusion")]
                return "["+(", ".join(reasons) or "observation")+"]"
            line=re.sub(r"\[([^]]*)\]",without_map_anchors,line)
        elif section == "OBJECTS":
            line = _LANE_OF_OBJECT.sub("", line)
        out.append(line)
    text = "\n".join(out).replace(CONTEXT_NOTE, "").replace(" OFFSET = distance from the lane center (+ = left).", "")
    return text.replace("EVENTS\n\n", "EVENTS\nnone\n\n")
