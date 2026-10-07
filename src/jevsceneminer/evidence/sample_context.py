"""Canonical numeric evidence shared by the readable script and processed samples."""
from __future__ import annotations

import bisect
import math
import numpy as np
from shapely.geometry import LineString, Point, Polygon

from jevsceneminer.evidence.facts import DT_S, LANE_EVERY
from jevsceneminer.evidence.lane_evidence import rectangle

BRAKING_ACCEL_MS2 = -0.3
LANE_CORRIDOR_HALF_WIDTH_M = 1.15  # explicit fixed-width proxy, not a mapped lane boundary


def lane_trace(tl, lo, hi):
    """10 Hz map observations, including missing samples so gaps cannot be bridged."""
    out = []
    for i in sorted({lo,hi,*range(lo + (-lo) % LANE_EVERY, hi + 1, LANE_EVERY)}):
        p = tl.lane(i) if tl.valid[i] else None
        out.append((i, p.lane_id if p else None))
    return out


def intersections(tl, lanemap, trace, i_now, lo, hi):
    spans, current, previous = [], None, None
    for i, lane_id in trace:
        lane = lanemap.lanes.get(lane_id)
        inside = lane is not None and lane.is_intersection
        contiguous = previous is not None and tl.valid[previous[0]:i + 1].all()
        if current is not None and (not inside or not contiguous):
            if contiguous and lane is not None and not inside:
                current['exit_i'] = i
            spans.append(current)
            current = None
        if inside:
            if current is None:
                current = dict(start_i=i, last_i=i, exit_i=None, directions=[],
                               entry_observed=bool(contiguous and previous[1] is not None
                                   and not lanemap.lanes[previous[1]].is_intersection))
            current['last_i'] = i
            current['directions'].append(lane.turn_direction.upper())
        previous = (i, lane_id)
    if current is not None:
        spans.append(current)
    out = []
    for span in spans:
        a, b, exit_i = span['start_i'], span['last_i'], span['exit_i']
        direction = max(dict.fromkeys(span['directions']), key=span['directions'].count)
        valid = bool(tl.valid[a:b + 1].all() and np.isfinite(tl.yaw[[a,b]]).all())
        out.append(dict(first_inside_s=(a-i_now)*DT_S, last_inside_s=(b-i_now)*DT_S,
                        first_outside_s=(exit_i-i_now)*DT_S if exit_i is not None else None,
                        entry_observed=span['entry_observed'], map_turn_direction=direction,
                        heading_change_deg=math.degrees(tl.yaw[b]-tl.yaw[a]) if valid else None,
                        starts_at_window_edge=not span['entry_observed'] and a == lo,
                        ends_at_window_edge=exit_i is None and b == hi))
    return out


def speed_context(tl, i_now, lo, hi):
    radius = round(2/DT_S)
    a, b = max(lo,i_now-radius), min(hi,i_now+radius)
    indices = np.arange(a,b+1)
    gaps = not bool(tl.valid[a:b+1].all())
    def minimum(values):
        good = indices[tl.valid[indices] & np.isfinite(values[indices])]
        if not len(good):
            return None
        k = int(good[np.argmin(values[good])])
        return dict(value=float(values[k]), time_s=(k-i_now)*DT_S)
    clean = not gaps and bool(np.isfinite(tl.v[a:b+1]).all())
    return dict(from_s=(a-i_now)*DT_S,to_s=(b-i_now)*DT_S,
                complete_window=a==i_now-radius and b==i_now+radius, missing_ego_data=gaps,
                speed_now_m_s=float(tl.v[i_now]),
                start_speed_m_s=float(tl.v[a]) if clean else None,
                end_speed_m_s=float(tl.v[b]) if clean else None,
                speed_change_m_s=float(tl.v[b]-tl.v[a]) if clean else None,
                minimum_acceleration_m_s2=minimum(tl.accel),minimum_jerk_m_s3=minimum(tl.jerk))


def window_context(tl, i_now, lo, hi, now_lane, names):
    indices=np.arange(lo,hi+1)
    def extreme(values, absolute=False):
        good=indices[tl.valid[indices] & np.isfinite(values[indices])]
        if not len(good):return None
        which=np.argmax(np.abs(values[good])) if absolute else np.argmin(values[good])
        k=int(good[which])
        return dict(value=float(values[k]),time_s=(k-i_now)*DT_S)
    onsets=[(i-i_now)*DT_S for i in range(lo+1,hi+1)
            if tl.valid[i-1:i+1].all() and tl.accel[i-1]>=BRAKING_ACCEL_MS2>tl.accel[i]]
    heading=math.degrees(tl.yaw[hi]-tl.yaw[lo]) if tl.valid[lo:hi+1].all() else None
    offset=None
    if now_lane is not None:
        samples=[(i,tl.lane(i).offset) for i,lane_id in lane_trace(tl,lo,hi)
                 if lane_id==now_lane.lane_id]
        if samples:
            vals=[float(v) for _,v in samples]
            offset=dict(lane=names(now_lane.lane_id),from_s=(samples[0][0]-i_now)*DT_S,
                        to_s=(samples[-1][0]-i_now)*DT_S,min_m=min(vals),max_m=max(vals))
    return dict(from_s=(lo-i_now)*DT_S,to_s=(hi-i_now)*DT_S,
                heading_change_deg=heading,minimum_acceleration_m_s2=extreme(tl.accel),
                minimum_jerk_m_s3=extreme(tl.jerk),
                peak_absolute_yaw_rate_rad_s=extreme(tl.yaw_rate,True),
                braking_threshold_m_s2=BRAKING_ACCEL_MS2,braking_onsets_s=onsets,lane_offset=offset)


def _footprint(obj):
    if (obj.length_m is None or obj.width_m is None
            or not all(math.isfinite(v) and v>0 for v in (obj.length_m,obj.width_m))):
        return None
    return rectangle(obj.x,obj.y,obj.yaw,obj.length_m,obj.width_m)


def key_observations(observations, event_times=(), use_map=True):
    """Event anchors, not uniform thinning; canonical records are rendered verbatim."""
    anchors={}
    def add(k,reason):
        anchors.setdefault(k,[]).append(reason)
    if not observations:return []
    nearest=min(range(len(observations)),key=lambda k:abs(observations[k]['time_s']))
    if abs(observations[nearest]['time_s'])<=.3:add(nearest,'NOW')
    for field,reason in [('ego_body_clearance_m','closest body passage'),
                         ('center_distance_m','closest center passage')]:
        valid=[k for k,p in enumerate(observations) if p.get(field) is not None]
        if valid:add(min(valid,key=lambda k:observations[k][field]),reason)
    closest=[p for p in observations if p.get('ego_body_clearance_m') is not None]
    pass_time=min(closest,key=lambda p:p['ego_body_clearance_m'])['time_s'] if closest else None
    for t,reason in event_times:
        if pass_time is None or abs(t-pass_time)>3:continue
        k=min(range(len(observations)),key=lambda k:abs(observations[k]['time_s']-t))
        if abs(observations[k]['time_s']-t)<=.3 and observations[k]['center_distance_m']<=30:add(k,reason)
    blockers=[k for k,p in enumerate(observations) if use_map and p['ahead_m']>0 and p['center_distance_m']<=30 and p.get('reference_lane_overlap_m2',0)]
    if blockers:
        add(blockers[0],'first observed ahead intrusion')
        add(blockers[-1],'last observed ahead intrusion')
    add(0,'first observation');add(len(observations)-1,'last observation')
    # Priority insertion preserves NOW/closest/event anchors before optional endpoints.
    return [dict(observations[k],anchor_reasons=anchors[k]) for k in sorted(list(anchors)[:9])]


def object_tracks(tl, lanemap, i_now, lo, hi, max_tracks=8, radius_m=30,
                  geometry=None, selection_info=None, map_free_tracks=None):
    from jevsceneminer.evidence.lane_evidence import LaneEvidence
    geometry=geometry or LaneEvidence(tl,lanemap,lo,hi,i_now)
    groups,seen,missing_ids={},set(),False
    corridors={}
    # Shapes are source observations and reusable across overlapping context windows.
    shape_cache=getattr(tl,'_object_footprints',None)
    if shape_cache is None:shape_cache={};tl._object_footprints=shape_cache
    first=bisect.bisect_left(tl.object_times,tl.time_ns(lo))
    last=bisect.bisect_right(tl.object_times,tl.time_ns(hi))
    for timestamp,objects in tl.session.objects[first:last]:
        i=tl.index(timestamp)
        if i is None:continue
        c,s=math.cos(tl.yaw[i]),math.sin(tl.yaw[i]);projection=tl.lane(i)
        corridor=None;lane_area=None
        if projection is not None:
            lid=projection.lane_id
            if lid not in corridors:
                corridors[lid]=LineString(lanemap.lanes[lid].centerline).buffer(LANE_CORRIDOR_HALF_WIDTH_M)
            corridor=corridors[lid];lane_area=geometry.polygons.get(lid)
        ego_body=geometry.ego_shape(i);centered=geometry.centered_shape(i)
        for obj in objects:
            if obj.existence<.5 or not all(math.isfinite(v) for v in (obj.x,obj.y,obj.yaw,obj.speed)):continue
            dx,dy=obj.x-tl.x[i],obj.y-tl.y[i];distance=math.hypot(dx,dy)
            if obj.track_id is None:
                missing_ids |= distance<=radius_m
                continue
            key=(timestamp,obj.track_id)
            if key in seen:continue
            seen.add(key)
            if key not in shape_cache:shape_cache[key]=_footprint(obj)
            footprint=shape_cache[key];shape=footprint if footprint is not None else Point(obj.x,obj.y)
            observation=dict(time_s=(timestamp-tl.time_ns(i_now))/1e9,
                ahead_m=float(c*dx+s*dy),left_m=float(-s*dx+c*dy),speed_m_s=float(obj.speed),
                center_distance_m=distance,length_m=obj.length_m,width_m=obj.width_m,
                relative_yaw_deg=math.degrees(math.atan2(math.sin(obj.yaw-tl.yaw[i]),math.cos(obj.yaw-tl.yaw[i]))),
                ego_reference_to_object_footprint_m=float(Point(tl.x[i],tl.y[i]).distance(footprint)) if footprint is not None else None,
                ego_body_clearance_m=float(ego_body.distance(footprint)) if ego_body is not None and footprint is not None else None,
                lane_centered_body_clearance_proxy_m=float(centered.distance(footprint)) if centered is not None and footprint is not None else None,
                matched_lane_overlap_m2=float(lane_area.intersection(footprint).area) if lane_area is not None and footprint is not None else None,
                reference_lane_overlap_m2=float(geometry.reference_area.intersection(footprint).area)
                    if geometry.reference_area is not None and footprint is not None else None,
                lane_corridor_overlap=bool(corridor.intersects(shape)) if corridor is not None else None,
                corridor_overlap_basis='footprint' if footprint is not None else 'object_center')
            groups.setdefault(obj.track_id,dict(track_id=obj.track_id,kind=obj.kind,observations=[]))['observations'].append(observation)
    tracks=[t for t in groups.values() if min(p['center_distance_m'] for p in t['observations'])<=radius_m]
    def priority(track):
        obs=track['observations']
        stopped_intruder=any(abs(p['time_s'])<=5 and 0<p['ahead_m']<=radius_m
                            and p['speed_m_s']<.5 and (p['reference_lane_overlap_m2'] or 0)>.01 for p in obs)
        close_interaction=any(abs(p['time_s'])<=5 and p['ego_body_clearance_m'] is not None
                             and p['ego_body_clearance_m']<3 for p in obs)
        tier=0 if stopped_intruder else 1 if close_interaction else 2
        local=[p['center_distance_m'] for p in obs if abs(p['time_s'])<=3]
        return tier,min(local,default=math.inf),min(p['center_distance_m'] for p in obs),track['track_id']
    if map_free_tracks is not None:
        independent=sorted(tracks,key=lambda t:(min(p["center_distance_m"] for p in t["observations"]),t["track_id"]))
        for k,track in enumerate(independent[:max_tracks],1):
            observations=sorted(track["observations"],key=lambda p:p["time_s"])
            map_free_tracks.append(dict(track_id=track["track_id"],kind=track["kind"],alias=f"O{k}",
                observations=observations,key_observations=key_observations(observations,use_map=False)))
    tracks.sort(key=priority)
    if selection_info is not None:
        selection_info.update(qualifying_tracks=len(tracks),retained_tracks=min(len(tracks),max_tracks),
            omitted_tracks=max(0,len(tracks)-max_tracks),policy='stopped-ahead reference-lane intruders, close body passages, then proximity to NOW')
    evidence=geometry.summary();events=[]
    for span in evidence['departure_intervals']:
        if span['entry_observed']:events.append((span['first_outside_s'],'body departure'))
        if span['return_s'] is not None:events.append((span['return_s'],'body return'))
    if evidence['lateral_excursion_peak'] is not None:
        events.append((evidence['lateral_excursion_peak']['time_s'],'lateral excursion peak'))
    for k,track in enumerate(tracks[:max_tracks],1):
        track['alias']=f'O{k}';track['observations'].sort(key=lambda p:p['time_s'])
        track['key_observations']=key_observations(track['observations'],events)
        track['selection_tier']=priority(track)[0]
    return tracks[:max_tracks],missing_ids
