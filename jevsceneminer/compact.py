"""Offline compact Jev inputs from canonical samples; never sets predicted labels.

Run: python -m jevsceneminer.compact SOURCE --out NEW --review REVIEW.json
Original source samples remain the audit record. Review intervals use seconds from
session start, applied to every observation (including neighboring windows).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sqlite3
import time

SCHEMA = 'jsm-sample-1.2'
FORMAT = 'compact-3'


def reviewed_state(now_ns, relative_s, review):
    if not review:
        return None
    absolute_ns = int(now_ns) + round(relative_s * 1e9)
    origin = int(review['start_ns'])
    lo = origin + round(review['from_s'] * 1e9)
    hi = origin + round(review['to_s'] * 1e9)
    margin = round(review.get('margin_s', 0) * 1e9)
    if lo <= absolute_ns <= hi:
        return 'core'
    if lo - margin <= absolute_ns <= hi + margin:
        return 'margin'
    return None


def _body_effective(body, state, review):
    if not state:
        return
    # Keep timestamps and availability; no native geometry survives this override.
    for key in list(body):
        if key not in ('time_s', 'ego_data_available'):
            body[key] = None
    body.update(physical_lane_chain=review['chain_id'] if state == 'core' else None,
                physical_lane_ownership_pct=100 if state == 'core' else None,
                lane_evidence_source=review['source'])


def effective_facts(native, now_ns, review):
    facts = copy.deepcopy(native)
    facts['schema_version'] = SCHEMA
    facts['script_format'] = FORMAT
    affected = False
    for ego in facts.get('ego_samples', []):
        state = reviewed_state(now_ns, ego['time_s'], review)
        ego['physical_lane_chain'] = None
        ego['physical_lane_ownership_pct'] = None
        if not state:
            continue
        affected = True
        ego.update(lane=review['chain_id'] if state == 'core' else None,
                   matched_lane_id=None, offset_m=None,
                   physical_lane_chain=review['chain_id'] if state == 'core' else None,
                   physical_lane_ownership_pct=100 if state == 'core' else None,
                   lane_evidence_source=review['source'],
                   road_context='road (camera-reviewed)' if state == 'core' else 'unknown',
                   intersection=False if state == 'core' else None,
                   map_turn_direction=None, light='unknown')
        _body_effective(ego.get('body_occupancy', {}), state, review)
    for body in facts.get('lane_evidence', {}).get('observations', []):
        state = reviewed_state(now_ns, body['time_s'], review)
        if state:
            affected = True
            _body_effective(body, state, review)
    for track in facts.get('object_tracks', []):
        for field in ('observations', 'key_observations'):
            for obs in track.get(field, []):
                if reviewed_state(now_ns, obs['time_s'], review):
                    for key in ('matched_lane_overlap_m2', 'reference_lane_overlap_m2',
                                'lane_corridor_overlap', 'corridor_overlap_basis',
                                'lane_centered_body_clearance_proxy_m'):
                        obs[key] = None
                    obs['anchor_reasons'] = [r for r in obs.get('anchor_reasons', [])
                                             if not any(word in r for word in ('intrusion','departure','return','excursion'))]
                    obs['lane_evidence_source'] = 'unavailable: reviewed native map mismatch'
    if affected:
        # These summaries refer to the original uncorrected lane geometry.
        facts['window'].pop('lane_offset', None)
        old = facts.get('lane_evidence', {})
        facts['lane_evidence'] = {key: old[key] for key in ('ego_geometry', 'observations') if key in old}
        intersections = []
        for span in facts.get('intersections', []):
            if not _interval_touches_review(now_ns, span['first_inside_s'], span['last_inside_s'], review):
                intersections.append(span)
                continue
            valid_times = [obs['time_s'] for obs in old.get('observations', [])
                if span['first_inside_s'] <= obs['time_s'] <= span['last_inside_s']
                and obs.get('ego_data_available') and not reviewed_state(now_ns,obs['time_s'],review)]
            boundary = (review['start_ns'] - now_ns)/1e9 + review['from_s'] - review.get('margin_s',0)
            for times in ([t for t in valid_times if t < boundary], [t for t in valid_times if t > boundary]):
                if not times:
                    continue
                clipped = dict(span,first_inside_s=min(times),last_inside_s=max(times),
                               heading_change_deg=None,review_clipped=True)
                if clipped['first_inside_s'] != span['first_inside_s']:
                    clipped['entry_observed'] = False
                    clipped['starts_at_window_edge'] = False
                if clipped['last_inside_s'] != span['last_inside_s']:
                    clipped['first_outside_s'] = None
                    clipped['ends_at_window_edge'] = False
                intersections.append(clipped)
        facts['intersections'] = intersections
        facts.pop('map_free_object_script', None)
        facts['lane_review'] = copy.deepcopy(review)
    return facts


def _interval_touches_review(now_ns, start, end, review):
    if not review:
        return False
    lo = review['start_ns'] + round((review['from_s'] - review.get('margin_s', 0)) * 1e9)
    hi = review['start_ns'] + round((review['to_s'] + review.get('margin_s', 0)) * 1e9)
    return now_ns + round(end * 1e9) >= lo and now_ns + round(start * 1e9) <= hi


def object_share(obs):
    area = obs.get('reference_lane_overlap_m2')
    length, width = obs.get('length_m'), obs.get('width_m')
    if area is None or length is None or width is None or length <= 0 or width <= 0:
        return None
    return min(100., max(0., 100 * area / (length * width)))


def road_shares(body, topology=None):
    shares={item['lane_id']:100*item['fraction'] for item in (body.get('lane_occupancy') or [])
        if item.get('intersection') is False and item.get('kind')=='road'
        and item.get('direction_relation')=='same-direction' and item.get('fraction') is not None
        and not (topology or {}).get(item['lane_id'],{}).get('connector',False)}
    # Do not turn overlapping polygons into exclusive ownership by normalizing them.
    return shares if shares and sum(shares.values())<=101 else {}


def _fmt(value, digits=1):
    return 'unknown' if value is None else f'{value:+.{digits}f}'


def _alias(index):
    result = ''
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        result = chr(65 + remainder) + result
    return result


def map_metadata(path):
    """Only native lane topology metadata; no changes to log or map geometry."""
    with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as db:
        result = {}
        by_left, by_right = {}, {}
        for fid, group, order, left, right in db.execute(
                'select lane_fid, lane_group_fid, lane_index, left_boundary_fid, right_boundary_fid from lanes_polygons'):
            result[int(fid)] = dict(id=int(fid), group=group, order=order, connector=False,
                                    following=[], previous=[])
            by_left[left], by_right[right] = int(fid), int(fid)
        rows = db.execute('select lane_fid,left_boundary_fid,right_boundary_fid from lanes_polygons').fetchall()
        for fid, left, right in rows:
            result[int(fid)].update(left=by_right.get(left), right=by_left.get(right))
        for fid, incoming, outgoing, group in db.execute(
                'select fid,exit_lane_fid,entry_lane_fid,lane_group_connector_fid from lane_connectors'):
            result[int(fid)] = dict(id=int(fid), group=group, order=None, connector=True,
                                    left=None, right=None, following=[int(outgoing)], previous=[int(incoming)])
            if int(incoming) in result:
                result[int(incoming)]['following'].append(int(fid))
            if int(outgoing) in result:
                result[int(outgoing)]['previous'].append(int(fid))
        return result


def _events(script):
    if '\nEVENTS\n' not in script:
        return []
    section = script.split('\nEVENTS\n', 1)[1].split('\n\n', 1)[0]
    result = []
    for line in section.splitlines():
        match = re.match(r't=([+-]?[\d.]+)s: (.*)', line)
        if match:
            result.append(dict(time_s=float(match[1]), text=match[2]))
    return result


def _lane_event(text):
    match=re.search(r'(?:moves(?: back)?|continues) from ([A-Z]+) into (?:its (left|right) neighbor )?([A-Z]+)',text)
    if match:
        a,side,b=match.groups()
        relationship=f'mapped {side} neighbor' if side else ('mapped predecessor' if 'moves back' in text else
                    ('mapped successor' if 'continues' in text else 'map continuity unverified'))
        return a,b,relationship
    match=re.match(r'enters mapped lane ([A-Z]+)',text)
    if match:
        return None,match[1],'map match becomes available'
    match=re.match(r'leaves the mapped lanes \(from ([A-Z]+)\)',text)
    if match:
        return match[1],None,'map match becomes unavailable'
    return None


def _fork_section(script):
    match=re.search(r'\nFORK CONTEXT[^\n]*\n(.*?)(?:\n\n|\Z)',script,re.S)
    return match[1] if match else None


def render_compact(facts, now_ns, meta, topology, events):
    review = facts.get('lane_review')
    lanes = {x['id']: x for x in facts.get('lanes', [])}
    names = {x['id']: x['alias'] for x in lanes.values()}
    name_ids = {v:k for k,v in names.items()}
    # Only occupied native segments outside the reviewed/margin interval are relevant.
    seeds = {e.get('matched_lane_id') for e in facts['ego_samples']} - {None}
    seeds |= {e.get('matched_lane_id') for e in facts.get('lane_evidence', {}).get('observations', [])} - {None}
    for ego in facts['ego_samples']:
        seeds.update(road_shares(ego.get('body_occupancy') or {},topology))
    selected_events = []
    fork=facts.get('fork_context_script')
    if fork:
        for match in re.finditer(r'(?:from |into |options |; )([A-Z]+)(?::|\b)',fork):
            if match[1] in name_ids:
                seeds.add(name_ids[match[1]])
    for event in events:
        text, t = event['text'], event['time_s']
        lane_event = _lane_event(text)
        is_lane = lane_event is not None
        in_review = reviewed_state(now_ns, t, review)
        # Lane events come from a 10 Hz trace: check both observations at the
        # transition, not the entire time since an unrelated earlier transition.
        bridges_review = is_lane and _interval_touches_review(now_ns, t - .1, t, review)
        if (in_review or bridges_review) and (is_lane or 'traffic light' in text):
            continue
        if is_lane:
            a,b,relationship = lane_event
            if any(alias is not None and alias not in name_ids for alias in (a,b)):
                continue
            seeds.update(name_ids[alias] for alias in (a,b) if alias is not None)
            endpoint=lambda alias: f'{alias} ({name_ids[alias]})' if alias is not None else 'unknown'
            text = f'map match {endpoint(a)} -> {endpoint(b)}: {relationship}'
        selected_events.append(dict(time_s=t, text=text))
    reference_ids = facts.get('lane_evidence', {}).get('reference_lane_ids', [])
    relevant = set(seeds) | set(reference_ids)
    alternatives=[]
    for event in selected_events:
        if 'mapped left neighbor' not in event['text'] and 'mapped right neighbor' not in event['text']:
            continue
        pair=re.findall(r'\((\d+)\)',event['text'])
        if len(pair)!=2:
            continue
        paths=[]
        for lane_id in map(int,pair):
            path=[lane_id]
            for _ in range(2):
                successors=topology.get(path[-1],{}).get('following',[])
                if len(successors)!=1:
                    break
                path.append(successors[0])
            relevant.update(path)
            paths.append(path)
        alternatives.append(paths)
    for lane_id in seeds:
        lane = topology.get(lane_id, {})
        relevant.update(x for x in (lane.get('left'), lane.get('right'), *lane.get('following', [])) if x is not None)
    # Successor endpoints also have neighbors. Resolve those aliases rather than
    # emitting a neighbor alias without a corresponding lane definition.
    while True:
        expanded = relevant | {value for lane_id in relevant for value in
            (topology.get(lane_id, {}).get('left'), topology.get(lane_id, {}).get('right')) if value is not None}
        if expanded == relevant:
            break
        relevant = expanded
    for lane_id in sorted(relevant):
        if lane_id not in names:
            index = 0
            while _alias(index) in names.values():
                index += 1
            names[lane_id] = _alias(index)
    if review:
        # No native alias or chain facts for the reviewed physical group.
        manual = f"{review['chain_id']}: same physical lane chain, ownership 100% within session {review['from_s']}–{review['to_s']}s; source: {review['source']}."
    else:
        manual = None
    elapsed = (now_ns - meta['start_ns']) / 1e9
    lines = [
        'TARGET: lateral maneuver active at NOW (t=0); longitudinal behavior over -2 to +2s.',
        f"NOW: session {elapsed:.2f}s; t_ns={now_ns}. WINDOW: past {meta['past_s']}s / future {meta['future_s']}s; ego timeline {1/meta['table_step_s']:g} Hz.",
        'Positive offset/yaw/relative heading = left. Relative heading is orientation minus NOW; its sign alone is not turn direction.',
        'LANE CONTEXT',
        'LANE is a matched map segment. Connected segments may follow one lane. ROAD_GROUP contains parallel lanes; lane order increases left to right.',
        'Native polygon alignment is unvalidated. unknown = missing evidence. Connector overlap is not exclusive physical ownership.',
        'ALIAS RAW_ID TYPE LEFT_NEIGHBOR RIGHT_NEIGHBOR ROAD_GROUP LANE_ORDER']
    geometry = meta.get('ego_geometry') or {}
    lines.insert(3, f"EGO BODY: {geometry.get('length_m','unknown')} x {geometry.get('width_m','unknown')} m; pose reference {geometry.get('pose_reference','unknown')}; body center {geometry.get('center_forward_m','unknown')} m forward of pose.")
    for lane_id in sorted(relevant, key=lambda x:names[x]):
        lane = topology.get(lane_id, {})
        source = lanes.get(lane_id, {})
        road = 'intersection_' + str(source.get('map_turn_direction') or 'unknown').upper() if source.get('intersection') else ('connector' if lane.get('connector') else 'road')
        def neighbor(side):
            if lane.get('connector') or not lane:
                return 'unknown'
            value = lane.get(side)
            return 'none' if value is None else names.get(value, f'raw_id:{value}')
        lines.append(f"{names[lane_id]} {lane_id} {road} {neighbor('left')} {neighbor('right')} {lane.get('group','unknown')} {lane.get('order') or 'unknown'}")
    lines.extend(['NEXT CONNECTED LANES (mapped successors; no unique keep-lane successor designated)'])
    for lane_id in sorted(relevant, key=lambda x:names[x]):
        successors = topology.get(lane_id, {}).get('following', [])
        represented=[x for x in successors if x in names and x in relevant]
        if represented:
            lines.append(f"{names[lane_id]} -> {', '.join(names[x] for x in represented)}")
    for paths in alternatives:
        lines.append('Adjacent-road connected paths: '+'; '.join(' -> '.join(names[x] for x in path) for path in paths)+'.')
    lines.append('Observed native reference chain: ' + (' -> '.join(names.get(x, f'raw_id:{x}') for x in reference_ids) or 'unknown in reviewed region'))
    if manual:
        lines.extend([manual, f"Within the {review.get('margin_s',0):g}s margins, native lane geometry is unknown; no ownership is asserted."])
    lines.extend(['', 'EGO TIMELINE',
                  'Speed km/h; accel m/s²; yaw deg/s; relative heading deg; offsets m.',
                  'REF_BODY%=ego footprint in native reference-chain polygon union; overlaps are not physical ownership.',
                  'ROAD_BODY%=ego footprint shares in same-direction ordinary-road polygons; overlapping connector polygons excluded.',
                  *(['LANE PHYS%=camera-reviewed physical-chain ownership, otherwise unknown.'] if review else []),
                  'TIME LANE ROAD_CONTEXT OFFSET SPEED ACCEL YAW HEADING LIGHT REF_OFFSET REF_BODY% '+('LANE_PHYS% ' if review else '')+'ROAD_BODY%'])
    for ego in facts['ego_samples']:
        body = ego.get('body_occupancy') or {}
        speed = ego.get('speed_m_s')
        yaw = ego.get('yaw_rate_rad_s')
        fraction = body.get('reference_lane_fraction')
        road = 'IS_'+str(ego.get('map_turn_direction') or 'unknown').upper() if ego.get('intersection') else ('road' if ego.get('intersection') is False else 'unknown')
        shares=road_shares(body,topology)
        body_text='/'.join(f'{names[lane_id]}:{fraction:.1f}' for lane_id,fraction in sorted(shares.items())) or 'unknown'
        lines.append(f"{ego['time_s']:+g}s {ego.get('lane') or 'unknown'} {road} {_fmt(ego.get('offset_m'))} {_fmt(None if speed is None else speed*3.6)} {_fmt(ego.get('acceleration_m_s2'),2)} {_fmt(None if yaw is None else math.degrees(yaw))} {_fmt(ego.get('relative_heading_deg'))} {ego.get('light','unknown')} {_fmt(body.get('reference_offset_m'))} {_fmt(None if fraction is None else 100*fraction)} "+(f"{_fmt(ego.get('physical_lane_ownership_pct'))} " if review else '')+body_text + (' <- NOW' if ego['time_s']==0 else ''))
    if not meta.get('has_indicator'):
        lines.append('Indicator unavailable throughout.')
    else:
        lines.append('Indicator observations: '+', '.join(f"{e['time_s']:+g}s:{e.get('indicator','unknown')}" for e in facts['ego_samples']))
    lines.extend(['', 'EVENTS'])
    lines.extend(f"{e['time_s']:+.2f}s: {e['text']}" for e in selected_events)
    # A short ordinary-road segment can fall between the 1 Hz rows. Preserve
    # body observations around actual mapped adjacent-road transitions, without
    # naming a maneuver or assigning exclusive physical ownership.
    bodies=facts.get('lane_evidence',{}).get('observations',[])
    for event in selected_events:
        if 'mapped left neighbor' not in event['text'] and 'mapped right neighbor' not in event['text']:
            continue
        pair=re.findall(r'\((\d+)\)',event['text'])
        if len(pair)!=2:
            continue
        a,b=map(int,pair)
        around=[row for row in bodies if event['time_s']-3 <= row['time_s'] <= event['time_s']+3
                and a in road_shares(row,topology) and b in road_shares(row,topology)]
        if not around:
            continue
        selected={row['time_s']:row for row in (around[0],min(around,key=lambda x:abs(x['time_s']-event['time_s'])),around[-1])}
        lines.append(f"Adjacent-road body samples {names[a]}/{names[b]} (mapped estimates, not a maneuver label): "+
            '; '.join(f"{row['time_s']:+.2f}s {names[a]}:{road_shares(row,topology)[a]:.1f}% {names[b]}:{road_shares(row,topology)[b]:.1f}%" for row in sorted(selected.values(),key=lambda x:x['time_s'])))
    for intersection in facts.get('intersections', []):
        lines.append(f"Intersection map occupancy {intersection['first_inside_s']:+.2f}s to {intersection['last_inside_s']:+.2f}s; first outside: {_fmt(intersection.get('first_outside_s'),2)}s; map turn {intersection['map_turn_direction']}; measured heading change {_fmt(intersection.get('heading_change_deg'))}deg. Occupancy boundaries are not maneuver boundaries." + (' Partial span: remaining map occupancy suppressed by camera review.' if intersection.get('review_clipped') else ''))
    # Preserve actual positive fork evidence, omit repetitive absence boilerplate.
    fork = facts.get('fork_context_script')
    if fork:
        lines.extend(['FORK CONTEXT',fork])
    local = facts.get('local_speed', {})
    local_accel=local.get('minimum_acceleration_m_s2') or {}
    if local_accel.get('value') is not None and local_accel['value'] < -2.5:
        jerk=local.get('minimum_jerk_m_s3') or {}
        lines.append(f"Local braking detail: minimum acceleration at {_fmt(local_accel.get('time_s'),2)}s; minimum jerk {_fmt(jerk.get('value'),2)}m/s³ at {_fmt(jerk.get('time_s'),2)}s (over local speed interval).")
    lines.extend(['',f"LOCAL LONGITUDINAL FEATURES: {_fmt(local.get('from_s'))} to {_fmt(local.get('to_s'))}s; complete={local.get('complete_window',False)}; missing_ego={local.get('missing_ego_data',False)}.",
                  f"speed_now_m_s={_fmt(local.get('speed_now_m_s'),4)}; speed_change_m_s={_fmt(local.get('speed_change_m_s'),4)}; minimum_acceleration_m_s2={_fmt((local.get('minimum_acceleration_m_s2') or {}).get('value'),4)}.",
                  'Acceleration is derived from smoothed speed; short peaks can be softened.',
                  '', 'OBJECT TRACKS',
                  'Ahead/left/body_gap in m; object speed m/s; size length x width m. body_gap=edge-to-edge ego/object distance.',
                  'object_footprint_in_reference_lane_pct is OBJECT area inside native reference-chain polygon union / object footprint area; map-derived, physical blockage unverified.',
                  'Reference polygon shares are unavailable in the reviewed map-mismatch region; object position and body gap remain observed.',
                  'ID TYPE TIME AHEAD LEFT SPEED SIZE BODY_GAP object_footprint_in_reference_lane_pct EVENT'])
    for track in facts.get('object_tracks', []):
        observations = track.get('observations', [])
        if not observations:
            continue
        keys = track.get('key_observations', [])
        # Merge anchors by timestamp; include actual NOW and closest passage for every retained track.
        selected = {}
        for obs in keys:
            selected[obs['time_s']] = copy.deepcopy(obs)
        now_obs = min(observations, key=lambda x:abs(x['time_s']))
        if abs(now_obs['time_s']) <= max(.6, meta['table_step_s'] / 2):
            selected.setdefault(now_obs['time_s'], copy.deepcopy(now_obs)).setdefault('anchor_reasons', []).append('NOW observation')
        valid = [o for o in observations if o.get('ego_body_clearance_m') is not None]
        if valid:
            closest = min(valid, key=lambda x:x['ego_body_clearance_m'])
            selected.setdefault(closest['time_s'], copy.deepcopy(closest)).setdefault('anchor_reasons', []).append('minimum body gap')
        for obs in sorted(selected.values(), key=lambda x:x['time_s']):
            normalize = {'NOW':'NOW observation', 'closest body passage':'minimum body gap'}
            reasons = list(dict.fromkeys(normalize.get(r,r) for r in obs.get('anchor_reasons', [])))
            lines.append(f"{track['alias']} {track.get('kind','unknown').replace(' ','_')} {obs['time_s']:+.2f}s {_fmt(obs.get('ahead_m'))} {_fmt(obs.get('left_m'))} {_fmt(obs.get('speed_m_s'),2)} {_fmt(obs.get('length_m'),2)}x{_fmt(obs.get('width_m'),2)} {_fmt(obs.get('ego_body_clearance_m'),2)} {_fmt(object_share(obs))} {'; '.join(reasons) or 'observed'}")
    selection = facts.get('object_selection', {})
    lines.append(f"Objects retained: {selection.get('retained_tracks','unknown')} of {selection.get('qualifying_tracks','unknown')} qualifying tracks. Geometry supports interaction evidence, not avoidance intent.")
    facts['events'] = selected_events
    facts['compact_lane_context'] = [dict(topology.get(x, {}),alias=names[x]) for x in sorted(relevant)]
    return '\n'.join(lines)


def generate(source, out, review_path=None):
    source, out = Path(source).resolve(), Path(out).resolve()
    if source == out or out.exists():
        raise ValueError('output must be a new directory, distinct from source')
    review = json.loads(Path(review_path).read_text()) if review_path else None
    if review and (review['to_s'] < review['from_s'] or review.get('margin_s',0) < 0 or not review.get('source')):
        raise ValueError('review needs an ordered interval, nonnegative margin and provenance source')
    started = time.perf_counter()
    out.mkdir(parents=True)
    for sub in ('meta','camera','indicator','tags'):
        if (source/sub).exists():
            shutil.copytree(source/sub,out/sub)
    (out/'steps').mkdir()
    report = dict(source=str(source),script_format=FORMAT,review=review,sessions=[],schema_version=SCHEMA)
    for path in sorted((source/'steps').glob('*.jsonl')):
        meta_path = out/'meta'/f'{path.stem}.json'
        meta = json.loads(meta_path.read_text())
        if review and (meta['start_ns'] != review['start_ns'] or review.get('session_id') != path.stem):
            raise ValueError('review file must explicitly match this session and start timestamp')
        topology = map_metadata(meta['map'])
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        meta.update(sample_schema_version=SCHEMA,script_format=FORMAT,prepared_from=str(source),lane_review=review)
        meta_path.write_text(json.dumps(meta,indent=2)+'\n')
        count, corrected, chars, native_chars = 0,0,0,0
        with path.open() as original, (out/'steps'/path.name).open('w') as target:
            for line in original:
                row = json.loads(line)
                facts = effective_facts(row['facts'],row['t_ns'],review)
                # Keep source traceability rather than silently overwriting native evidence.
                facts['native_audit'] = dict(path=str(path),sha256=sha,t_ns=row['t_ns'])
                fork=_fork_section(row['script'])
                if fork and not fork.startswith('no clear'):
                    retained=[]
                    for fork_line in fork.splitlines():
                        event_time=re.match(r't=([+-]?[\d.]+)s:',fork_line)
                        if not facts.get('lane_review') or (event_time and not reviewed_state(row['t_ns'],float(event_time[1]),review)):
                            retained.append(fork_line)
                    if retained:
                        facts['fork_context_script']='\n'.join(retained)
                script = render_compact(facts,row['t_ns'],meta,topology,_events(row['script']))
                target.write(json.dumps(dict(t_ns=row['t_ns'],script=script,facts=facts),ensure_ascii=False)+'\n')
                count+=1;corrected+=bool(facts.get('lane_review'));chars+=len(script);native_chars+=len(row['script'])
        report['sessions'].append(dict(id=path.stem,samples=count,samples_with_reviewed_context=corrected,
                                       source_sha256=sha,total_script_chars=chars,previous_script_chars=native_chars))
    report['preparation_wall_seconds']=time.perf_counter()-started
    (out/'preparation.json').write_text(json.dumps(report,indent=2)+'\n')
    if review_path:
        shutil.copy2(review_path,out/'lane_review.json')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source');parser.add_argument('--out',required=True);parser.add_argument('--review')
    args=parser.parse_args()
    generate(args.source,args.out,args.review)
