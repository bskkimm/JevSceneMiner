"""Correct saved verbose inputs using an explicit camera-reviewed lane interval.

This changes evidence, never predicted labels. Original samples remain the audit
record; callers choose which NOWs to reclassify. Motion and body-to-body object
clearance stay native. Map-derived ownership within the reviewed interval is
replaced rather than presented alongside contradictory native geometry.
"""
from copy import deepcopy
import re

from .compact import effective_facts, reviewed_state, _interval_touches_review, _lane_event
from .script import _interaction_context


def apply_lane_review(row, review):
    now = row['t_ns']
    facts = effective_facts(row['facts'], now, review)
    if 'lane_review' not in facts:
        return deepcopy(row)
    result = deepcopy(row)
    facts['schema_version'] = 'jsm-sample-1.1-reviewed'
    facts['script_format'] = 'verbose-reviewed-1'
    result['facts'] = facts
    state = lambda t: reviewed_state(now, t, review)
    rel_from = (review['start_ns'] - now)/1e9 + review['from_s']
    rel_to = (review['start_ns'] - now)/1e9 + review['to_s']
    core_note = (f"{review['chain_id']}: same physical lane chain from t={rel_from:+.2f}s to t={rel_to:+.2f}s; "
                 f"ego ownership 100% within this interval, source: {review['source']}. "
                 "This is a manual physical-lane annotation, not a measured polygon fraction or a maneuver label.")
    note = (core_note + f" The {review.get('margin_s',0):g} s margins have unknown lane evidence. "
            "Native lane definitions and geometry below apply only outside the reviewed interval and its margins.")
    blocks = row['script'].split('\n\n')
    rendered = []
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            rendered.append(block)
            continue
        title = lines[0]
        if title == 'LANES':
            block = '\n'.join([title, note, *lines[1:]])
        elif title.startswith('TIME') and 'ROAD CONTEXT' in title:
            a,b,c,d,e = [title.index(k) for k in ('LANE','ROAD CONTEXT','OFFSET','SPEED','LIGHT')]
            replaced = [title]
            for line in lines[1:]:
                match = re.match(r't=([+-]?[\d.]+)s', line)
                observed = state(float(match[1])) if match else None
                if observed and '(no data)' not in line:
                    lane = review['chain_id'] if observed == 'core' else 'unknown'
                    road = 'road (camera-reviewed)' if observed == 'core' else 'unknown (review margin)'
                    marker = '   <- NOW' if '<- NOW' in line else ''
                    line = line[:a] + lane.ljust(b-a) + road.ljust(c-b) + '-'.ljust(d-c) + line[d:e] + 'unknown' + marker
                replaced.append(line)
            block = '\n'.join(replaced)
        elif title == 'EVENTS':
            kept = []
            for line in lines[1:]:
                match = re.match(r't=([+-]?[\d.]+)s: (.*)', line)
                if match:
                    t, text = float(match[1]), match[2]
                    lane_event = _lane_event(text) is not None
                    if (lane_event or 'traffic light' in text) and (
                            state(t) or lane_event and _interval_touches_review(now,t-.1,t,review)):
                        continue
                kept.append(line)
            block = '\n'.join([title, *(kept or ['none'])])
        elif title == 'SUMMARY':
            # Whole-window map summaries would mix reviewed and native evidence.
            lines = [line for line in lines if not line.startswith((
                'offset while in lane ', 'enters the intersection lane ', 'leaves the intersection lane ',
                'NOW is inside intersection lane '))]
            block = '\n'.join(lines)
        elif title.startswith('FORK CONTEXT'):
            block = title + '\nNative fork evidence is unavailable across the reviewed lane interval and margins.'
        elif title.startswith('ROUTE CONTEXT'):
            route = []
            for span in facts.get('intersections', []):
                direction = span['map_turn_direction']
                route.append(f"native mapped intersection ({direction}), outside the reviewed interval: "
                             f"t={span['first_inside_s']:+.2f}s to t={span['last_inside_s']:+.2f}s; "
                             "map occupancy only; a clipped boundary is unknown.")
            block = '\n'.join([title, note, *route])
        elif title == 'EGO BODY AND LANE OCCUPANCY':
            body = [title, note]
            for line in lines[1:]:
                match = re.match(r'body t=([+-]?[\d.]+)s', line)
                if not match:
                    # Keep the native per-observation header; discard aggregate
                    # departure, reset and reference claims spanning the correction.
                    if line.startswith(('BODY TIME', 'Ego body:', 'Map-derived', 'Margins are')):
                        body.append(line)
                    continue
                observed = state(float(match[1]))
                if observed == 'core':
                    body.append(f"body t={float(match[1]):+g}s: physical lane chain {review['chain_id']}; "
                                'ego ownership 100% (manual review); native segment shares/offset/margins unavailable')
                elif observed:
                    body.append(f"body t={float(match[1]):+g}s: lane evidence unknown (review margin)")
                else:
                    body.append(line)
            block = '\n'.join(body)
        elif title.startswith('OBJECTS AT NOW') and state(0):
            lines = [re.sub(r', (?:same lane(?: [A-Z]+)?|other lane|map relationship unknown|left neighbor lane|right neighbor lane|lane ahead|lane behind|in a lane),',
                            ', lane relationship unknown (reviewed map mismatch),', line) for line in lines]
            block = '\n'.join(lines)
        elif title.startswith('OBJECT INTERACTION CONTEXT'):
            block = '\n'.join([title, *_interaction_context(facts.get('object_tracks',[]),
                               facts.get('objects_missing_ids',False), facts.get('object_selection'))])
        rendered.append(block)
    result['script'] = '\n\n'.join(rendered)
    return result
