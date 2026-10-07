"""Add source-grounded moving-vehicle history to prepared nuPlan samples."""

import copy
import math
import re

import numpy as np

from .lane_evidence import rectangle

def derive_observations(actors, ego, connector, now_ns):
    """Actors: [us,x,y,yaw,vx,vy,length,width]; ego: [us,x,y,yaw]."""
    actors=np.asarray(actors,dtype=float);ego=np.asarray(ego,dtype=float)
    if not len(actors) or not len(ego):return []
    yaw=np.unwrap(ego[:,3]);result=[]
    for us,x,y,angle,vx,vy,length,width in actors:
        if not np.isfinite([us,x,y,angle,vx,vy,length,width]).all() or min(length,width)<=0:continue
        if us<ego[0,0] or us>ego[-1,0] or np.min(abs(ego[:,0]-us))>200000:continue
        right=min(int(np.searchsorted(ego[:,0],us)),len(ego)-1)
        left=max(0,right-1)
        if abs(ego[right,0]-us)>1e-6 and ego[right,0]-ego[left,0]>200000:continue
        ex=np.interp(us,ego[:,0],ego[:,1]);ey=np.interp(us,ego[:,0],ego[:,2]);heading=np.interp(us,ego[:,0],yaw)
        c,s=math.cos(heading),math.sin(heading);dx,dy=x-ex,y-ey
        body=rectangle(ex,ey,heading,5.176,2.297,1.461)
        footprint=rectangle(x,y,angle,length,width)
        overlap=float(connector.intersection(footprint).area)
        result.append(dict(time_s=(us*1000-now_ns)/1e9,ahead_m=c*dx+s*dy,left_m=-s*dx+c*dy,
            speed_m_s=math.hypot(vx,vy),center_distance_m=math.hypot(dx,dy),length_m=length,width_m=width,
            ego_body_clearance_m=float(body.distance(footprint)),
            connector_fraction=overlap/footprint.area,connector_overlap_m2=overlap,
            relative_yaw_deg=math.degrees(math.atan2(math.sin(angle-heading),math.cos(angle-heading)))))
    return result


def _anchors(observations):
    reasons={}
    def add(k,reason):reasons.setdefault(k,[]).append(reason)
    add(0,'first observation');add(len(observations)-1,'last observation')
    nearest=min(range(len(observations)),key=lambda k:abs(observations[k]['time_s']))
    if abs(observations[nearest]['time_s'])<=.3:add(nearest,'NOW')
    add(min(range(len(observations)),key=lambda k:observations[k]['ego_body_clearance_m']),'closest body passage')
    add(max(range(len(observations)),key=lambda k:observations[k]['connector_fraction']),'maximum overlap')
    # Preserve entry and clearing even when the whole overlap lasts <1s.
    inside=[k for k,p in enumerate(observations) if p['connector_fraction']>.01]
    if inside:
        first,last=inside[0],inside[-1]
        add(first,'first observed overlap');add(last,'last observed overlap')
        if first>0:add(first-1,'last clear before overlap')
        if last+1<len(observations):add(last+1,'first observed clear')
    return [dict(observations[k],anchor_reasons=reasons[k]) for k in sorted(reasons)]


def augment_sample(row, track):
    """Replace one existing low-priority history, retaining its alias and limit.

    Selection is explicit and auditable. It supplies observations, never an
    expected maneuver label or a requested change to Jev's answer.
    """
    result=copy.deepcopy(row);facts=result['facts'];tracks=facts['object_tracks']
    if not tracks or not track['observations']:raise ValueError('Existing tracks and source observations required')
    observations=sorted(track['observations'],key=lambda p:p['time_s'])
    slot=next((i for i,t in enumerate(tracks) if t['track_id']==track['track_id']),len(tracks)-1)
    old=tracks[slot];alias=old['alias'];new=copy.deepcopy(track)
    new.update(alias=alias,key_observations=_anchors(observations),observations=observations,
               selection_tier=old.get('selection_tier',2))
    tracks[slot]=new
    facts['moving_history_review']=dict(track_id=new['track_id'],alias=alias,
        replaced_track_id=old['track_id'] if old['track_id']!=new['track_id'] else None,
        source=new['source'],connector_id=new['connector_id'],overlap_reporting_fraction=.01,
        method='Native object footprints and interpolated native ego poses; mapped connector overlap is not exclusive physical lane ownership')
    facts['object_selection']['policy']=('existing selection; a retained moving actor has expanded mapped-path history'
        if old['track_id']==new['track_id'] else 'existing selection with a moving mapped-path intruder retained in place of the lowest-priority history')
    text=result['script'];before,interaction=text.split('\nOBJECT INTERACTION CONTEXT',1)
    prefix,snapshot=before.split('\nOBJECTS AT NOW',1)
    lines=snapshot.splitlines();nearest=min(observations,key=lambda p:abs(p['time_s']))
    known=[i for i,line in enumerate(lines) if old['track_id']==new['track_id'] and line.startswith(alias+' car: ')]
    signature=new.get('snapshot_signature',nearest)
    snapshot_observation=new.get('snapshot_observation',nearest)
    matched=known
    if not matched and abs(signature['time_s'])<=.3:
        pattern=r'^(?:O\d+ )?car: (\d+) m (ahead|behind), (\d+) m (left|right), .*?, (\d+) km/h(?:, oncoming)?$'
        for i,line in enumerate(lines):
            m=re.match(pattern,line)
            if not m:continue
            a=float(m[1])*(1 if m[2]=='ahead' else -1);l=float(m[3])*(1 if m[4]=='left' else -1)
            if abs(a-signature['ahead_m'])<=.75 and abs(l-signature['left_m'])<=.75 and abs(float(m[5])-signature['speed_m_s']*3.6)<=.75:
                matched.append(i)
    if len(matched)>1:raise ValueError('Ambiguous source identity in NOW snapshot')
    # Keep the original snapshot's observation time and rounded positions.
    # An actor omitted from this snapshot remains available in its history;
    # never duplicate it or silently overwrite an unrelated actor.
    if old['track_id']!=new['track_id']:
        for i,line in enumerate(lines):
            if line.startswith(alias+' '):lines[i]=line[len(alias)+1:]
    if matched:
        i=matched[0];line=re.sub(r'^O\d+ ','',lines[i])
        line=line.replace('other lane, ','').replace('map relationship unknown, ','')
        lines[i]=f"{alias} {line}; observation t={snapshot_observation['time_s']:+.2f}s; mapped connector {new['connector_alias']} footprint overlap {snapshot_observation['connector_fraction']*100:.0f}% (nonexclusive)"
    before=prefix+'\nOBJECTS AT NOW'+'\n'.join(lines)
    history=[f"track {alias} (car): source ID {new['track_id']}; observed t={observations[0]['time_s']:+.2f}s to t={observations[-1]['time_s']:+.2f}s.",
        f"SHARE = this car's footprint area inside ego's mapped connector {new['connector_alias']}; overlapping virtual paths are not exclusive physical lanes.",
        'LEFT positive = left. BODY GAP is the shortest 2D edge-to-edge distance. Event overlap tolerance: 1% actor area.',
        'TIME(s) AHEAD(m) LEFT(m) SPEED(km/h) BODY_GAP(m) SHARE(%) EVENT']
    for p in new['key_observations']:
        history.append(f"{p['time_s']:+.2f} {p['ahead_m']:+.1f} {p['left_m']:+.1f} {p['speed_m_s']*3.6:.1f} {p['ego_body_clearance_m']:.2f} {p['connector_fraction']*100:.1f} {', '.join(p['anchor_reasons'])}")
    pattern=rf'^track {re.escape(alias)}\s*\(.*?(?=^track (?!{re.escape(alias)}\b)O\d+\b|\Z)'
    interaction,count=re.subn(pattern,lambda _:'\n'.join(history)+'\n',interaction,flags=re.M|re.S)
    if count!=1:raise ValueError('Exactly one existing history must be replaced')
    interaction=re.sub(r'(?m)^(Tracks retained .*?; priority: ).*$',lambda m:m[1]+facts['object_selection']['policy']+'.',interaction)
    result['script']=before.rstrip()+'\n\nOBJECT INTERACTION CONTEXT'+interaction
    return result
