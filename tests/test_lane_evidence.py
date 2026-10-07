import math
from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import box

from jevsceneminer.lanes import Lane, LaneMap, Projection
from jevsceneminer.script import render_sample, strip_map
from jevsceneminer.facts import build_timeline
from conftest import T0, OFFSET_X, OFFSET_Y


def fixture_map(overlap=False):
    lanes = {
        1: Lane(1, 'road', None, 2, True, None, False, (3,), (), np.array([[0., 0.], [10., 0.]])),
        2: Lane(2, 'road', None, None, False, 1, True, (), (), np.array([[0., 3.], [20., 3.]])),
        3: Lane(3, 'road', None, None, False, None, False, (), (1,), np.array([[10., 0.], [20., 0.]])),
    }
    polygons = {1: box(0, -1.5, 10, 1.5), 2: box(0, 1. if overlap else 1.5, 20, 4.5), 3: box(10, -1.5, 20, 1.5)}
    lm = LaneMap(lanes, lambda x, y, r: [(p.distance(__import__('shapely').Point(x,y)), i)
                 for i,p in polygons.items() if p.distance(__import__('shapely').Point(x,y)) <= r])
    lm.polygons = polygons
    return lm


def timeline(y=1., geometry=True):
    n=151; x=np.linspace(7,13,n); yy=np.full(n,y)
    lanes=[Projection(1 if k<15 else 3,0,y,0) for k in range(31)]
    return SimpleNamespace(x=x,y=yy,yaw=np.zeros(n),valid=np.ones(n,bool),lanes=lanes,
        session=SimpleNamespace(ego_geometry={'length_m':4.,'width_m':2.,'center_forward_m':0.,'pose_reference':'body center','source':'test'} if geometry else None),
        lane=lambda i: lanes[min(30,round(i/5))])


def evidence(tl,lm):
    from jevsceneminer.lane_evidence import LaneEvidence
    return LaneEvidence(tl,lm,0,150,75)


def test_straddling_uses_body_area_not_pose_point():
    # Body y=0..2, boundary y=1.5: 75% in original lane, 25% in left lane.
    ev=evidence(timeline(),fixture_map()); row=ev.at(50)
    assert row['reference_lane_fraction']==pytest.approx(.75)
    assert row['outside_reference_fraction']==pytest.approx(.25)
    assert row['left_boundary_margin_m']==pytest.approx(-.5)
    assert row['right_boundary_margin_m']==pytest.approx(1.5)
    assert row['mapped_fraction']==pytest.approx(1)
    assert any(p['lane_id']==2 and p['fraction']==pytest.approx(.25) for p in row['lane_occupancy'])


def test_occupied_opposing_lane_is_distinguished_from_neighbor_lane():
    lm=fixture_map();lm.lanes[2].centerline[:]=lm.lanes[2].centerline[::-1]
    row=evidence(timeline(),lm).at(50)
    other=next(p for p in row['lane_occupancy'] if p['lane_id']==2)
    assert other['direction_relation']=='opposing'


def test_successor_end_cap_does_not_become_lane_departure():
    ev=evidence(timeline(y=0),fixture_map());row=ev.at(75)
    assert row['reference_lane_fraction']==pytest.approx(1)
    assert row['outside_reference_fraction']==pytest.approx(0)
    assert len(row['lane_occupancy'])==2
    assert ev.summary()['departure_intervals']==[]


def test_body_uses_known_successor_even_when_pose_never_enters_it():
    tl=timeline(y=0);tl.x[:]=9.5;tl.lanes[:]=[Projection(1,0,0,0)]*len(tl.lanes)
    ev=evidence(tl,fixture_map());row=ev.at(75)
    assert row['reference_lane_fraction']==pytest.approx(1)
    assert ev.summary()['departure_intervals']==[]


def test_native_longitudinal_gap_is_unknown_area_not_lateral_departure():
    tl=timeline(y=0);lm=fixture_map();lm.polygons[1]=box(0,-1.5,9.5,1.5);lm.polygons[3]=box(10.5,-1.5,20,1.5)
    ev=evidence(tl,lm);row=ev.at(75)
    assert row['outside_reference_fraction']==pytest.approx(.25)
    assert row['lateral_departure'] is None  # a gap cannot establish a return
    assert ev.summary()['departure_intervals']==[]


def test_source_boundary_gap_does_not_crash_cross_section():
    tl=timeline(y=0);lm=fixture_map();lm.polygons[1]=box(0,-1.5,9,1.5);lm.polygons[3]=box(11,-1.5,20,1.5)
    ev=evidence(tl,lm)
    assert ev.at(75)['left_boundary_margin_m'] is None


def test_gap_never_manufactures_return_for_fully_outside_body():
    tl=timeline(y=4);lm=fixture_map();lm.polygons[1]=box(0,-1.5,10,1.5);lm.polygons[3]=box(11,-1.5,20,1.5)
    spans=evidence(tl,lm).summary()['departure_intervals']
    assert spans and all(p['return_s'] is None for p in spans)


def test_map_free_event_anchors_ignore_lane_intrusion_fields():
    from jevsceneminer.sample_context import key_observations
    observations=[dict(time_s=t,ego_body_clearance_m=abs(t-.5),center_distance_m=abs(t-1),
                       ahead_m=1,reference_lane_overlap_m2=1 if t==-.5 else 0)
                  for t in [-2,-1,-.5,0,.5,1,2]]
    selected=key_observations(observations,use_map=False)
    assert -.5 not in [p['time_s'] for p in selected]


def test_overlapping_polygons_are_flagged_not_double_counted_as_exclusive_lanes():
    row=evidence(timeline(),fixture_map(overlap=True)).at(50)
    assert row['mapped_fraction']==pytest.approx(1)
    assert row['ambiguous_overlap_fraction']==pytest.approx(.25)


def test_reference_does_not_reset_with_matched_centerline():
    lm=fixture_map();lm.lanes[3].centerline[:,1]=.8
    ev=evidence(timeline(y=.5),lm)
    # Joining centerlines gives a continuous proxy, unlike resetting at x=10.
    before,after=ev.at(74)['reference_offset_m'],ev.at(76)['reference_offset_m']
    assert abs(after-before)<.1


def test_missing_body_dimensions_remain_unknown():
    ev=evidence(timeline(geometry=False),fixture_map())
    assert ev.ego_shape(50) is None
    assert ev.at(50)['outside_reference_fraction'] is None
    assert ev.at(50)['reference_offset_m']==pytest.approx(1)


def test_pose_gap_breaks_departure_interval_and_preserves_unknown_return():
    tl=timeline();tl.valid[70:81]=False
    spans=evidence(tl,fixture_map()).summary()['departure_intervals']
    assert len(spans)==2
    assert spans[0]['return_s'] is None
    assert spans[1]['entry_observed'] is False


def test_footprint_requires_finite_pose():
    from jevsceneminer.sample_context import _footprint
    obj=SimpleNamespace(length_m=4,width_m=2,x=0,y=0,yaw=math.nan)
    assert _footprint(obj) is None


def test_lanelet_map_exposes_real_polygons(lanemap):
    assert lanemap.polygons[10].area==pytest.approx(175,abs=.02)


def test_body_clearance_and_key_pass_observations_reach_script(lanemap,lane_change_session):
    from jevsceneminer.bag import DetectedObject
    s=lane_change_session;s.ego.y[:]=OFFSET_Y+2.8;s.ego.yaw[:]=0
    s.ego_geometry={'length_m':4.,'width_m':2.,'center_forward_m':0.,'pose_reference':'body center','source':'test'}
    # Stopped box y=.25..2.25: body y=1.8..3.8 overlaps laterally;
    # x at NOW=20, object x=24, bodies separated by 0 m at this instant.
    s.objects=[(T0+int(t*1e9),[DetectedObject('car',OFFSET_X+24,OFFSET_Y+1.25,0,0,1,'blocker',4,2)])
               for t in np.arange(0,10.01,.5)]
    tl=build_timeline(s,lanemap);sample=render_sample(tl,lanemap,T0+5*10**9,past_s=5,future_s=5)
    tr=sample['facts']['object_tracks'][0];atnow=next(p for p in tr['observations'] if p['time_s']==0)
    assert atnow['ego_body_clearance_m']==pytest.approx(0)
    assert atnow['length_m']==4 and atnow['width_m']==2
    assert any(p['time_s']==0 and 'NOW' in p['anchor_reasons'] for p in tr['key_observations'])
    assert any('closest body passage' in p['anchor_reasons'] for p in tr['key_observations'])
    assert 'EGO BODY AND LANE OCCUPANCY' in sample['script']
    assert 'body clearance 0.00 m' in sample['script']
    assert 'O1 car:' in sample['script']
    assert 'EGO BODY AND LANE OCCUPANCY' not in strip_map(sample['script'])
    assert 'body clearance' in strip_map(sample['script'])


def test_stopped_lane_intruder_survives_closer_side_objects(lanemap,lane_change_session):
    from jevsceneminer.bag import DetectedObject
    s=lane_change_session;s.ego.y[:]=OFFSET_Y+1.75;s.ego.yaw[:]=0
    nearby=[DetectedObject('car',OFFSET_X+20,OFFSET_Y+5+j*.1,0,0,1,f'side{j}',1,1) for j in range(9)]
    blocker=DetectedObject('car',OFFSET_X+30,OFFSET_Y+1.75,0,0,1,'blocker',4,2)
    s.objects=[(T0+5*10**9,nearby+[blocker])]
    sample=render_sample(build_timeline(s,lanemap),lanemap,T0+5*10**9,past_s=1,future_s=1)
    assert 'blocker' in {t['track_id'] for t in sample['facts']['object_tracks']}
    assert sample['facts']['object_selection']['omitted_tracks']==2


def test_no_map_actor_selection_is_independent_of_reference_polygons(lanemap,lane_change_session):
    s=lane_change_session
    from jevsceneminer.bag import DetectedObject
    s.ego.y[:]=OFFSET_Y+1.75;s.ego.yaw[:]=0
    s.objects=[(T0+5*10**9,[DetectedObject('car',OFFSET_X+20,OFFSET_Y+5+j*.1,0,0,1,f'side{j}',1,1)
                          for j in range(9)]+[DetectedObject('car',OFFSET_X+30,OFFSET_Y+1.75,0,0,1,'blocker',4,2)])]
    mapped=render_sample(build_timeline(s,lanemap),lanemap,T0+5*10**9,past_s=1,future_s=1)
    missing_polygons=LaneMap(lanemap.lanes,lanemap._find_within)
    unknown=render_sample(build_timeline(s,missing_polygons),missing_polygons,T0+5*10**9,past_s=1,future_s=1)
    assert mapped['facts']['map_free_object_script']==unknown['facts']['map_free_object_script']
    text=strip_map(mapped['script'],mapped['facts']['map_free_object_script'])
    assert '10 m ahead' not in text  # farther blocker was selected only in mapped mode


def test_geometry_override_is_used_in_prepared_facts(tmp_path,monkeypatch,lanemap,lane_change_session):
    import json
    from jevsceneminer import cli
    lane_change_session.topics={}
    monkeypatch.setattr(cli,'_read',lambda *args:(lane_change_session,lanemap,'right',{}))
    assert cli.main(['run','synthetic','--dry-run','--max-steps','1','--out',str(tmp_path),
                     '--ego-length','4','--ego-width','2','--ego-center-offset','0'])==0
    meta=json.loads(next((tmp_path/'meta').glob('*.json')).read_text())
    row=json.loads(next((tmp_path/'steps').glob('*.jsonl')).read_text().splitlines()[0])
    assert row['facts']['lane_evidence']['ego_geometry']['width_m']==2
    assert meta['ego_geometry']['center_forward_m']==0
    assert meta['sample_schema_version']==row['facts']['schema_version']


@pytest.mark.parametrize('flags',[
    ['--ego-length','4'],
    ['--ego-length','4','--ego-width','nan','--ego-center-offset','0'],
    ['--ego-length','-4','--ego-width','2','--ego-center-offset','0'],
])
def test_geometry_override_rejects_missing_or_invalid_reference(flags,tmp_path):
    from jevsceneminer import cli
    with pytest.raises(SystemExit) as exc:cli.main(['run','unused','--dry-run','--out',str(tmp_path),*flags])
    assert exc.value.code==2


def test_event_anchors_do_not_invent_now_when_track_is_absent():
    from jevsceneminer.sample_context import key_observations
    observations=[dict(time_s=t,ego_body_clearance_m=abs(t),center_distance_m=abs(t),ahead_m=1,
                       reference_lane_overlap_m2=0) for t in [-2.,-1.,1.,2.]]
    assert all('NOW' not in p['anchor_reasons'] for p in key_observations(observations))


def test_no_map_retains_body_clearance_but_not_lane_event_anchors(lanemap,lane_change_session):
    from jevsceneminer.bag import DetectedObject
    s=lane_change_session;s.ego_geometry=dict(length_m=4,width_m=2,center_forward_m=0,pose_reference='body center',source='test')
    s.objects=[(T0+int(t*1e9),[DetectedObject('car',OFFSET_X+25,OFFSET_Y+1.75,0,0,1,'x',4,2)]) for t in np.arange(0,10,.5)]
    text=strip_map(render_sample(build_timeline(s,lanemap),lanemap,T0+5*10**9,past_s=5,future_s=5)['script'])
    assert 'body clearance' in text
    for phrase in ['body departure','body return','lateral excursion peak','reference-lane','lane-centered','REFERENCE']:
        assert phrase not in text
