import math

import numpy as np

from conftest import OFFSET_X, OFFSET_Y, T0
from jevsceneminer.facts import build_timeline
from jevsceneminer.script import render, strip_map


def test_topology(lanemap):
    a, b, c = lanemap.lanes[10], lanemap.lanes[11], lanemap.lanes[12]
    assert a.right == 11 and b.left == 10
    assert a.left is None and b.right is None
    assert a.right_crossable and not a.left_crossable
    assert a.following == (12,) and c.previous == (10,)
    assert lanemap.relation(10, 12) == "following"
    assert lanemap.relation(10, 11) == "right"
    assert lanemap.relation(11, 10) == "left"
    assert lanemap.relation(11, 12) == "other"


def test_coordinates_follow_local_xy(lanemap):
    center = lanemap.lanes[10].centerline
    assert np.allclose(center[0], [OFFSET_X, OFFSET_Y + 1.75], atol=1e-3)


def test_candidates_offset_sign(lanemap):
    (p,) = [c for c in lanemap.candidates(OFFSET_X + 25, OFFSET_Y + 2.75, 0.0) if c.lane_id == 10]
    assert math.isclose(p.offset, 1.0, abs_tol=1e-3)   # left of center is positive
    (q,) = [c for c in lanemap.candidates(OFFSET_X + 25, OFFSET_Y + 0.75, 0.0) if c.lane_id == 10]
    assert math.isclose(q.offset, -1.0, abs_tol=1e-3)
    assert lanemap.candidates(OFFSET_X + 25, OFFSET_Y + 1.75, math.pi) == []   # wrong direction


def test_match_track_follows_the_road(lanemap):
    xs = OFFSET_X + np.arange(1, 100, 2.0)
    path = lanemap.match_track(xs, np.full_like(xs, OFFSET_Y + 1.75), np.zeros_like(xs), np.ones_like(xs, bool))
    ids = [p.lane_id for p in path]
    assert set(ids[:20]) == {10} and set(ids[-20:]) == {12}


def test_script_describes_the_lane_change(lanemap, lane_change_session):
    tl = build_timeline(lane_change_session, lanemap)
    text = render(tl, lanemap, T0 + 2 * 10**9, past_s=5, future_s=5)   # NOW = 2 s, before the lane change
    assert "t=+2.0s: moves from A into its right neighbor B" in text
    assert "t=+0.5s: indicator RIGHT on" in text
    assert "indicator during the table: RIGHT from t=+0.5s to t=+5.0s" in text
    assert "A: road; left neighbor: none; right neighbor: B (dashed line, can be crossed)" in text
    assert "car: 15 m ahead, 0 m left, same lane, 14 km/h" in text
    assert "t=-5s   (no data)" in text
    assert "<- NOW" in text
    # Only relative facts: no absolute coordinates or clock times leak into the script.
    assert str(int(OFFSET_X)) not in text and str(T0)[:6] not in text


def test_render_returns_none_without_ego_data(lanemap, lane_change_session):
    tl = build_timeline(lane_change_session, lanemap)
    assert render(tl, lanemap, T0 + 60 * 10**9) is None


def test_script_says_when_the_indicator_is_never_on(lanemap, lane_change_session):
    s = lane_change_session
    s.indicator[:] = 1                       # indicator off the whole time
    text = render(build_timeline(s, lanemap), lanemap, T0 + 2 * 10**9, past_s=5, future_s=10)
    assert "indicator during the table: never on" in text
    assert "t=+10s" in text and "t=-5s" in text   # asymmetric table: -5 s .. +10 s


def test_strip_map_removes_every_map_fact(lanemap, lane_change_session):
    tl = build_timeline(lane_change_session, lanemap)
    full = render(tl, lanemap, T0 + 2 * 10**9, past_s=5, future_s=10)
    bare = strip_map(full)
    for gone in ("LANES", "OFFSET", "neighbor", "moves from", "same lane", "offset while", "lane center"):
        assert gone not in bare, gone
    for kept in ("SPEED", "YAW RATE", "indicator RIGHT on", "car: 15 m ahead, 0 m left, 14 km/h", "<- NOW"):
        assert kept in bare, kept
    row = next(l for l in bare.splitlines() if l.startswith("t=+0s"))
    assert row.split()[1] == "14"   # speed now follows the time column directly


def test_the_light_ahead_is_picked_from_all_recognized_groups():
    from types import SimpleNamespace

    from jevsceneminer.facts import LIGHT_AHEAD_M, lights_ahead, resolve_lights
    from jevsceneminer.lanes import Projection

    lanemap = SimpleNamespace(lanes={1: SimpleNamespace(traffic_lights=()), 2: SimpleNamespace(traffic_lights=(7,))})
    p1, p2 = Projection(1, 0, 0, 0), Projection(2, 0, 0, 0)
    x = np.array([0.0, 10.0, 20.0, 20.0 + LIGHT_AHEAD_M + 1])
    ahead = lights_ahead([p1, p1, p2, p1], x, np.zeros(4), lanemap)
    assert ahead == [(7,), (7,), (7,), ()]
    far = lights_ahead([p1, p2], np.array([0.0, LIGHT_AHEAD_M + 1]), np.zeros(2), lanemap)
    assert far == [(), (7,)]                          # too far along the path: no light yet
    session = SimpleNamespace(lights=[(T0, {7: "red", 9: "green"}), (T0 + 10**8, {9: "green"}),
                                      (T0 + 2 * 10**8, {None: "amber"})])
    assert [text for _, text in resolve_lights(session, T0, ahead)] == ["red", "unknown", "amber"]



def test_script_names_the_traffic_side(lanemap, lane_change_session):
    tl = build_timeline(lane_change_session, lanemap)
    assert "(left-hand traffic)" in render(tl, lanemap, T0 + 2 * 10**9)
    assert "(right-hand traffic)" in render(tl, lanemap, T0 + 2 * 10**9, traffic_side="right")


def test_default_script_has_ten_seconds_each_side_at_two_hz(lanemap, lane_change_session):
    import re

    text = render(build_timeline(lane_change_session, lanemap), lanemap, T0 + 5 * 10**9)
    table = text.split("TIME ", 1)[1].split("\nEVENTS", 1)[0]
    times = [float(t) for t in re.findall(r"^t=([+-][\d.]+)s", table, re.M)]
    assert len(times) == 41
    assert times[:3] == [-10.0, -9.5, -9.0]
    assert times[-3:] == [9.0, 9.5, 10.0]
    assert sum("<- NOW" in line for line in table.splitlines()) == 1
    half_row = next(line for line in table.splitlines() if line.startswith("t=+0.5s"))
    assert half_row.split()[1] in {"A", "B"}  # time and lane remain separate columns
    bare = strip_map(text)
    half_bare = next(line for line in bare.splitlines() if line.startswith("t=+0.5s"))
    assert half_bare.split()[1:3] == ["14", "km/h"]


def test_route_context_stays_inside_requested_sample_window(lanemap, lane_change_session):
    from dataclasses import replace
    from jevsceneminer.lanes import LaneMap, Projection

    lanes = dict(lanemap.lanes)
    lanes[12] = replace(lanes[12], turn_direction="straight")
    mapped = LaneMap(lanes, lanemap._find_within)
    tl = build_timeline(lane_change_session, mapped)
    # NOW=5 s, context=[3,6] s. Only the intersection at [4,4.5] is in the window.
    for k in range(len(tl.lanes)):
        t = k * 0.1
        lane_id = 12 if 1 <= t <= 2 or 4 <= t <= 4.5 or 7 <= t <= 8 else 10
        tl.lanes[k] = Projection(lane_id, 0.0, 0.0, 0.0)
    tl.yaw[:] = 0.0
    tl.yaw[tl.index(T0 + int(6.1e9)):] = 1.0  # outside future limit; must not affect route heading
    text = render(tl, mapped, T0 + 5 * 10**9, past_s=2, future_s=1)
    route = text.split("ROUTE CONTEXT", 1)[1].split("OBJECTS AT NOW", 1)[0]
    intersections = [line for line in route.splitlines() if line.startswith("intersection")]
    assert len(intersections) == 1
    assert "last inside at t=-0.5s" in intersections[0]
    assert "first outside at t=-0.4s" in intersections[0]
    assert "heading change +0°" in intersections[0]
    assert "t=-3.0s" not in route and "t=+2.0s" not in route


def test_route_context_does_not_invent_entry_or_exit_at_window_edges(lanemap, lane_change_session):
    from dataclasses import replace
    from jevsceneminer.lanes import LaneMap, Projection

    lanes = dict(lanemap.lanes)
    lanes[12] = replace(lanes[12], turn_direction="straight")
    mapped = LaneMap(lanes, lanemap._find_within)
    tl = build_timeline(lane_change_session, mapped)
    # Actual intersection [2,8] s extends beyond the requested [3,6] s window.
    tl.lanes = [Projection(12 if 2 <= k * 0.1 <= 8 else 10, 0.0, 0.0, 0.0)
                for k in range(len(tl.lanes))]
    text = render(tl, mapped, T0 + 5 * 10**9, past_s=2, future_s=1)
    route = text.split("ROUTE CONTEXT", 1)[1].split("OBJECTS AT NOW", 1)[0]
    assert "already inside at window start (t=-2.0s)" in route
    assert "still inside at window end (t=+1.0s)" in route
    assert "entered at t=-2.0s" not in route and "leaves at t=+1.0s" not in route


def test_table_states_intersection_and_missing_map_on_each_row(lanemap, lane_change_session):
    from dataclasses import replace
    from jevsceneminer.lanes import LaneMap, Projection

    lanes=dict(lanemap.lanes)
    lanes[12]=replace(lanes[12],turn_direction='left')
    mapped=LaneMap(lanes,lanemap._find_within)
    tl=build_timeline(lane_change_session,mapped)
    tl.lanes[40]=Projection(12,0,0,0)
    tl.lanes[45]=None
    tl.lanes[50]=Projection(10,0,0,0)
    tl.yaw[:]=np.linspace(0,math.pi,len(tl.yaw))  # heading alone must not set the map flag
    text=render(tl,mapped,T0+5*10**9,past_s=1,future_s=1)
    header=next(l for l in text.splitlines() if l.startswith('TIME '))
    a,b=header.index('ROAD CONTEXT'),header.index('OFFSET')
    rows={l.split()[0]:l[a:b].strip() for l in text.splitlines() if l.startswith('t=') and '<- NOW' in l or l.startswith(('t=-1s ', 't=-0.5s '))}
    assert rows['t=-1s']=='intersection (LEFT)'
    assert rows['t=-0.5s']=='map match unavailable'
    assert rows['t=+0s']=='road'
    bare=strip_map(text)
    assert 'ROAD CONTEXT' not in bare
    assert 'intersection (LEFT)' not in bare
    assert 'map match unavailable' not in bare


def test_table_cadence_can_change_without_changing_window(lanemap, lane_change_session):
    import re
    tl=build_timeline(lane_change_session,lanemap)
    for step,want in [(1,[-1,0,1]),(.25,[-1,-.75,-.5,-.25,0,.25,.5,.75,1]),(.75,[-1,-.75,0,.75,1])]:
        text=render(tl,lanemap,T0+5*10**9,past_s=1,future_s=1,table_step_s=step)
        table=text.split('TIME ',1)[1].split('\nEVENTS',1)[0]
        assert [float(x) for x in re.findall(r'^t=([+-][\d.]+)s',table,re.M)]==want
        assert sum('<- NOW' in l for l in table.splitlines())==1


def test_cli_table_step_is_independent_of_inference_step(tmp_path,monkeypatch,lanemap,lane_change_session):
    import json
    from jevsceneminer import cli
    lane_change_session.topics={}
    monkeypatch.setattr(cli,'_read',lambda *args:(lane_change_session,lanemap,'right',{}))
    assert cli.main(['run','synthetic','--out',str(tmp_path),'--dry-run','--step','.5','--table-step','1','--past','1','--future','1'])==0
    meta=json.loads(next((tmp_path/'meta').glob('*.json')).read_text())
    rows=[json.loads(l) for l in next((tmp_path/'steps').glob('*.jsonl')).read_text().splitlines()]
    assert meta['step_s']==.5 and meta['table_step_s']==1
    assert len(rows)==21 and rows[1]['t_ns']-rows[0]['t_ns']==500_000_000
    table=rows[10]['script'].split('TIME ',1)[1].split('\nEVENTS',1)[0]
    assert sum(l.startswith('t=') for l in table.splitlines())==3


def test_fractional_table_step_keeps_columns_aligned_and_strips_map(lanemap,lane_change_session):
    tl=build_timeline(lane_change_session,lanemap)
    for step in (.125,.033):
        text=render(tl,lanemap,T0+5*10**9,past_s=1,future_s=1,table_step_s=step)
        table=text.split('TIME ',1)[1].split('\nEVENTS',1)[0]
        header=next(l for l in text.splitlines() if l.startswith('TIME '))
        a,b=header.index('ROAD CONTEXT'),header.index('OFFSET')
        for line in table.splitlines():
            if not line.startswith('t='):continue
            assert line.split()[0].endswith('s')
            assert line.split()[1] in {'A','B'}
            assert line[a:b].strip()=='road'
        stripped=strip_map(text).split('TIME ',1)[1].split('\nEVENTS',1)[0]
        for line in stripped.splitlines():
            if not line.startswith('t='):continue
            assert line.split()[0].endswith('s')
            assert line.split()[1:3]==['14','km/h']



def test_local_speed_summary_uses_precise_now_and_local_extrema(lanemap, lane_change_session):
    tl = build_timeline(lane_change_session, lanemap)
    tl.v[:] = 0.2
    tl.accel[:] = 0.0
    tl.jerk[:] = 0.0
    # NOW=5 s, local context=[3,7] s. A larger event outside must not contaminate it.
    tl.v[150] = 1.0
    tl.v[350] = 3.0
    tl.accel[100], tl.accel[225] = -9.0, -3.5
    tl.jerk[100], tl.jerk[220] = -20.0, -4.0
    text = render(tl, lanemap, T0 + 5 * 10**9)
    assert 'LOCAL SPEED CONTEXT' in text
    local = text.split('LOCAL SPEED CONTEXT', 1)[1].split('\nSUMMARY', 1)[0]
    assert 'speed at NOW: 0.20 m/s (0.72 km/h)' in local
    assert 'speed change: +2.00 m/s' in local
    assert 'minimum acceleration: -3.50 m/s² at t=-0.5s' in local
    assert 'minimum jerk: -4.00 m/s³ at t=-0.6s' in local
    assert '-9.00' not in local and '-20.00' not in local
    assert 'LOCAL SPEED CONTEXT' in strip_map(text)


def test_local_speed_summary_flags_gaps_and_excludes_invalid_values(lanemap, lane_change_session):
    tl = build_timeline(lane_change_session, lanemap)
    tl.accel[:], tl.jerk[:] = 0.0, 0.0
    tl.valid[200] = False
    tl.accel[200], tl.jerk[200] = -99.0, -99.0
    text = render(tl, lanemap, T0 + 5 * 10**9)
    assert 'LOCAL SPEED CONTEXT' in text
    local = text.split('LOCAL SPEED CONTEXT', 1)[1].split('\nSUMMARY', 1)[0]
    assert 'missing ego data' in local
    assert 'speed change: unknown' in local
    assert '-99.00' not in local
    clipped = render(tl, lanemap, T0, past_s=1, future_s=1)
    assert 'partial window' in clipped.split('LOCAL SPEED CONTEXT', 1)[1].split('\nSUMMARY', 1)[0]


def _fork_fixture(lanemap, lane_change_session, intersection=False, parallel=False):
    from dataclasses import replace
    from jevsceneminer.lanes import LaneMap, Projection
    lanes = dict(lanemap.lanes)
    origin = lanes[10].centerline[-1].copy()
    lanes[10] = replace(lanes[10], following=(12, 13))
    lanes[12] = replace(lanes[12], centerline=np.array([origin, origin+[20,10]]),
                        following=(), turn_direction='left' if intersection else None)
    lanes[13] = replace(lanes[12], id=13, centerline=np.array([origin,origin+[20,10 if parallel else -10]]),
                        turn_direction='right' if intersection else None)
    mapped = LaneMap(lanes, lanemap._find_within)
    tl = build_timeline(lane_change_session, lanemap)
    tl.lanes = [Projection(10 if k < 60 else 12, 0, 0, 0) for k in range(len(tl.lanes))]
    return mapped, tl


def test_fork_context_orders_map_branches_and_observes_chosen_left(lanemap, lane_change_session):
    mapped, tl = _fork_fixture(lanemap, lane_change_session)
    text = render(tl, mapped, T0+5*10**9)
    assert 'FORK CONTEXT' in text
    fork = text.split('FORK CONTEXT', 1)[1].split('\nROUTE CONTEXT', 1)[0]
    assert 't=+1.0s' in fork and 'chosen branch: LEFT' in fork
    assert 'B: LEFT' in fork and 'C: RIGHT' in fork
    assert '+27°' in fork and '-27°' in fork
    assert 'FORK CONTEXT' not in strip_map(text)
    assert 'chosen branch' not in strip_map(text)
    # A direct successor on the right is branch evidence, not a neighbor-lane crossing.
    from jevsceneminer.lanes import Projection
    tl.lanes = [Projection(10 if k < 60 else 13, 0, 0, 0) for k in range(len(tl.lanes))]
    right = render(tl, mapped, T0+5*10**9)
    assert 'chosen branch: RIGHT' in right


def test_fork_context_does_not_invent_choice_at_intersections_or_after_gaps(lanemap, lane_change_session):
    for intersection, parallel in ((True,False),(False,True)):
        mapped, tl = _fork_fixture(lanemap, lane_change_session, intersection, parallel)
        text = render(tl, mapped, T0+5*10**9)
        assert 'FORK CONTEXT' in text
        assert 'chosen branch:' not in text
    mapped, tl = _fork_fixture(lanemap, lane_change_session)
    tl.valid[295:301] = False  # unknown map transition; never jump over the gap
    text = render(tl, mapped, T0+5*10**9)
    assert 'chosen branch:' not in text
    # A transition outside the configured context must also be absent.
    text = render(tl, mapped, T0+8*10**9, past_s=1,future_s=1)
    assert 'chosen branch:' not in text



def test_fork_context_compares_branches_over_common_supported_distance(lanemap, lane_change_session):
    from dataclasses import replace
    from jevsceneminer.lanes import LaneMap
    mapped, tl = _fork_fixture(lanemap, lane_change_session)
    lanes = dict(mapped.lanes)
    origin = lanes[10].centerline[-1]
    # These two map pieces overlap completely for the shorter piece's 5 m.
    lanes[12] = replace(lanes[12], centerline=np.array([origin, origin+[5,0]]))
    lanes[13] = replace(lanes[13], centerline=np.array([origin, origin+[5,0], origin+[20,10]]))
    text = render(tl, LaneMap(lanes, mapped._find_within), T0+5*10**9)
    assert 'chosen branch:' not in text
    # Tiny angled pieces are also insufficient to establish a branch direction.
    lanes[12] = replace(lanes[12], centerline=np.array([origin, origin+[1,.5]]))
    lanes[13] = replace(lanes[13], centerline=np.array([origin, origin+[1,-.5]]))
    text = render(tl, LaneMap(lanes, mapped._find_within), T0+5*10**9)
    assert 'chosen branch:' not in text



def test_intersection_heading_and_exit_share_explicit_map_interval(lanemap,lane_change_session):
    from dataclasses import replace
    from jevsceneminer.lanes import LaneMap,Projection
    lanes=dict(lanemap.lanes);lanes[12]=replace(lanes[12],turn_direction='right')
    mapped=LaneMap(lanes,lanemap._find_within)
    tl=build_timeline(lane_change_session,mapped)
    tl.lanes=[Projection(12 if 30<=k<=60 else 10,0,0,0) for k in range(len(tl.lanes))]
    tl.yaw[:]=np.arange(tl.n)*-.01
    assert 'first outside at t=+1.1s' in render(tl,mapped,T0+5*10**9)
    from jevsceneminer.script import render_sample
    sample=render_sample(tl,mapped,T0+5*10**9)
    span,=sample['facts']['intersections']
    assert span['first_inside_s']==-2.0 and span['last_inside_s']==1.0
    assert span['first_outside_s']==1.1
    assert math.isclose(span['heading_change_deg'],-85.94366926962348)
    route=sample['script'].split('ROUTE CONTEXT',1)[1].split('OBJECTS AT NOW',1)[0]
    assert 'last inside at t=+1.0s' in route and 'first outside at t=+1.1s' in route
    assert 'heading change -86° over t=-2.0s to t=+1.0s' in route
    assert 't=+1.1s: moves back from A into B' in sample['script']
    assert 'Map occupancy boundaries do not define maneuver start/end.' in sample['script']


def test_summary_jerk_is_not_automatically_braking_onset(lanemap,lane_change_session):
    tl=build_timeline(lane_change_session,lanemap)
    tl.accel[:]=1.0;tl.jerk[:]=0.0;tl.jerk[300]=-2.0
    assert 'minimum longitudinal jerk' in render(tl,lanemap,T0+5*10**9)
    from jevsceneminer.script import render_sample
    sample=render_sample(tl,lanemap,T0+5*10**9)
    text=sample['script'];facts=sample['facts']
    assert 'minimum longitudinal jerk: -2.0 m/s³ at t=+1.0s' in text
    assert 'sharpest braking onset' not in text
    assert facts['window']['braking_onsets_s']==[]
    assert 'peak absolute yaw rate' in text and 'over t=-5.0s to t=+5.0s' in text
    assert 'unwrapped yaw(t) - yaw(NOW)' in text
    assert 'Classify the observed lateral maneuver active at t=0' in text
    # Braking begins only on an observed threshold crossing, not a jerk minimum.
    tl.accel[300:]=-1.0
    sample=render_sample(tl,lanemap,T0+5*10**9)
    assert sample['facts']['window']['braking_onsets_s']==[1.0]


def test_unmatched_object_is_unknown_map_relationship(lanemap,lane_change_session):
    from jevsceneminer.bag import DetectedObject
    from jevsceneminer.script import _object_lane,Aliases
    tl=build_timeline(lane_change_session,lanemap)
    obj=DetectedObject('car',OFFSET_X-100,OFFSET_Y+100,0,0,1)
    assert _object_lane(lanemap,Aliases(),tl.lane(250),obj)=='map relationship unknown'


def test_object_tracks_use_stable_ids_and_geometry_without_invented_associations(lanemap,lane_change_session):
    from jevsceneminer.bag import DetectedObject
    assert 'OBJECT INTERACTION CONTEXT' in render(build_timeline(lane_change_session,lanemap),lanemap,T0+5*10**9)
    from jevsceneminer.script import render_sample
    tl=build_timeline(lane_change_session,lanemap)
    # Keep ego in lane A, directly behind one stopped, known-size object.
    tl.y[:]=OFFSET_Y+1.75;tl.yaw[:]=0
    from jevsceneminer.lanes import Projection
    tl.lanes=[Projection(10,0,0,0)]*len(tl.lanes)
    lane_change_session.objects=[(T0+int(t*1e9),[DetectedObject('car',OFFSET_X+24,OFFSET_Y+1.75,0,0,1,
                      track_id='stable-1',length_m=4,width_m=2)]) for t in (3.,5.,7.)]
    tl.object_times=[t for t,_ in lane_change_session.objects]
    sample=render_sample(tl,lanemap,T0+5*10**9,past_s=2,future_s=2)
    track,=sample['facts']['object_tracks']
    assert track['track_id']=='stable-1'
    assert [p['time_s'] for p in track['observations']]==[-2.,0.,2.]
    assert [p['ahead_m'] for p in track['observations']]==[12.,4.,-4.]
    assert all(p['lane_corridor_overlap'] for p in track['observations'])
    assert track['observations'][1]['ego_reference_to_object_footprint_m']==2.0
    assert 'OBJECT INTERACTION CONTEXT' in sample['script']
    bare=strip_map(sample['script'])
    assert 'lane corridor' not in bare and 'lane_corridor' not in bare
    assert 'track O1' in bare and 'footprint' in bare
    # No source ID: do not fabricate a trajectory by matching nearby positions.
    for _,objects in lane_change_session.objects:objects[0].track_id=None
    sample=render_sample(tl,lanemap,T0+5*10**9,past_s=2,future_s=2)
    assert sample['facts']['object_tracks']==[]
    assert 'stable object IDs unavailable' in sample['script']



def test_ros_object_decoder_preserves_id_and_dimensions():
    from types import SimpleNamespace as N
    from jevsceneminer.bag import _objects
    obj=N(classification=[N(label=1,probability=1)],existence_probability=.9,
          object_id=N(uuid=[1]+[0]*15),shape=N(dimensions=N(x=4,y=2)),
          kinematics=N(pose_with_covariance=N(pose=N(position=N(x=1,y=2),orientation=N(w=1,x=0,y=0,z=0))),
                       twist_with_covariance=N(twist=N(linear=N(x=3,y=4)))))
    decoded,=_objects(N(objects=[obj]))
    assert decoded.track_id=='01'+'00'*15
    assert decoded.length_m==4 and decoded.width_m==2 and decoded.speed==5
    obj.object_id.uuid=[0]*16
    decoded,=_objects(N(objects=[obj]))
    assert decoded.track_id is None



def test_map_occupancy_checks_actual_window_boundaries(lanemap,lane_change_session):
    from dataclasses import replace
    from jevsceneminer.lanes import LaneMap,Projection
    from jevsceneminer.script import render_sample
    lanes=dict(lanemap.lanes);lanes[12]=replace(lanes[12],turn_direction='right')
    mapped=LaneMap(lanes,lanemap._find_within)
    tl=build_timeline(lane_change_session,mapped)
    tl.lanes=[Projection(12 if k in (50,51) else 10,0,0,0) for k in range(len(tl.lanes))]
    sample=render_sample(tl,mapped,T0+5*10**9,past_s=.06,future_s=.16)
    route=sample['script'].split('ROUTE CONTEXT',1)[1].split('OBJECTS AT NOW',1)[0]
    assert 'already inside at window start' not in route
    assert 'still inside at window end' not in route
    span,=sample['facts']['intersections']
    assert span['entry_observed'] is True
    assert span['first_outside_s']==.16
