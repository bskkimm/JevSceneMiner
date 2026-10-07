import copy
import numpy as np
from shapely.geometry import box
from jevsceneminer.moving_history import augment_sample, derive_observations


def sample():
    return dict(t_ns=10000000000,script='Header\n\nOBJECTS AT NOW (up to 8 within 30 m)\ncar: 17 m ahead, 2 m right, other lane, 26 km/h\nO8 car: 20 m behind, 9 m right, other lane, stopped\n\nOBJECT INTERACTION CONTEXT\nGeometry supports interaction evidence, not a causal claim of avoidance.\ntrack O1 (car): body clearance 3 m\ntrack O8 (car): body clearance 10 m\n',facts=dict(object_selection=dict(qualifying_tracks=20,retained_tracks=8,omitted_tracks=12,policy='old'),object_tracks=[dict(alias=f'O{i}',track_id=str(i),observations=[]) for i in range(1,9)]))


def track():
    observations=[dict(time_s=t,ahead_m=17,left_m=l,speed_m_s=7.2,ego_body_clearance_m=11,
                       connector_fraction=p,relative_yaw_deg=-81) for t,l,p in [(-1,3,0),(-.55,1,.03),(0,-2,.8),(.15,-3,.97),(.8,-6,.03),(1,-8,0),(2,-14,0)]]
    return dict(track_id='MOVING',kind='car',connector_alias='A',connector_id=123,
                observations=observations,source='native nuPlan lidar boxes and interpolated ego poses')


def test_moving_intruder_replaces_low_priority_history_and_links_now_identity():
    old=sample();before=copy.deepcopy(old);new=augment_sample(old,track())
    assert len(new['facts']['object_tracks'])==8
    assert new['facts']['object_tracks'][-1]['track_id']=='MOVING'
    assert 'O8 car: 17 m ahead, 2 m right' in new['script']
    assert 'O8 car: 20 m behind' not in new['script']
    assert 'mapped connector A footprint overlap 80%' in new['script']
    assert 'track O1 (car): body clearance 3 m' in new['script']
    assert old==before


def test_brief_intrusion_retains_event_anchors_and_actor_area_shares():
    new=augment_sample(sample(),track());text=new['script']
    assert '-0.55' in text and '+0.15' in text and '+0.80' in text and '+1.00' in text
    assert 'first observed overlap' in text and 'maximum overlap' in text and 'first observed clear' in text
    assert '97' in text and 'SHARE' in text
    assert 'Geometry supports interaction evidence, not a causal claim of avoidance.' in text


def test_native_actor_footprint_shares_and_body_gap_use_interpolated_ego_pose():
    actors=np.array([[0,10,4,0,7,0,4,2],[500000,10,0,0,7,0,4,2],[1000000,10,4,0,7,0,4,2]])
    ego=np.array([[t,0,0,0] for t in range(0,1000001,100000)])
    obs=derive_observations(actors,ego,box(-50,-2,50,2),500000000)
    assert [p['connector_fraction'] for p in obs]==[0,1,0]
    assert [p['time_s'] for p in obs]==[-.5,0,.5]
    assert abs(obs[1]['ego_body_clearance_m']-3.951)<1e-6
    assert obs[1]['ahead_m']==10 and obs[1]['left_m']==0


def test_missing_ego_support_is_not_interpolated_into_an_obstacle_interaction():
    actors=np.array([[100000,10,0,0,7,0,4,2],[500000,10,0,0,7,0,4,2]])
    ego=np.array([[0,0,0,0],[1000000,0,0,0]])
    assert derive_observations(actors,ego,box(-50,-2,50,2),500000000)==[]


def test_replacement_removes_the_old_track_summary_as_well_as_its_header():
    old=sample();old['script']+='track O8: minimum body clearance 10 m.\n'
    new=augment_sample(old,track())
    assert 'track O8: minimum body clearance 10 m.' not in new['script']
    assert 'track O1 (car): body clearance 3 m' in new['script']


def test_already_retained_moving_actor_updates_its_history_without_replacing_another_track():
    old=sample();old['facts']['object_tracks'][0]['track_id']='MOVING'
    new=augment_sample(old,track())
    assert new['facts']['object_tracks'][1:]==old['facts']['object_tracks'][1:]
    assert 'track O8 (car): body clearance 10 m' in new['script']
    assert 'track O1 (car): source ID MOVING' in new['script']


def test_existing_actor_snapshot_is_linked_by_source_alias_despite_older_coordinates():
    old=sample();old['facts']['object_tracks'][0]['track_id']='MOVING'
    old['script']=old['script'].replace('car: 17 m ahead, 2 m right, other lane, 26 km/h','O1 car: 14 m ahead, 12 m right, other lane, 31 km/h')
    new=augment_sample(old,track())
    assert 'O1 car: 14 m ahead, 12 m right' in new['script']
    assert 'car: 17 m ahead, 2 m right' not in new['script']
    assert 'O8 car: 20 m behind, 9 m right' in new['script']
    assert 'retained moving actor' in new['facts']['object_selection']['policy']


def test_an_old_source_observation_is_not_labeled_as_now():
    t=track();t['observations']=t['observations'][:1];t['observations'][0]['time_s']=-5
    new=augment_sample(sample(),t)
    assert not any('NOW' in p['anchor_reasons'] for p in new['facts']['object_tracks'][-1]['key_observations'])
