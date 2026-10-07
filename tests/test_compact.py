import copy
from jevsceneminer.compact import reviewed_state, effective_facts, object_share, render_compact, _fork_section, road_shares

REVIEW = {'start_ns': 1_000_000_000, 'from_s': 613.9, 'to_s': 631.8,
          'margin_s': 2.0, 'chain_id': 'R1', 'source': 'manual camera review / user-reviewed'}

def fixture():
    return {'ego_samples': [{'time_s': 10., 'lane': 'A', 'matched_lane_id': 99,
        'offset_m': -1.5, 'speed_m_s': 5., 'acceleration_m_s2': .2,
        'yaw_rate_rad_s': 0., 'relative_heading_deg': 0., 'light': 'green',
        'road_context': 'road', 'intersection': False, 'body_occupancy': {
        'time_s':10., 'reference_lane_fraction': .4, 'lane_occupancy': [{'lane_id': 99, 'fraction': .4}]}}],
        'lane_evidence': {'observations': [], 'reference_lane_ids': [99]},
        'object_tracks': [{'alias':'O1','observations':[{'time_s':10., 'reference_lane_overlap_m2':2.,
        'matched_lane_overlap_m2':2., 'lane_corridor_overlap':True}], 'key_observations':[]}],
        'lanes':[{'alias':'A','id':99}], 'local_speed':{}, 'window':{}, 'intersections':[]}

def test_review_uses_observation_absolute_time_not_only_now():
    native=fixture(); before=copy.deepcopy(native)
    # NOW is before reviewed interval; its +10s row is within it.
    result=effective_facts(native, 605_000_000_000, REVIEW)
    ego=result['ego_samples'][0]
    assert ego['lane']=='R1' and ego['physical_lane_ownership_pct']==100
    assert ego['offset_m'] is None and ego['matched_lane_id'] is None
    assert ego['speed_m_s']==5.
    assert result['object_tracks'][0]['observations'][0]['reference_lane_overlap_m2'] is None
    assert native==before

def test_margin_does_not_invent_full_lane_ownership():
    assert reviewed_state(613_000_000_000, 0, REVIEW)=='margin'
    assert reviewed_state(615_000_000_000, 0, REVIEW)=='core'
    assert reviewed_state(635_000_000_000, 0, REVIEW) is None
    r=effective_facts(fixture(), 603_000_000_000, REVIEW)['ego_samples'][0]
    assert r['lane'] is None and r['physical_lane_ownership_pct'] is None

def test_object_footprint_share_is_object_area_based_and_unknown_without_geometry():
    assert object_share({'length_m':4.,'width_m':2.,'reference_lane_overlap_m2':2.})==25.
    assert object_share({'length_m':4.,'width_m':2.,'reference_lane_overlap_m2':None}) is None
    assert object_share({'reference_lane_overlap_m2':0.}) is None

def test_compact_input_hides_reviewed_native_transitions_but_keeps_motion():
    native=fixture()
    native['ego_samples'][0]['ego_data_available']=True
    result=effective_facts(native,605_000_000_000,REVIEW)
    meta={'start_ns':1_000_000_000,'past_s':10,'future_s':15,'table_step_s':1.,'has_indicator':False}
    script=render_compact(result,605_000_000_000,meta,{99:{'following':[], 'left':None,'right':None}},
                         [{'time_s':10.,'text':'moves from A into its right neighbor B'},
                          {'time_s':11.,'text':'comes to a stop'}])
    assert '99' not in script and 'right neighbor' not in script
    assert '+10s R1 road unknown +18.0 +0.20' in script
    assert 'comes to a stop' in script and 'ownership 100%' in script

def test_compact_input_preserves_partial_native_body_occupancy_and_body_gap():
    native=fixture()
    native['ego_samples'][0]['body_occupancy']['reference_lane_fraction']=.4
    native['object_tracks'][0]['observations'][0].update(length_m=4.,width_m=2.,
        ahead_m=5.,left_m=1.,speed_m_s=0.,ego_body_clearance_m=.5)
    meta={'start_ns':1_000_000_000,'past_s':10,'future_s':15,'table_step_s':1.,'has_indicator':False}
    script=render_compact(native,605_000_000_000,meta,{99:{'following':[],'left':None,'right':None}},[])
    assert '+40.0 unknown' in script  # native reference fraction never becomes physical ownership
    assert '+0.50 +25.0' in script  # body clearance and OBJECT footprint percentage

def test_review_endpoints_are_inclusive_and_neighboring_future_is_masked():
    assert reviewed_state(614_900_000_000,0,REVIEW)=='core'
    assert reviewed_state(632_800_000_000,0,REVIEW)=='core'
    assert reviewed_state(632_800_000_001,0,REVIEW)=='margin'

def test_all_native_transition_forms_are_hidden_inside_review():
    facts=effective_facts(fixture(),605_000_000_000,REVIEW)
    meta={'start_ns':1_000_000_000,'past_s':10,'future_s':15,'table_step_s':1.,'has_indicator':False}
    events=[{'time_s':10.,'text':text} for text in ('enters mapped lane A',
        'leaves the mapped lanes (from A)', 'moves back from A into B')]
    script=render_compact(facts,605_000_000_000,meta,{},events)
    assert facts['events']==[] and '99' not in script

def test_review_clips_intersection_without_erasing_valid_prior_evidence():
    native=fixture()
    native['intersections']=[{'first_inside_s':5.,'last_inside_s':10.,'first_outside_s':10.1,
                             'map_turn_direction':'LEFT','heading_change_deg':50.,'entry_observed':True}]
    native['lane_evidence']['observations']=[{'time_s':t,'ego_data_available':True,'matched_lane_id':99}
                                            for t in [5.,6.,7.,8.,9.,10.]]
    facts=effective_facts(native,605_000_000_000,REVIEW)
    # Margin begins at relative +7.9: +5,+6,+7 remain valid intersection evidence.
    span=facts['intersections'][0]
    assert (span['first_inside_s'],span['last_inside_s'])==(5.,7.)
    assert span['heading_change_deg'] is None and span['first_outside_s'] is None
    assert span['review_clipped'] is True

def test_fork_evidence_is_read_with_actual_annotated_heading():
    script='\nFORK CONTEXT (map alternatives and observed path within the sample window)\nt=+2.0s: mapped non-intersection fork from A into B\n\nOTHER\nignored'
    assert _fork_section(script)=='t=+2.0s: mapped non-intersection fork from A into B'

def test_road_footprint_transfer_survives_compaction_without_connector_overlap():
    native=fixture()
    native['lanes'].append({'alias':'B','id':100})
    native['ego_samples'][0]['body_occupancy']['lane_occupancy']=[
        {'lane_id':99,'fraction':.4,'intersection':False,'kind':'road','direction_relation':'same-direction'},
        {'lane_id':100,'fraction':.6,'intersection':False,'kind':'road','direction_relation':'same-direction'},
        {'lane_id':101,'fraction':1.,'intersection':True,'kind':'road','direction_relation':'same-direction'}]
    meta={'start_ns':1_000_000_000,'past_s':10,'future_s':15,'table_step_s':1.,'has_indicator':False}
    topology={99:{'following':[],'right':100},100:{'following':[],'left':99}}
    script=render_compact(native,605_000_000_000,meta,topology,[])
    assert 'A:40.0/B:60.0' in script
    assert '101:100' not in script

def test_nonintersection_connector_is_not_presented_as_ordinary_road_share():
    body={'lane_occupancy':[{'lane_id':99,'fraction':1.,'intersection':False,
                            'kind':'road','direction_relation':'same-direction'}]}
    assert road_shares(body,{99:{'connector':True}})=={}
