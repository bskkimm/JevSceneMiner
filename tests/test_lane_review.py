from copy import deepcopy

from jevsceneminer.lane_review import apply_lane_review


def sample():
    header = ('TIME'.ljust(8)+'LANE'.ljust(9)+'ROAD CONTEXT'.ljust(26)+'OFFSET'.ljust(9)
              +'SPEED'.ljust(10)+'ACCEL'.ljust(12)+'YAW RATE'.ljust(10)
              +'RELATIVE HEADING'.ljust(18)+'INDICATOR'.ljust(11)+'LIGHT')
    ego = [dict(time_s=t,ego_data_available=True,lane='A',matched_lane_id=1,
                road_context='road',offset_m=1.2,light='green',body_occupancy={})
           for t in [-2,0,3]]
    rows = [f't={t:+g}s'.ljust(8)+'A'.ljust(9)+'road'.ljust(26)+'+1.2 m'.ljust(9)
            +'20 km/h'.ljust(10)+'+0.1 m/s²'.ljust(12)+'+1 °/s'.ljust(10)
            +'+0°'.ljust(18)+'n/a'.ljust(11)+'green'+('   <- NOW' if t==0 else '') for t in [-2,0,3]]
    script='Driving log\n\nLANES\nA: road; right neighbor: B\n\n'+header+'\n'+'\n'.join(rows)
    script+='\n\nEVENTS\nt=-0.1s: moves from A into its right neighbor B\nt=+0.0s: starts moving\nt=+3.0s: continues from B into C'
    script+='\n\nLOCAL SPEED CONTEXT\nspeed at NOW: 5.56 m/s\n\nSUMMARY\nheading change over the window: +1°\noffset while in lane A: -1 to 2 m'
    script+='\n\nFORK CONTEXT\nno mapped fork\n\nROUTE CONTEXT\nNOW inside an intersection'
    script+='\n\nEGO BODY AND LANE OCCUPANCY\nREFERENCE: A\nBODY TIME REF IN/OUT\nbody t=-2s +1.2 80/20\nbody t=+0s +1.2 80/20\nbody t=+3s +1.2 80/20\nfirst boundary departure: t=-0.1s'
    script+='\n\nOBJECTS AT NOW\nO1 car: 5 m ahead, 1 m right, same lane A, stopped\nO2 car: 8 m ahead, 3 m right, right neighbor lane, stopped\n\nOBJECT INTERACTION CONTEXT\nno tracks'
    facts=dict(ego_samples=ego,window={},lane_evidence=dict(observations=[]),object_tracks=[],intersections=[])
    return dict(t_ns=2_000_000_000,script=script,facts=facts,lateral='lane_change_right',lateral_probs={'lane_change_right':.8},longitudinal='cruising')


REVIEW=dict(start_ns=0,from_s=1,to_s=3,margin_s=.5,chain_id='R1',source='manual camera review / user-reviewed')


def test_override_changes_observed_lane_evidence_without_forcing_answer():
    original=sample(); frozen=deepcopy(original)
    result=apply_lane_review(original,REVIEW)
    assert original==frozen
    assert result['lateral']=='lane_change_right' and result['lateral_probs']==original['lateral_probs']
    assert result['facts']['ego_samples'][1]['physical_lane_chain']=='R1'
    assert result['facts']['ego_samples'][1]['physical_lane_ownership_pct']==100
    now=next(line for line in result['script'].splitlines() if '<- NOW' in line)
    assert 'R1' in now and 'A ' not in now and '+1.2 m' not in now
    assert '20 km/h' in now and '+0.1 m/s²' in now
    assert 'moves from A into its right neighbor B' not in result['script']
    assert 'starts moving' in result['script'] and 'continues from B into C' in result['script']
    assert '80/20' not in next(line for line in result['script'].splitlines() if line.startswith('body t=+0s'))
    assert 'same lane A' not in result['script']
    assert 'right neighbor lane' not in result['script']
    assert 'offset while in lane A' not in result['script']
    assert 'speed at NOW: 5.56 m/s' in result['script']


def test_context_outside_review_keeps_exact_source_rows_and_margin_is_unknown():
    original=sample(); review=dict(REVIEW,from_s=2,to_s=3,margin_s=1)
    original['facts']['ego_samples'][1]['time_s']=-.5
    original['script']=original['script'].replace('t=+0s','t=-0.5s')
    result=apply_lane_review(original,review)
    assert result['facts']['ego_samples'][1]['physical_lane_chain'] is None
    line=next(line for line in result['script'].splitlines() if '<- NOW' in line)
    assert 'unknown' in line and 'R1' not in line
    old=next(line for line in original['script'].splitlines() if line.startswith('t=+3s'))
    assert old in result['script']


def test_unaffected_sample_is_byte_equivalent():
    original=sample();original['t_ns']=100_000_000_000
    assert apply_lane_review(original,REVIEW)==original


def test_reviewed_road_drops_native_now_intersection_summary():
    original=sample()
    original['script']=original['script'].replace('SUMMARY\n','SUMMARY\nNOW is inside intersection lane A\n')
    summary=apply_lane_review(original,REVIEW)['script'].split('SUMMARY\n')[1].split('\n\n')[0]
    assert 'NOW is inside intersection lane' not in summary
