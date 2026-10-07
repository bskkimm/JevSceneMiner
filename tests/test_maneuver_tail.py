from copy import deepcopy
from jevsceneminer.jev import Answer
from jevsceneminer.scenes import Step, stitch, extend_maneuver_ends


def scenes(labels, longitudinal=None, times=None):
    times=times or list(range(len(labels)))
    longitudinal=longitudinal or ['cruising']*len(labels)
    steps=[Step(int((10+t)*1e9),Answer(label,{label:1.0},lon,{lon:1.0}))
           for t,label,lon in zip(times,labels,longitudinal)]
    return stitch(steps,1,0,10_000_000_000,30_000_000_000),steps


def test_tail_borrows_following_speed_phases_and_preserves_raw_answers():
    original,steps=scenes(['turn_right']*2+['keep_lane']*5,
                         ['decelerating']*2+['accelerating','stopped','stopped','cruising','cruising'])
    frozen=deepcopy(steps)
    result=extend_maneuver_ends(original,2)
    assert [(s.lateral,s.start_ns,s.end_ns) for s in result]==[
        ('turn_right',10_000_000_000,13_500_000_000),('keep_lane',13_500_000_000,16_500_000_000)]
    assert [(p.decision,p.end_ns) for p in result[0].longitudinal_phases]==[
        ('decelerating',11_500_000_000),('accelerating',12_500_000_000),('stopped',13_500_000_000)]
    assert [(p.decision,p.end_ns) for p in result[1].longitudinal_phases]==[
        ('stopped',14_500_000_000),('cruising',16_500_000_000)]
    assert steps==frozen


def test_tail_stops_at_next_maneuver_and_does_not_bridge_a_gap():
    original,_=scenes(['turn_right']*2+['keep_lane']+['lane_change_left']*2)
    result=extend_maneuver_ends(original,2)
    assert [(s.lateral,s.start_ns,s.end_ns) for s in result]==[
        ('turn_right',10_000_000_000,12_500_000_000),('lane_change_left',12_500_000_000,14_500_000_000)]
    gap,_=scenes(['turn_right']*2+['keep_lane']*2,times=[0,1,4,5])
    assert extend_maneuver_ends(gap,2)==gap


def test_tail_stops_at_review_boundary_and_never_enters_protected_region():
    original,_=scenes(['turn_right']*2+['keep_lane']*5)
    result=extend_maneuver_ends(original,2,protected_intervals=[(12_000_000_000,15_000_000_000)])
    assert result[0].end_ns==12_000_000_000
    assert result[1].start_ns==12_000_000_000
    assert extend_maneuver_ends(original,2,protected_intervals=[(11_000_000_000,15_000_000_000)])==original


def test_tail_coalesces_touching_equal_maneuvers_and_clips_log_end():
    original,_=scenes(['turn_right']*2+['keep_lane']+['turn_right']*2+['keep_lane'])
    result=extend_maneuver_ends(original,2)
    assert len(result)==1 and result[0].end_ns==15_500_000_000
    assert result[0].longitudinal_phases[-1].end_ns==result[0].end_ns


def test_zero_tail_preserves_previous_stitching():
    original,_=scenes(['turn_right']*2+['keep_lane']*5)
    assert extend_maneuver_ends(original,0)==original


def test_unknown_lateral_does_not_claim_a_maneuver_tail():
    original,_=scenes(['unknown']*2+['keep_lane']*3)
    assert extend_maneuver_ends(original,2)==original
