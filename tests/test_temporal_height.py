import numpy as np
from jevsceneminer.temporal_height import TemporalHeightReference
from jevsceneminer.camera_projection import project_path
from test_camera_projection import CAM,POSE


def actors(end=6,noise=False,dropout=False):
    tracks={}
    for i,(x,y) in enumerate([(-10,-10),(-10,10),(10,-10),(10,10),(0,15),(20,0)]):
        rows=[]
        for t in np.arange(0,end+.01,.05):
            if dropout and 2<t<4:continue
            z=1+t*(-.02*x+.01*y-.4)
            if noise:z+=.04*np.sin(28*t+i)
            rows.append([t,x,y,z,4,2,2])
        tracks[i]=np.array(rows)
    return tracks


def test_shared_reference_preserves_true_grade_and_source_xy():
    ref=TemporalHeightReference(actors(end=2),origin=[0,0])
    poses=np.array([[1,10,0,-.8,1,0,0,0],[2,20,0,-2.,1,0,0,0]])
    drift,info=ref.offsets(0,POSE,poses)
    assert info['available'] and np.allclose(drift,[-.6,-1.6],atol=.03)
    projected=project_path(0,POSE,poses,CAM,width_m=2,ground_offset_m=.24,height_drift_m=drift)
    assert np.allclose(projected['sections'][1]['left'],[495,510.7],atol=.2)
    assert projected['world_sections'][1]['left']==[20,1]
    assert poses[1,3]==-2.


def test_observation_dropout_does_not_cut_or_switch_the_ribbon():
    ref=TemporalHeightReference(actors(dropout=True),origin=[0,0])
    poses=np.array([[t,10*t,0,0,1,0,0,0] for t in np.arange(.1,6.01,.1)])
    drift,info=ref.offsets(0,POSE,poses)
    assert np.isfinite(drift).all() and info['approximate_fraction']>0
    out=project_path(0,POSE,poses,CAM,height_drift_m=drift,max_gap_s=.5)
    assert len(out['sections'])>40
    assert all(section['connect'] for section in out['sections'][1:])


def test_noisy_objects_use_one_smooth_reference_for_adjacent_images():
    ref=TemporalHeightReference(actors(noise=True),origin=[0,0])
    heights=[]
    target=np.array([[5,20,0,-2,1,0,0,0]])
    for t in np.arange(0,1.01,.05):
        drift,info=ref.offsets(t,POSE,target)
        heights.append(float(drift[0]))
    # The actual datum changes .8m/s at this fixed point, so .04m per
    # image is real motion. Test the added noise after removing that trend.
    assert max(abs(np.diff(heights)-.04))<.005
    # Seeking in either direction must use the same field, without state lag.
    a=ref.offsets(.35,POSE,target)[0]
    ref.offsets(.8,POSE,target)
    assert np.array_equal(a,ref.offsets(.35,POSE,target)[0])


def test_weak_spatial_support_blends_to_translation_instead_of_inventing_tilt():
    tracks={i:np.array([[t,x,y,1+t*.002*x,4,2,2] for t in np.arange(0,10.01,.05)])
            for i,(x,y) in enumerate([(-2,-2),(-2,2),(2,-2),(2,2)])}
    poses=np.array([[t,10*t,0,0,1,0,0,0] for t in np.arange(.1,10.01,.1)])
    drift,info=TemporalHeightReference(tracks,origin=[0,0]).offsets(0,POSE,poses)
    assert np.isfinite(drift).all()
    assert abs(drift[-1])<.1 and info['approximate_fraction']>0


def test_source_pose_gaps_still_break_projection_with_a_shared_height_reference():
    ref=TemporalHeightReference(actors(),origin=[0,0])
    poses=np.array([[.1,10,0,0,1,0,0,0],[.2,11,0,0,1,0,0,0],[2,20,0,0,1,0,0,0]])
    drift,info=ref.offsets(0,POSE,poses)
    out=project_path(0,POSE,poses,CAM,height_drift_m=drift,max_gap_s=.5)
    assert out['horizon_s']==.2


def test_no_landmarks_keep_legacy_projection_without_creating_a_model():
    poses=np.array([[1,10,0,0,1,0,0,0]])
    drift,info=TemporalHeightReference({},origin=[0,0]).offsets(0,POSE,poses)
    assert drift is None and not info['available']


def test_identical_fractional_times_have_zero_height_difference_uncertainty():
    tracks={i:np.array([[t,x,y,1,4,2,2] for t in np.arange(0,2.01,.05)])
            for i,(x,y) in enumerate([(-2,-2),(-2,2),(2,-2),(2,2)])}
    ref=TemporalHeightReference(tracks,origin=[0,0])
    drift,info=ref.offsets(.25,POSE,np.array([[.25,100,0,0,1,0,0,0]]))
    assert drift[0]==0 and info['max_prediction_std_m']<1e-6


def test_approximation_reports_missing_observations_anywhere_in_the_interval():
    tracks=actors(end=3)
    tracks={k:rows[(rows[:,0]<=.6)|(rows[:,0]>=1.4)] for k,rows in tracks.items()}
    ref=TemporalHeightReference(tracks,origin=[0,0])
    drift,info=ref.offsets(0,POSE,np.array([[2,20,0,0,1,0,0,0]]))
    assert np.isfinite(drift[0])
    assert info['approximate_fraction']==1 and info['supported_fraction']==0
