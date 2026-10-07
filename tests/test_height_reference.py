import numpy as np
from jevsceneminer.height_reference import HeightReference
from jevsceneminer.camera_projection import project_path
from test_camera_projection import CAM, POSE


def landmarks(outlier=False):
    tracks={}
    for i,(x,y) in enumerate([(-10,-10),(-10,10),(10,-10),(10,10),(0,15),(20,0)]):
        rows=[]
        for t in np.arange(0,2.01,.1):
            drift=t*(-.02*x+.01*y-.4)
            if outlier and i==5:drift+=2*t
            rows.append([t,x,y,1+drift,4,2,2])
        tracks[i]=np.array(rows)
    return tracks


def test_static_landmarks_remove_temporal_height_tilt_preserving_true_grade():
    reference=HeightReference(landmarks())
    poses=np.array([[1,10,0,-.8,1,0,0,0],[2,20,0,-2.,1,0,0,0]])
    drift,info=reference.offsets(0,POSE,poses)
    assert info['available']
    assert np.allclose(drift,[-.6,-1.6],atol=.02)
    out=project_path(0,POSE,poses,CAM,width_m=2,ground_offset_m=.24,height_drift_m=drift)
    assert np.allclose(out['sections'][0]['left'],[490,519.4],atol=.2)
    assert np.allclose(out['sections'][1]['left'],[495,510.7],atol=.2)
    assert out['world_sections'][1]['left']==[20,1]
    assert poses[1,3]==-2.


def test_height_reference_rejects_an_inconsistent_object_height():
    reference=HeightReference(landmarks(outlier=True))
    poses=np.array([[2,20,0,-2.,1,0,0,0]])
    drift,info=reference.offsets(0,POSE,poses)
    assert info['available'] and np.allclose(drift,[-1.6],atol=.03)


def test_neighbor_frames_bridge_landmarks_entering_and_leaving_view():
    tracks=landmarks()
    # The initial landmarks disappear halfway through, while another set
    # is already visible. Their common observations transfer the reference.
    later={i+10:rows[rows[:,0]>=.5].copy() for i,rows in tracks.items()}
    tracks={i:rows[rows[:,0]<=1].copy() for i,rows in tracks.items()}
    tracks.update(later)
    poses=np.array([[t,t*10,0,-t*.2+t*(-.02*t*10-.4),1,0,0,0]
                    for t in np.arange(.1,2.01,.1)])
    drift,info=HeightReference(tracks).offsets(0,POSE,poses)
    assert info['available'] and info['supported_fraction']==1
    assert info['bridged_grid_nodes']>0
    assert np.isclose(drift[-1],-1.6,atol=.03)


def test_unconnected_landmark_sets_cannot_transfer_height_reference():
    tracks=landmarks()
    later={i+10:rows[rows[:,0]>=1.5].copy() for i,rows in tracks.items()}
    tracks={i:rows[rows[:,0]<=.5].copy() for i,rows in tracks.items()}
    tracks.update(later)
    poses=np.array([[t,t*10,0,0,1,0,0,0] for t in np.arange(.1,2.01,.1)])
    drift,info=HeightReference(tracks).offsets(0,POSE,poses)
    assert info['available']
    assert np.isnan(drift[poses[:,0]>=1]).all()


def test_clustered_landmarks_cannot_support_far_away_height_extrapolation():
    tracks={}
    for i,(x,y) in enumerate([(-2,-2),(-2,2),(2,-2),(2,2)]):
        tracks[i]=np.array([[t,x,y,1+t*.002*x,4,2,2]
                            for t in np.arange(0,10.01,.1)])
    poses=np.array([[t,t*10,0,0,1,0,0,0] for t in np.arange(.1,10.01,.1)])
    drift,info=HeightReference(tracks).offsets(0,POSE,poses)
    assert info['available']
    assert np.isfinite(drift[0])
    # Just4cm differential height noise at the landmarks would otherwise
    # invent a2m correction100m away. The prediction lacks spatial support.
    assert np.isnan(drift[-1])
    assert info['unsupported_uncertainty_fraction']>0


def test_height_uncertainty_accumulates_when_reference_is_bridged():
    tracks={i:np.array([[t,x,y,1-.1*t,4,2,2] for t in np.arange(0,2.01,.1)])
            for i,(x,y) in enumerate([(-2,-2),(-2,2),(2,-2),(2,2)])}
    later={i+10:rows[rows[:,0]>=.5].copy() for i,rows in tracks.items()}
    early={i:rows[rows[:,0]<=1].copy() for i,rows in tracks.items()}
    poses=np.array([[1.5,15,0,0,1,0,0,0],[2,15,0,0,1,0,0,0]])
    direct,unused=HeightReference(tracks).offsets(0,POSE,poses)
    bridged,info=HeightReference({**early,**later}).offsets(0,POSE,poses)
    assert np.isfinite(direct).all()
    assert np.isnan(bridged[-1]) and info['bridged_grid_nodes']>0


def test_missing_or_collinear_landmarks_do_not_invent_a_plane():
    poses=np.array([[1,10,0,0,1,0,0,0]])
    for tracks in [{},{k:v for k,v in landmarks().items() if k<2},
                   {i:np.array([[0,i*5,0,1,4,2,2],[1,i*5,0,.5,4,2,2]]) for i in range(6)}]:
        drift,info=HeightReference(tracks).offsets(0,POSE,poses)
        assert not info['available'] and drift is None


def test_unobserved_gap_breaks_camera_ribbon_without_removing_bev_path():
    poses=np.array([[.1,10,0,0,1,0,0,0],[.2,11,0,0,1,0,0,0],[.3,12,0,0,1,0,0,0]])
    out=project_path(0,POSE,poses,CAM,height_drift_m=np.array([0,np.nan,0]))
    assert len(out['world_sections'])==3
    assert len(out['sections'])==2 and not out['sections'][1]['connect']
