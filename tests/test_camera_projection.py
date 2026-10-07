import numpy as np
from jevsceneminer.camera_projection import project_path

CAM = dict(translation=[0,0,1.5], rotation=[.5,-.5,.5,-.5], intrinsic=[[100,0,500],[0,100,500],[0,0,1]], distortion=[0]*5, width=1000,height=1000)
POSE = np.array([0,0,0,1,0,0,0.])

def test_vehicle_width_and_perspective():
    poses = np.array([[1,10,0,0,1,0,0,0],[2,20,0,0,1,0,0,0]])
    out = project_path(0,POSE,poses,CAM,width_m=2)
    assert np.allclose(out['sections'][0]['left'],[490,515])
    assert np.allclose(out['sections'][0]['right'],[510,515])
    assert np.allclose(out['sections'][1]['left'],[495,507.5])
    assert len(out['markers']) == 2

def test_future_horizon_and_behind_camera():
    poses = np.array([[1,-10,0,0,1,0,0,0],[2,20,0,0,1,0,0,0],[11,110,0,0,1,0,0,0]])
    out = project_path(0,POSE,poses,CAM,width_m=2)
    assert len(out['sections']) == 1
    assert out['sections'][0]['time_s'] == 2
    assert out['horizon_s'] == 2

def test_stops_at_missing_pose_gap():
    poses = np.array([[.1,1,0,0,1,0,0,0],[.2,2,0,0,1,0,0,0],[2,20,0,0,1,0,0,0]])
    out = project_path(0,POSE,poses,CAM,max_gap_s=.5)
    assert out['horizon_s'] == .2

def test_frame_uses_linked_capture_pose_and_stops_at_session_end(tmp_path):
    import pickle,sqlite3
    from jevsceneminer.camera_projection import CameraPath
    source=tmp_path/'log.db'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE camera (token BLOB,channel TEXT,translation BLOB,rotation BLOB,intrinsic BLOB,distortion BLOB,width INT,height INT)')
        db.execute('CREATE TABLE ego_pose (token BLOB,timestamp INT,x REAL,y REAL,z REAL,qw REAL,qx REAL,qy REAL,qz REAL)')
        db.execute('CREATE TABLE image (filename_jpg TEXT,timestamp INT,camera_token BLOB,ego_pose_token BLOB)')
        db.execute('INSERT INTO camera VALUES (?,?,?,?,?,?,?,?)',(b'c','CAM_F0',*[pickle.dumps(CAM[k]) for k in ('translation','rotation','intrinsic','distortion')],1000,1000))
        db.executemany('INSERT INTO ego_pose VALUES (?,?,?,?,?,?,?,?,?)',[(b'capture',0,5,0,0,1,0,0,0),(b'a',1100000,15,0,0,1,0,0,0),(b'b',1200000,25,0,0,1,0,0,0),(b'outside',1300000,35,0,0,1,0,0,0)])
        db.execute('INSERT INTO image VALUES (?,?,?,?)',('front.jpg',1000000,b'c',b'capture'))
    path=CameraPath(dict(sources=[str(source)],end_ns=1200000000))
    out=path.frame(1000000000,'front.jpg')
    assert out['horizon_s']==.2
    assert len(out['sections'])==2
    assert np.allclose(out['sections'][0]['left'],[494.258,515])
    assert out['timestamp_ns']=='1000000000'


def test_lens_extrapolation_cannot_flip_offscreen_path_into_sky():
    camera = {**CAM, 'distortion': [-.356123,.172545,-.00213,.000464,-.05231]}
    # This ray is in front of the camera, but beyond the valid lens domain:
    # the extrapolated radial scale becomes negative and mirrors it skyward.
    poses = np.array([[.1,1.05,1.2,0,1,0,0,0], [.2,10,0,0,1,0,0,0]])
    out = project_path(0,POSE,poses,camera,width_m=.1)
    assert [s['time_s'] for s in out['sections']] == [.2]
    assert out['sections'][0]['connect'] is False


def test_positive_lens_scale_with_folded_radius_is_also_rejected():
    camera = {**CAM, 'distortion': [-.356123,.172545,-.00213,.000464,-.05231]}
    # Here the radial scale remains positive, but increasing ray angle maps
    # backwards toward the image center. Keeping it creates a doubled branch.
    poses = np.array([[.1,1.05,0,0,1,0,0,0], [.2,10,0,0,1,0,0,0]])
    out = project_path(0,POSE,poses,camera,width_m=.1)
    assert [s['time_s'] for s in out['sections']] == [.2]
    assert out['sections'][0]['connect'] is False


def test_tangential_lens_correction_cannot_fold_an_edge_ray():
    camera = {**CAM, 'distortion': [-.356123,.172545,-.00213,.000464,-.05231]}
    # The radial slope is still positive here; the small tangential terms
    # nevertheless fold the image mapping at the edge of the usable lens.
    poses = np.array([[.1,1.19,0,0,1,0,0,0], [.2,10,0,0,1,0,0,0]])
    out = project_path(0,POSE,poses,camera,width_m=.001)
    assert [s['time_s'] for s in out['sections']] == [.2]


def test_camera_and_bev_share_world_sections_including_near_camera_path():
    poses=np.array([[.1,.5,0,0,1,0,0,0],[1,10,2,0,1,0,0,0],[2,20,4,0,1,0,0,0]])
    out=project_path(0,POSE,poses,CAM,width_m=2)
    # The first world sample is retained even when camera clipping omits it.
    assert len(out['world_sections'])==3
    assert out['world_sections'][0]['time_s']==.1
    section=out['world_sections'][1]
    assert np.allclose(section['left'],[10,3])
    assert np.allclose(section['right'],[10,1])
    assert out['world_pose']==[0.,0.,0.]
    assert out['world_sections'][0]['connect'] is False


def test_world_sections_stop_at_the_same_pose_gap_as_camera_path():
    poses=np.array([[.1,1,0,0,1,0,0,0],[2,20,0,0,1,0,0,0]])
    out=project_path(0,POSE,poses,CAM,max_gap_s=.5)
    assert [s['time_s'] for s in out['world_sections']]==[.1]


def test_ground_offset_lowers_only_display_height_and_preserves_road_grade():
    pose=np.array([0,0,2,1,0,0,0.])
    poses=np.array([[1,10,0,2.3,1,0,0,0],[2,20,0,2.6,1,0,0,0]])
    original=poses.copy()
    native=project_path(0,pose,poses,CAM,width_m=2)
    lowered=project_path(0,pose,poses,CAM,width_m=2,ground_offset_m=.24)
    assert np.allclose(lowered['sections'][0]['left'],[490,514.4])
    assert np.allclose(lowered['sections'][1]['left'],[495,505.7])
    assert lowered['world_sections']==native['world_sections']
    assert lowered['world_pose']==native['world_pose']
    assert np.array_equal(poses,original)
