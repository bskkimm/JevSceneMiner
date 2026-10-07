"""Project the recorded next ten seconds onto the calibrated nuPlan front camera."""
from __future__ import annotations

import io
import pickle
import sqlite3
from pathlib import Path

import numpy as np

# nuPlan's Pacifica bounding-box width (vehicle_parameters.get_pacifica_parameters).
VEHICLE_WIDTH_M = 2.297
RIBBON_WIDTH_M = VEHICLE_WIDTH_M / 2


class _CalibrationUnpickler(pickle.Unpickler):
    """Decode only nuPlan's list wrappers and numpy numeric calibration scalars."""
    def find_class(self, module, name):
        if module == 'nuplan.database.common.data_types' and name in {'Translation', 'Rotation', 'CameraIntrinsic'}:
            return list
        if module == 'numpy' and name == 'dtype':
            return np.dtype
        if module in {'numpy.core.multiarray', 'numpy._core.multiarray'} and name == 'scalar':
            return np.core.multiarray.scalar
        raise pickle.UnpicklingError(f'unsupported calibration type: {module}.{name}')


def _decode(blob):
    return _CalibrationUnpickler(io.BytesIO(blob)).load()


def rotation(q):
    """Unit quaternion (w,x,y,z) to local-to-world rotation."""
    w,x,y,z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def project_path(time_s, pose, poses, camera, width_m=RIBBON_WIDTH_M, max_gap_s=1.5,
                 ground_offset_m=0.0, height_drift_m=None):
    """Return image-space cross sections; poses contain [time_s,x,y,z,qw,qx,qy,qz].

    The ribbon follows the recorded ego reference positions, including their
    orientation and road grade. An explicit display-only ground offset lowers
    the reference height without changing the logged poses or BEV XY positions.
    It is an approximate height adjustment, not a measured road surface at every
    future point. Sections too close to/behind the camera are omitted.
    A break flag prevents joining across an invisible portion of the path.
    """
    pose = np.asarray(pose, dtype=float)
    if not np.isfinite(ground_offset_m) or ground_offset_m < 0:
        raise ValueError('ground offset must be finite and nonnegative')
    transform = rotation(camera['rotation']).T @ rotation(pose[3:]).T
    offset = np.asarray(camera['translation'], dtype=float)
    intrinsic = np.asarray(camera['intrinsic'], dtype=float)
    distortion = np.zeros(5)
    values = camera.get('distortion', [])
    distortion[:min(5,len(values))] = values[:5]
    k1,k2,p1,p2,k3=distortion
    # r_distorted = r * (1 + k1*r² + k2*r⁴ + k3*r⁶). Beyond its first
    # stationary point this calibration folds off-camera rays back into view.
    # Keep only the increasing branch connected to the optical axis.
    roots=np.roots([7*k3,5*k2,3*k1,1])
    radius_sq_limit=min((float(r.real) for r in roots
                         if abs(r.imag)<1e-9 and r.real>0),default=float('inf'))
    def pixel(world):
        p = transform @ (world-pose[:3]) - rotation(camera['rotation']).T @ offset
        if p[2] < 1: return None
        x,y = p[:2]/p[2]
        r2=x*x+y*y
        if r2 >= radius_sq_limit: return None
        radial=1+k1*r2+k2*r2*r2+k3*r2*r2*r2
        slope=k1+2*k2*r2+3*k3*r2*r2
        # The full lens mapping must also stay locally increasing. Its
        # symmetric Jacobian includes the tangential terms near the rim.
        jxx=radial+2*x*x*slope+2*p1*y+6*p2*x
        jyy=radial+2*y*y*slope+6*p1*y+2*p2*x
        jxy=2*x*y*slope+2*p1*x+2*p2*y
        if jxx<=0 or jxx*jyy-jxy*jxy<=0: return None
        xy=np.array([x*radial+2*p1*x*y+p2*(r2+2*x*x),y*radial+p1*(r2+2*y*y)+2*p2*x*y,1])
        result=intrinsic@xy
        return np.round(result[:2]/result[2],3).tolist()
    sections, markers, world_sections = [], [], []
    previous, horizon, last_sample = time_s, 0., -1.
    connected = False
    marker_times = set()
    if height_drift_m is not None and len(height_drift_m)!=len(poses):
        raise ValueError('height drift must match the future poses')
    for index,row in enumerate(poses):
        dt=float(row[0]-time_s)
        if dt < 0: continue
        if dt > 10+1e-6: break
        if row[0]-previous > max_gap_s: break
        previous=float(row[0]); horizon=max(horizon,dt)
        if dt-last_sample < .09 and dt < 9.99: continue
        last_sample=dt
        lateral=rotation(row[4:])[:,1]*width_m/2
        center=row[1:4]
        # Keep the same sampled positions for the BEV, including points outside
        # the camera frustum. This is a second view of the recorded path.
        world_sections.append(dict(time_s=round(dt,3),
            left=np.round((center+lateral)[:2],3).tolist(),
            right=np.round((center-lateral)[:2],3).tolist(),connect=bool(world_sections)))
        drift=float(height_drift_m[index]) if height_drift_m is not None else 0.
        if not np.isfinite(drift):
            connected=False
            continue
        ground_center=center-np.array([0.,0.,ground_offset_m+drift])
        left,right=pixel(ground_center+lateral),pixel(ground_center-lateral)
        if left is None or right is None:
            connected=False
            continue
        sections.append(dict(time_s=round(dt,3),left=left,right=right,connect=connected))
        connected=True
        second=int(round(dt))
        if second>=1 and abs(dt-second)<.06 and second not in marker_times:
            point=pixel(ground_center)
            if point is not None:
                markers.append(dict(time_s=second,xy=point))
                marker_times.add(second)
    return dict(available=True,width=camera['width'],height=camera['height'],width_m=width_m,
                horizon_s=round(horizon,2),sections=sections,markers=markers,
                world_sections=world_sections,ground_offset_m=float(ground_offset_m),
                world_pose=[float(pose[0]),float(pose[1]),
                            float(np.arctan2(rotation(pose[3:])[1,0],rotation(pose[3:])[0,0]))])


class CameraPath:
    """Read calibration, image capture poses and full ego poses once per session."""
    def __init__(self, meta):
        self.images, self.cameras, all_poses = {}, {}, {}
        self.end_s = int(meta['end_ns'])/1e9
        self.ground_offset_m = float(meta.get('camera_ground_offset_m',0.0))
        self.height_reference=None
        landmark_rows={}
        align_height=meta.get('camera_height_reference')=='stationary_landmarks'
        for source in meta['sources']:
            with sqlite3.connect(f'file:{Path(source).resolve()}?mode=ro',uri=True) as db:
                db.row_factory=sqlite3.Row
                for row in db.execute("SELECT * FROM camera WHERE channel='CAM_F0'"):
                    self.cameras[row['token']]={key: _decode(row[key]) for key in ('translation','rotation','intrinsic','distortion')}
                    self.cameras[row['token']].update(width=row['width'],height=row['height'])
                for row in db.execute('SELECT timestamp,x,y,z,qw,qx,qy,qz FROM ego_pose ORDER BY timestamp'):
                    all_poses[row['timestamp']]=list(row)[1:]
                for row in db.execute("SELECT i.filename_jpg,i.timestamp,i.camera_token,e.x,e.y,e.z,e.qw,e.qx,e.qy,e.qz FROM image i JOIN camera c ON c.token=i.camera_token JOIN ego_pose e ON e.token=i.ego_pose_token WHERE c.channel='CAM_F0'"):
                    self.images[(row['timestamp']*1000,row['filename_jpg'])]=(row['camera_token'],np.array(list(row)[3:],dtype=float))
                if align_height:
                    query="""SELECT lb.track_token,lp.timestamp/1000000.0,lb.x,lb.y,lb.z,lb.length,lb.width,lb.height
                             FROM lidar_box lb JOIN lidar_pc lp ON lp.token=lb.lidar_pc_token
                             JOIN track t ON t.token=lb.track_token JOIN category c ON c.token=t.category_token
                             WHERE lb.vx*lb.vx+lb.vy*lb.vy<=.04 AND lb.height>0 AND lb.height<2.6
                             AND lb.length>0 AND lb.width>0
                             AND c.name IN ('vehicle','traffic_cone','barrier','generic_object') ORDER BY lp.timestamp"""
                    for row in db.execute(query):
                        landmark_rows.setdefault(row[0],[]).append(tuple(row)[1:])
        self.poses=np.array([[t/1e6,*p] for t,p in sorted(all_poses.items())],dtype=float)
        if align_height:
            from jevsceneminer.viewer.temporal_height import TemporalHeightReference
            self.height_reference=TemporalHeightReference(
                {key:np.asarray(sorted(rows),dtype=float) for key,rows in landmark_rows.items()},
                origin=self.poses[0,1:3],time_bounds=(self.poses[0,0],self.poses[-1,0]))

    def frame(self, timestamp_ns, filename):
        image=self.images.get((timestamp_ns,filename))
        if image is None:
            return dict(available=False,reason='No calibration or capture pose for this image')
        token,pose=image
        time_s=timestamp_ns/1e9
        times=self.poses[:,0]
        start=np.searchsorted(times,time_s)
        stop=np.searchsorted(times,min(self.end_s,time_s+10),side='right')
        future=self.poses[start:stop]
        drift,info=self.height_reference.offsets(time_s,pose,future) if self.height_reference else (None,None)
        result=project_path(time_s,pose,future,self.cameras[token],max_gap_s=.5,
                            ground_offset_m=self.ground_offset_m,height_drift_m=drift)
        if info is not None:
            result['height_reference']=info
        result['timestamp_ns']=str(timestamp_ns)
        return result
