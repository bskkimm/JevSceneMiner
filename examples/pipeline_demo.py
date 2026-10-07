"""Generate synthetic source logs, processed samples and cached illustrative labels.

No private files or API credentials are needed. Both sources share a tiny
synthetic Lanelet2 map fixture; the nuPlan production GeoPackage map reader is
tested separately. These deterministic labels demonstrate the pipeline, not
Jev accuracy. Every unexpected API request is blocked.
"""
import argparse
import functools
import json
import math
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from mcap_ros2.writer import Writer

from jevsceneminer import cli
from jevsceneminer import lanes as lane_geometry
from jevsceneminer.bag import read_session
from jevsceneminer.jev import Answer, AnswerCache, JevClassifier, cache_key, load_labels
from jevsceneminer.lanes import LaneMap
from jevsceneminer.nuplan import read_nuplan

T0 = 1_767_000_000_000_000_000
ODOMETRY = '''std_msgs/Header header
string child_frame_id
geometry_msgs/PoseWithCovariance pose
geometry_msgs/TwistWithCovariance twist
================================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
string frame_id
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
================================================================================
MSG: geometry_msgs/PoseWithCovariance
geometry_msgs/Pose pose
float64[36] covariance
================================================================================
MSG: geometry_msgs/Pose
geometry_msgs/Point position
geometry_msgs/Quaternion orientation
================================================================================
MSG: geometry_msgs/Point
float64 x
float64 y
float64 z
================================================================================
MSG: geometry_msgs/Quaternion
float64 x
float64 y
float64 z
float64 w
================================================================================
MSG: geometry_msgs/TwistWithCovariance
geometry_msgs/Twist twist
float64[36] covariance
================================================================================
MSG: geometry_msgs/Twist
geometry_msgs/Vector3 linear
geometry_msgs/Vector3 angular
================================================================================
MSG: geometry_msgs/Vector3
float64 x
float64 y
float64 z
'''
INDICATOR = '''builtin_interfaces/Time stamp
uint8 report
================================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
'''


def write_map(path):
    nodes = [(1, 0, 3.5), (2, 200, 3.5), (3, 0, 0), (4, 200, 0), (5, 0, -3.5), (6, 200, -3.5)]
    text = ['<osm version="0.6" generator="jevsceneminer-synthetic-demo">']
    for node, x, y in nodes:
        text.append(f'<node id="{node}" lat="35" lon="139" version="1"><tag k="local_x" v="{90000+x}"/>'
                    f'<tag k="local_y" v="{44000+y}"/><tag k="ele" v="0"/></node>')
    for way, a, b, boundary in [(101,1,2,'solid'), (102,3,4,'dashed'), (103,5,6,'solid')]:
        text.append(f'<way id="{way}" version="1"><nd ref="{a}"/><nd ref="{b}"/>'
                    f'<tag k="type" v="line_thin"/><tag k="subtype" v="{boundary}"/></way>')
    for lane, left, right in [(10,101,102), (11,102,103)]:
        text.append(f'<relation id="{lane}" version="1"><member type="way" role="left" ref="{left}"/>'
                    f'<member type="way" role="right" ref="{right}"/><tag k="type" v="lanelet"/>'
                    '<tag k="subtype" v="road"/><tag k="location" v="urban"/><tag k="one_way" v="yes"/></relation>')
    path.write_text('\n'.join([*text, '</osm>'])+'\n')


def trajectory():
    for i in range(2251):
        t = i*.02
        y = 1.75 if t < 20 else -1.75 if t > 23 else 1.75-3.5*(t-20)/3
        yaw = math.atan2(-3.5/3,4) if 20<t<23 else 0
        yield T0+i*20_000_000, 90000+4*t, 44000+y, yaw


def write_mcap(folder):
    folder.mkdir()
    with (folder/'synthetic.mcap').open('wb') as stream:
        writer = Writer(stream)
        odom = writer.register_msgdef('nav_msgs/msg/Odometry', ODOMETRY)
        indicator = writer.register_msgdef('autoware_vehicle_msgs/msg/TurnIndicatorsReport', INDICATOR)
        for timestamp, x, y, yaw in trajectory():
            stamp = dict(sec=timestamp//10**9, nanosec=timestamp%10**9)
            vector = dict(x=0.,y=0.,z=0.)
            pose = dict(position=dict(x=x,y=y,z=0.), orientation=dict(x=0.,y=0.,z=math.sin(yaw/2),w=math.cos(yaw/2)))
            writer.write_message('/localization/kinematic_state', odom,
                dict(header=dict(stamp=stamp,frame_id='map'), child_frame_id='base_link',
                     pose=dict(pose=pose,covariance=[0.]*36),
                     twist=dict(twist=dict(linear=dict(x=4.,y=0.,z=0.),angular=vector),covariance=[0.]*36)), log_time=timestamp)
            writer.write_message('/vehicle/status/turn_indicators_status', indicator,
                dict(stamp=stamp,report=3 if 19<=(timestamp-T0)/1e9<=23 else 1), log_time=timestamp)
        writer.finish()


def write_nuplan(path):
    with sqlite3.connect(path) as db:
        db.executescript('''CREATE TABLE log(map_version TEXT);
CREATE TABLE ego_pose(timestamp INTEGER,x REAL,y REAL,qw REAL,qx REAL,qy REAL,qz REAL,vx REAL,angular_rate_z REAL);
CREATE TABLE lidar_pc(token BLOB,timestamp INTEGER);
CREATE TABLE lidar_box(lidar_pc_token BLOB,track_token BLOB,x REAL,y REAL,yaw REAL,vx REAL,vy REAL,confidence REAL,length REAL,width REAL);
CREATE TABLE track(token BLOB,category_token BLOB);
CREATE TABLE category(token BLOB,name TEXT);
CREATE TABLE traffic_light_status(lidar_pc_token BLOB,lane_connector_id INTEGER,status TEXT);
CREATE TABLE scenario_tag(lidar_pc_token BLOB,type TEXT);
CREATE TABLE image(timestamp INTEGER,filename_jpg TEXT,camera_token BLOB);
CREATE TABLE camera(token BLOB,channel TEXT);
INSERT INTO log VALUES ('synthetic');''')
        db.executemany('INSERT INTO ego_pose VALUES (?,?,?,?,?,?,?,?,?)',
            [(t//1000,x,y,math.cos(yaw/2),0,0,math.sin(yaw/2),4.,0.) for t,x,y,yaw in trajectory()])


def forbid_api():
    raise RuntimeError('Synthetic demo attempted an API request; blocked')


def build_demo(out, source):
    if out.exists():
        raise ValueError('Choose a new output directory; existing files are never overwritten')
    out.mkdir(parents=True)
    map_path = out/'synthetic_map.osm'
    write_map(map_path)
    if source == 'rosbag':
        folder = out/'source'
        write_mcap(folder)
        session = read_session(folder)
    else:
        path = out/'synthetic.db'
        write_nuplan(path)
        session = read_nuplan(path).session
    session.name = 'synthetic-'+source
    args = SimpleNamespace(step=1.,table_step=1.,past=10,future=15,max_steps=0,first_step=0,
        model='synthetic-demo',workers=1,ego_length=5.176,ego_width=2.297,ego_center_offset=1.461,
        min_scene=2.,min_phase=1.,min_prob=.4,maneuver_tail=0.,require_indicator=False)
    original_cache = lane_geometry.CACHE_DIR
    try:
        lane_geometry.CACHE_DIR = out/'map_cache'
        lane_map = LaneMap.load(map_path)
    finally:
        lane_geometry.CACHE_DIR = original_cache
    meta, scripts = cli._prepare(session, lane_map, 'right', {'map':str(map_path.resolve())}, args, out)
    sid = meta['date']+'_'+meta['name']
    labels = load_labels(cli.DEFAULT_LABELS, traffic_side='right', has_indicator=session.has_indicator)
    cache = AnswerCache(out/'cache'/f'{sid}.jsonl')
    for t, script in scripts.items():
        decision = 'lane_change_right' if 20 <= (t-T0)/1e9 <= 23 else 'keep_lane'
        cache.put(cache_key(args.model,labels,script), Answer(decision,
            {key: float(key==decision) for key in labels.lateral}, 'cruising',
            {key: float(key=='cruising') for key in labels.longitudinal}, args.model, 0))
    original = cli.JevClassifier
    try:
        cli.JevClassifier = functools.partial(JevClassifier, client_factory=forbid_api)
        cli._classify(out,meta,scripts,labels,args)
    finally:
        cli.JevClassifier = original
    row = next(json.loads(line) for line in (out/'steps'/f'{sid}.jsonl').read_text().splitlines()
               if json.loads(line)['t_ns']==T0+22*10**9)
    (out/'sample.json').write_text(json.dumps(row,indent=2)+'\n')
    (out/'sample.txt').write_text(row['script']+'\n')
    (out/'demo.json').write_text(json.dumps(dict(source='synthetic',api_calls=0,
        probabilities='illustrative, deterministic; not Jev predictions',sample_at_s=22,adapter=source),indent=2)+'\n')
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',choices=['rosbag','nuplan'],default='rosbag')
    parser.add_argument('--out',type=Path,required=True)
    arguments = parser.parse_args()
    print(build_demo(arguments.out,arguments.source))
