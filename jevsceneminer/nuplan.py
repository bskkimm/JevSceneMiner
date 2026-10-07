"""Read nuPlan logs (SQLite ``.db`` files) into the same ``Session`` as the rosbag reader.

    ego      ego_pose               ~100 Hz, UTM x/y, quaternion, body-frame velocity
    objects  lidar_box + track      20 Hz, map-frame boxes with a category
    lights   traffic_light_status   20 Hz, red/green per lane connector id
    tags     scenario_tag           nuPlan's automatic scenario labels (a reference, not GT)
    camera   image (CAM_F0)         front camera frames, for the viewer

nuPlan does not record the turn indicator, so the session says so (``has_indicator``).
nuPlan timestamps are microseconds; everything here is nanoseconds.
"""

from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .bag import BagError, DetectedObject, EgoTrack, Session

OBJECT_KINDS = {"vehicle": "car", "bicycle": "bicycle", "pedestrian": "pedestrian", "traffic_cone": "traffic cone",
                "barrier": "barrier", "czone_sign": "construction sign", "generic_object": "object"}
CAMERA = "CAM_F0"
_SLICE = re.compile(r"^(.*)_(\d{5})_(\d{5})$")


@dataclass
class NuplanLog:
    session: Session
    location: str                 # map folder name, e.g. "us-ma-boston" (the log's map_version field)
    tags: list[tuple[int, str]] = field(default_factory=list)       # (t_ns, scenario type)
    camera: list[tuple[int, str]] = field(default_factory=list)     # (t_ns, path under sensor_blobs)


def _connect(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)


def _yaw(qw, qx, qy, qz) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def session_name(paths: list[Path]) -> str:
    """One log: its name. Consecutive slices of one recording: the recording with the joint range."""
    stems = [Path(p).stem for p in paths]
    if len(stems) == 1:
        return stems[0]
    parts = [_SLICE.match(s) for s in stems]
    if all(parts) and len({m.group(1) for m in parts}) == 1:
        return f"{parts[0].group(1)}_{parts[0].group(2)}_{parts[-1].group(3)}"
    return "+".join(stems)


def read_nuplan(paths, object_interval_s: float = 0.45) -> NuplanLog:
    """Read one session from one log, or from consecutive slices of the same recording."""
    paths = sorted(Path(p) for p in ([paths] if isinstance(paths, (str, Path)) else paths))
    ego_rows, boxes, light_rows, tags, camera = [], [], [], [], []
    logs = set()
    for path in paths:
        db = _connect(path)
        logs.add(db.execute("select map_version from log").fetchone()[0])
        ego_rows += db.execute("select timestamp, x, y, qw, qx, qy, qz, vx, angular_rate_z from ego_pose").fetchall()
        boxes += db.execute(
            "select lp.timestamp, lb.x, lb.y, lb.yaw, lb.vx, lb.vy, lb.confidence, c.name, hex(lb.track_token), lb.length, lb.width "
            "from lidar_box lb join lidar_pc lp on lb.lidar_pc_token = lp.token "
            "join track t on lb.track_token = t.token join category c on t.category_token = c.token").fetchall()
        light_rows += db.execute(
            "select lp.timestamp, s.lane_connector_id, s.status from traffic_light_status s "
            "join lidar_pc lp on s.lidar_pc_token = lp.token").fetchall()
        tags += db.execute("select lp.timestamp, t.type from scenario_tag t "
                           "join lidar_pc lp on t.lidar_pc_token = lp.token").fetchall()
        camera += db.execute("select i.timestamp, i.filename_jpg from image i join camera c on i.camera_token = c.token "
                             "where c.channel = ?", (CAMERA,)).fetchall()
        db.close()
    name = session_name(paths)
    if len(logs) != 1:
        raise BagError(f"{name}: the logs are from different maps: {sorted(logs)}")
    if not ego_rows:
        raise BagError(f"{name}: no ego poses")
    (location,) = logs

    ego_rows.sort()
    arr = np.array([(x, y, _yaw(qw, qx, qy, qz), vx, wz) for _, x, y, qw, qx, qy, qz, vx, wz in ego_rows])
    ego = EgoTrack(t=np.array([r[0] for r in ego_rows], dtype=np.int64) * 1000,
                   x=arr[:, 0], y=arr[:, 1], yaw=np.unwrap(arr[:, 2]), v=arr[:, 3], yaw_rate=arr[:, 4])

    # Objects only every object_interval_s (the script needs them at NOW steps only).
    frames: dict[int, list[DetectedObject]] = {}
    for t, x, y, yaw, vx, vy, conf, kind, track_id, length, width in boxes:
        frames.setdefault(int(t) * 1000, []).append(DetectedObject(
            kind=OBJECT_KINDS.get(kind, "object"), x=x, y=y, yaw=yaw, speed=math.hypot(vx, vy),
            existence=conf if conf is not None else 1.0, track_id=track_id or None,
            length_m=length, width_m=width))
    objects, next_ns = [], 0
    for t in sorted(frames):
        if t >= next_ns:
            objects.append((t, frames[t]))
            next_ns = t + int(object_interval_s * 1e9)

    lights: dict[int, dict] = {}
    for t, connector, status in light_rows:
        lights.setdefault(int(t) * 1000, {})[int(connector)] = status

    first = datetime.fromtimestamp(ego.t[0] / 1e9, timezone.utc)
    session = Session(
        name=name, date=first.strftime("%Y-%m-%d"), ego=ego, objects=objects,
        lights=sorted(lights.items()),
        indicator_t=np.zeros(0, dtype=np.int64), indicator=np.zeros(0, dtype=np.int64),
        topics={"source": "nuplan", "logs": [p.stem for p in paths]}, has_indicator=False,
        ego_geometry=dict(length_m=5.176,width_m=2.297,center_forward_m=1.461,
                          pose_reference="rear axle",source="nuPlan Pacifica vehicle model"))
    return NuplanLog(session, location,
                     tags=sorted((int(t) * 1000, kind) for t, kind in tags),
                     camera=sorted((int(t) * 1000, f) for t, f in camera))
