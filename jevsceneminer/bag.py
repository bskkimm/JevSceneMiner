"""Read one driving session (a folder of MCAP chunks) into plain arrays.

Messages are decoded with the definitions stored inside each MCAP file, so no ROS
installation or sourcing is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

# Default topics of the current Autoware (autoware_msgs / AD API). Each role lists its
# topics in order of preference; a session uses the first one it has. Other recordings
# can pass their own lists (see ``read_session(topics=...)`` and ``--topics``).
DEFAULT_TOPICS = {
    "ego": ("/localization/kinematic_state",      # nav_msgs/Odometry, 50 Hz
            "/api/vehicle/kinematics"),           # autoware_adapi_v1_msgs/VehicleKinematics
    # The tracking topic carries the same objects as the prediction topic without the
    # predicted paths, and decodes about 10x faster.
    "objects": ("/perception/object_recognition/tracking/objects",
                "/perception/object_recognition/objects"),
    "lights": ("/perception/traffic_light_recognition/traffic_signals",),  # TrafficLightGroupArray
    "indicator": ("/vehicle/status/turn_indicators_status",),             # TurnIndicatorsReport
}
REQUIRED_ROLES = ("ego", "indicator")

OBJECT_KINDS = {
    0: "unknown", 1: "car", 2: "truck", 3: "bus", 4: "trailer", 5: "motorcycle",
    6: "bicycle", 7: "pedestrian", 8: "animal", 9: "hazard", 10: "object", 11: "object",
}
LIGHT_COLORS = {0: "unknown", 1: "red", 2: "amber", 3: "green", 4: "white"}
LIGHT_ARROWS = {
    2: "left arrow", 3: "right arrow", 4: "up arrow", 5: "up-left arrow",
    6: "up-right arrow", 7: "down arrow", 8: "down-left arrow", 9: "down-right arrow",
}
INDICATOR_OFF, INDICATOR_LEFT, INDICATOR_RIGHT = 1, 2, 3

class BagError(RuntimeError):
    """A session is missing data the pipeline needs."""


@dataclass
class EgoTrack:
    t: np.ndarray  # ns since epoch, int64
    x: np.ndarray  # m, map frame
    y: np.ndarray
    yaw: np.ndarray  # rad, unwrapped
    v: np.ndarray  # m/s
    yaw_rate: np.ndarray  # rad/s


@dataclass
class DetectedObject:
    kind: str
    x: float  # m, map frame
    y: float
    yaw: float  # rad
    speed: float  # m/s
    existence: float


@dataclass
class Session:
    name: str  # the session folder's name unless given
    date: str  # UTC YYYY-MM-DD of the first message unless given
    ego: EgoTrack
    objects: list[tuple[int, list[DetectedObject]]]
    # Per message: traffic light group id -> text. A topic that already carries only the
    # light ahead of the ego (one group, no id) is stored under the key None.
    lights: list[tuple[int, dict[int | None, str]]]
    indicator_t: np.ndarray
    indicator: np.ndarray  # INDICATOR_OFF / INDICATOR_LEFT / INDICATOR_RIGHT
    topics: dict | None = None  # role -> topic actually used
    has_indicator: bool = True  # False when the log does not record the turn indicator

    @property
    def start_ns(self) -> int:
        return int(self.ego.t[0])

    @property
    def end_ns(self) -> int:
        return int(self.ego.t[-1])


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def light_text(msg) -> str:
    """Describe one traffic light group as e.g. "red", "red + green right arrow" or "none"."""
    if not msg.elements:
        return "none"
    circles, arrows = [], []
    for e in msg.elements:
        color = LIGHT_COLORS.get(e.color, "unknown")
        if e.shape in LIGHT_ARROWS:
            arrows.append(f"{color} {LIGHT_ARROWS[e.shape]}")
        else:
            circles.append(color)
    parts = sorted(set(circles)) + sorted(set(arrows))
    return " + ".join(parts) if parts else "unknown"


def _lights(msg) -> dict[int | None, str]:
    """TrafficLightGroupArray -> {group id: text}; a single group message -> {None: text}."""
    if hasattr(msg, "traffic_light_groups"):
        return {int(g.traffic_light_group_id): light_text(g) for g in msg.traffic_light_groups}
    return {None: light_text(msg)}


def _objects(msg) -> list[DetectedObject]:
    out = []
    for o in msg.objects:
        best = max(o.classification, key=lambda c: c.probability) if o.classification else None
        kin = o.kinematics   # tracked objects: pose_with_covariance; predicted: initial_pose_with_covariance
        pose = (getattr(kin, "pose_with_covariance", None) or kin.initial_pose_with_covariance).pose
        twist = (getattr(kin, "twist_with_covariance", None) or kin.initial_twist_with_covariance).twist
        out.append(DetectedObject(
            kind=OBJECT_KINDS.get(best.label, "unknown") if best else "unknown",
            x=pose.position.x,
            y=pose.position.y,
            yaw=yaw_from_quaternion(pose.orientation),
            speed=math.hypot(twist.linear.x, twist.linear.y),
            existence=o.existence_probability,
        ))
    return out


def _files_in_time_order(paths) -> list[tuple[int, int, Path, set]]:
    files = []
    for path in paths:
        with open(path, "rb") as fh:
            summary = make_reader(fh).get_summary()
        st = summary.statistics
        files.append((st.message_start_time, st.message_end_time, Path(path),
                      {c.topic for c in summary.channels.values()}))
    return sorted(files, key=lambda f: f[0])


def _ego_row(msg) -> tuple:
    if hasattr(msg.pose.pose, "pose"):      # AD API VehicleKinematics: one level deeper than Odometry
        p, tw = msg.pose.pose.pose, msg.twist.twist.twist
    else:
        p, tw = msg.pose.pose, msg.twist.twist
    return p.position.x, p.position.y, yaw_from_quaternion(p.orientation), tw.linear.x, tw.angular.z


def read_session(source, object_interval_s: float = 0.45, start_ns: int | None = None,
                 end_ns: int | None = None, name: str | None = None, date: str | None = None,
                 topics: dict | None = None) -> Session:
    """Read one session: a folder of MCAP chunks, or a list of MCAP files cut to a time range.

    Without ``name`` the session is named after its folder; without ``date``, the UTC date
    of its first message is used. ``topics`` overrides ``DEFAULT_TOPICS`` role by role.
    Objects are only decoded every ``object_interval_s`` seconds: they are the slowest
    topic and the script only needs them at NOW steps.
    """
    if isinstance(source, (str, Path)) and Path(source).is_dir():
        label, paths = Path(source).name, sorted(Path(source).glob("*.mcap"))
    else:
        paths = [Path(p) for p in source]
        label = name or (paths[0].parent.name if paths else "?")
    files = [f for f in _files_in_time_order(paths)
             if (start_ns is None or f[1] >= start_ns) and (end_ns is None or f[0] <= end_ns)]
    if not files:
        raise BagError(f"{label}: no .mcap files in range")
    available = set().union(*(f[3] for f in files))
    wanted_topics = {**DEFAULT_TOPICS, **{k: tuple(v) for k, v in (topics or {}).items()}}
    chosen = {role: next((t for t in wanted_topics[role] if t in available), None) for role in DEFAULT_TOPICS}
    missing = [role for role in REQUIRED_ROLES if chosen[role] is None]
    if missing:
        raise BagError(f"{label}: no topic for {', '.join(missing)}")
    first_ns = start_ns if start_ns is not None else files[0][0]
    first = datetime.fromtimestamp(first_ns / 1e9, timezone.utc)

    ego_rows, objects, lights, ind_t, ind = [], [], [], [], []
    object_step_ns = int(object_interval_s * 1e9)
    next_object_ns = 0
    wanted = [t for t in chosen.values() if t]
    # Some bags store a message definition that does not match the data: such messages
    # are skipped and counted, not fatal.
    decode_errors: dict[str, int] = {}
    for _, _, path, _ in files:
        with open(path, "rb") as fh:
            reader = make_reader(fh)
            # A new factory per file: it caches decoders by schema id, and the ids differ
            # between chunks of the same session (schema 5 is Odometry in one, a traffic
            # light in the next).
            factory = DecoderFactory()
            decoders = {}
            for schema, channel, message in reader.iter_messages(
                    topics=wanted, start_time=start_ns, end_time=end_ns, log_time_order=True):
                topic = channel.topic
                if topic == chosen["objects"]:
                    if message.log_time < next_object_ns:
                        continue
                    next_object_ns = message.log_time + object_step_ns
                decode = decoders.get(channel.id)
                if decode is None:
                    decode = factory.decoder_for(channel.message_encoding, schema)
                    decoders[channel.id] = decode
                try:
                    msg = decode(message.data)
                except Exception:           # noqa: BLE001 - struct/index errors from bad data
                    decode_errors[topic] = decode_errors.get(topic, 0) + 1
                    continue
                t = message.log_time
                if topic == chosen["ego"]:
                    ego_rows.append((t, *_ego_row(msg)))
                elif topic == chosen["objects"]:
                    objects.append((t, _objects(msg)))
                elif topic == chosen["lights"]:
                    lights.append((t, _lights(msg)))
                else:
                    ind_t.append(t)
                    ind.append(msg.report)

    empty = [role for role, rows in (("ego", ego_rows), ("indicator", ind)) if not rows]
    if empty:
        raise BagError(f"{label}: no messages on {', '.join(chosen[r] for r in empty)}")

    ego_rows.sort(key=lambda r: r[0])
    arr = np.array(ego_rows, dtype=np.float64)
    ego = EgoTrack(
        t=np.array([r[0] for r in ego_rows], dtype=np.int64),
        x=arr[:, 1], y=arr[:, 2], yaw=np.unwrap(arr[:, 3]), v=arr[:, 4], yaw_rate=arr[:, 5],
    )
    order = np.argsort(np.array(ind_t, dtype=np.int64), kind="stable")
    return Session(
        name=name or label,
        date=date or first.strftime("%Y-%m-%d"),
        ego=ego,
        objects=sorted(objects, key=lambda r: r[0]),
        lights=sorted(lights, key=lambda r: r[0]),
        indicator_t=np.array(ind_t, dtype=np.int64)[order],
        indicator=np.array(ind, dtype=np.int64)[order],
        topics=dict(chosen, decode_errors=decode_errors) if decode_errors else chosen,
    )
