"""Lane map lookups in the same x/y frame as the ego pose.

``LaneMap`` holds the lanes and does the lane matching; a loader fills it from a map
file: ``LaneMap.load`` for Lanelet2 (below), ``nuplan_map.load_nuplan_map`` for nuPlan.

Autoware maps store each node's map-frame position in ``local_x`` / ``local_y`` tags.
The map is loaded so that lanelet2 coordinates equal those tags: every node's lat/lon
is rewritten as ``projector.reverse(local_x, local_y)`` and the rewritten file is loaded
with the same projector. A UTM projector is used because its x/y do not depend on the
altitude, so the round trip is exact even ~100 km from the origin (a local tangent-plane
projector drifts by metres there). The rewritten copy is cached under ~/.cache.
"""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import lanelet2
import numpy as np
from shapely.geometry import Polygon
from lanelet2.core import BasicPoint2d, BasicPoint3d
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector

VEHICLE_SUBTYPES = {"road", "road_shoulder", "highway", "bus_lane"}
CACHE_DIR = Path.home() / ".cache" / "jevsceneminer"

# Viterbi costs for lane matching (unitless; emission ~1 per 1.5 m off-center).
_OFF_MAP_COST = 3.0
_TRANSITION_COST = {"same": 0.0, "following": 0.3, "previous": 0.3, "near": 0.6,
                    "left": 1.0, "right": 1.0, "other": 5.0}
_OFF_MAP_SWITCH_COST = 2.0


@dataclass(frozen=True)
class Lane:
    id: int
    kind: str                   # "road", "shoulder", "highway", "bus lane"
    turn_direction: str | None  # "left" / "right" / "straight" for intersection lanes
    left: int | None            # same-direction neighbor sharing our left boundary
    left_crossable: bool        # that boundary is dashed or virtual
    right: int | None
    right_crossable: bool
    following: tuple[int, ...]
    previous: tuple[int, ...]
    centerline: np.ndarray      # (N, 2) map-frame points in driving direction
    traffic_lights: tuple[int, ...] = ()   # ids of the traffic_light regulatory elements that apply

    @property
    def is_intersection(self) -> bool:
        return self.turn_direction is not None


@dataclass(frozen=True)
class Projection:
    lane_id: int
    distance: float      # m outside the lane polygon (0 inside)
    offset: float        # m from the centerline, + = left
    heading_diff: float  # rad, (heading - lane direction) wrapped to [-pi, pi]


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _crossable(linestring) -> bool:
    attrs = linestring.attributes
    kind = attrs["type"] if "type" in attrs else ""
    subtype = attrs["subtype"] if "subtype" in attrs else ""
    return kind == "virtual" or "dashed" in subtype


def _rewrite_to_local(osm_path: Path) -> tuple[Path, Origin]:
    stat = osm_path.stat()
    key = hashlib.sha1(f"utm-v1:{osm_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()[:16]
    out, meta = CACHE_DIR / f"{key}.osm", CACHE_DIR / f"{key}.origin"
    if out.exists() and meta.exists():
        lat, lon = (float(v) for v in meta.read_text().split())
        return out, Origin(lat, lon)

    tree = ET.parse(osm_path)
    nodes = tree.getroot().findall("node")
    lat0, lon0 = float(nodes[0].get("lat")), float(nodes[0].get("lon"))
    projector = UtmProjector(Origin(lat0, lon0))
    missing = 0
    for node in nodes:
        tags = {t.get("k"): t.get("v") for t in node.findall("tag")}
        if "local_x" not in tags or "local_y" not in tags:
            missing += 1
            continue
        gps = projector.reverse(BasicPoint3d(float(tags["local_x"]), float(tags["local_y"]),
                                             float(tags.get("ele", 0.0))))
        node.set("lat", repr(gps.lat))
        node.set("lon", repr(gps.lon))
    if missing:
        raise ValueError(f"{osm_path}: {missing} nodes have no local_x/local_y tags")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tree.write(tmp, encoding="utf-8", xml_declaration=True)
    tmp.rename(out)
    meta.write_text(f"{lat0!r} {lon0!r}")
    return out, Origin(lat0, lon0)


class LaneMap:
    def __init__(self, lanes: dict[int, Lane], find_within):
        """``find_within(x, y, max_distance)`` -> ``[(distance, lane_id), ...]``: the lanes whose
        area is within ``max_distance`` of the point (distance 0 inside the lane)."""
        self.lanes = lanes
        self._find_within = find_within
        self._relations: dict[tuple[int, int], str] = {}

    @classmethod
    def load(cls, osm_path: Path) -> "LaneMap":
        """Load a Lanelet2 map (Autoware format, with local_x / local_y node tags)."""
        path, origin = _rewrite_to_local(Path(osm_path))
        # Autoware-only regulatory elements (road_marking, ...) fail to parse; lanes still load.
        lmap, _errors = lanelet2.io.loadRobust(str(path), UtmProjector(origin))

        def find_within(x, y, max_distance):
            found = lanelet2.geometry.findWithin2d(lmap.laneletLayer, BasicPoint2d(x, y), max_distance)
            return [(float(d), ll.id) for d, ll in found]
        result = cls(_build_lanes(lmap), find_within)
        result.polygons = {ll.id: Polygon([(p.x,p.y) for p in ll.leftBound]
                            + [(p.x,p.y) for p in reversed(list(ll.rightBound))])
                           for ll in lmap.laneletLayer if ll.id in result.lanes}
        return result

    # ---- geometry -------------------------------------------------------------

    def project(self, lane_id: int, x: float, y: float, yaw: float, distance: float = 0.0) -> Projection:
        pts = self.lanes[lane_id].centerline
        seg = pts[1:] - pts[:-1]
        seg_len2 = np.maximum((seg ** 2).sum(axis=1), 1e-12)
        w = np.array([x, y]) - pts[:-1]
        u = np.clip((w * seg).sum(axis=1) / seg_len2, 0.0, 1.0)
        foot = pts[:-1] + seg * u[:, None]
        d2 = ((np.array([x, y]) - foot) ** 2).sum(axis=1)
        i = int(np.argmin(d2))
        cross = seg[i, 0] * (y - pts[i, 1]) - seg[i, 1] * (x - pts[i, 0])
        offset = math.copysign(math.sqrt(d2[i]), cross)
        direction = math.atan2(seg[i, 1], seg[i, 0])
        return Projection(lane_id, distance, offset, wrap(yaw - direction))

    def candidates(self, x: float, y: float, yaw: float, max_distance: float = 1.0,
                   max_heading_diff: float = math.radians(60)) -> list[Projection]:
        out = []
        for distance, lane_id in self._find_within(x, y, max_distance):
            if lane_id not in self.lanes:
                continue
            p = self.project(lane_id, x, y, yaw, distance)
            if abs(p.heading_diff) <= max_heading_diff:
                out.append(p)
        return out

    @staticmethod
    def emission_cost(p: Projection) -> float:
        return abs(p.offset) / 1.5 + abs(p.heading_diff) / math.radians(30) + 2.0 * p.distance

    def lane_at(self, x: float, y: float, yaw: float) -> int | None:
        """Best lane for a single pose (no history). Used for surrounding objects."""
        cands = self.candidates(x, y, yaw)
        return min(cands, key=self.emission_cost).lane_id if cands else None

    # ---- topology -------------------------------------------------------------

    def relation(self, a: int, b: int) -> str:
        """How lane ``b`` relates to lane ``a``."""
        key = (a, b)
        if key not in self._relations:
            self._relations[key] = self._relation(a, b)
        return self._relations[key]

    def _relation(self, a: int, b: int) -> str:
        if a == b:
            return "same"
        la = self.lanes[a]
        if b in la.following:
            return "following"
        if b in la.previous:
            return "previous"
        if la.left == b:
            return "left"
        if la.right == b:
            return "right"
        for f in la.following:
            if b in self.lanes[f].following:
                return "near"
        for p in la.previous:
            if b in self.lanes[p].previous:
                return "near"
        return "other"

    # ---- lane matching over a track -------------------------------------------

    def match_track(self, xs, ys, yaws, valid) -> list[Projection | None]:
        """Pick one lane per sample with a Viterbi pass (no teleporting between lanes).

        Overlapping intersection lanes are resolved by what the ego does next, since the
        whole track is known offline.
        """
        n = len(xs)
        layers: list[list[tuple[Projection | None, float]]] = []
        for i in range(n):
            cands = self.candidates(xs[i], ys[i], yaws[i]) if valid[i] else []
            states = [(p, self.emission_cost(p)) for p in cands]
            states.append((None, _OFF_MAP_COST if cands else 0.0))
            layers.append(states)

        cost = [c for _, c in layers[0]]
        back: list[list[int]] = [[-1] * len(layers[0])]
        for i in range(1, n):
            prev, cur = layers[i - 1], layers[i]
            new_cost, new_back = [], []
            for p_cur, c_cur in cur:
                best, arg = math.inf, 0
                for j, (p_prev, _) in enumerate(prev):
                    c = cost[j] + self._transition(p_prev, p_cur)
                    if c < best:
                        best, arg = c, j
                new_cost.append(best + c_cur)
                new_back.append(arg)
            cost, back = new_cost, back + [new_back]

        k = int(np.argmin(cost))
        path: list[Projection | None] = [None] * n
        for i in range(n - 1, -1, -1):
            path[i] = layers[i][k][0]
            k = back[i][k]
        return path

    def _transition(self, a: Projection | None, b: Projection | None) -> float:
        if a is None and b is None:
            return 0.0
        if a is None or b is None:
            return _OFF_MAP_SWITCH_COST
        return _TRANSITION_COST[self.relation(a.lane_id, b.lane_id)]


def _build_lanes(lmap) -> dict[int, Lane]:
    vehicle = []
    for ll in lmap.laneletLayer:
        subtype = ll.attributes["subtype"] if "subtype" in ll.attributes else ""
        if subtype in VEHICLE_SUBTYPES and len(ll.centerline) >= 2:
            vehicle.append(ll)

    starts: dict[tuple[int, int], list[int]] = {}
    as_right: dict[tuple[int, bool], list[int]] = {}
    as_left: dict[tuple[int, bool], list[int]] = {}
    for ll in vehicle:
        starts.setdefault((ll.leftBound[0].id, ll.rightBound[0].id), []).append(ll.id)
        as_right.setdefault((ll.rightBound.id, ll.rightBound.inverted()), []).append(ll.id)
        as_left.setdefault((ll.leftBound.id, ll.leftBound.inverted()), []).append(ll.id)

    following = {ll.id: tuple(i for i in starts.get((ll.leftBound[-1].id, ll.rightBound[-1].id), [])
                              if i != ll.id) for ll in vehicle}
    previous: dict[int, list[int]] = {ll.id: [] for ll in vehicle}
    for a, fs in following.items():
        for b in fs:
            previous[b].append(a)

    lanes = {}
    for ll in vehicle:
        attrs = ll.attributes
        subtype = attrs["subtype"]
        left = [i for i in as_right.get((ll.leftBound.id, ll.leftBound.inverted()), []) if i != ll.id]
        right = [i for i in as_left.get((ll.rightBound.id, ll.rightBound.inverted()), []) if i != ll.id]
        lanes[ll.id] = Lane(
            id=ll.id,
            kind={"road_shoulder": "shoulder", "bus_lane": "bus lane"}.get(subtype, subtype),
            turn_direction=attrs["turn_direction"] if "turn_direction" in attrs else None,
            left=left[0] if left else None,
            left_crossable=_crossable(ll.leftBound),
            right=right[0] if right else None,
            right_crossable=_crossable(ll.rightBound),
            following=following[ll.id],
            previous=tuple(previous[ll.id]),
            centerline=np.array([[p.x, p.y] for p in ll.centerline], dtype=np.float64),
            traffic_lights=tuple(r.id for r in ll.regulatoryElements
                                 if "subtype" in r.attributes and r.attributes["subtype"] == "traffic_light"),
        )
    return lanes
