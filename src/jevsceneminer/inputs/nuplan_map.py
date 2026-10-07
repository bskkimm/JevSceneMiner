"""nuPlan map (GeoPackage ``map.gpkg``) as a ``LaneMap``.

The map stores lanes (``lanes_polygons``), the connections between them
(``lane_connectors``, mostly through intersections), their center lines
(``baseline_paths``) and lane boundaries in WGS 84 longitude/latitude. They are projected
to the map's UTM zone (``meta.projectedCoordSystem``), the frame of nuPlan's ego poses.

Both lanes and connectors become ``Lane`` entries. Neighbors share a boundary, as in
Lanelet2; a shared boundary of type 0 only ever appears between two lanes, so it is taken
as crossable (type 2 also bounds the road edge: solid). A connector is an intersection
lane when it belongs to an intersection; its turn direction comes from its heading
change. nuPlan gives traffic light states per connector, so a connector in a signalized
intersection (type 1) lists its own id in ``traffic_lights``.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import numpy as np
import pyproj
import shapely
from shapely import wkb

from jevsceneminer.evidence.lanes import Lane, LaneMap, wrap

CROSSABLE_BOUNDARY_TYPES = {0}
TRAFFIC_LIGHT_INTERSECTION_TYPE = 1   # every connector with light states in the logs is in one
STRAIGHT_BELOW_DEG = 30.0
U_TURN_ABOVE_DEG = 135.0

# City -> side traffic drives on (nuPlan's four locations).
TRAFFIC_SIDE = {"sg-one-north": "left", "us-ma-boston": "right", "us-pa-pittsburgh-hazelwood": "right",
                "us-nv-las-vegas-strip": "right"}


def gpkg_geometry(blob: bytes):
    """GeoPackage geometry blob -> shapely geometry (skips the GP header and envelope)."""
    flags = blob[3]
    envelope = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(flags >> 1) & 7]
    return wkb.loads(bytes(blob[8 + envelope:]))


def find_map(maps_root: Path, location: str, version: str | None = None) -> Path:
    """``<maps_root>/<location>/<version>/map.gpkg``; the newest version when ``version`` is not
    one of the folders (logs store the location in their ``map_version`` field)."""
    folder = Path(maps_root) / location
    versions = sorted((p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else [],
                      key=lambda v: [int(x) for x in v.split(".") if x.isdigit()])
    if version in versions:
        versions.append(version)
    for v in reversed(versions):
        path = folder / v / "map.gpkg"
        if path.exists():
            return path
    raise FileNotFoundError(f"no map.gpkg for {location} under {maps_root}")


def _heading(coords: np.ndarray, at_end: bool) -> float:
    a, b = (coords[-min(4, len(coords))], coords[-1]) if at_end else (coords[0], coords[min(3, len(coords) - 1)])
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _turn_direction(coords: np.ndarray) -> str:
    change = math.degrees(wrap(_heading(coords, True) - _heading(coords, False)))
    if abs(change) < STRAIGHT_BELOW_DEG:
        return "straight"
    if abs(change) > U_TURN_ABOVE_DEG:
        return "u-turn"
    return "left" if change > 0 else "right"


def load_nuplan_map(path: Path) -> LaneMap:
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    meta = dict(db.execute("select key, value from meta"))
    to_utm = pyproj.Transformer.from_crs(4326, meta["projectedCoordSystem"].upper(), always_xy=True)

    def utm(blob):
        return shapely.transform(gpkg_geometry(blob),
                                 lambda xy: np.column_stack(to_utm.transform(xy[:, 0], xy[:, 1])))

    boundary_type = dict(db.execute("select fid, boundary_type_fid from boundaries"))
    centerline = {}
    for blob, lane, connector in db.execute("select geom, lane_fid, lane_connector_fid from baseline_paths"):
        centerline[int(lane if lane is not None else connector)] = np.asarray(utm(blob).coords)[:, :2]

    lanes_rows = db.execute("select fid, geom, lane_fid, lane_type_fid, left_boundary_fid, right_boundary_fid "
                            "from lanes_polygons").fetchall()
    conn_rows = db.execute("select fid, exit_lane_fid, entry_lane_fid, intersection_fid "
                           "from lane_connectors").fetchall()
    signalized = {fid for fid, kind in db.execute("select fid, intersection_type_fid from intersections")
                  if kind == TRAFFIC_LIGHT_INTERSECTION_TYPE}
    conn_polys = {int(c): utm(g) for g, c in
                  db.execute("select geom, lane_connector_fid from gen_lane_connectors_scaled_width_polygons")}
    lane_ids = {int(r[2]) for r in lanes_rows}
    if lane_ids & {int(r[0]) for r in conn_rows}:
        raise ValueError(f"{path}: lane and lane connector ids overlap")

    following: dict[int, list[int]] = {i: [] for i in lane_ids}
    previous: dict[int, list[int]] = {i: [] for i in lane_ids}
    for fid, exit_lane, entry_lane, _ in conn_rows:
        following.setdefault(int(exit_lane), []).append(int(fid))
        previous[int(fid)] = [int(exit_lane)]
        following[int(fid)] = [int(entry_lane)]
        previous.setdefault(int(entry_lane), []).append(int(fid))

    by_left = {int(r[4]): int(r[2]) for r in lanes_rows}     # lane whose left boundary is b
    by_right = {int(r[5]): int(r[2]) for r in lanes_rows}
    lanes: dict[int, Lane] = {}
    polygons: dict[int, object] = {}
    for _, blob, lane, lane_type, left_b, right_b in lanes_rows:
        lane = int(lane)
        if lane not in centerline:
            continue
        left, right = by_right.get(int(left_b)), by_left.get(int(right_b))
        lanes[lane] = Lane(
            id=lane, kind="road", turn_direction=None,
            left=left if left != lane else None,
            left_crossable=boundary_type.get(int(left_b)) in CROSSABLE_BOUNDARY_TYPES,
            right=right if right != lane else None,
            right_crossable=boundary_type.get(int(right_b)) in CROSSABLE_BOUNDARY_TYPES,
            following=tuple(following[lane]), previous=tuple(previous[lane]),
            centerline=centerline[lane])
        polygons[lane] = utm(blob)
    for fid, _, _, intersection in conn_rows:
        fid = int(fid)
        if fid not in centerline or fid not in conn_polys:
            continue
        lanes[fid] = Lane(
            id=fid, kind="road",
            turn_direction=_turn_direction(centerline[fid]) if intersection is not None else None,
            left=None, left_crossable=False, right=None, right_crossable=False,
            following=tuple(following[fid]), previous=tuple(previous[fid]),
            centerline=centerline[fid],
            traffic_lights=(fid,) if intersection in signalized else ())
        polygons[fid] = conn_polys[fid]

    ids = list(polygons)
    geoms = [polygons[i] for i in ids]
    tree = shapely.STRtree(geoms)

    def find_within(x, y, max_distance):
        point = shapely.Point(x, y)
        hits = tree.query(point, predicate="dwithin", distance=max_distance)
        return [(float(geoms[k].distance(point)), ids[k]) for k in hits]

    lanemap = LaneMap(lanes, find_within)
    lanemap.polygons = polygons          # for the viewer's map layer
    return lanemap
