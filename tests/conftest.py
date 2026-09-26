"""Shared fixtures: a tiny synthetic Lanelet2 map and a synthetic lane-change session."""

from __future__ import annotations

import math

import numpy as np
import pytest

from jevsceneminer import lanes as lanes_mod
from jevsceneminer.bag import INDICATOR_OFF, INDICATOR_RIGHT, DetectedObject, EgoTrack, Session

# Three lanes, driving direction +x (left = +y):
#   A (id 10): y 0..3.5,  x 0..50      C (id 12) follows A: x 50..100
#   B (id 11): y -3.5..0, x 0..50      A and B share the dashed line y = 0
NODES = {1: (0, 3.5), 2: (50, 3.5), 3: (0, 0), 4: (50, 0), 5: (0, -3.5), 6: (50, -3.5),
         7: (100, 3.5), 8: (100, 0)}
WAYS = {101: ((1, 2), "solid"), 102: ((3, 4), "dashed"), 103: ((5, 6), "solid"),
        104: ((2, 7), "solid"), 105: ((4, 8), "dashed")}
LANELETS = {10: (101, 102), 11: (102, 103), 12: (104, 105)}
OFFSET_X, OFFSET_Y = 90000.0, 44000.0   # map-frame coordinates are large, like the real map


def write_osm(path):
    out = ['<?xml version="1.0" encoding="UTF-8"?>', '<osm version="0.6" generator="test">']
    for nid, (x, y) in NODES.items():
        out.append(f'<node id="{nid}" lat="35.0" lon="139.0" visible="true" version="1">'
                   f'<tag k="local_x" v="{x + OFFSET_X}"/><tag k="local_y" v="{y + OFFSET_Y}"/>'
                   f'<tag k="ele" v="5.0"/></node>')
    for wid, ((a, b), subtype) in WAYS.items():
        out.append(f'<way id="{wid}" visible="true" version="1"><nd ref="{a}"/><nd ref="{b}"/>'
                   f'<tag k="type" v="line_thin"/><tag k="subtype" v="{subtype}"/></way>')
    for rid, (left, right) in LANELETS.items():
        out.append(f'<relation id="{rid}" visible="true" version="1">'
                   f'<member type="way" role="left" ref="{left}"/><member type="way" role="right" ref="{right}"/>'
                   f'<tag k="type" v="lanelet"/><tag k="subtype" v="road"/><tag k="location" v="urban"/>'
                   f'<tag k="one_way" v="yes"/></relation>')
    out.append("</osm>")
    path.write_text("\n".join(out))


@pytest.fixture(scope="session")
def lanemap(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("map")
    old = lanes_mod.CACHE_DIR
    lanes_mod.CACHE_DIR = tmp / "cache"
    try:
        osm = tmp / "lanelet2_map.osm"
        write_osm(osm)
        yield lanes_mod.LaneMap.load(osm)
    finally:
        lanes_mod.CACHE_DIR = old


T0 = 1_767_000_000_000_000_000


@pytest.fixture()
def lane_change_session():
    """10 s at 4 m/s in lane A; moves into its right neighbor B between t=3 s and t=5 s."""
    t = np.arange(0, 10.0 + 1e-9, 0.02)
    x = 4.0 * t + OFFSET_X
    y_rel = np.where(t < 3, 1.75, np.where(t > 5, -1.75, 1.75 - 3.5 * (t - 3) / 2))
    vy = np.gradient(y_rel, t)
    yaw = np.arctan2(vy, 4.0)
    ego = EgoTrack(t=(T0 + t * 1e9).astype(np.int64), x=x, y=y_rel + OFFSET_Y, yaw=yaw,
                   v=np.full_like(t, 4.0), yaw_rate=np.gradient(yaw, t))
    ind_t = (T0 + np.arange(0, 10.0 + 1e-9, 0.1) * 1e9).astype(np.int64)
    ind = np.where((ind_t - T0) / 1e9 >= 2.5, INDICATOR_RIGHT, INDICATOR_OFF)
    obj_t = (T0 + np.arange(0, 10.0 + 1e-9, 0.5) * 1e9).astype(np.int64)
    objects = [(int(ts), [DetectedObject("car", float(4.0 * (ts - T0) / 1e9 + 15 + OFFSET_X), 1.75 + OFFSET_Y,
                                         0.0, 4.0, 0.9)]) for ts in obj_t]
    lights = [(int(ts), {None: "green"}) for ts in obj_t]
    return Session(name="12-00-00", date="2026-01-01", ego=ego, objects=objects,
                   lights=lights, indicator_t=ind_t, indicator=ind)


def seconds(s: float) -> int:
    return T0 + int(round(s * 1e9))


__all__ = ["seconds", "T0", "OFFSET_X", "OFFSET_Y", "math"]
