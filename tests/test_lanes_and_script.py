import math

import numpy as np

from conftest import OFFSET_X, OFFSET_Y, T0
from jevsceneminer.facts import build_timeline
from jevsceneminer.script import render, strip_map


def test_topology(lanemap):
    a, b, c = lanemap.lanes[10], lanemap.lanes[11], lanemap.lanes[12]
    assert a.right == 11 and b.left == 10
    assert a.left is None and b.right is None
    assert a.right_crossable and not a.left_crossable
    assert a.following == (12,) and c.previous == (10,)
    assert lanemap.relation(10, 12) == "following"
    assert lanemap.relation(10, 11) == "right"
    assert lanemap.relation(11, 10) == "left"
    assert lanemap.relation(11, 12) == "other"


def test_coordinates_follow_local_xy(lanemap):
    center = lanemap.lanes[10].centerline
    assert np.allclose(center[0], [OFFSET_X, OFFSET_Y + 1.75], atol=1e-3)


def test_candidates_offset_sign(lanemap):
    (p,) = [c for c in lanemap.candidates(OFFSET_X + 25, OFFSET_Y + 2.75, 0.0) if c.lane_id == 10]
    assert math.isclose(p.offset, 1.0, abs_tol=1e-3)   # left of center is positive
    (q,) = [c for c in lanemap.candidates(OFFSET_X + 25, OFFSET_Y + 0.75, 0.0) if c.lane_id == 10]
    assert math.isclose(q.offset, -1.0, abs_tol=1e-3)
    assert lanemap.candidates(OFFSET_X + 25, OFFSET_Y + 1.75, math.pi) == []   # wrong direction


def test_match_track_follows_the_road(lanemap):
    xs = OFFSET_X + np.arange(1, 100, 2.0)
    path = lanemap.match_track(xs, np.full_like(xs, OFFSET_Y + 1.75), np.zeros_like(xs), np.ones_like(xs, bool))
    ids = [p.lane_id for p in path]
    assert set(ids[:20]) == {10} and set(ids[-20:]) == {12}


def test_script_describes_the_lane_change(lanemap, lane_change_session):
    tl = build_timeline(lane_change_session, lanemap)
    text = render(tl, lanemap, T0 + 2 * 10**9, past_s=5, future_s=5)   # NOW = 2 s, before the lane change
    assert "t=+2.0s: moves from A into its right neighbor B" in text
    assert "t=+0.5s: indicator RIGHT on" in text
    assert "indicator during the table: RIGHT from t=+0.5s to t=+5.0s" in text
    assert "A: road; left neighbor: none; right neighbor: B (dashed line, can be crossed)" in text
    assert "car: 15 m ahead, 0 m left, same lane, 14 km/h" in text
    assert "t=-5s   (no data)" in text
    assert "<- NOW" in text
    # Only relative facts: no absolute coordinates or clock times leak into the script.
    assert str(int(OFFSET_X)) not in text and str(T0)[:6] not in text


def test_render_returns_none_without_ego_data(lanemap, lane_change_session):
    tl = build_timeline(lane_change_session, lanemap)
    assert render(tl, lanemap, T0 + 60 * 10**9) is None


def test_script_says_when_the_indicator_is_never_on(lanemap, lane_change_session):
    s = lane_change_session
    s.indicator[:] = 1                       # indicator off the whole time
    text = render(build_timeline(s, lanemap), lanemap, T0 + 2 * 10**9, past_s=5, future_s=10)
    assert "indicator during the table: never on" in text
    assert "t=+10s" in text and "t=-5s" in text   # asymmetric table: -5 s .. +10 s


def test_strip_map_removes_every_map_fact(lanemap, lane_change_session):
    tl = build_timeline(lane_change_session, lanemap)
    full = render(tl, lanemap, T0 + 2 * 10**9, past_s=5, future_s=10)
    bare = strip_map(full)
    for gone in ("LANES", "OFFSET", "neighbor", "moves from", "same lane", "offset while", "lane center"):
        assert gone not in bare, gone
    for kept in ("SPEED", "YAW RATE", "indicator RIGHT on", "car: 15 m ahead, 0 m left, 14 km/h", "<- NOW"):
        assert kept in bare, kept
    row = next(l for l in bare.splitlines() if l.startswith("t=+0s"))
    assert row.split()[1] == "14"   # speed now follows the time column directly


def test_the_light_ahead_is_picked_from_all_recognized_groups():
    from types import SimpleNamespace

    from jevsceneminer.facts import LIGHT_AHEAD_M, lights_ahead, resolve_lights
    from jevsceneminer.lanes import Projection

    lanemap = SimpleNamespace(lanes={1: SimpleNamespace(traffic_lights=()), 2: SimpleNamespace(traffic_lights=(7,))})
    p1, p2 = Projection(1, 0, 0, 0), Projection(2, 0, 0, 0)
    x = np.array([0.0, 10.0, 20.0, 20.0 + LIGHT_AHEAD_M + 1])
    ahead = lights_ahead([p1, p1, p2, p1], x, np.zeros(4), lanemap)
    assert ahead == [(7,), (7,), (7,), ()]
    far = lights_ahead([p1, p2], np.array([0.0, LIGHT_AHEAD_M + 1]), np.zeros(2), lanemap)
    assert far == [(), (7,)]                          # too far along the path: no light yet
    session = SimpleNamespace(lights=[(T0, {7: "red", 9: "green"}), (T0 + 10**8, {9: "green"}),
                                      (T0 + 2 * 10**8, {None: "amber"})])
    assert [text for _, text in resolve_lights(session, T0, ahead)] == ["red", "unknown", "amber"]

