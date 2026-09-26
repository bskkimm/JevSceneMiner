import struct

import numpy as np
import shapely

from conftest import T0
from jevsceneminer.evaluate import maneuvers, match, score_against_gt, score_against_tags, tag_events
from jevsceneminer.facts import build_timeline
from jevsceneminer.jev import load_labels
from jevsceneminer.nuplan import session_name
from jevsceneminer.nuplan_map import _turn_direction, gpkg_geometry
from jevsceneminer.rules import lateral_spans


def doc(*scenes):
    return {"scenes": [{"start_ns": str(int(a * 1e9)), "end_ns": str(int(b * 1e9)), "lateral": lat,
                        "longitudinal": lon} for a, b, lat, lon in scenes]}


def test_maneuvers_merge_scenes_split_by_speed():
    d = doc((0, 2, "keep_lane", "cruising"), (2, 4, "turn_left", "decelerating"), (4, 6, "turn_left", "stopped"),
            (6, 9, "keep_lane", "accelerating"))
    assert maneuvers(d) == [(2.0, 6.0, "turn_left")]


def test_match_is_one_to_one_and_needs_overlap():
    gt = [(0, 10, "turn_left"), (20, 25, "lane_change_right")]
    run = [(1, 4, "turn_left"), (5, 9, "turn_left"), (24.8, 30, "lane_change_right")]
    assert match(gt, run) == [(0, 1)]          # 0.2 s overlap is too little; one run maneuver per GT one


def test_score_against_gt_counts_and_times():
    gt = {"s": doc((0, 10, "keep_lane", "cruising"), (10, 20, "turn_left", "cruising"), (20, 30, "keep_lane", "cruising"))}
    run = {"s": doc((0, 11, "keep_lane", "cruising"), (11, 19, "turn_left", "cruising"),
                    (19, 25, "keep_lane", "cruising"), (25, 30, "lane_change_left", "cruising"))}
    r = score_against_gt(gt, run)
    assert r["labels"]["turn_left"] == {"TP": 1, "FN": 0, "FP": 0, "precision": 1.0, "recall": 1.0}
    assert r["labels"]["lane_change_left"]["FP"] == 1
    assert r["timing_s"]["start_median"] == 1.0 and r["timing_s"]["end_median"] == -1.0
    assert r["agreement"]["lateral"] == 0.767 and r["agreement"]["longitudinal"] == 1.0


def test_tag_check_clusters_repeated_tags():
    tags = {"s": [(int(t * 1e9), "starting_left_turn") for t in (10.0, 10.05, 10.1, 40.0)]}
    assert tag_events(tags["s"], {"starting_left_turn"}) == {"starting_left_turn": [10.0, 40.0]}
    run = {"s": doc((0, 12, "keep_lane", "cruising"), (12, 18, "turn_left", "cruising"),
                    (18, 60, "keep_lane", "cruising"))}
    assert score_against_tags(tags, run)["starting_left_turn"] == {"events": 2, "found": 1, "share": 0.5}


def test_gpkg_geometry_skips_the_header():
    point = shapely.Point(139.7, 35.6)
    header = b"GP" + bytes([0, 0b00000011]) + struct.pack("<i", 4326) + struct.pack("<4d", 139.7, 139.7, 35.6, 35.6)
    assert gpkg_geometry(header + shapely.to_wkb(point)).equals(point)


def test_turn_direction_from_heading_change():
    left = np.array([[0, 0], [5, 0], [10, 0], [15, 0], [20, 5], [20, 10], [20, 15], [20, 20]], float)
    assert _turn_direction(left) == "left"
    assert _turn_direction(left * [1, -1]) == "right"
    assert _turn_direction(np.array([[0, 0], [5, 0.1], [10, 0], [15, 0.2]], float)) == "straight"


def test_consecutive_slices_become_one_session_name():
    assert session_name(["a/2021.07.16.20.45.29_veh-35_00600_01084.db",
                         "a/2021.07.16.20.45.29_veh-35_01095_01486.db"]) == "2021.07.16.20.45.29_veh-35_00600_01486"


def test_rules_find_the_lane_change(lanemap, lane_change_session):
    tl = build_timeline(lane_change_session, lanemap)
    spans = lateral_spans(tl, lanemap)
    assert [label for _, _, label in spans] == ["lane_change_right"]
    a, b, _ = spans[0]
    assert 1.0 < a < 4.0 < b < 7.0          # the switch is between t=3 s and t=5 s


def test_labels_without_the_indicator_drop_its_requirement(tmp_path):
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "labels.yaml"
    with_ind, without = load_labels(path), load_labels(path, has_indicator=False)
    assert "LEFT indicator" in with_ind.lateral["turn_left"]
    assert "indicator" not in without.lateral["turn_left"] and "[[" not in without.lateral["turn_left"]
    assert "starts moving toward the left road edge" in without.lateral["pull_over"]
    assert without.indicator == {} and with_ind.indicator["turn_left"] == 2


def test_script_says_the_indicator_is_not_recorded(lanemap, lane_change_session):
    from jevsceneminer.script import render

    lane_change_session.has_indicator = False
    text = render(build_timeline(lane_change_session, lanemap), lanemap, T0 + 2 * 10**9)
    assert "indicator during the table: not recorded in this log" in text
    assert "RIGHT" not in text.split("EVENTS")[0]
