from conftest import T0, seconds
from jevsceneminer.jev import Answer
from jevsceneminer.scenes import Step, demote_unsure, gate_by_indicator, session_document, smooth_runs, stitch, write_scenes


def answer(lat, lon="cruising", p=0.8):
    return Answer(lat, {lat: p, "keep_lane": 1 - p} if lat != "keep_lane" else {lat: p}, lon, {lon: 1.0})


def test_smooth_runs_removes_flicker():
    assert smooth_runs(["a", "a", "b", "a", "a"], 2) == ["a"] * 5
    assert smooth_runs(["a"] * 4 + ["b"] * 4, 2) == ["a"] * 4 + ["b"] * 4
    assert len(set(smooth_runs(["a", "b"], 2))) == 1
    assert smooth_runs(["a", "a", "a", "b", "c", "c", "c", "c"], 2) == ["a"] * 3 + ["c"] * 5


def test_stitch_splits_on_label_change_with_half_step_edges():
    labels = ["keep_lane"] * 4 + ["turn_left"] * 4 + ["keep_lane"] * 4
    steps = [Step(seconds(0.5 * i), answer(lab)) for i, lab in enumerate(labels)]
    scenes = stitch(steps, 0.5, 1.0, start_ns=T0, end_ns=seconds(10))
    assert [s.lateral for s in scenes] == ["keep_lane", "turn_left", "keep_lane"]
    assert scenes[0].start_ns == T0                      # clipped to the session start
    assert scenes[1].start_ns == seconds(1.75) and scenes[1].end_ns == seconds(3.75)
    assert abs(scenes[1].lateral_prob - 0.8) < 1e-9


def test_stitch_never_bridges_a_data_gap():
    times = [0, 0.5, 1.0, 1.5, 3.0, 3.5, 4.0, 4.5]
    steps = [Step(seconds(t), answer("keep_lane")) for t in times]
    scenes = stitch(steps, 0.5, 1.0, start_ns=T0, end_ns=seconds(5))
    assert len(scenes) == 2
    assert scenes[0].end_ns == seconds(1.75) and scenes[1].start_ns == seconds(2.75)


def test_stitch_nests_longitudinal_changes():
    steps = [Step(seconds(0.5 * i), answer("turn_left", lon)) for i, lon in
             enumerate(["decelerating"] * 3 + ["accelerating"] * 3)]
    scenes = stitch(steps, 0.5, 1.0, start_ns=T0, end_ns=seconds(5))
    assert len(scenes) == 1
    assert [(p.end_ns, p.decision) for p in scenes[0].longitudinal_phases] == [(seconds(1.25), "decelerating"), (seconds(2.75), "accelerating")]


def test_session_document_keeps_exact_times(tmp_path):
    steps = [Step(seconds(0.5 * i), answer("turn_left")) for i in range(4)]
    doc = session_document("2026-01-07", "drive1", T0, seconds(5), "jsm-1.0",
                           stitch(steps, 0.5, 1.0, T0, seconds(5)), {"name": "jevsceneminer"})
    (scene,) = doc["scenes"]
    assert set(scene) == {"start_ns", "end_ns", "driving_decision", "longitudinal_phases"}
    assert scene["start_ns"] == str(T0) and int(scene["end_ns"]) == seconds(1.75)
    assert scene["driving_decision"] == "turn_left"
    assert scene["longitudinal_phases"] == [{"end_ns": str(seconds(1.75)), "decision": "cruising"}]
    assert write_scenes(tmp_path, doc) == tmp_path / "scenes" / "2026-01-07_drive1.json"


def test_unsure_maneuvers_become_keep_lane():
    steps = [Step(seconds(0.5 * i), answer(lab, p=p)) for i, (lab, p) in enumerate(
        [("keep_lane", 1.0)] * 2 + [("turn_left", 0.3)] * 4 + [("lane_change_left", 0.9)] * 4)]
    lat = demote_unsure(steps, [s.answer.lateral for s in steps], 0.4)
    assert lat == ["keep_lane"] * 6 + ["lane_change_left"] * 4
    scenes = stitch(steps, 0.5, 1.0, T0, seconds(10), min_prob=0.4)
    assert [s.lateral for s in scenes] == ["keep_lane", "lane_change_left"]


def test_indicator_gate_keeps_only_signalled_maneuvers():
    labels = ["turn_left"] * 4 + ["keep_lane"] * 4 + ["lane_change_right"] * 4
    steps = [Step(seconds(0.5 * i), answer(lab)) for i, lab in enumerate(labels)]
    t = lambda x: (T0 / 1e9) + x   # noqa: E731
    left_only = [[t(-1.0), t(1.0), 2]]          # LEFT on around the turn; no RIGHT at all
    needs = {"turn_left": 2, "lane_change_right": 3}
    gated = gate_by_indicator(steps, labels, left_only, 0.5, needs)
    assert gated == ["turn_left"] * 4 + ["keep_lane"] * 8
    assert gate_by_indicator(steps, labels, [], 0.5, needs) == ["keep_lane"] * 12


def test_phase_smoothing_has_its_own_duration():
    steps = [Step(seconds(0.5 * i), answer("turn_left", lon)) for i, lon in
             enumerate(["cruising"] * 4 + ["stopped"] * 2 + ["cruising"] * 4)]
    scene, = stitch(steps, 0.5, 2.0, T0, seconds(5))
    assert [p.decision for p in scene.longitudinal_phases] == ["cruising", "stopped", "cruising"]
    scene, = stitch(steps, 0.5, 2.0, T0, seconds(5), min_phase_s=2.0)
    assert [p.decision for p in scene.longitudinal_phases] == ["cruising"]


def test_missing_longitudinal_is_unknown():
    steps = [Step(seconds(0.5 * i), Answer("keep_lane", {"keep_lane": 1.0}, None, {})) for i in range(5)]
    scene, = stitch(steps, 0.5, 2.0, seconds(0.5), seconds(1.5))
    assert [(p.end_ns, p.decision) for p in scene.longitudinal_phases] == [(seconds(1.5), "unknown")]


def test_scene_rejects_nonincreasing_or_incomplete_phases():
    import pytest
    from jevsceneminer.scenes import Scene, LongitudinalPhase
    for ends in [(T0, seconds(2)), (seconds(2), seconds(1)), (seconds(1),)]:
        with pytest.raises(ValueError, match="phase"):
            Scene(T0, seconds(2), "keep_lane", "cruising", 1.0, 1.0,
                  tuple(LongitudinalPhase(end, "cruising") for end in ends))


def test_document_validation_checks_nested_shape_and_phase_bounds():
    import pytest
    import jevsceneminer.scenes as module
    valid = {"scenes": [{"start_ns": "100", "end_ns": "300", "driving_decision": "keep_lane",
                        "longitudinal_phases": [{"end_ns": "200", "decision": "cruising"},
                                                {"end_ns": "300", "decision": "stopped"}]}]}
    module.validate_scene_document(valid)
    from copy import deepcopy
    for field, value in [("start_ns", "300"), ("driving_decision", ""), ("longitudinal_phases", []),
                         ("longitudinal_phases", [{"end_ns": "200", "decision": "cruising"}]),
                         ("longitudinal_phases", [{"end_ns": "100", "decision": "unknown"}]),
                         ("longitudinal_phases", [{"end_ns": "300", "decision": ""}])]:
        invalid = deepcopy(valid)
        invalid["scenes"][0][field] = value
        with pytest.raises(ValueError):
            module.validate_scene_document(invalid)
    with pytest.raises(ValueError):
        module.validate_scene_document({"scenes": [{"start_ns": "100", "end_ns": "300", "lateral": "keep_lane"}]})


def test_restitch_uses_saved_answers_and_exposes_phase_duration(tmp_path):
    import json
    from jevsceneminer.cli import main
    source, destination = tmp_path / "source", tmp_path / "nested"
    (source / "steps").mkdir(parents=True)
    (source / "meta").mkdir()
    sid = "2026-01-07_drive1"
    meta = {"date": "2026-01-07", "name": "drive1", "start_ns": T0,
            "end_ns": seconds(5), "step_s": 0.5}
    (source / "meta" / f"{sid}.json").write_text(json.dumps(meta))
    rows = [{"t_ns": seconds(0.5 * i), "lateral": "keep_lane", "lateral_probs": {"keep_lane": 1.0}}
            for i in range(10)]
    (source / "steps" / f"{sid}.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
    assert main(["restitch", str(source), "--out", str(destination), "--min-phase", "1.5"]) == 0
    document = json.loads((destination / "scenes" / f"{sid}.json").read_text())
    assert document["generator"]["min_phase_s"] == 1.5
    assert document["scenes"][0]["longitudinal_phases"] == [{"end_ns": str(seconds(4.75)), "decision": "unknown"}]


def test_phase_smoothing_preserves_missing_longitudinal_boundaries():
    labels = ["cruising", "cruising", None, "cruising", "cruising"]
    steps = [Step(seconds(0.5 * i), Answer("keep_lane", {"keep_lane": 1.0}, label,
                                          {label: 1.0} if label else {})) for i, label in enumerate(labels)]
    scene, = stitch(steps, 0.5, 2.0, T0, seconds(3), min_phase_s=2.0)
    assert [(p.end_ns, p.decision) for p in scene.longitudinal_phases] == [
        (seconds(0.75), "cruising"), (seconds(1.25), "unknown"), (seconds(2.25), "cruising")]
