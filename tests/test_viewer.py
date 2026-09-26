import json

import pytest

from jevsceneminer.viewer import Site


@pytest.fixture()
def out(tmp_path):
    root = tmp_path / "out"
    for sub in ("meta", "steps", "scenes", "camera", "tags"):
        (root / sub).mkdir(parents=True)
    (root / "meta" / "s1.json").write_text(json.dumps({"date": "2021-01-01", "name": "s1", "start_ns": 0,
                                                      "end_ns": 10**10}))
    (root / "steps" / "s1.jsonl").write_text("\n".join(json.dumps({"t_ns": t * 10**9, "script": f"script at {t}"})
                                                       for t in range(10)))
    (root / "scenes" / "s1.json").write_text(json.dumps({"scenes": []}))
    (root / "camera" / "s1.json").write_text(json.dumps([[0, "log/CAM_F0/a.jpg"], [10**9, "../../etc/passwd"]]))
    sensors = tmp_path / "sensors" / "log" / "CAM_F0"
    sensors.mkdir(parents=True)
    (sensors / "a.jpg").write_bytes(b"jpg")
    return root


def test_session_script_and_camera(out):
    site = Site(out, None, out.parent / "sensors", {"jev": out / "scenes"})
    s = site.session("s1")
    assert len(s["steps"]) == 10 and "script" not in s["steps"][0]
    assert s["runs"]["jev"] == {"scenes": []} and s["gt"] is None and not s["gt_writable"]
    assert site.script("s1", int(3.4e9)) == "script at 3"
    assert site.camera_file("s1", 0).read_bytes() == b"jpg"
    assert site.camera_file("s1", 1) is None          # outside the sensor root
    assert site.bev("s1")["available"] is False        # not a nuPlan session


def test_gt_is_saved_only_with_a_gt_folder(out, tmp_path):
    with pytest.raises(PermissionError):
        Site(out, None, None, {}).save_gt("s1", {})
    path = Site(out, tmp_path / "gt", None, {}).save_gt("s1", {"scenes": []})
    assert json.loads(path.read_text()) == {"scenes": []}
