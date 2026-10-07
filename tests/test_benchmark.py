"""Frozen saved-result scoring must fail clearly rather than omit bad inputs."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from jevsceneminer.scenes.benchmark import run_benchmark

T0 = 1_767_000_000_000_000_000


def stamp(seconds):
    return str(T0 + int(seconds * 1e9))


def document(change_start=3):
    return {"scenes": [
        {"start_ns": stamp(a), "end_ns": stamp(b), "driving_decision": label,
         "longitudinal_phases": [{"end_ns": stamp(b), "decision": "cruising"}]}
        for a, b, label in [(0, change_start, "keep_lane"),
                            (change_start, 7, "lane_change_right"), (7, 12, "keep_lane")]
    ]}


def frozen_file(root, name, doc):
    path = root / name
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return {"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def fixture(tmp_path):
    manifest = {"schema_version": "jsm-benchmark-1", "name": "Synthetic contract",
                "kind": "synthetic", "provenance": "Hand-written synthetic intervals; no Jev.",
                "run_configuration": {"delayed": {"source": "synthetic", "start_delay_s": 1}},
                "sessions": [{"id": "synthetic", "start_ns": stamp(0), "end_ns": stamp(12),
                              "gt": frozen_file(tmp_path, "gt.json", document()),
                              "runs": {"delayed": frozen_file(tmp_path, "delayed.json", document(4))}}]}
    path = tmp_path / "manifest.json"
    return path, manifest


def execute(fixture, out=None):
    path, manifest = fixture
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return run_benchmark(path, out or path.parent / "results")


def test_known_delay_and_deterministic_reports_without_api(fixture, monkeypatch):
    from jevsceneminer.inference.jev import JevClassifier
    def forbidden(*args, **kwargs):
        raise AssertionError("Benchmark must not instantiate Jev")
    monkeypatch.setattr(JevClassifier, "__init__", forbidden)
    report = execute(fixture)
    score = report["gt"]["delayed"]
    assert score["total"] == {"TP": 1, "FN": 0, "FP": 0, "precision": 1.0, "recall": 1.0}
    assert score["timing_s"]["start_median"] == 1.0
    assert score["agreement"] == {"lateral": 0.917, "longitudinal": 1.0}
    assert report["kind"] == "synthetic"
    assert report["inputs"][0]["gt"]["sha256"] == fixture[1]["sessions"][0]["gt"]["sha256"]
    first = (fixture[0].parent / "results/report.json").read_bytes()
    assert execute(fixture) == report
    assert (fixture[0].parent / "results/report.json").read_bytes() == first
    assert "synthetic" in (fixture[0].parent / "results/report.md").read_text().lower()


@pytest.mark.parametrize("duration", [0.55, 1.55])
def test_identical_complete_half_grid_intervals_have_full_agreement(fixture, duration):
    path, manifest = fixture
    doc = {"scenes": [{"start_ns": stamp(0), "end_ns": stamp(duration),
                        "driving_decision": "keep_lane",
                        "longitudinal_phases": [{"end_ns": stamp(duration), "decision": "cruising"}]}]}
    session = manifest["sessions"][0]
    session["end_ns"] = stamp(duration)
    session["gt"] = frozen_file(path.parent, "gt.json", doc)
    session["runs"]["delayed"] = frozen_file(path.parent, "delayed.json", doc)
    result = execute(fixture)["gt"]["delayed"]
    assert result["agreement"] == {"lateral": 1.0, "longitudinal": 1.0}
    assert result["confusion_lateral_s"] == {}


@pytest.mark.parametrize(("overlap_ns", "matched"), [(500_000_000, 1), (499_999_999, 0)])
def test_maneuver_match_threshold_uses_nanosecond_precision(fixture, overlap_ns, matched):
    path, manifest = fixture
    end = str(T0 + 200_000_000 + overlap_ns)
    doc = {"scenes": [
        {"start_ns": a, "end_ns": b, "driving_decision": label,
         "longitudinal_phases": [{"end_ns": b, "decision": "cruising"}]}
        for a, b, label in [(stamp(0), stamp(0.2), "keep_lane"),
                            (stamp(0.2), end, "lane_change_right"), (end, stamp(1), "keep_lane")]
    ]}
    session = manifest["sessions"][0]
    session["end_ns"] = stamp(1)
    session["gt"] = frozen_file(path.parent, "gt.json", doc)
    session["runs"]["delayed"] = frozen_file(path.parent, "delayed.json", doc)
    assert execute(fixture)["gt"]["delayed"]["total"]["TP"] == matched


@pytest.mark.parametrize("mutation", ["missing", "hash", "duplicate", "run_missing"])
def test_input_failure_is_not_silently_skipped(fixture, mutation):
    path, manifest = fixture
    session = manifest["sessions"][0]
    if mutation == "missing":
        (path.parent / "gt.json").unlink()
    elif mutation == "hash":
        (path.parent / "delayed.json").write_text("{}")
    elif mutation == "duplicate":
        manifest["sessions"].append(deepcopy(session))
    else:
        session["runs"] = {}
    with pytest.raises(ValueError):
        execute(fixture)
    assert not (path.parent / "results").exists()


@pytest.mark.parametrize(("source", "defect"), [
    ("gt", "gap"), ("gt", "overlap"), ("gt", "phase"),
    ("gt", "empty"), ("gt", "unknown"), ("delayed", "truncated"),
])
def test_incomplete_or_invalid_coverage_is_rejected(fixture, source, defect):
    path, manifest = fixture
    doc = document(4 if source == "delayed" else 3)
    if defect == "gap":
        doc["scenes"][1]["start_ns"] = stamp(4)
    elif defect == "overlap":
        doc["scenes"][1]["start_ns"] = stamp(2)
    elif defect == "phase":
        doc["scenes"][0]["longitudinal_phases"][0]["end_ns"] = stamp(2)
    elif defect == "empty":
        doc["scenes"] = []
    elif defect == "unknown":
        doc["scenes"][0]["driving_decision"] = "unknown"
    else:
        doc["scenes"].pop()
    entry = frozen_file(path.parent, source + ".json", doc)
    if source == "gt":
        manifest["sessions"][0]["gt"] = entry
    else:
        manifest["sessions"][0]["runs"][source] = entry
    with pytest.raises(ValueError):
        execute(fixture)
    assert not (path.parent / "results").exists()


def test_output_cannot_overwrite_a_frozen_input(fixture):
    path, manifest = fixture
    gt = path.parent / "gt.json"
    protected = path.parent / "report.json"
    gt.rename(protected)
    manifest["sessions"][0]["gt"]["path"] = protected.name
    original = protected.read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        execute(fixture, path.parent)
    assert protected.read_bytes() == original


@pytest.mark.parametrize("defect", ["different_runs", "reused_file", "cross_role_reuse",
                                  "reserved_name", "unknown_phase", "unlabeled"])
def test_manifest_integrity_across_runs_and_sessions(fixture, defect):
    path, manifest = fixture
    first = manifest["sessions"][0]
    if defect in {"different_runs", "reused_file", "cross_role_reuse"}:
        second = deepcopy(first)
        second["id"] = "second"
        if defect == "different_runs":
            second["runs"] = {}
        elif defect == "cross_role_reuse":
            second["gt"] = deepcopy(first["runs"]["delayed"])
            second["runs"]["delayed"] = deepcopy(first["gt"])
        manifest["sessions"].append(second)
    elif defect == "reserved_name":
        manifest["run_configuration"] = {"gt": {}}
        first["runs"] = {"gt": first["gt"]}
    else:
        doc = document()
        doc["scenes"][0]["longitudinal_phases"][0]["decision"] = (
            "Unlabeled" if defect == "unlabeled" else "unknown")
        first["gt"] = frozen_file(path.parent, "gt.json", doc)
    with pytest.raises(ValueError):
        execute(fixture)
    assert not (path.parent / "results").exists()


def test_cli_reports_invalid_input_without_writing(fixture, monkeypatch, capsys):
    from jevsceneminer import cli
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    path, manifest = fixture
    manifest["sessions"][0]["gt"]["sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    assert cli.main(["benchmark", str(path), "--out", str(path.parent / "results")]) == 2
    assert "hash mismatch" in capsys.readouterr().err
    assert not (path.parent / "results").exists()


def test_cli_scores_checked_in_contract_fixture(tmp_path, monkeypatch):
    from jevsceneminer import cli
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    manifest = Path(__file__).resolve().parents[1] / "benchmarks/synthetic/manifest.json"
    assert cli.main(["benchmark", str(manifest), "--out", str(tmp_path / "report")]) == 0
    assert json.loads((tmp_path / "report/report.json").read_text())["kind"] == "synthetic"
