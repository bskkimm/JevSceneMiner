"""Import compatibility and packaged resource contracts after source grouping."""
import importlib
import subprocess
import sys
from unittest.mock import patch

import pytest


@pytest.mark.parametrize(("legacy", "canonical"), [
    ("bag", "inputs.bag"), ("nuplan", "inputs.nuplan"),
    ("nuplan_map", "inputs.nuplan_map"), ("lanes", "evidence.lanes"),
    ("facts", "evidence.facts"), ("script", "evidence.script"),
    ("sample_context", "evidence.sample_context"),
    ("lane_evidence", "evidence.lane_evidence"),
    ("lane_review", "evidence.lane_review"),
    ("moving_history", "evidence.moving_history"),
    ("compact", "evidence.compact"), ("jev", "inference.jev"),
    ("evaluate", "scenes.evaluate"), ("rules", "scenes.rules"),
    ("camera_projection", "viewer.camera_projection"),
    ("height_reference", "viewer.height_reference"),
    ("temporal_height", "viewer.temporal_height"),
])
def test_old_imports_share_the_canonical_module(legacy, canonical):
    old = importlib.import_module("jevsceneminer." + legacy)
    try:
        new = importlib.import_module("jevsceneminer." + canonical)
    except ModuleNotFoundError:
        pytest.fail("Grouped source import is unavailable: " + canonical)
    assert old is new  # Patches and source types must not fork between paths.


def test_legacy_viewer_page_override_reaches_server(monkeypatch, tmp_path):
    import jevsceneminer.viewer as viewer
    try:
        from jevsceneminer.viewer import server
    except ImportError:
        pytest.fail("Viewer server package is unavailable")
    replacement = tmp_path / "index.html"
    monkeypatch.setattr(viewer, "PAGE", replacement)
    assert server.PAGE == replacement
    assert viewer.Site is server.Site
    assert viewer.serve is server.serve


def test_scene_api_keeps_the_same_classes():
    import jevsceneminer.scenes as scenes
    try:
        from jevsceneminer.scenes import merge
    except ImportError:
        pytest.fail("Scene merge package is unavailable")
    assert scenes.Step is merge.Step
    assert scenes.Scene is merge.Scene
    assert scenes.stitch is merge.stitch


@pytest.mark.parametrize(("package_name", "module_name", "attribute"), [
    ("viewer", "viewer.server", "PAGE"), ("scenes", "scenes.merge", "Step"),
])
def test_temporary_legacy_override_restores_canonical_attribute(package_name, module_name, attribute):
    package = importlib.import_module("jevsceneminer." + package_name)
    implementation = importlib.import_module("jevsceneminer." + module_name)
    original = getattr(implementation, attribute)
    replacement = object()
    try:
        with patch.object(package, attribute, replacement):
            assert getattr(implementation, attribute) is replacement
        assert getattr(implementation, attribute) is original
        assert getattr(package, attribute) is original
    finally:
        setattr(implementation, attribute, original)


def test_compact_module_entry_points_remain_usable():
    for module in ("jevsceneminer.compact", "jevsceneminer.evidence.compact"):
        result = subprocess.run([sys.executable, "-m", module, "--help"],
                                capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert "--out" in result.stdout


def test_cli_finds_source_labels_and_viewer_assets():
    from jevsceneminer.cli import DEFAULT_LABELS
    from jevsceneminer.viewer import PAGE
    assert DEFAULT_LABELS.is_file()
    assert PAGE.is_file()
    assert all(PAGE.with_name(name).is_file() for name in
               ("bev_geometry.js", "camera_decisions.js", "editor.js"))
