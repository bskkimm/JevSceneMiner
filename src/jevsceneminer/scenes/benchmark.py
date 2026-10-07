"""Reproducible offline scoring of explicitly listed, hash-locked scene files."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from pathlib import Path

from jevsceneminer import __version__
from . import evaluate
from .merge import validate_scene_document

SCHEMA = "jsm-benchmark-1"
KINDS = {"synthetic", "development", "held_out"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _timestamp(value):
    _require(isinstance(value, str) and value.isascii() and value.isdigit(),
             "benchmark timestamps must be nanosecond integer strings")
    return int(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"non-finite JSON number: {value}")


def _read_json(path):
    try:
        data = path.read_bytes()
        doc = json.loads(data, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError(f"cannot read {path.name}: {exc}") from exc
    _require(isinstance(doc, dict), f"{path.name}: expected a JSON object")
    return data, doc


def _frozen_document(root, ref, start, end):
    _require(isinstance(ref, dict) and _text(ref.get("path")), "input requires path and sha256")
    expected = ref.get("sha256")
    _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
             "input sha256 must contain 64 lowercase hexadecimal characters")
    path = (root / Path(ref["path"]).expanduser()).resolve()
    data, doc = _read_json(path)
    _require(hashlib.sha256(data).hexdigest() == expected, f"hash mismatch: {ref['path']}")
    try:
        validate_scene_document(doc)
        _require(bool(doc["scenes"]), "empty scenes")
        cursor = start
        for scene in doc["scenes"]:
            a, b = _timestamp(scene["start_ns"]), _timestamp(scene["end_ns"])
            _require(a == cursor and b <= end, "scenes must cover the declared interval without gaps or overlaps")
            decisions = [scene["driving_decision"]] + [p["decision"] for p in scene["longitudinal_phases"]]
            _require(all(label.strip().lower() not in {"unknown", "(none)", "unlabeled", "unlabelled", ""}
                         for label in decisions),
                     "all assessed lateral and longitudinal decisions must be labeled")
            cursor = b
        _require(cursor == end, "scenes must cover the entire declared interval")
    except (TypeError, KeyError, ValueError) as exc:
        raise ValueError(f"{ref['path']}: {exc}") from exc
    return path, doc


def _relative_document(doc, origin):
    # Score with small relative seconds, avoiding loss of precision at epoch times.
    scenes = []
    for scene in doc["scenes"]:
        scenes.append({**scene, "start_ns": str(int(scene["start_ns"]) - origin),
                       "end_ns": str(int(scene["end_ns"]) - origin),
                       "longitudinal_phases": [
                           {**phase, "end_ns": str(int(phase["end_ns"]) - origin)}
                           for phase in scene["longitudinal_phases"]]})
    return {"scenes": scenes}


def _atomic_write(path, contents):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(contents)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_benchmark(manifest_path: Path, out: Path) -> dict:
    """Validate every frozen input, then write deterministic JSON/Markdown reports.

    No sessions are discovered, skipped or filled with inferred keep-lane labels.
    All ground truth and candidates must label exactly the declared interval.
    """
    manifest_path = Path(manifest_path).resolve()
    data, manifest = _read_json(manifest_path)
    _require(manifest.get("schema_version") == SCHEMA, f"expected {SCHEMA}")
    _require(_text(manifest.get("name")), "benchmark requires a name")
    _require(isinstance(manifest.get("kind"), str) and manifest["kind"] in KINDS,
             "kind must be synthetic, development or held_out")
    _require(_text(manifest.get("provenance")), "benchmark requires ground-truth provenance")
    configurations = manifest.get("run_configuration")
    _require(isinstance(configurations, dict) and bool(configurations), "run_configuration must list every run")
    _require(all(name != "gt" and re.fullmatch(r"[A-Za-z0-9_.-]+", name) and isinstance(config, dict)
                 for name, config in configurations.items()), "runs require simple names and configuration objects")
    names = sorted(configurations)
    sessions = manifest.get("sessions")
    _require(isinstance(sessions, list) and bool(sessions), "benchmark must explicitly list sessions")
    gt, runs, inputs = {}, {name: {} for name in names}, []
    protected = {manifest_path}
    seen_files = {}
    for session in sessions:
        _require(isinstance(session, dict) and _text(session.get("id")), "session requires an id")
        sid = session["id"]
        _require(sid not in gt, f"duplicate session id: {sid}")
        start, end = _timestamp(session.get("start_ns")), _timestamp(session.get("end_ns"))
        _require(start < end, "session start must precede end")
        candidates = session.get("runs")
        _require(isinstance(candidates, dict) and set(candidates) == set(names),
                 f"{sid}: every session must contain every declared run")
        for name, ref in [("gt", session.get("gt")), *sorted(candidates.items())]:
            path, doc = _frozen_document(manifest_path.parent, ref, start, end)
            _require(path not in seen_files or seen_files[path] == sid,
                     f"{sid}: file reused across sessions: {ref['path']}")
            seen_files[path] = sid
            protected.add(path)
            target = gt if name == "gt" else runs[name]
            target[sid] = _relative_document(doc, start)
        inputs.append({"id": sid, "start_ns": session["start_ns"], "end_ns": session["end_ns"],
                       "gt": session["gt"], "runs": candidates})
    out = Path(out).resolve()
    for filename in ("report.json", "report.md"):
        _require((out / filename).resolve() not in protected, "report would overwrite a frozen input or manifest")
    report = {
        "schema_version": SCHEMA, "name": manifest["name"], "kind": manifest["kind"],
        "provenance": manifest["provenance"], "manifest_sha256": hashlib.sha256(data).hexdigest(),
        "tool": {"version": __version__,
                 "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "scorer_sha256": hashlib.sha256(Path(evaluate.__file__).read_bytes()).hexdigest()},
        "scoring": {"minimum_overlap_s": evaluate.MIN_OVERLAP_S, "agreement_grid_s": evaluate.GRID_S,
                    "coverage": "complete declared intervals; no inferred labels"},
        "run_configuration": configurations, "inputs": sorted(inputs, key=lambda entry: entry["id"]),
        "runs": names, "gt": {name: evaluate.score_against_gt(gt, runs[name]) for name in names},
    }
    header = (f"# {manifest['name']}\n\n"
              f"Evaluation kind: **{manifest['kind']}**. {manifest['provenance']}\n\n"
              "Synthetic or development scores do not measure held-out Jev accuracy.\n\n"
              f"Manifest SHA-256: `{report['manifest_sha256']}`. Input hashes and run settings are in `report.json`.\n\n")
    markdown = header + evaluate.render_markdown(report).removeprefix("# JevSceneMiner report\n\n")
    out.mkdir(parents=True, exist_ok=True)
    _atomic_write(out / "report.json", json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    _atomic_write(out / "report.md", markdown)
    return report
