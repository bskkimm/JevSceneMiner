"""Turn per-step answers into scenes and write one JSON file per session.

A step's label holds for step/2 on either side of it. Runs shorter than ``min_scene_s``
are merged into a neighbor (flicker removal), maneuvers Jev is unsure about (mean
probability below ``min_prob``) become follow_lane, and a new scene starts whenever the
lateral or longitudinal label changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .jev import Answer


@dataclass(frozen=True)
class Step:
    t_ns: int
    answer: Answer


@dataclass(frozen=True)
class Scene:
    start_ns: int
    end_ns: int
    lateral: str
    longitudinal: str
    lateral_prob: float
    longitudinal_prob: float


def smooth_runs(labels: list[str], min_len: int) -> list[str]:
    """Merge runs shorter than ``min_len`` steps into their longer neighbor, shortest first."""
    runs = []
    for label in labels:
        if runs and runs[-1][0] == label:
            runs[-1][1] += 1
        else:
            runs.append([label, 1])
    while len(runs) > 1:
        short = [i for i, (_, n) in enumerate(runs) if n < min_len]
        if not short:
            break
        i = min(short, key=lambda k: runs[k][1])
        left = runs[i - 1] if i > 0 else None
        right = runs[i + 1] if i + 1 < len(runs) else None
        target = i - 1 if right is None or (left is not None and left[1] >= right[1]) else i + 1
        runs[target][1] += runs[i][1]
        del runs[i]
        merged = []
        for run in runs:
            if merged and merged[-1][0] == run[0]:
                merged[-1][1] += run[1]
            else:
                merged.append(run)
        runs = merged
    return [label for label, n in runs for _ in range(n)]


def _segments(steps: list[Step], step_s: float) -> list[list[Step]]:
    """Split where steps are missing (ego data gaps), so scenes never bridge a gap."""
    out: list[list[Step]] = []
    for s in steps:
        if out and s.t_ns - out[-1][-1].t_ns <= 1.5 * step_s * 1e9:
            out[-1].append(s)
        else:
            out.append([s])
    return out


def demote_unsure(steps: list[Step], lat: list[str], min_prob: float, fallback: str = "follow_lane") -> list[str]:
    """Relabel lateral runs whose mean Jev probability is below ``min_prob`` as ``fallback``."""
    lat = list(lat)
    i = 0
    while i < len(lat):
        j = i
        while j + 1 < len(lat) and lat[j + 1] == lat[i]:
            j += 1
        if lat[i] != fallback:
            mean = sum(s.answer.lateral_probs.get(lat[i], 0.0) for s in steps[i:j + 1]) / (j - i + 1)
            if mean < min_prob:
                lat[i:j + 1] = [fallback] * (j - i + 1)
        i = j + 1
    return lat


# Maneuvers that only exist with the matching indicator (2 = LEFT, 3 = RIGHT), which is
# normally on from a few seconds before the maneuver starts. Pull-over/out are for
# left-hand traffic (pull over to the left); swap them for right-hand traffic.
NEEDS_INDICATOR = {"turn_left": 2, "turn_right": 3, "change_lane_left": 2, "change_lane_right": 3,
                   "pulling_over": 2, "pulling_out": 3}


def gate_by_indicator(steps: list[Step], lat: list[str], segments: list, step_s: float,
                      lead_s: float = 3.0, fallback: str = "follow_lane") -> list[str]:
    """Relabel maneuvers with no matching indicator (on from ``lead_s`` before) as ``fallback``.

    ``segments`` are indicator periods ``[start_s, end_s, 2|3]``. This is a rule on top of
    Jev, so it is off unless asked for.
    """
    lat = list(lat)
    half = step_s / 2
    i = 0
    while i < len(lat):
        j = i
        while j + 1 < len(lat) and lat[j + 1] == lat[i]:
            j += 1
        want = NEEDS_INDICATOR.get(lat[i])
        if want is not None:
            a = steps[i].t_ns / 1e9 - half - lead_s
            b = steps[j].t_ns / 1e9 + half
            if not any(s0 <= b and s1 >= a and st == want for s0, s1, st in segments):
                lat[i:j + 1] = [fallback] * (j - i + 1)
        i = j + 1
    return lat


def stitch(steps: list[Step], step_s: float, min_scene_s: float, start_ns: int, end_ns: int,
           min_prob: float = 0.0, indicator: list | None = None) -> list[Scene]:
    min_len = max(1, int(round(min_scene_s / step_s)))
    half = int(step_s * 1e9 / 2)
    scenes: list[Scene] = []
    for seg in _segments(sorted(steps, key=lambda s: s.t_ns), step_s):
        lat = smooth_runs([s.answer.lateral for s in seg], min_len)
        if min_prob > 0:
            lat = demote_unsure(seg, lat, min_prob)
        if indicator is not None:
            lat = gate_by_indicator(seg, lat, indicator, step_s)
        lon = smooth_runs([s.answer.longitudinal for s in seg], min_len)
        i = 0
        while i < len(seg):
            j = i
            while j + 1 < len(seg) and lat[j + 1] == lat[i] and lon[j + 1] == lon[i]:
                j += 1
            chunk = seg[i:j + 1]
            scenes.append(Scene(
                start_ns=max(start_ns, chunk[0].t_ns - half),
                end_ns=min(end_ns, chunk[-1].t_ns + half),
                lateral=lat[i],
                longitudinal=lon[i],
                lateral_prob=sum(s.answer.lateral_probs.get(lat[i], 0.0) for s in chunk) / len(chunk),
                longitudinal_prob=sum(s.answer.longitudinal_probs.get(lon[i], 0.0) for s in chunk) / len(chunk),
            ))
            i = j + 1
    return scenes


def session_document(date: str, name: str, start_ns: int, end_ns: int, taxonomy_version: str,
                     scenes: list[Scene], generator: dict) -> dict:
    """One session's scenes; times are integer nanoseconds since the epoch, as strings."""
    return {
        "date": date,
        "session": name,
        "start_ns": str(start_ns),
        "end_ns": str(end_ns),
        "taxonomy_version": taxonomy_version,
        "generator": generator,
        "scenes": [{
            "scene_id": f"scene_{k:03d}",
            "start_ns": str(sc.start_ns),
            "end_ns": str(sc.end_ns),
            "lateral": sc.lateral,
            "longitudinal": sc.longitudinal,
            "lateral_prob": round(sc.lateral_prob, 3),
            "longitudinal_prob": round(sc.longitudinal_prob, 3),
        } for k, sc in enumerate(scenes, start=1)],
    }


def write_scenes(out_dir: Path, doc: dict) -> Path:
    """Write ``<out_dir>/scenes/<date>_<session>.json``."""
    path = Path(out_dir) / "scenes" / f"{doc['date']}_{doc['session']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    return path
