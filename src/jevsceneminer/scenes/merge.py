"""Turn per-step answers into scenes and write one JSON file per session.

A step's label holds for step/2 on either side of it. Runs shorter than ``min_scene_s``
are merged into a neighbor (flicker removal), maneuvers Jev is unsure about (mean
probability below ``min_prob``) become keep_lane, and a new scene starts whenever the
lateral label changes. Longitudinal changes become phases within each scene.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
import math
from pathlib import Path

from jevsceneminer.inference.jev import Answer


@dataclass(frozen=True)
class Step:
    t_ns: int
    answer: Answer


@dataclass(frozen=True)
class LongitudinalPhase:
    end_ns: int
    decision: str


@dataclass(frozen=True)
class Scene:
    start_ns: int
    end_ns: int
    lateral: str
    longitudinal: str
    lateral_prob: float
    longitudinal_prob: float
    longitudinal_phases: tuple[LongitudinalPhase, ...] = ()

    def __post_init__(self):
        previous = self.start_ns
        for phase in self.longitudinal_phases:
            if not previous < phase.end_ns <= self.end_ns:
                raise ValueError("phase ends must increase within the scene")
            previous = phase.end_ns
        if self.longitudinal_phases and previous != self.end_ns:
            raise ValueError("final phase must end at the scene end")


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


def demote_unsure(steps: list[Step], lat: list[str], min_prob: float, fallback: str = "keep_lane") -> list[str]:
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


def gate_by_indicator(steps: list[Step], lat: list[str], segments: list, step_s: float, needs: dict[str, int],
                      lead_s: float = 3.0, fallback: str = "keep_lane") -> list[str]:
    """Relabel maneuvers with no matching indicator (on from ``lead_s`` before) as ``fallback``.

    ``segments`` are indicator periods ``[start_s, end_s, 2|3]``; ``needs`` maps a label to
    the indicator it requires (``Labels.indicator``). This is a rule on top of Jev, so it
    is off unless asked for.
    """
    lat = list(lat)
    half = step_s / 2
    i = 0
    while i < len(lat):
        j = i
        while j + 1 < len(lat) and lat[j + 1] == lat[i]:
            j += 1
        want = needs.get(lat[i])
        if want is not None:
            a = steps[i].t_ns / 1e9 - half - lead_s
            b = steps[j].t_ns / 1e9 + half
            if not any(s0 <= b and s1 >= a and st == want for s0, s1, st in segments):
                lat[i:j + 1] = [fallback] * (j - i + 1)
        i = j + 1
    return lat


def stitch(steps: list[Step], step_s: float, min_scene_s: float, start_ns: int, end_ns: int,
           min_prob: float = 0.0, indicator: list | None = None, needs: dict[str, int] | None = None,
           min_phase_s: float = 1.0) -> list[Scene]:
    min_len = max(1, int(round(min_scene_s / step_s)))
    phase_min_len = max(1, int(round(min_phase_s / step_s)))
    half = int(step_s * 1e9 / 2)
    scenes: list[Scene] = []
    for seg in _segments(sorted(steps, key=lambda s: s.t_ns), step_s):
        lat = smooth_runs([s.answer.lateral for s in seg], min_len)
        if min_prob > 0:
            lat = demote_unsure(seg, lat, min_prob)
        if indicator is not None:
            lat = gate_by_indicator(seg, lat, indicator, step_s, needs or {})
        lon = [s.answer.longitudinal or "unknown" for s in seg]
        i = 0
        while i < len(seg):
            j = i
            while j + 1 < len(seg) and lat[j + 1] == lat[i]:
                j += 1
            chunk = seg[i:j + 1]
            a = max(start_ns, chunk[0].t_ns - half)
            b = min(end_ns, chunk[-1].t_ns + half)
            if a >= b:
                i = j + 1
                continue
            phase_labels = lon[i:j + 1]
            known_start = 0
            while known_start < len(phase_labels):
                if phase_labels[known_start] == "unknown":
                    known_start += 1
                    continue
                known_end = known_start
                while known_end < len(phase_labels) and phase_labels[known_end] != "unknown":
                    known_end += 1
                phase_labels[known_start:known_end] = smooth_runs(
                    phase_labels[known_start:known_end], phase_min_len)
                known_start = known_end
            phases = []
            k = 0
            phase_start = a
            while k < len(chunk):
                last = k
                while last + 1 < len(chunk) and phase_labels[last + 1] == phase_labels[k]:
                    last += 1
                phase_end = min(b, chunk[last].t_ns + half)
                if phase_end > phase_start:
                    phases.append(LongitudinalPhase(phase_end, phase_labels[k]))
                    phase_start = phase_end
                k = last + 1
            scenes.append(Scene(
                start_ns=a,
                end_ns=b,
                lateral=lat[i],
                longitudinal=phases[0].decision,
                lateral_prob=sum(s.answer.lateral_probs.get(lat[i], 0.0) for s in chunk) / len(chunk),
                longitudinal_prob=sum(s.answer.longitudinal_probs.get(lon[i], 0.0) for s in chunk) / len(chunk),
                longitudinal_phases=tuple(phases),
            ))
            i = j + 1
    return scenes


def extend_maneuver_ends(scenes, tail_s, protected_intervals=()):
    """Borrow up to ``tail_s`` from an immediately following keep_lane scene.

    Never bridge a gap or enter protected lane-review intervals. Longitudinal
    phases come from the same time spans in the existing merged output. Raw
    answers remain untouched; support fields are retained from the original
    scene and are not probabilities for the padded boundary.
    """
    if not math.isfinite(tail_s) or tail_s < 0:
        raise ValueError('maneuver tail must be finite and nonnegative')
    if tail_s == 0:
        return list(scenes)
    tail_ns = round(tail_s * 1e9)
    work = list(scenes)

    def phases_in(scene, start, end):
        phases = scene.longitudinal_phases or (LongitudinalPhase(scene.end_ns,scene.longitudinal or 'unknown'),)
        previous = scene.start_ns
        result = []
        for phase in phases:
            if min(phase.end_ns,end) > max(previous,start):
                result.append(LongitudinalPhase(min(phase.end_ns,end),phase.decision))
            previous = phase.end_ns
        return result

    def join(phases):
        result = []
        for phase in phases:
            if result and result[-1].decision == phase.decision:
                result[-1] = phase
            else:
                result.append(phase)
        return tuple(result)

    for i,scene in enumerate(work[:-1]):
        following = work[i+1]
        if scene is None or following is None or scene.lateral in ('keep_lane','unknown','unlabeled') or following.lateral != 'keep_lane':
            continue
        if scene.end_ns != following.start_ns:
            continue
        end = min(scene.end_ns + tail_ns,following.end_ns)
        for lo,hi in protected_intervals:
            if lo <= scene.end_ns < hi:
                end = scene.end_ns
            elif scene.end_ns < lo < end:
                end = lo
        if end <= scene.end_ns:
            continue
        phases = join([*phases_in(scene,scene.start_ns,scene.end_ns),
                       *phases_in(following,following.start_ns,end)])
        work[i] = replace(scene,end_ns=end,longitudinal_phases=phases)
        work[i+1] = (replace(following,start_ns=end,longitudinal=phases_in(following,end,following.end_ns)[0].decision,
                            longitudinal_phases=tuple(phases_in(following,end,following.end_ns)))
                     if end < following.end_ns else None)
    result = []
    for scene in work:
        if scene is None:
            continue
        if result and result[-1].lateral == scene.lateral and result[-1].end_ns == scene.start_ns:
            previous = result[-1]
            result[-1] = replace(previous,end_ns=scene.end_ns,
                longitudinal_phases=join([*phases_in(previous,previous.start_ns,previous.end_ns),
                                          *phases_in(scene,scene.start_ns,scene.end_ns)]))
        else:
            result.append(scene)
    return result


def session_document(date: str, name: str, start_ns: int, end_ns: int, labels_version: str,
                     scenes: list[Scene], generator: dict) -> dict:
    """One session's scenes; times are integer nanoseconds since the epoch, as strings."""
    return {
        "date": date,
        "session": name,
        "start_ns": str(start_ns),
        "end_ns": str(end_ns),
        "labels_version": labels_version,
        "generator": generator,
        "scenes": [{
            "start_ns": str(sc.start_ns),
            "end_ns": str(sc.end_ns),
            "driving_decision": sc.lateral,
            "longitudinal_phases": [{"end_ns": str(p.end_ns), "decision": p.decision}
                                    for p in (sc.longitudinal_phases or
                                              (LongitudinalPhase(sc.end_ns, sc.longitudinal or "unknown"),))],
        } for sc in scenes],
    }


def validate_scene_document(doc: dict) -> None:
    """Reject malformed nested scenes before writing; legacy documents remain readable."""
    def timestamp(value):
        if not isinstance(value, str) or not value.isdigit():
            raise ValueError("timestamps must be nanosecond integer strings")
        return int(value)

    if not isinstance(doc, dict) or not isinstance(doc.get("scenes"), list):
        raise ValueError("document must contain a scenes list")
    for scene in doc["scenes"]:
        if not isinstance(scene, dict) or set(scene) != {"start_ns", "end_ns", "driving_decision", "longitudinal_phases"}:
            raise ValueError("scene must contain only the nested scene fields")
        start, end = timestamp(scene["start_ns"]), timestamp(scene["end_ns"])
        if start >= end:
            raise ValueError("scene start must precede its end")
        if not isinstance(scene["driving_decision"], str) or not scene["driving_decision"].strip():
            raise ValueError("scene driving_decision must be a nonempty string")
        phases = scene["longitudinal_phases"]
        if not isinstance(phases, list) or not phases:
            raise ValueError("scene must contain longitudinal phases")
        previous = start
        for phase in phases:
            if not isinstance(phase, dict) or set(phase) != {"end_ns", "decision"}:
                raise ValueError("phase must contain only end_ns and decision")
            phase_end = timestamp(phase["end_ns"])
            if not previous < phase_end <= end:
                raise ValueError("phase ends must increase within the scene")
            if not isinstance(phase["decision"], str) or not phase["decision"].strip():
                raise ValueError("phase decision must be a nonempty string")
            previous = phase_end
        if previous != end:
            raise ValueError("final phase must end at the scene end")


def write_scenes(out_dir: Path, doc: dict) -> Path:
    """Write ``<out_dir>/scenes/<date>_<session>.json``."""
    validate_scene_document(doc)
    path = Path(out_dir) / "scenes" / f"{doc['date']}_{doc['session']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    return path
