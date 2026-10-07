"""Score runs: against hand-labeled GT when there is some, and against nuPlan's scenario tags.

A run and the GT use the same file format (``scenes/<session>.json``, see scenes.py).

With GT, per lateral maneuver label (every label except keep_lane):
    a maneuver is a run of consecutive scenes with that label; a run maneuver matches a GT
    maneuver of the same label when they overlap by at least MIN_OVERLAP_S (one-to-one,
    largest overlap first). precision = matched / run maneuvers, recall = matched / GT ones.
    Also: start/end error of matched maneuvers, seconds of confusion between labels,
    agreement over time (lateral and longitudinal), and calibration of Jev's step
    probabilities (how often a step is right when Jev says p).

Without GT, nuPlan's scenario tags give a rough check: e.g. is there a turn_left within
TAG_WINDOW_S of each "starting_left_turn" tag?
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

MIN_OVERLAP_S = 0.5
GRID_S = 0.1
TAG_WINDOW_S = 5.0
TAG_CLUSTER_S = 3.0            # nuPlan repeats a tag on consecutive frames: one event per cluster
FALLBACK = "keep_lane"

# nuPlan scenario tag -> what the run should show near it (lateral labels, or a longitudinal one).
TAG_CHECKS = {
    "starting_left_turn": ("lateral", {"turn_left", "u_turn"}),
    "starting_right_turn": ("lateral", {"turn_right", "u_turn"}),
    "changing_lane_to_left": ("lateral", {"lane_change_left"}),
    "changing_lane_to_right": ("lateral", {"lane_change_right"}),
    "high_lateral_acceleration": ("lateral", {"turn_left", "turn_right", "u_turn", "lane_change_left",
                                              "lane_change_right", "avoidance"}),
    "stationary": ("longitudinal", {"stopped"}),
    "stationary_at_traffic_light_without_lead": ("longitudinal", {"stopped"}),
    "stopping_with_lead": ("longitudinal", {"decelerating", "hard_braking", "stopped"}),
}


def load_scenes(folder: Path) -> dict[str, dict]:
    """``{session id: document}`` for every ``<folder>/*.json``."""
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(folder).glob("*.json"))}


def spans(doc: dict, key: str = "lateral") -> list[tuple[float, float, str]]:
    """(start_s, end_s, label) per scene, epoch seconds."""
    out = []
    for scene in doc["scenes"]:
        a, b = int(scene["start_ns"]) / 1e9, int(scene["end_ns"]) / 1e9
        if key == "longitudinal" and "longitudinal_phases" in scene:
            for phase in scene["longitudinal_phases"]:
                end = int(phase["end_ns"]) / 1e9
                out.append((a, end, phase["decision"]))
                a = end
        else:
            label = scene["driving_decision"] if key == "lateral" and "driving_decision" in scene else scene.get(key, "unknown")
            out.append((a, b, label or "unknown"))
    return out


def maneuvers(doc: dict) -> list[tuple[float, float, str]]:
    """Consecutive scenes with the same lateral label merged; keep_lane left out."""
    out: list[list] = []
    for a, b, label in spans(doc):
        if out and out[-1][2] == label and a - out[-1][1] < 1e-6:
            out[-1][1] = b
        else:
            out.append([a, b, label])
    return [tuple(m) for m in out if m[2] != FALLBACK]


def match(gt: list, run: list) -> list[tuple[int, int]]:
    """One-to-one (gt index, run index) pairs of the same label, largest overlap first."""
    pairs = []
    for i, (ga, gb, gl) in enumerate(gt):
        for j, (ra, rb, rl) in enumerate(run):
            overlap = min(gb, rb) - max(ga, ra)
            if gl == rl and overlap >= MIN_OVERLAP_S:
                pairs.append((overlap, i, j))
    used_g, used_r, out = set(), set(), []
    for _, i, j in sorted(pairs, reverse=True):
        if i not in used_g and j not in used_r:
            used_g.add(i)
            used_r.add(j)
            out.append((i, j))
    return out


def _label_at(sp: list, t: float) -> str | None:
    for a, b, label in sp:
        if a <= t < b:
            return label
    return None


def confusion(gt_doc: dict, run_doc: dict, key: str) -> collections.Counter:
    """Seconds per (GT label, run label) over the GT's scenes."""
    g, r = spans(gt_doc, key), spans(run_doc, key)
    out: collections.Counter = collections.Counter()
    for a, b, gl in g:
        for k in range(int(round((b - a) / GRID_S))):
            t = a + (k + 0.5) * GRID_S
            if t < b:  # Rounding can place the last midpoint at the excluded end.
                out[(gl, _label_at(r, t) or "(none)")] += GRID_S
    return out


def _pr(tp: int, n_gt: int, n_run: int) -> dict:
    return {"TP": tp, "FN": n_gt - tp, "FP": n_run - tp,
            "precision": round(tp / n_run, 3) if n_run else None,
            "recall": round(tp / n_gt, 3) if n_gt else None}


def _median(values):
    v = sorted(values)
    return round(v[len(v) // 2], 2) if v else None


def score_against_gt(gt: dict, run: dict, steps: dict | None = None) -> dict:
    """Metrics of one run over the sessions both have."""
    sessions = sorted(set(gt) & set(run))
    per_label: dict = collections.defaultdict(lambda: [0, 0, 0])      # tp, n_gt, n_run
    starts, ends = [], []
    conf_lat: collections.Counter = collections.Counter()
    conf_lon: collections.Counter = collections.Counter()
    for sid in sessions:
        g, r = maneuvers(gt[sid]), maneuvers(run[sid])
        for _, _, label in g:
            per_label[label][1] += 1
        for _, _, label in r:
            per_label[label][2] += 1
        for i, j in match(g, r):
            per_label[g[i][2]][0] += 1
            starts.append(r[j][0] - g[i][0])
            ends.append(r[j][1] - g[i][1])
        conf_lat += confusion(gt[sid], run[sid], "lateral")
        conf_lon += confusion(gt[sid], run[sid], "longitudinal")
    tp, n_gt, n_run = (sum(v[k] for v in per_label.values()) for k in range(3))

    def agreement(conf):
        total = sum(conf.values())
        return round(sum(s for (a, b), s in conf.items() if a == b) / total, 3) if total else None

    report = {
        "sessions": sessions,
        "labels": {label: _pr(*v) for label, v in sorted(per_label.items())},
        "total": _pr(tp, n_gt, n_run),
        "timing_s": {"n": len(starts), "start_abs_median": _median([abs(x) for x in starts]),
                     "end_abs_median": _median([abs(x) for x in ends]),
                     "start_median": _median(starts), "end_median": _median(ends)},
        "agreement": {"lateral": agreement(conf_lat), "longitudinal": agreement(conf_lon)},
        "confusion_lateral_s": {f"{a} -> {b}": round(s, 1) for (a, b), s in conf_lat.most_common(12) if a != b},
    }
    if steps:
        report["calibration"] = calibration(gt, steps)
    return report


def calibration(gt: dict, steps: dict) -> list[dict]:
    """Per probability bin: how often Jev's lateral label at a step matches the GT."""
    bins: dict = collections.defaultdict(lambda: [0, 0])
    for sid, rows in steps.items():
        if sid not in gt:
            continue
        g = spans(gt[sid])
        for row in rows:
            if "lateral" not in row:
                continue
            truth = _label_at(g, row["t_ns"] / 1e9)
            if truth is None:
                continue
            p = row["lateral_probs"].get(row["lateral"], 0.0)
            k = min(9, int(p * 10))
            bins[k][0] += 1
            bins[k][1] += row["lateral"] == truth
    return [{"prob": f"{k / 10:.1f}-{(k + 1) / 10:.1f}", "steps": n, "correct": round(c / n, 3)}
            for k, (n, c) in sorted(bins.items())]


def tag_events(tags: list, types) -> dict[str, list[float]]:
    """Tag type -> event times (epoch s), repeated tags clustered into one event."""
    out: dict[str, list[float]] = {}
    for t_ns, kind in sorted(tags):
        if kind not in types:
            continue
        t = t_ns / 1e9
        times = out.setdefault(kind, [])
        if not times or t - times[-1] > TAG_CLUSTER_S:
            times.append(t)
    return out


def score_against_tags(tags: dict, run: dict) -> dict:
    """Per tag type: share of tag events with the expected label within TAG_WINDOW_S."""
    hits: dict = collections.defaultdict(lambda: [0, 0])
    for sid, session_tags in tags.items():
        if sid not in run:
            continue
        for kind, times in tag_events(session_tags, TAG_CHECKS).items():
            key, wanted = TAG_CHECKS[kind]
            sp = spans(run[sid], key)
            for t in times:
                near = {label for a, b, label in sp if a <= t + TAG_WINDOW_S and b >= t - TAG_WINDOW_S}
                if key == "longitudinal":   # a state tag: the label at the tag itself
                    near = {_label_at(sp, t)}
                hits[kind][0] += 1
                hits[kind][1] += bool(near & wanted)
    return {kind: {"events": n, "found": f, "share": round(f / n, 3) if n else None}
            for kind, (n, f) in sorted(hits.items())}


def render_markdown(report: dict) -> str:
    lines = ["# JevSceneMiner report", ""]
    runs = report["runs"]
    if report.get("tags"):
        lines += ["## nuPlan scenario tags (rough check, not GT)", "",
                  f"Share of tag events with the expected label within {TAG_WINDOW_S:.0f} s "
                  "(for state tags: at the tag itself).", "",
                  "| tag | expects | events | " + " | ".join(runs) + " |",
                  "|---|---|---|" + "---|" * len(runs)]
        kinds = sorted({k for r in report["tags"].values() for k in r})
        for kind in kinds:
            first = next(r[kind] for r in report["tags"].values() if kind in r)
            cells = [f"{report['tags'][name].get(kind, {}).get('share', '-')}" for name in runs]
            lines.append(f"| {kind} | {', '.join(sorted(TAG_CHECKS[kind][1]))} | {first['events']} | "
                         + " | ".join(cells) + " |")
        lines.append("")
    if report.get("gt"):
        lines += ["## Against the GT", ""]
        for name, r in report["gt"].items():
            t = r["total"]
            lines += [f"### {name}: precision {t['precision']}, recall {t['recall']} "
                      f"({len(r['sessions'])} sessions)", "",
                      "| label | GT | TP | FP | precision | recall |", "|---|---|---|---|---|---|"]
            for label, v in r["labels"].items():
                lines.append(f"| {label} | {v['TP'] + v['FN']} | {v['TP']} | {v['FP']} | "
                             f"{v['precision']} | {v['recall']} |")
            tm = r["timing_s"]
            lines += ["", f"Timing of matched maneuvers (run minus GT): median |start| {tm['start_abs_median']} s, "
                          f"median |end| {tm['end_abs_median']} s (n={tm['n']}).",
                      f"Agreement over time: lateral {r['agreement']['lateral']}, "
                      f"longitudinal {r['agreement']['longitudinal']}.", ""]
            if r["confusion_lateral_s"]:
                lines += ["Biggest lateral confusions (GT -> run, seconds): "
                          + "; ".join(f"{k} {v}" for k, v in r["confusion_lateral_s"].items()), ""]
            if r.get("calibration"):
                lines += ["Calibration (Jev's lateral probability -> share of steps right): "
                          + "; ".join(f"{c['prob']}: {c['correct']} (n={c['steps']})" for c in r["calibration"]), ""]
    else:
        lines += ["No GT given: only the tag check above.", ""]
    return "\n".join(lines)


def _steps(out: Path) -> dict:
    return {p.stem: [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
            for p in sorted((out / "steps").glob("*.jsonl"))}


def score(out: Path, gt_dir: Path | None, runs: dict[str, Path], report_dir: Path) -> dict:
    loaded = {name: load_scenes(folder) for name, folder in runs.items()}
    tags = {p.stem: json.loads(p.read_text()) for p in sorted((out / "tags").glob("*.json"))}
    report: dict = {"runs": list(runs), "tags": {}, "gt": {}}
    if tags:
        report["tags"] = {name: score_against_tags(tags, run) for name, run in loaded.items()}
    if gt_dir is not None:
        gt = load_scenes(gt_dir)
        steps = _steps(out)
        report["gt"] = {name: score_against_gt(gt, run, steps if name == "jev" else None)
                        for name, run in loaded.items()}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "report.json").write_text(json.dumps(report, indent=1))
    (report_dir / "report.md").write_text(render_markdown(report))
    return report
