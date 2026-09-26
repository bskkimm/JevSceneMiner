"""Command line: ``jevsceneminer run`` (logs -> scripts -> Jev -> scenes), ``classify``, ``restitch``, ``score``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

from . import __version__
from .bag import INDICATOR_LEFT, INDICATOR_RIGHT, BagError, read_session
from .evaluate import score
from .facts import build_timeline
from .jev import Answer, AnswerCache, JevClassifier, JevError, cache_key, load_labels
from .lanes import LaneMap
from .nuplan import read_nuplan
from .nuplan_map import TRAFFIC_SIDE, find_map, load_nuplan_map
from .rules import rule_scenes
from .scenes import Step, session_document, stitch, write_scenes
from .script import render, strip_map

REPO_ROOT = Path(__file__).resolve().parents[1]
PRICE_PER_M_INPUT_TOKENS = 0.042  # USD, TypeSafe's published price; output tokens are free


def load_dotenv(paths=(Path.cwd() / ".env", REPO_ROOT / ".env")) -> None:
    for path in paths:
        if path.exists():
            for line in path.read_text().splitlines():
                key, sep, value = line.strip().partition("=")
                if sep and not key.startswith("#") and key not in os.environ:
                    os.environ[key] = value.strip()


def indicator_segments(session) -> list[list]:
    """Periods with the indicator on, as ``[start_s, end_s, 2|3]`` (2 = LEFT, 3 = RIGHT)."""
    segments, state, start, last = [], None, None, None
    for t, report in zip(session.indicator_t, session.indicator):
        ts = int(t) / 1e9
        if report != state:
            if state in (INDICATOR_LEFT, INDICATOR_RIGHT):
                segments.append([start, ts, int(state)])
            state, start = report, ts
        last = ts
    if state in (INDICATOR_LEFT, INDICATOR_RIGHT):
        segments.append([start, last, int(state)])
    return segments


def _sources(args):
    """(kind, source, read kwargs, map path) per session: bag folders, --spans entries, --nuplan logs."""
    for d in args.sessions:
        yield "bag", Path(d), {}, args.map
    if args.spans:
        for sp in json.loads(Path(args.spans).read_text()):
            yield ("bag", [Path(f) for f in sp["files"]],
                   {"start_ns": int(sp["start_s"] * 1e9), "end_ns": int(sp["end_s"] * 1e9),
                    "name": sp["name"], "date": sp["date"]},
                   sp.get("map") or args.map)
    for spec in args.nuplan or []:
        yield "nuplan", [Path(p) for p in spec.split(",")], {}, None


def _nuplan_maps_root(args, db: Path) -> Path:
    """--nuplan-maps, or the ``maps`` folder next to an ancestor of the log (nuPlan's layout)."""
    if args.nuplan_maps:
        return Path(args.nuplan_maps)
    for parent in db.resolve().parents:
        if (parent / "maps").is_dir():
            return parent / "maps"
    raise SystemExit(f"no nuPlan maps folder found above {db}: pass --nuplan-maps")


def _read(kind, source, read_kwargs, map_path, args, maps: dict):
    """(session, lane map, traffic side, extras) for one source; maps are loaded once."""
    interval = min(0.45, args.step * 0.9)
    if kind == "nuplan":
        log = read_nuplan(source, object_interval_s=interval)
        map_path = find_map(_nuplan_maps_root(args, source[0]), log.location)
        side = args.traffic_side or TRAFFIC_SIDE.get(log.location, "right")
        extras = {"location": log.location, "tags": log.tags, "camera": log.camera,
                  "sources": [str(Path(x).resolve()) for x in source]}
        loader = load_nuplan_map
    else:
        if map_path is None:
            raise SystemExit("no map: pass --map, or a \"map\" in each --spans entry")
        log = None
        side, extras, loader = None, {}, LaneMap.load
    if map_path not in maps:
        t0 = time.time()
        maps[map_path] = loader(map_path)
        print(f"map {map_path}: {len(maps[map_path].lanes)} lanes ({time.time() - t0:.1f} s)", flush=True)
    session = log.session if log else read_session(source, object_interval_s=interval, topics=args.topic_lists,
                                                   **read_kwargs)
    extras["map"] = str(map_path)
    return session, maps[map_path], side, extras


def _prepare(session, lanemap, traffic_side: str, extras: dict, args, out: Path) -> tuple[dict, dict]:
    """Write one session's scripts (steps), indicator periods, rule baseline and metadata."""
    t1 = time.time()
    timeline = build_timeline(session, lanemap)
    span = (session.end_ns - session.start_ns) / 1e9
    print(f"{session.date} {session.name}: {span:.0f} s of driving ({time.time() - t1:.1f} s), "
          f"{', '.join(f'{k} {v}' for k, v in session.topics.items() if k != 'decode_errors')}", flush=True)

    step_ns = int(round(args.step * 1e9))
    first = session.start_ns + (-session.start_ns) % step_ns   # steps on a fixed clock grid
    times = list(range(first, session.end_ns + 1, step_ns))
    if args.max_steps:
        times = times[args.first_step:args.first_step + args.max_steps]
    scripts = {}
    for t in times:
        text = render(timeline, lanemap, t, past_s=args.past, future_s=args.future, traffic_side=traffic_side)
        if text is not None:
            scripts[t] = text

    sid = f"{session.date}_{session.name}"
    for sub in ("steps", "indicator", "meta"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    (out / "indicator" / f"{sid}.json").write_text(json.dumps(indicator_segments(session)))
    meta = {"date": session.date, "name": session.name, "start_ns": session.start_ns, "end_ns": session.end_ns,
            "step_s": args.step, "past_s": args.past, "future_s": args.future, "topics": session.topics,
            "traffic_side": traffic_side, "has_indicator": session.has_indicator,
            "map": extras.get("map"), "location": extras.get("location"), "sources": extras.get("sources")}
    (out / "meta" / f"{sid}.json").write_text(json.dumps(meta))
    for sub in ("tags", "camera"):
        if extras.get(sub):
            (out / sub).mkdir(exist_ok=True)
            (out / sub / f"{sid}.json").write_text(json.dumps(extras[sub]))
    rules = rule_scenes(timeline, lanemap, list(scripts), args.step, args.min_scene, session.start_ns, session.end_ns)
    write_scenes(out / "rules", session_document(session.date, session.name, session.start_ns, session.end_ns,
                                                 "rules", rules, {"name": "jevsceneminer rules", "version": __version__}))
    with open(out / "steps" / f"{sid}.jsonl", "w") as fh:
        for t, text in scripts.items():
            fh.write(json.dumps({"t_ns": t, "script": text}, ensure_ascii=False) + "\n")
    return meta, scripts


def _labels_for(args, meta: dict, lateral_only: bool = False):
    """The questions for one session: its traffic side and whether it has the indicator."""
    return load_labels(args.labels, lateral_only=lateral_only,
                       traffic_side=args.traffic_side or meta.get("traffic_side"),
                       has_indicator=meta.get("has_indicator", True))


def _classify(out: Path, meta: dict, scripts: dict, labels, args) -> None:
    """Ask Jev about every script of one session, then stitch and write its run file."""
    sid = f"{meta['date']}_{meta['name']}"
    cache = AnswerCache(out / "cache" / f"{sid}.jsonl")
    answers = JevClassifier(labels, cache, model=args.model, workers=args.workers).classify(scripts)
    with open(out / "steps" / f"{sid}.jsonl", "w") as fh:
        for t, text in scripts.items():
            a = answers[t]
            fh.write(json.dumps({"t_ns": t, "script": text, "lateral": a.lateral, "lateral_probs": a.lateral_probs,
                                 "longitudinal": a.longitudinal, "longitudinal_probs": a.longitudinal_probs,
                                 "model": a.model, "input_tokens": a.input_tokens,
                                 "cache_key": cache_key(args.model, labels, text)}, ensure_ascii=False) + "\n")
    indicator = None
    if args.require_indicator:
        indicator = json.loads((out / "indicator" / f"{sid}.json").read_text())
    steps = [Step(t, answers[t]) for t in scripts]
    scenes = stitch(steps, meta["step_s"], args.min_scene, meta["start_ns"], meta["end_ns"], args.min_prob,
                    indicator=indicator, needs=labels.indicator)
    models = sorted({a.model for a in answers.values() if a.model})
    doc = session_document(meta["date"], meta["name"], meta["start_ns"], meta["end_ns"], labels.version,
                           scenes, {
        "name": "jevsceneminer", "version": __version__, "models": models,
        "step_s": meta["step_s"], "past_s": meta.get("past_s"), "future_s": meta.get("future_s"),
        "min_scene_s": args.min_scene, "min_prob": args.min_prob,
        "require_indicator": args.require_indicator, "traffic_side": labels.traffic_side})
    path = write_scenes(out, doc)
    n_tokens = sum(a.input_tokens or 0 for a in answers.values())
    print(f"  {len(scenes)} scenes -> {path} (models {models}, {n_tokens} input tokens)", flush=True)


def cmd_run(args) -> int:
    out = Path(args.out)
    maps: dict = {}
    totals = {"calls": 0, "chars": 0, "sessions": 0, "skipped": 0}
    for kind, source, read_kwargs, map_path in _sources(args):
        try:
            session, lanemap, side, extras = _read(kind, source, read_kwargs, map_path, args, maps)
        except BagError as exc:
            if not args.keep_going:
                raise
            totals["skipped"] += 1
            print(f"  [skip] {exc}", flush=True)
            continue
        if side is None:
            side = load_labels(args.labels).traffic_side if not args.traffic_side else args.traffic_side
        meta, scripts = _prepare(session, lanemap, side, extras, args, out)
        totals["calls"] += len(scripts)
        totals["chars"] += sum(len(x) for x in scripts.values())
        totals["sessions"] += 1
        if args.dry_run:
            print(f"  dry run: {len(scripts)} scripts written", flush=True)
        else:
            _classify(out, meta, scripts, _labels_for(args, meta, args.lateral_only), args)

    # Script chars / 4, plus the question definitions sent with every call (measured ~1.4k tokens).
    est_tokens = totals["chars"] / 4 + totals["calls"] * 1400
    print(f"total: {totals['sessions']} sessions ({totals['skipped']} skipped), {totals['calls']} Jev calls, "
          f"~{est_tokens / 1e6:.1f}M input tokens, ~${est_tokens / 1e6 * PRICE_PER_M_INPUT_TOKENS:.2f}")
    return 0


def cmd_classify(args) -> int:
    """Jev + stitching for every session a `run --dry-run` prepared under <out>."""
    out = Path(args.out)
    failed = []
    for meta_path in sorted((out / "meta").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        sid = meta_path.stem
        rows = [json.loads(line) for line in (out / "steps" / f"{sid}.jsonl").read_text().splitlines() if line.strip()]
        print(f"{meta['date']} {meta['name']}: {len(rows)} scripts", flush=True)
        scripts = {r["t_ns"]: r["script"] for r in rows}
        if args.strip_map:
            scripts = {t: strip_map(x) for t, x in scripts.items()}
        try:
            _classify(out, meta, scripts, _labels_for(args, meta, args.lateral_only), args)
        except JevError as exc:     # answers so far are cached: re-running finishes the session
            failed.append(sid)
            print(f"  [failed] {exc}", flush=True)
    if failed:
        print(f"{len(failed)} session(s) failed (re-run to finish from the cache): {', '.join(failed)}")
        return 1
    return 0


def cmd_restitch(args) -> int:
    """Rebuild scenes from the saved per-step answers with other settings (no Jev calls)."""
    src, dst = Path(args.src), Path(args.out)
    for steps_path in sorted((src / "steps").glob("*.jsonl")):
        rows = [json.loads(line) for line in steps_path.read_text().splitlines() if line.strip()]
        if not rows or "lateral" not in rows[0]:
            continue   # dry-run output has no answers
        sid = steps_path.stem
        meta = json.loads((src / "meta" / f"{sid}.json").read_text())
        labels = _labels_for(args, meta)
        old = src / "scenes" / f"{sid}.json"
        generator = dict(json.loads(old.read_text()).get("generator") or {}) if old.exists() else {}
        steps = [Step(r["t_ns"], Answer(r["lateral"], r["lateral_probs"], r["longitudinal"], r["longitudinal_probs"],
                                        r.get("model"), r.get("input_tokens"))) for r in rows]
        indicator = None
        if args.require_indicator:
            indicator = json.loads((src / "indicator" / f"{sid}.json").read_text())
        scenes = stitch(steps, meta["step_s"], args.min_scene, meta["start_ns"], meta["end_ns"],
                        args.min_prob, indicator=indicator, needs=labels.indicator)
        generator.update(min_scene_s=args.min_scene, min_prob=args.min_prob,
                         require_indicator=args.require_indicator, restitched_from=str(src))
        write_scenes(dst, session_document(meta["date"], meta["name"], meta["start_ns"], meta["end_ns"],
                                           labels.version, scenes, generator))
        print(f"{meta['date']} {meta['name']}: {len(scenes)} scenes")
    for sub in ("meta", "indicator", "tags", "camera", "rules"):
        if (src / sub).exists():
            shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
    return 0


def cmd_score(args) -> int:
    out = Path(args.out)
    runs = {}
    for spec in args.run or ["jev=scenes", "rules=rules/scenes"]:
        name, _, folder = spec.partition("=")
        path = Path(folder) if Path(folder).is_absolute() else out / folder
        if path.is_dir() and any(path.glob("*.json")):
            runs[name] = path
    if not runs:
        raise SystemExit(f"no scenes to score under {out}")
    report_dir = Path(args.report) if args.report else out / "report"
    score(out, Path(args.gt) if args.gt else None, runs, report_dir)
    print((report_dir / "report.md").read_text())
    return 0


def cmd_view(args) -> int:
    from .viewer import serve

    out = Path(args.out)
    runs = {}
    for spec in args.run or ["jev=scenes", "rules=rules/scenes"]:
        name, _, folder = spec.partition("=")
        path = Path(folder) if Path(folder).is_absolute() else out / folder
        runs[name] = path
    serve(out, Path(args.gt) if args.gt else None, Path(args.sensor_root).expanduser() if args.sensor_root else None,
          runs, host=args.host, port=args.port)
    return 0


def main(argv=None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(prog="jevsceneminer", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="bags -> scripts -> Jev -> scenes")
    r.add_argument("sessions", nargs="*", help="session folders with .mcap chunks")
    r.add_argument("--spans", default=None,
                   help="JSON list of sessions as {name, date, start_s, end_s, files[, map]}")
    r.add_argument("--map", default=None, help="lanelet2_map.osm (unless every --spans entry has one)")
    r.add_argument("--nuplan", action="append", metavar="DB[,DB...]",
                   help="a nuPlan log .db (repeatable); comma-join consecutive slices to read them as one session")
    r.add_argument("--nuplan-maps", default=None,
                   help="nuPlan maps folder (default: the 'maps' folder above the log, as nuPlan unpacks)")
    r.add_argument("--topics", default=None,
                   help="YAML {role: [topic, ...]} overriding the default Autoware topics "
                        "(roles: ego, objects, lights, indicator)")
    r.add_argument("--keep-going", action="store_true", help="skip sessions whose bags lack required topics")
    r.add_argument("--out", required=True)
    r.add_argument("--labels", default=str(REPO_ROOT / "labels.yaml"))
    r.add_argument("--traffic-side", choices=("left", "right"), default=None,
                   help="side traffic drives on (default: the nuPlan city, else labels.yaml's traffic_side)")
    r.add_argument("--step", type=float, default=0.5, help="seconds between NOW steps (0.5 = 2 Hz)")
    r.add_argument("--past", type=int, default=5, help="seconds of PAST in each script's table")
    r.add_argument("--future", type=int, default=10, help="seconds of FUTURE in each script's table")
    r.add_argument("--min-scene", type=float, default=2.0, help="shorter label runs are merged away")
    r.add_argument("--min-prob", type=float, default=0.4,
                   help="maneuvers with a lower mean Jev probability become keep_lane")
    r.add_argument("--workers", type=int, default=8, help="parallel Jev calls")
    r.add_argument("--model", default=None, help="Jev model (default: the API's default)")
    r.add_argument("--lateral-only", action="store_true", help="ask Jev only the lateral question")
    r.add_argument("--require-indicator", action="store_true",
                   help="turns, lane changes and pull-over/out need the matching indicator (a rule on top of Jev)")
    r.add_argument("--dry-run", action="store_true", help="write scripts and a cost estimate, no Jev calls")
    r.add_argument("--max-steps", type=int, default=0, help="only this many steps (smoke tests)")
    r.add_argument("--first-step", type=int, default=0, help="with --max-steps: index of the first step")
    r.set_defaults(func=cmd_run)

    c = sub.add_parser("classify", help="Jev + scenes for sessions prepared by `run --dry-run`")
    c.add_argument("out", help="the --out folder of `run --dry-run`")
    c.add_argument("--labels", default=str(REPO_ROOT / "labels.yaml"))
    c.add_argument("--traffic-side", choices=("left", "right"), default=None,
                   help="side traffic drives on (default: labels.yaml's traffic_side)")
    c.add_argument("--min-scene", type=float, default=2.0)
    c.add_argument("--min-prob", type=float, default=0.4)
    c.add_argument("--require-indicator", action="store_true")
    c.add_argument("--lateral-only", action="store_true", help="ask Jev only the lateral question")
    c.add_argument("--strip-map", action="store_true",
                   help="remove every map-derived fact from the scripts first (no-map ablation)")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--model", default=None)
    c.set_defaults(func=cmd_classify)

    t = sub.add_parser("restitch", help="rebuild scenes from saved answers with other settings (no Jev calls)")
    t.add_argument("src", help="an --out folder of `jevsceneminer run`")
    t.add_argument("--out", required=True)
    t.add_argument("--labels", default=str(REPO_ROOT / "labels.yaml"))
    t.add_argument("--traffic-side", choices=("left", "right"), default=None,
                   help="side traffic drives on (default: labels.yaml's traffic_side)")
    t.add_argument("--min-scene", type=float, default=2.0)
    t.add_argument("--min-prob", type=float, default=0.4)
    t.add_argument("--require-indicator", action="store_true")
    t.set_defaults(func=cmd_restitch)

    sc = sub.add_parser("score", help="score runs against GT scenes and nuPlan's scenario tags")
    sc.add_argument("out", help="an --out folder of `jevsceneminer run`")
    sc.add_argument("--gt", default=None, help="folder of GT scene files (<session>.json, same format as scenes/)")
    sc.add_argument("--run", action="append", metavar="NAME=FOLDER",
                    help="runs to score (default: jev=scenes rules=rules/scenes, relative to OUT)")
    sc.add_argument("--report", default=None, help="report folder (default: OUT/report)")
    sc.set_defaults(func=cmd_score)

    v = sub.add_parser("view", help="local review / GT labeling site for a run folder")
    v.add_argument("out", help="an --out folder of `jevsceneminer run`")
    v.add_argument("--gt", default=None, help="GT folder to show and save edits to")
    v.add_argument("--sensor-root", default=None, help="nuPlan sensor_blobs folder (front camera frames)")
    v.add_argument("--run", action="append", metavar="NAME=FOLDER",
                   help="runs to show (default: jev=scenes rules=rules/scenes, relative to OUT)")
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--port", type=int, default=8650)
    v.set_defaults(func=cmd_view)

    args = p.parse_args(argv)
    if args.command == "run":
        args.topic_lists = yaml.safe_load(Path(args.topics).read_text()) if args.topics else None
    needs_key = (args.command == "run" and not args.dry_run) or args.command == "classify"
    if needs_key and not os.environ.get("TYPESAFE_API_KEY"):
        print("TYPESAFE_API_KEY is not set (put it in .env), or use --dry-run", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
