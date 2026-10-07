"""A local review and labeling site for one ``jevsceneminer run`` output folder.

    jevsceneminer view out/run1 --sensor-root ~/dataset/nuplan/sensor_blobs --gt gt/ --port 8650

The page (viewer/index.html) shows, per session: the front camera, a bird's-eye view of
the map with the ego and nearby objects, a compact Jev/GT timeline, and the script at
NOW. Extra signal tracks are available under Details. The GT editor supports labels,
boundaries, add/split/merge/delete and Undo; Save writes ``<gt>/<session>.json``
(the same format as ``scenes/``).

The bird's-eye data (map polygons, ego track, objects) depends only on the log and map
files and any explicitly configured local display alignment, so it is built once and cached under ``~/.cache/jevsceneminer/bev``
(keyed by those files' paths, sizes and times). A new run reuses it; its labels and
probabilities are drawn on top by the page. At start, missing caches are built in the
background.

Everything is served from this machine; nothing is uploaded. Standard library only.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import shapely

from .lanes import CACHE_DIR

PAGE = Path(__file__).resolve().parent / "viewer" / "index.html"
BEV_CACHE = CACHE_DIR / "bev"
BEV_FORMAT = 3
MAP_MARGIN_M = 60.0
TRACK_EVERY_S = 0.1


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


class Site:
    def __init__(self, out: Path, gt_dir: Path | None, sensor_root: Path | None, runs: dict[str, Path]):
        self.out, self.gt_dir, self.sensor_root, self.runs = out, gt_dir, sensor_root, runs
        self._bev: dict[str, dict] = {}
        self._scripts: dict[str, tuple] = {}
        self._camera_paths = {}
        self._camera_lock = threading.Lock()
        self._script_lock = threading.Lock()
        self._lock = threading.Lock()

    def sessions(self) -> list[dict]:
        return [dict(_read_json(p), sid=p.stem) for p in sorted((self.out / "meta").glob("*.json"))]

    def session(self, sid: str) -> dict:
        meta = _read_json(self.out / "meta" / f"{sid}.json")
        steps = []
        path = self.out / "steps" / f"{sid}.jsonl"
        for line in path.read_text().splitlines() if path.exists() else []:
            row = json.loads(line)
            row.pop("script", None)
            row.pop("facts", None)
            steps.append(row)
        runs = {name: _read_json(folder / f"{sid}.json") for name, folder in self.runs.items()}
        gt = _read_json(self.gt_dir / f"{sid}.json") if self.gt_dir else None
        return {"meta": meta, "steps": steps, "runs": runs, "gt": gt,
                "gt_writable": self.gt_dir is not None,
                "tags": _read_json(self.out / "tags" / f"{sid}.json", []),
                "camera": [t for t, _ in _read_json(self.out / "camera" / f"{sid}.json", [])],
                "indicator": _read_json(self.out / "indicator" / f"{sid}.json", [])}

    def script(self, sid: str, t_ns: int) -> str:
        path = self.out / "steps" / f"{sid}.jsonl"
        # Audit facts can dwarf the displayed script; read them only on a dataset change.
        with self._script_lock:
            stat = path.stat()
            stamp = (stat.st_mtime_ns, stat.st_size)
            cached = self._scripts.get(sid)
            if cached is None or cached[0] != stamp:
                rows = []
                with path.open() as fh:
                    for line in fh:
                        row = json.loads(line)
                        rows.append((row["t_ns"], row["script"]))
                cached = (stamp, rows)
                self._scripts[sid] = cached
        rows = cached[1]
        return min(rows, key=lambda row: abs(row[0]-t_ns))[1] if rows else ""

    def camera_file(self, sid: str, index: int) -> Path | None:
        frames = _read_json(self.out / "camera" / f"{sid}.json", [])
        if not self.sensor_root or not 0 <= index < len(frames):
            return None
        path = (self.sensor_root / frames[index][1]).resolve()
        return path if path.is_file() and self.sensor_root.resolve() in path.parents else None

    def camera_path(self, sid: str, index: int) -> dict:
        frames = _read_json(self.out / "camera" / f"{sid}.json", [])
        meta = _read_json(self.out / "meta" / f"{sid}.json", {})
        if not 0 <= index < len(frames) or meta.get("topics", {}).get("source") != "nuplan":
            return {"available": False, "reason": "Recorded camera projection requires nuPlan calibration"}
        from .camera_projection import CameraPath
        with self._camera_lock:
            if sid not in self._camera_paths:
                self._camera_paths[sid] = CameraPath(meta)
            path = self._camera_paths[sid]
        return path.frame(*frames[index])

    def bev(self, sid: str) -> dict:
        """Map polygons near the path, the ego track and objects, relative to the first pose."""
        with self._lock:
            if sid not in self._bev:
                self._bev[sid] = self._cached_bev(sid)
            return self._bev[sid]

    def prebuild(self) -> None:
        """Build the missing bird's-eye caches of all sessions (run in a background thread)."""
        for p in sorted((self.out / "meta").glob("*.json")):
            try:
                self.bev(p.stem)
            except Exception as exc:          # noqa: BLE001 - one bad session must not stop the rest
                print(f"bird's-eye view of {p.stem} failed: {exc}", flush=True)

    @staticmethod
    def bev_key(meta: dict) -> str | None:
        """Cache key from the source files (paths, sizes, modification times) and the format."""
        if not meta or not meta.get("sources") or not meta.get("map"):
            return None
        parts = [f"bev-v{BEV_FORMAT}"]
        for f in [*meta["sources"], meta["map"], *([meta["bev_alignment"]] if meta.get("bev_alignment") else [])]:
            st = os.stat(f)
            parts.append(f"{Path(f).resolve()}:{st.st_size}:{st.st_mtime_ns}")
        return hashlib.sha1("|".join(parts).encode()).hexdigest()[:20]

    def _cached_bev(self, sid: str) -> dict:
        meta = _read_json(self.out / "meta" / f"{sid}.json")
        key = self.bev_key(meta)
        if key is None:
            return {"available": False, "reason": "the map view is available for nuPlan sessions"}
        path = BEV_CACHE / f"{key}.json.gz"
        if path.exists():
            return json.loads(gzip.decompress(path.read_bytes()))
        doc = self._build_bev(meta)
        BEV_CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(gzip.compress(json.dumps(doc).encode(), 6))
        tmp.rename(path)
        print(f"bird's-eye view of {sid} cached: {path}", flush=True)
        return doc

    @staticmethod
    def _build_bev(meta: dict) -> dict:
        from .nuplan import read_nuplan
        from .nuplan_map import load_nuplan_map

        log = read_nuplan(meta["sources"], object_interval_s=0.2)
        s = log.session
        lanemap = load_nuplan_map(Path(meta["map"]))
        x0, y0 = float(s.ego.x[0]), float(s.ego.y[0])
        step = max(1, int(round(TRACK_EVERY_S / (np.median(np.diff(s.ego.t)) / 1e9))))
        e = s.ego
        track = [[int(t), round(x - x0, 2), round(y - y0, 2), round(float(w), 3)]
                 for t, x, y, w in zip(e.t[::step], e.x[::step], e.y[::step], e.yaw[::step])]
        area = shapely.LineString(np.column_stack([e.x[::step], e.y[::step]])).buffer(MAP_MARGIN_M)
        alignment = None
        overrides = {}
        if meta.get("bev_alignment"):
            alignment = _read_json(Path(meta["bev_alignment"]))
            if not alignment or alignment.get("format") != "jsm-bev-alignment-1":
                raise ValueError("unsupported local BEV alignment format")
            if Path(alignment["source_map"]).resolve() != Path(meta["map"]).resolve():
                raise ValueError("local BEV alignment belongs to another source map")
            if sorted(Path(f).name for f in meta["sources"]) != sorted(alignment["source_logs"]):
                raise ValueError("local BEV alignment belongs to another source log")
            for row in alignment["lanes"]:
                lane_id = row["id"]
                if lane_id not in lanemap.polygons or lane_id in overrides:
                    raise ValueError(f"unknown or repeated local BEV lane {lane_id}")
                poly = shapely.Polygon(row["polygon_xy"])
                if not poly.is_valid or poly.is_empty or poly.area <= 0:
                    raise ValueError(f"invalid polygon for local BEV lane {lane_id}")
                overrides[lane_id] = poly
        lanes, original_lanes = [], []
        for lane_id, poly in lanemap.polygons.items():
            if not poly.intersects(area):
                continue
            coords = np.asarray(poly.simplify(0.2).exterior.coords)[:, :2] - [x0, y0]
            row = {"id": lane_id, "intersection": lanemap.lanes[lane_id].is_intersection,
                   "xy": np.round(coords, 1).ravel().tolist()}
            original_lanes.append(row)
            if lane_id in overrides:
                coords = np.asarray(overrides[lane_id].simplify(0.02).exterior.coords)[:, :2] - [x0, y0]
                row = dict(row, xy=np.round(coords, 2).ravel().tolist())
            lanes.append(row)
        objects = [[int(t), [[o.kind, round(o.x - x0, 1), round(o.y - y0, 1), round(o.yaw, 2), round(o.speed, 1),
                                  round(o.length_m, 2) if o.length_m else None,
                                  round(o.width_m, 2) if o.width_m else None]
                             for o in objs]] for t, objs in s.objects]
        doc = {"available": True, "lanes": lanes, "track": track, "objects": objects,
               "origin": [x0, y0],
               "ego_vehicle": {"length_m": 5.176, "width_m": 2.297, "rear_axle_to_center_m": 1.461}}
        if alignment:
            doc["original_lanes"] = original_lanes
            doc["map_alignment"] = {"description": alignment["description"], "approximate": True,
                                    "lane_ids": sorted(overrides)}
        return doc

    def save_gt(self, sid: str, doc: dict) -> Path:
        if not self.gt_dir:
            raise PermissionError("start the viewer with --gt to save GT")
        from .scenes import validate_scene_document
        validate_scene_document(doc)
        self.gt_dir.mkdir(parents=True, exist_ok=True)
        path = self.gt_dir / f"{sid}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
        return path


def make_handler(site: Site):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):   # quiet
            pass

        def _send(self, code: int, body: bytes, kind: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Seeking can cancel an in-flight image/projection request.


        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            parts = [unquote(p) for p in self.path.split("?")[0].strip("/").split("/")]
            try:
                if parts == [""]:
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif len(parts) == 1 and parts[0] in {"editor.js", "camera_decisions.js", "bev_geometry.js"}:
                    self._send(200, PAGE.with_name(parts[0]).read_bytes(), "text/javascript; charset=utf-8")
                elif parts == ["api", "sessions"]:
                    self._json({"sessions": site.sessions(), "runs": list(site.runs)})
                elif parts[:2] == ["api", "session"] and len(parts) == 3:
                    self._json(site.session(parts[2]))
                elif parts[:2] == ["api", "camera-path"] and len(parts) == 4:
                    self._json(site.camera_path(parts[2], int(parts[3])))
                elif parts[:2] == ["api", "bev"] and len(parts) == 3:
                    self._json(site.bev(parts[2]))
                elif parts[:2] == ["api", "script"] and len(parts) == 4:
                    self._send(200, site.script(parts[2], int(parts[3])).encode(), "text/plain; charset=utf-8")
                elif parts[0] == "camera" and len(parts) == 3:
                    path = site.camera_file(parts[1], int(parts[2]))
                    if path is None:
                        self._send(404, b"no image")
                    else:
                        self._send(200, path.read_bytes(), mimetypes.guess_type(path.name)[0] or "image/jpeg")
                else:
                    self._send(404, b"not found", "text/plain")
            except (FileNotFoundError, ValueError) as exc:
                self._json({"error": str(exc)}, 404)

        def do_PUT(self):
            parts = [unquote(p) for p in self.path.strip("/").split("/")]
            if parts[:2] != ["api", "gt"] or len(parts) != 3 or "/" in parts[2] or parts[2].startswith("."):
                return self._send(404, b"not found", "text/plain")
            try:
                doc = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                path = site.save_gt(parts[2], doc)
                self._json({"saved": str(path)})
            except PermissionError as exc:
                self._json({"error": str(exc)}, 403)
            except ValueError as exc:
                self._json({"error": str(exc)}, 400)

    return Handler


def serve(out: Path, gt_dir: Path | None, sensor_root: Path | None, runs: dict[str, Path],
          host: str = "127.0.0.1", port: int = 8650) -> None:
    site = Site(out, gt_dir, sensor_root, runs)
    threading.Thread(target=site.prebuild, daemon=True).start()
    server = ThreadingHTTPServer((host, port), make_handler(site))
    print(f"serving {out} on http://{host}:{port}/ (ctrl-c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
