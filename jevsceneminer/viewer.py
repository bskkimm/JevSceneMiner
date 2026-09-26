"""A local review and labeling site for one ``jevsceneminer run`` output folder.

    jevsceneminer view out/run1 --sensor-root ~/dataset/nuplan/sensor_blobs --gt gt/ --port 8650

The page (viewer/index.html) shows, per session: the front camera, a bird's-eye view of
the map with the ego and nearby objects, timeline bars for the GT, Jev and the rule
baseline, Jev's probability, nuPlan's scenario tags, and the script Jev saw. The GT row
can be edited and saved to ``<gt>/<session>.json`` (the same format as ``scenes/``).

Everything is served from this machine; nothing is uploaded. Standard library only.
"""

from __future__ import annotations

import json
import mimetypes
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import shapely

PAGE = Path(__file__).resolve().parent / "viewer" / "index.html"
MAP_MARGIN_M = 60.0
TRACK_EVERY_S = 0.1


def _read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


class Site:
    def __init__(self, out: Path, gt_dir: Path | None, sensor_root: Path | None, runs: dict[str, Path]):
        self.out, self.gt_dir, self.sensor_root, self.runs = out, gt_dir, sensor_root, runs
        self._bev: dict[str, dict] = {}
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
        best, best_dt = "", None
        for line in path.read_text().splitlines():
            row = json.loads(line)
            dt = abs(row["t_ns"] - t_ns)
            if best_dt is None or dt < best_dt:
                best, best_dt = row["script"], dt
        return best

    def camera_file(self, sid: str, index: int) -> Path | None:
        frames = _read_json(self.out / "camera" / f"{sid}.json", [])
        if not self.sensor_root or not 0 <= index < len(frames):
            return None
        path = (self.sensor_root / frames[index][1]).resolve()
        return path if path.is_file() and self.sensor_root.resolve() in path.parents else None

    def bev(self, sid: str) -> dict:
        """Map polygons near the path, the ego track and objects, relative to the first pose."""
        with self._lock:
            if sid not in self._bev:
                self._bev[sid] = self._build_bev(sid)
            return self._bev[sid]

    def _build_bev(self, sid: str) -> dict:
        from .nuplan import read_nuplan
        from .nuplan_map import load_nuplan_map

        meta = _read_json(self.out / "meta" / f"{sid}.json")
        if not meta or not meta.get("sources") or not meta.get("map"):
            return {"available": False, "reason": "the map view is available for nuPlan sessions"}
        log = read_nuplan(meta["sources"], object_interval_s=0.2)
        s = log.session
        lanemap = load_nuplan_map(Path(meta["map"]))
        x0, y0 = float(s.ego.x[0]), float(s.ego.y[0])
        step = max(1, int(round(TRACK_EVERY_S / (np.median(np.diff(s.ego.t)) / 1e9))))
        e = s.ego
        track = [[int(t), round(x - x0, 2), round(y - y0, 2), round(float(w), 3)]
                 for t, x, y, w in zip(e.t[::step], e.x[::step], e.y[::step], e.yaw[::step])]
        area = shapely.LineString(np.column_stack([e.x[::step], e.y[::step]])).buffer(MAP_MARGIN_M)
        lanes = []
        for lane_id, poly in lanemap.polygons.items():
            if not poly.intersects(area):
                continue
            coords = np.asarray(poly.simplify(0.2).exterior.coords)[:, :2] - [x0, y0]
            lanes.append({"id": lane_id, "intersection": lanemap.lanes[lane_id].is_intersection,
                          "xy": np.round(coords, 1).ravel().tolist()})
        objects = [[int(t), [[o.kind, round(o.x - x0, 1), round(o.y - y0, 1), round(o.yaw, 2), round(o.speed, 1)]
                             for o in objs]] for t, objs in s.objects]
        return {"available": True, "lanes": lanes, "track": track, "objects": objects}

    def save_gt(self, sid: str, doc: dict) -> Path:
        if not self.gt_dir:
            raise PermissionError("start the viewer with --gt to save GT")
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
            self.wfile.write(body)

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj).encode())

        def do_GET(self):
            parts = [unquote(p) for p in self.path.split("?")[0].strip("/").split("/")]
            try:
                if parts == [""]:
                    self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
                elif parts == ["api", "sessions"]:
                    self._json({"sessions": site.sessions(), "runs": list(site.runs)})
                elif parts[:2] == ["api", "session"] and len(parts) == 3:
                    self._json(site.session(parts[2]))
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
    server = ThreadingHTTPServer((host, port), make_handler(site))
    print(f"serving {out} on http://{host}:{port}/ (ctrl-c to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
