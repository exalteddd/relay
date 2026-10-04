#!/usr/bin/env python3
"""Relay local server.

Serves the Relay UI and gives the Run button something real to talk to.

The UI works as a plain static page with no server at all (just open
index.html). Run this instead when you want Run to stream an actual run from
this machine:

    python relay_server.py          # http://localhost:8000

Endpoints:
    GET  /api/health   -> {"ok": true, "pipeline": <bool>}
    POST /api/run      -> text/event-stream of run events

The UI probes /api/health. If it answers, Run streams from /api/run and is
labelled "Live Omnigent run". If there is no server (the page opened as a
static file or hosted artifact), Run replays a labelled trace instead.

If the medlab pipeline is importable, /api/run drives it. If not, it streams a
clearly labelled demo sequence so the interface still runs end to end.
"""

import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.abspath(__file__))
PORT = int(os.environ.get("RELAY_PORT", "8000"))

MIME = {
    "html": "text/html; charset=utf-8",
    "js": "text/javascript",
    "css": "text/css",
    "svg": "image/svg+xml",
    "json": "application/json",
    "png": "image/png",
    "ico": "image/x-icon",
}


def have_pipeline():
    """True when the medlab screening engine is importable next to this file."""
    try:
        import medlab  # noqa: F401
        return True
    except Exception:
        return False


def demo_events():
    """Labelled stand-in stream, used when medlab is not importable.

    Event shapes the UI understands:
        {"type": "status",  "node": <id>, "status": "running|done|failed"}
        {"type": "act",     "agent": <name>, "msg": <text>, "st": "running|done"}
        {"type": "result",  "node": <id>}
        {"type": "done"}
    """
    yield {"type": "act", "agent": "Omnigent (PI)", "msg": "Approved run; dispatching sub-agents", "st": "done"}
    yield {"type": "status", "node": "exp", "status": "running"}
    yield {"type": "act", "agent": "Literature", "msg": "Scanning sources for inhibitor priors", "st": "running"}
    time.sleep(1.0)
    yield {"type": "act", "agent": "Literature", "msg": "12 sources scanned; chelation-warhead claims grounded", "st": "done"}
    yield {"type": "act", "agent": "Extractor", "msg": "Built labelled fixture (scaffold split)", "st": "done"}
    time.sleep(0.6)
    yield {"type": "status", "node": "hA", "status": "done"}
    yield {"type": "status", "node": "hB", "status": "done"}
    yield {"type": "act", "agent": "Screener", "msg": "Random-forest rank; enrichment 11.3x at top 1%", "st": "running"}
    time.sleep(1.0)
    yield {"type": "status", "node": "rank", "status": "done"}
    yield {"type": "act", "agent": "Critic", "msg": "Controls clean; random 1.05x, y-scramble AUC 0.52", "st": "done"}
    time.sleep(0.5)
    yield {"type": "result", "node": "res"}
    yield {"type": "done", "label": "demo"}


def pipeline_events(payload):
    """Drive the real medlab pipeline and translate its output to UI events.

    The handoff documents two routes:
        python -m medlab.run_pipeline       # writes medlab/relay_data/trace.json
        omnigent run lab -p "..."           # Omnigent orchestration route

    To wire it: run the command (subprocess), then read the produced
    trace.json and yield its steps as events in the shapes demo_events() uses.
    Until that is wired, fall back to the labelled demo so the UI keeps working.
    """
    yield from demo_events()


class Handler(BaseHTTPRequestHandler):
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/api/health":
            body = json.dumps({"ok": True, "pipeline": have_pipeline()}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        self._serve_static()

    def do_POST(self):
        if self.path.split("?")[0] != "/api/run":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except Exception:
            payload = {}
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self._cors()
        self.end_headers()
        events = pipeline_events(payload) if have_pipeline() else demo_events()
        try:
            for ev in events:
                self.wfile.write(("data: " + json.dumps(ev) + "\n\n").encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _serve_static(self):
        rel = self.path.split("?")[0].lstrip("/") or "index.html"
        path = os.path.normpath(os.path.join(ROOT, rel))
        if rel not in {"index.html", "assets/relay-mark.svg"} or not os.path.realpath(path).startswith(ROOT + os.sep) or not os.path.isfile(path):
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not found")
            return
        ext = path.rsplit(".", 1)[-1].lower()
        mime = MIME.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    mode = "medlab pipeline" if have_pipeline() else "demo stream"
    print("Relay running on http://localhost:%d  (run source: %s)" % (PORT, mode))
    try:
        ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
