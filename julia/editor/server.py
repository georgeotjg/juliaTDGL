#!/usr/bin/env python3
"""Local server for the shape editor. 127.0.0.1 only, standard library only.

    python3 julia/editor/server.py               # opens the browser
    python3 julia/editor/server.py --port 8765 --no-browser
    python3 julia/editor/server.py --idle 0           # never stop on its own

It is a plain Python process (no npm, nothing installed). It stops by itself
after --idle minutes without requests (default 30); the editor page polls it
while it is open, so it stays up as long as a tab shows the editor.

GET  /                  the editor
GET  /shapes            list of julia/shapes/*.toml (JSON)
GET  /shape?name=x      shape x as JSON (tomllib) plus its text
POST /save?name=x       writes shapes/x.toml. If it exists, the previous version
                        is MOVED to trash_/shapes/ (never deleted).
"""
import argparse
import json
import threading
import re
import shutil
import time
import tomllib
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # repository root
SHAPES = HERE.parent / "shapes"
TRASH = ROOT / "trash_" / "shapes"
NAME_OK = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


LAST = [time.time()]                        # time of the last request


SRV = [None]


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, kind="application/json; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _name(self, q):
        n = (q.get("name") or [""])[0]
        if not NAME_OK.match(n):
            self._send(400, json.dumps({"error": "name: letters, digits, _ and - (max. 64)"}))
            return None
        return n

    def do_GET(self):
        LAST[0] = time.time()
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/shape_editor.html"):
            self._send(200, (HERE / "shape_editor.html").read_bytes(), "text/html; charset=utf-8")
        elif u.path == "/ping":
            self._send(200, json.dumps({"ok": True}))
        elif u.path == "/quit":
            # the "close server" button of the page
            self._send(200, json.dumps({"ok": True}))
            print("closed from the page: stopping the editor server")
            threading.Thread(target=SRV[0].shutdown, daemon=True).start()
        elif u.path == "/shapes":
            SHAPES.mkdir(exist_ok=True)
            self._send(200, json.dumps(sorted(p.stem for p in SHAPES.glob("*.toml"))))
        elif u.path == "/shape":
            n = self._name(q)
            if n is None:
                return
            p = SHAPES / f"{n}.toml"
            if not p.is_file():
                self._send(404, json.dumps({"error": f"{p.name} does not exist"}))
                return
            text = p.read_text(encoding="utf-8")
            try:
                data = tomllib.loads(text)
            except tomllib.TOMLDecodeError as e:
                self._send(422, json.dumps({"error": f"invalid TOML: {e}"}))
                return
            self._send(200, json.dumps({"toml": data, "text": text}))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_POST(self):
        LAST[0] = time.time()
        u = urlparse(self.path)
        if u.path != "/save":
            self._send(404, json.dumps({"error": "not found"}))
            return
        n = self._name(parse_qs(u.query))
        if n is None:
            return
        size = int(self.headers.get("Content-Length", "0"))
        if size <= 0 or size > 2_000_000:
            self._send(400, json.dumps({"error": "empty or too large body"}))
            return
        text = self.rfile.read(size).decode("utf-8")
        try:
            tomllib.loads(text)                       # a broken TOML is not saved
        except tomllib.TOMLDecodeError as e:
            self._send(422, json.dumps({"error": f"invalid TOML: {e}"}))
            return
        SHAPES.mkdir(exist_ok=True)
        dest = SHAPES / f"{n}.toml"
        moved = None
        if dest.exists():
            TRASH.mkdir(parents=True, exist_ok=True)
            moved = TRASH / f"{n}_{time.strftime('%Y%m%d_%H%M%S')}.toml"
            shutil.move(str(dest), str(moved))
        dest.write_text(text, encoding="utf-8")
        rel = dest.relative_to(ROOT)
        self._send(200, json.dumps({"ok": True, "path": str(rel),
                                    "previous": str(moved.relative_to(ROOT)) if moved else None}))

    def log_message(self, fmt, *args):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--idle", type=float, default=30.0,
                    help="minutes without requests before stopping (0 = never)")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    SRV[0] = srv
    if a.idle > 0:
        def vigilar():
            while time.time() - LAST[0] < 60 * a.idle:
                time.sleep(min(10.0, 15 * a.idle))
            print("idle: stopping the editor server")
            srv.shutdown()
        threading.Thread(target=vigilar, daemon=True).start()
    url = f"http://127.0.0.1:{a.port}/"
    print(f"shape editor: {url}   (Ctrl+C to close)")
    print(f"saves to: {SHAPES}")
    if not a.no_browser:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
