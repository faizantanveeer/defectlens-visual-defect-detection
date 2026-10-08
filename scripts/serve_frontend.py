"""Serve the test UI (``frontend/``) plus the example images. Standard library only.

    python scripts/serve_frontend.py --port 5173

Routes:
    /                     -> frontend/index.html (and its assets)
    /samples/index.json   -> list of example images
    /samples/<name>       -> examples/images/<name>
Binds to 127.0.0.1 by default: this is a local testing tool, not a public web server.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend"
SAMPLES = ROOT / "examples" / "images"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path == "/samples/index.json":
            names = sorted(p.name for p in SAMPLES.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            body = json.dumps(names).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def translate_path(self, path: str) -> str:
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean.startswith("/samples/"):
            # .name strips any directory component, so "../" cannot escape the folder.
            return str(SAMPLES / pathlib.PurePosixPath(clean).name)
        return super().translate_path(path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[ui] {self.address_string()} {fmt % args}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5173)
    args = ap.parse_args()
    handler = functools.partial(Handler, directory=str(FRONTEND))
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as server:
        print(f"[ui] serving {FRONTEND} at http://{args.host}:{args.port}  (Ctrl+C to stop)")
        with contextlib.suppress(KeyboardInterrupt):
            server.serve_forever()


if __name__ == "__main__":
    main()
