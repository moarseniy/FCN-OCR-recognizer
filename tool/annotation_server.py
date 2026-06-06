from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

from .markup import (
    discover_images,
    ensure_item,
    item_by_image,
    load_document,
    normalize_item,
    safe_image_path,
    save_document,
)


STATIC_ROOT = Path(__file__).with_name("static")


class AnnotationStore:
    def __init__(
        self,
        images_root: Path,
        output_path: Path,
        recursive: bool,
    ) -> None:
        self.images_root = images_root.expanduser().resolve()
        self.output_path = output_path.expanduser().resolve()
        self.images = discover_images(self.images_root, recursive=recursive)
        self.document = load_document(self.output_path, self.images_root)
        self.lock = threading.RLock()

        known = set(self.images)
        self.document["items"] = [
            item
            for item in self.document.get("items", [])
            if isinstance(item, dict) and item.get("image") in known
        ]
        save_document(self.output_path, self.document)

    def state(self) -> dict:
        with self.lock:
            annotations = item_by_image(self.document)
            images = []
            marked_count = 0
            for relative_path in self.images:
                item = annotations.get(relative_path)
                cuts = len(item.get("cuts", [])) if item else 0
                top_points = len((item.get("baselines") or {}).get("top", [])) if item else 0
                bottom_points = len((item.get("baselines") or {}).get("bottom", [])) if item else 0
                marked = bool(cuts or top_points or bottom_points)
                marked_count += int(marked)
                images.append(
                    {
                        "path": relative_path,
                        "marked": marked,
                        "cuts": cuts,
                        "top_points": top_points,
                        "bottom_points": bottom_points,
                    }
                )
            return {
                "images": images,
                "marked": marked_count,
                "total": len(images),
                "output": str(self.output_path),
            }

    def get_annotation(self, relative_path: str) -> dict:
        with self.lock:
            item = ensure_item(self.document, self.images_root, relative_path)
            return dict(item)

    def update_annotation(self, relative_path: str, payload: dict) -> dict:
        with self.lock:
            item = ensure_item(self.document, self.images_root, relative_path)
            image_path = safe_image_path(self.images_root, relative_path)
            with Image.open(image_path) as image:
                width, height = image.size
            normalized = normalize_item(payload, width, height)
            item.update(normalized)
            save_document(self.output_path, self.document)
            return dict(item)


class AnnotationHandler(BaseHTTPRequestHandler):
    server: "AnnotationHTTPServer"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            return self._serve_static("index.html")
        if parsed.path.startswith("/static/"):
            return self._serve_static(parsed.path.removeprefix("/static/"))
        if parsed.path == "/api/state":
            return self._send_json(self.server.store.state())
        if parsed.path == "/api/annotation":
            relative_path = self._query_value(parsed.query, "image")
            return self._send_json(self.server.store.get_annotation(relative_path))
        if parsed.path == "/api/image":
            relative_path = self._query_value(parsed.query, "image")
            return self._serve_image(relative_path)
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/api/annotation":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            relative_path = self._query_value(parsed.query, "image")
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            item = self.server.store.update_annotation(relative_path, payload)
            self._send_json({"ok": True, "item": item})
        except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.BAD_REQUEST)

    def log_message(self, fmt: str, *args) -> None:
        print(f"[annotation] {self.address_string()} {fmt % args}")

    def _serve_static(self, relative_path: str) -> None:
        candidate = (STATIC_ROOT / relative_path).resolve()
        if STATIC_ROOT.resolve() not in candidate.parents or not candidate.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = candidate.read_bytes()
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        self._send_bytes(content, content_type)

    def _serve_image(self, relative_path: str) -> None:
        try:
            path = safe_image_path(self.server.store.images_root, relative_path)
        except (ValueError, FileNotFoundError) as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), content_type)

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(
            json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _send_bytes(
        self,
        content: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    @staticmethod
    def _query_value(query: str, name: str) -> str:
        values = parse_qs(query).get(name)
        if not values or not values[0]:
            raise ValueError(f"Missing query parameter: {name}")
        return values[0]


class AnnotationHTTPServer(ThreadingHTTPServer):
    def __init__(self, address: tuple[str, int], store: AnnotationStore):
        super().__init__(address, AnnotationHandler)
        self.store = store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browser editor for FCN OCR cuts and baseline markup.")
    parser.add_argument("--images", required=True, help="Directory containing images to annotate.")
    parser.add_argument(
        "--output",
        default="output/manual_markup.json",
        help="JSON file that receives annotations.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-recursive", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = AnnotationStore(
        images_root=Path(args.images),
        output_path=Path(args.output),
        recursive=not args.no_recursive,
    )
    server = AnnotationHTTPServer((args.host, args.port), store)
    url = f"http://{args.host}:{args.port}/"
    print(f"Images:      {store.images_root}")
    print(f"Annotations: {store.output_path}")
    print(f"Samples:     {len(store.images)}")
    print(f"Open:        {url}")
    if args.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping annotation server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
