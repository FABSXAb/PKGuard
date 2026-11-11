from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pipeline.dm_runtime import model_class, runtime

DM = model_class()
DEFAULT_BATCH_SIZE = runtime.DEFAULT_BATCH_SIZE
MAX_VIDEO_BYTES = runtime.MAX_VIDEO_BYTES
VERIFY_MODES = runtime.VERIFY_MODES

HEALTH_PATH = "/health"
MODELS_PATH = "/v1/models"
SYSTEM_ONE_PATH = "/v1/systemone"
ROUTES = {HEALTH_PATH: "GET", MODELS_PATH: "GET", SYSTEM_ONE_PATH: "POST"}

PROTECTED = (MODELS_PATH, SYSTEM_ONE_PATH)
KEY_VARIABLE = "DECISION_API_KEY"
MEDIA_FIELDS = ("images", "videos")
ACCEPTED_FIELDS = ("state", "questions", *MEDIA_FIELDS, "model")
JSON_TYPE = "application/json"
DESCRIPTION = "Typed decisions with a probability for every option"
IDLE_SECONDS = 120


class Refusal(Exception):

    def __init__(self, status: HTTPStatus, detail: Any, headers: dict[str, str] | None = None):
        super().__init__(str(detail))
        self.status = status
        self.detail = detail
        self.headers = headers or {}


class Framing(Exception):
    """A request whose body cannot be delimited (bad Content-Length or chunk framing): 400, connection closed."""


def _no_constants(name: str) -> Any:
    raise ValueError(f"{name} is not valid JSON")


def _unprocessable(detail: Any) -> Refusal:
    return Refusal(HTTPStatus.UNPROCESSABLE_ENTITY, detail)


class Decider:

    def __init__(self, model: DM):
        self.model = model
        self.turn = threading.Lock()

    def modalities(self) -> list[str]:
        kinds = ["text"]
        if self.model.image_unavailable is None:
            kinds.append("image")
        if self.model.video_unavailable is None:
            kinds.append("video")
        return kinds

    def health(self, authentication: bool) -> dict[str, Any]:
        return {
            "status": "ready",
            "model": self.model.model_name,
            "max_input_tokens": self.model.max_length,
            "modalities": self.modalities(),
            "authentication": authentication,
        }

    def models(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "name": self.model.model_name,
            "description": DESCRIPTION,
            "max_input_tokens": self.model.max_length,
            "modalities": self.modalities(),
        }
        if self.model.image_unavailable is None:
            entry["image_max_pixels"] = self.model.image_contract()["max_pixels"]
        if self.model.video_unavailable is None:
            video = self.model.video_contract()
            entry["video"] = {
                "fps": video["fps"],
                "max_frames": video["max_frames"],
                "max_pixels_per_frame": video["max_pixels_per_frame"],
                "max_tokens_per_request": video["max_tokens_per_request"],
                "max_bytes": MAX_VIDEO_BYTES,
            }
        return {"models": [entry]}

    def arguments(self, body: Any) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise _unprocessable("expected a JSON object such as {\"questions\": {...}, \"state\": ...}")
        unknown = sorted(set(body) - set(ACCEPTED_FIELDS))
        if unknown:
            raise _unprocessable(
                f"{', '.join(map(repr, unknown))} not understood; a request may carry {', '.join(ACCEPTED_FIELDS)}"
            )
        questions = body.get("questions")
        if not isinstance(questions, dict) or not questions:
            raise _unprocessable("questions is required: an object of one or more question IDs, each to a question")
        call: dict[str, Any] = {"state": body.get("state"), "questions": questions, "images": None, "videos": None}
        for name in MEDIA_FIELDS:
            items = body.get(name)
            if items is None:
                continue
            if not isinstance(items, list):
                raise _unprocessable(f"{name} takes a list of base64 data URLs")
            loader = self.model.load_images if name == "images" else self.model.load_videos
            try:
                call[name] = loader(items, strict=True) or None
            except ValueError as exc:
                raise _unprocessable(str(exc)) from exc
        return call

    def answer(self, body: Any) -> dict[str, Any]:
        call = self.arguments(body)
        with self.turn:
            try:
                prepared = self.model.prepare(call["state"], call["questions"], call["images"], call["videos"])
                invalid = {
                    key: error["message"]
                    for key, error in prepared.errors.items()
                    if error.get("error") == "invalid_question"
                }
                if invalid:
                    raise _unprocessable({"invalid_questions": invalid})
                probabilities, tokens = self.model.run(prepared)
                return self.model.respond(prepared, probabilities, tokens)
            except ValueError as exc:
                raise _unprocessable(str(exc)) from exc


class SystemOneServer(ThreadingHTTPServer):

    daemon_threads = True

    request_queue_size = 128

    def __init__(self, address: tuple[str, int], decider: Decider, *, api_key: str | None = None):
        super().__init__(address, SystemOneHandler)
        self.decider = decider
        self.key_digest = hashlib.sha256(api_key.encode("utf-8")).digest() if api_key else None


class SystemOneHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "dm-server"
    sys_version = ""
    timeout = IDLE_SECONDS

    disable_nagle_algorithm = True

    default_request_version = "HTTP/1.1"
    error_content_type = "text/plain; charset=utf-8"
    error_message_format = "%(code)d: %(explain)s\n"

    def send_error(self, code: int, message: str | None = None, explain: str | None = None) -> None:

        if code == HTTPStatus.HTTP_VERSION_NOT_SUPPORTED:
            code = HTTPStatus.BAD_REQUEST
        super().send_error(code, message, explain)

    def do_GET(self) -> None:
        self.route("GET")

    def do_POST(self) -> None:
        self.route("POST")

    def do_PUT(self) -> None:
        self.route("PUT")

    def do_PATCH(self) -> None:
        self.route("PATCH")

    def do_DELETE(self) -> None:
        self.route("DELETE")

    def do_OPTIONS(self) -> None:
        self.route("OPTIONS")

    def do_HEAD(self) -> None:
        self.route("HEAD")

    def route(self, method: str) -> None:

        path, mark, query = self.requestline.split()[1].partition("?")
        try:
            raw = self.read_body()
        except Framing as exc:
            self.close_connection = True
            self.send_text(HTTPStatus.BAD_REQUEST, str(exc))
            return
        try:
            allowed = ROUTES.get(path)
            if allowed is None:
                target = path.rstrip("/")
                if path.endswith("/") and target in ROUTES:
                    self.redirect(target + mark + query)
                    return
                raise Refusal(HTTPStatus.NOT_FOUND, f"nothing is served at {path}")
            if method != allowed:
                raise Refusal(
                    HTTPStatus.METHOD_NOT_ALLOWED, f"{path} answers {allowed} only", {"Allow": allowed}
                )
            if path in PROTECTED and not self.authorized():
                raise Refusal(HTTPStatus.UNAUTHORIZED, "a valid bearer key is required", {"WWW-Authenticate": "Bearer"})
            decider = self.server.decider
            if path == HEALTH_PATH:
                reply = decider.health(self.server.key_digest is not None)
            elif path == MODELS_PATH:
                reply = decider.models()
            else:
                reply = decider.answer(self.parse(raw))
        except Refusal as refusal:
            self.send_json(refusal.status, {"detail": refusal.detail}, refusal.headers)
            return
        except Exception as exc:
            self.log_error("System One request failed: %s: %s", type(exc).__name__, exc)
            self.send_text(HTTPStatus.INTERNAL_SERVER_ERROR, f"the server failed on this request ({type(exc).__name__})")
            return
        self.send_json(HTTPStatus.OK, reply)

    def authorized(self) -> bool:
        expected = self.server.key_digest
        if expected is None:
            return True
        scheme, space, credential = self.headers.get("Authorization", "").partition(" ")
        offered = hashlib.sha256(credential.encode("utf-8")).digest()
        return scheme == "Bearer" and space == " " and hmac.compare_digest(offered, expected)

    def read_body(self) -> bytes:
        coding = self.headers.get("Transfer-Encoding")
        size = self.headers.get("Content-Length")
        if coding is not None:
            if size is not None or coding.strip().lower() != "chunked":
                raise Framing("use either Content-Length or Transfer-Encoding: chunked")
            return self.read_chunks()
        if size is None:
            return b""
        size = size.strip()
        if not size.isdigit():
            raise Framing("Content-Length must be a byte count")
        raw = self.rfile.read(int(size))
        if len(raw) != int(size):
            raise Framing("the connection closed before Content-Length bytes arrived")
        return raw

    def read_chunks(self) -> bytes:
        parts = []
        while True:
            line = self.rfile.readline(1 << 16)
            digits = line.split(b";", 1)[0].strip()
            if not digits or any(c not in b"0123456789abcdefABCDEF" for c in digits):
                raise Framing("a chunk must start with its size in hexadecimal")
            size = int(digits, 16)
            if size == 0:
                while self.rfile.readline(1 << 16) not in (b"\r\n", b"\n", b""):
                    pass
                return b"".join(parts)
            chunk = self.rfile.read(size)
            if len(chunk) != size or self.rfile.readline(3) not in (b"\r\n", b"\n"):
                raise Framing("a chunk is shorter than its size")
            parts.append(chunk)

    def parse(self, raw: bytes) -> Any:
        try:
            return json.loads(raw, parse_constant=_no_constants)
        except ValueError as exc:
            raise _unprocessable(f"could not read the body as JSON ({exc})") from exc

    def redirect(self, target: str) -> None:
        host = self.headers.get("Host") or "%s:%d" % self.server.server_address[:2]
        self.send_response(HTTPStatus.TEMPORARY_REDIRECT)
        self.send_header("Location", f"http://{host}{target}")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def send_text(self, status: HTTPStatus, text: str) -> None:
        data = f"{int(status)}: {text}\n".encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_json(self, status: HTTPStatus, body: Any, headers: dict[str, str] | None = None) -> None:
        try:
            data = json.dumps(body, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
        except ValueError:
            status, headers = HTTPStatus.INTERNAL_SERVER_ERROR, None
            data = b'{"detail":"the reply holds a number that JSON cannot carry"}'
        self.send_response(status)
        self.send_header("Content-Type", JSON_TYPE)
        self.send_header("Content-Length", str(len(data)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)


def serve(model: DM, host: str = "127.0.0.1", port: int = 8000, *, api_key: str | None = None) -> SystemOneServer:
    return SystemOneServer((host, port), Decider(model), api_key=api_key)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Serve the System One decisions from a decision model over HTTP.")
    ap.add_argument("--model", default=HERE, help="package directory or Hub id (default: the package of this file)")
    ap.add_argument("--revision", help="Hub revision (branch, tag or commit)")
    ap.add_argument("--device", help="default: cuda:0 when a GPU is visible, else cpu")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="questions per forward pass")
    ap.add_argument("--verify", choices=VERIFY_MODES, default="fast", help="package check against MODEL_MANIFEST.json")
    ap.add_argument("--name", default="dm", help="model name the server reports (default: dm)")
    ap.add_argument("--no-warmup", action="store_true", help="serve without first compiling the kernels")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    model = DM.from_pretrained(
        args.model,
        revision=args.revision,
        device=args.device,
        batch_size=args.batch_size,
        verify=args.verify,
    )
    if args.name:
        model.model_name = args.name
    if not args.no_warmup:
        model.warmup()
    api_key = os.environ.get(KEY_VARIABLE) or None
    server = serve(model, args.host, args.port, api_key=api_key)
    host, port = server.server_address[:2]
    access = "bearer key required" if api_key else f"open (set {KEY_VARIABLE} to require a bearer key)"
    print(f"DM System One server: {model.model_name} on http://{host}:{port}{SYSTEM_ONE_PATH}; {access}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
