import json
import logging
import signal
import sys
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .. import __version__
from ..calibration import Calibration
from ..report import assess, sentence_level
from .engine import EngineLink, EngineError
from .auth import check_request
from .config import ConfigError, WorkerSettings, load_settings
from .gate import GateBusy, GateSettings, GpuGate
from .humanise import MAX_SEGMENT_CHARS, Rewriter
from .records import ScoringInterrupted
from .session import GpuSession, WorkerBusy

log = logging.getLogger("aichecker.server")

MAX_EMBED_TEXTS = 64
MAX_EMBED_TEXT_CHARS = 4000
MAX_PROFILE_KEYS = 40
JOB_WAIT_SECONDS = 150
RETRY_AFTER_SECONDS = 15


class RequestRefused(Exception):
    def __init__(self, status: int, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.message = message
        self.retry_after = retry_after


@dataclass
class Services:
    settings: WorkerSettings
    calibration: Calibration
    session: GpuSession
    link: EngineLink
    rewriter: Rewriter


def free_gpu_mb() -> int:
    import torch
    free_bytes, _ = torch.cuda.mem_get_info()
    return int(free_bytes / (1024 * 1024))


def restore_engine_models(link: EngineLink):
    def restore(models: list[str]) -> None:
        for name in models:
            link.load_engine_model(name)
    return restore


def build_services(settings: WorkerSettings) -> Services:
    from .scorer import PairScorer
    calibration = Calibration.load(settings.calibration_file)
    link = EngineLink(settings.controller_url, settings.engine_url)
    gate = GpuGate(GateSettings(settings.lock_file, settings.quiet_seconds, settings.gate_wait_seconds, 2.0,
                                settings.min_free_gpu_mb), link, free_gpu_mb)
    scorer = PairScorer(settings.model_dir / settings.observer_name, settings.model_dir / settings.performer_name)
    session = GpuSession(gate, scorer, settings.idle_unload_seconds, JOB_WAIT_SECONDS, restore=restore_engine_models(link))
    return Services(settings, calibration, session, link, Rewriter(link, settings.rewrite_model))


def read_json_object(handler: BaseHTTPRequestHandler, limit: int) -> dict:
    if handler.headers.get("Transfer-Encoding"):
        raise RequestRefused(411, "send a Content-Length, chunked bodies are not accepted")
    try:
        length = int(handler.headers.get("Content-Length") or "")
    except ValueError:
        raise RequestRefused(411, "send a Content-Length header") from None
    if length < 0 or length > limit:
        raise RequestRefused(413, f"the request body may be at most {limit} bytes")
    try:
        body = json.loads(handler.rfile.read(length))
    except (ValueError, UnicodeDecodeError):
        raise RequestRefused(400, "the body must be a JSON object") from None
    if not isinstance(body, dict):
        raise RequestRefused(400, "the body must be a JSON object")
    return body


def require_text(body: dict, key: str, limit: int) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RequestRefused(400, f"{key} must be a non-empty string")
    if len(value) > limit:
        raise RequestRefused(413, f"{key} may be at most {limit} characters")
    return value


def score_text(services: Services, text: str) -> dict:
    calibration = services.calibration
    with services.session.scoring() as scorer:
        document = scorer.score(text, services.session.should_yield)
    sentences = [{"start": stat.start, "end": stat.end, "tokens": stat.tokens, "score": round(stat.binoculars, 3),
                  "level": sentence_level(stat.binoculars, stat.tokens, calibration)} for stat in document.sentences]
    return {"assessment": assess(document.features, document.scored_tokens, calibration),
            "features": {name: round(value, 4) for name, value in document.features.items()},
            "tokens": document.tokens, "scored_tokens": document.scored_tokens, "truncated": document.truncated,
            "characters_scored": document.characters_scored, "sentences": sentences,
            "calibration_created": calibration.created}


def embed_texts(services: Services, body: dict) -> dict:
    texts = body.get("texts")
    if not isinstance(texts, list) or not texts or len(texts) > MAX_EMBED_TEXTS:
        raise RequestRefused(400, f"texts must be a list of 1 to {MAX_EMBED_TEXTS} strings")
    if any(not isinstance(text, str) or not text.strip() or len(text) > MAX_EMBED_TEXT_CHARS for text in texts):
        raise RequestRefused(400, f"each text must be a non-empty string of at most {MAX_EMBED_TEXT_CHARS} characters")
    prefixed = [f"clustering: {text}" for text in texts]
    with services.session.engine_work():
        vectors = services.link.embed(services.settings.embed_model, prefixed)
    return {"vectors": vectors, "model": services.settings.embed_model}


def rewrite_segment(services: Services, body: dict) -> dict:
    text = require_text(body, "text", MAX_SEGMENT_CHARS)
    samples = body.get("samples")
    if not isinstance(samples, list) or not samples or any(not isinstance(item, str) or not item.strip() for item in samples):
        raise RequestRefused(400, "samples must be a non-empty list of strings")
    profile = body.get("profile")
    if profile is not None and (not isinstance(profile, dict) or len(profile) > MAX_PROFILE_KEYS):
        raise RequestRefused(400, "profile must be an object")
    with services.session.engine_work():
        result = services.rewriter.rewrite(text, samples, profile)
    return {"text": result.text, "warnings": result.warnings, "kept_original": result.kept_original,
            "model": services.settings.rewrite_model}


def health(services: Services) -> dict:
    return {"ok": True, "version": __version__, "pair_loaded": services.session.scorer.loaded,
            "gate_held": services.session.gate.holding, "calibration_created": services.calibration.created}


def make_handler(services: Services):
    settings = services.settings

    class Handler(BaseHTTPRequestHandler):
        server_version = "ai-checker-worker"
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            log.info("%s %s", self.address_string(), format % args)

        def _reply(self, status: int, body: dict, retry_after: int | None = None) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            if retry_after is not None:
                self.send_header("Retry-After", str(retry_after))
            self.end_headers()
            self.wfile.write(data)

        def _guard(self) -> bool:
            refusal = check_request(self.client_address[0], self.headers, settings.secret, settings.allowed_peers)
            if refusal is None:
                return True
            self.close_connection = True
            self._reply(refusal.status, {"error": refusal.message})
            return False

        def _run(self, action) -> None:
            try:
                self._reply(200, action())
            except RequestRefused as refusal:
                self.close_connection = True
                self._reply(refusal.status, {"error": refusal.message}, refusal.retry_after)
            except (GateBusy, WorkerBusy) as busy:
                self._reply(503, {"error": str(busy), "busy": True}, getattr(busy, "retry_after", RETRY_AFTER_SECONDS))
            except ScoringInterrupted as interrupted:
                self._reply(503, {"error": str(interrupted), "busy": True}, RETRY_AFTER_SECONDS)
            except EngineError as error:
                self._reply(502, {"error": str(error)})
            except ValueError as error:
                self._reply(400, {"error": str(error)})
            except Exception:
                log.exception("unhandled error in %s", self.path)
                self._reply(500, {"error": "the checker failed; see its log"})

        def do_GET(self):
            if not self._guard():
                return
            routes = {"/v1/health": lambda: health(services), "/v1/calibration": lambda: services.calibration.to_dict()}
            action = routes.get(self.path)
            if action is None:
                self._reply(404, {"error": "unknown path"})
                return
            self._run(action)

        def do_POST(self):
            if not self._guard():
                return
            routes = {
                "/v1/score": lambda body: score_text(services, require_text(body, "text", settings.max_text_chars)),
                "/v1/embed": lambda body: embed_texts(services, body),
                "/v1/rewrite": lambda body: rewrite_segment(services, body),
            }
            action = routes.get(self.path)
            if action is None:
                self._reply(404, {"error": "unknown path"})
                return
            self._run(lambda: action(read_json_object(self, settings.max_request_bytes)))

    return Handler


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    try:
        settings = load_settings()
    except ConfigError as error:
        raise SystemExit(f"configuration problem: {error}") from error
    services = build_services(settings)
    services.session.start_watcher()
    server = ThreadingHTTPServer((settings.bind_host, settings.port), make_handler(services))
    server.daemon_threads = True

    def stop(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    log.info("worker %s listening on %s:%d for %s", __version__, settings.bind_host, settings.port, ",".join(settings.allowed_peers))
    try:
        server.serve_forever()
    finally:
        services.session.stop()
        server.server_close()


if __name__ == "__main__":
    main()
