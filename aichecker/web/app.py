import logging
from pathlib import Path
from urllib.parse import urlsplit

from flask import Flask, Response, g, jsonify, redirect, request, send_from_directory
from werkzeug.exceptions import HTTPException

from .. import __version__
from ..calibration import Calibration, CalibrationError
from .auth import csrf_matches, csrf_token
from .config import AppSettings
from .extract import MAX_UPLOAD_BYTES, SUPPORTED_EXTENSIONS, ExtractError, extract_text
from .folder import MAX_FILE_CHARACTERS, rescan_folder
from .humanise import humanise
from .jobs import JobRunner, TooManyJobs
from .signin import SIGN_IN_PATHS, SIGN_OUT_PATH, PasswordAuth, add_sign_in_pages
from .upstream import UpstreamError, WorkerClient
from .voice import VoiceError, VoiceLibrary, index_text

STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_REQUEST_BYTES = 40 * 1024 * 1024
CSRF_HEADER = "X-CSRF-Token"
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")
PUBLIC_PATHS = ("/api/health",)
APP_NAME = "AI Checker"
SIGNED_OUT_TEXT = "Sign in through the main sign-in page first."
WRONG_HOST_TEXT = ("this app does not answer on that host name (AUTH_MODE=local answers on localhost, 127.0.0.1 or "
                   "[::1] at its own port; the own sign-in answers on the names in ALLOWED_HOSTS, or with it empty on "
                   "localhost, IP addresses and this machine's name)")
MISDIRECTED_REQUEST = 421
REFERRER_POLICY = "no-referrer"
SIGN_IN_REFERRER_POLICY = "same-origin"
CONTENT_SECURITY_POLICY = "; ".join((
    "default-src 'self'", "script-src 'self'", "style-src 'self'", "font-src 'self'", "img-src 'self' data:",
    "connect-src 'self'", "object-src 'none'", "frame-ancestors 'none'", "base-uri 'none'", "form-action 'self'",
))

log = logging.getLogger("aichecker.web")


def _error(message: str, status: int, retry_after: int | None = None, busy: bool = False):
    body = {"error": message}
    if busy:
        body["busy"] = True
    response = jsonify(body)
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = str(retry_after)
    return response


def _is_same_origin() -> bool:
    origin = request.headers.get("Origin")
    return origin is None or urlsplit(origin).netloc == request.host


def _json_body() -> dict:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise ExtractError("send a JSON object", 400)
    return data


def _text_field(body: dict, limit: int) -> str:
    text = body.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ExtractError("text must be a non-empty string", 400)
    if len(text) > limit:
        raise ExtractError(f"that text is {len(text):,} characters; the limit is {limit:,}", 413)
    return text


def load_calibration(path: Path) -> Calibration | None:
    try:
        return Calibration.load(path)
    except CalibrationError as error:
        log.warning("running without accuracy numbers: %s", error)
        return None


def create_app(settings: AppSettings, auth, secret: bytes, library: VoiceLibrary, worker: WorkerClient,
               calibration: Calibration | None, jobs: JobRunner | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    runner = jobs or JobRunner()
    has_own_sign_in = isinstance(auth, PasswordAuth)
    if has_own_sign_in:
        add_sign_in_pages(app, auth, APP_NAME)

    @app.before_request
    def identify_and_guard_writes():
        if request.path in PUBLIC_PATHS:
            return None
        if not auth.accepts_host(request.host):
            return _error(WRONG_HOST_TEXT, MISDIRECTED_REQUEST)
        if has_own_sign_in and request.path in SIGN_IN_PATHS:
            return None
        user = auth.user_for(request.remote_addr or "", request.headers)
        if user is None:
            if request.path.startswith("/api/"):
                return _error("sign in first", 401)
            if has_own_sign_in:
                return redirect(auth.entry_path(), code=303)
            return Response(SIGNED_OUT_TEXT, status=401, mimetype="text/plain")
        g.user = user
        if request.method in SAFE_METHODS:
            return None
        if not _is_same_origin():
            return _error("this request came from another site", 403)
        if not csrf_matches(secret, user, request.headers.get(CSRF_HEADER)):
            return _error("missing or wrong CSRF token, reload the page", 403)
        return None

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        response.headers["Content-Security-Policy"] = CONTENT_SECURITY_POLICY
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = SIGN_IN_REFERRER_POLICY if request.path in SIGN_IN_PATHS else REFERRER_POLICY
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.errorhandler(ExtractError)
    def extract_refused(error: ExtractError):
        return _error(str(error), error.status)

    @app.errorhandler(VoiceError)
    def voice_refused(error: VoiceError):
        return _error(str(error), error.status)

    @app.errorhandler(UpstreamError)
    def upstream_refused(error: UpstreamError):
        return _error(str(error), error.status, error.retry_after, error.busy)

    @app.errorhandler(TooManyJobs)
    def too_many(error: TooManyJobs):
        return _error(str(error), 429, 10)

    @app.errorhandler(HTTPException)
    def http_error(error: HTTPException):
        if error.code == 413:
            return _error("that request is too large", 413)
        if request.path.startswith("/api/"):
            return _error(error.description or error.name, error.code or 500)
        return error

    @app.get("/")
    def index():
        response = send_from_directory(STATIC_DIR, "index.html", max_age=0)
        response.headers["Cache-Control"] = "no-cache"
        return response

    @app.get("/static/<path:name>")
    def static_file(name: str):
        return send_from_directory(STATIC_DIR, name, max_age=0)

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True})

    @app.get("/api/session")
    def session_state():
        return jsonify({"user": g.user, "csrfToken": csrf_token(secret, g.user), "version": __version__,
                        "accepts": list(SUPPORTED_EXTENSIONS), "maxUploadBytes": MAX_UPLOAD_BYTES,
                        "maxCheckCharacters": settings.max_check_characters, "folderConfigured": settings.voice_folder is not None,
                        "signOutPath": SIGN_OUT_PATH if has_own_sign_in else None})

    @app.get("/api/status")
    def status():
        try:
            reading = worker.health()
            return jsonify({"worker": "up", "pairLoaded": reading.get("pair_loaded"), "gateHeld": reading.get("gate_held")})
        except UpstreamError as error:
            return jsonify({"worker": "down", "reason": str(error)})

    @app.get("/api/accuracy")
    def accuracy():
        if calibration is None:
            return _error("the checker has not been calibrated yet", 404)
        return jsonify({"accuracy": calibration.accuracy, "pair": calibration.pair, "created": calibration.created,
                        "minTokens": calibration.min_tokens, "windowTokens": calibration.window_tokens,
                        "maxTokens": calibration.max_tokens, "thresholds": calibration.thresholds})

    @app.post("/api/extract")
    def extract():
        upload = request.files.get("file")
        if upload is None or not upload.filename:
            raise ExtractError("choose a file to read", 400)
        extracted = extract_text(upload.filename, upload.read(MAX_UPLOAD_BYTES + 1), settings.max_check_characters)
        return jsonify({"name": upload.filename, "text": extracted.text, "characters": extracted.characters,
                        "truncated": extracted.truncated})

    @app.post("/api/check")
    def check():
        return jsonify(worker.score(_text_field(_json_body(), settings.max_check_characters)))

    @app.get("/api/voice")
    def voice_state():
        return jsonify({"totals": library.totals(), "sources": library.sources(),
                        "folderConfigured": settings.voice_folder is not None,
                        "folderName": settings.voice_folder.name if settings.voice_folder else None})

    @app.post("/api/voice/files")
    def voice_add_files():
        uploads = [item for item in request.files.getlist("files") if item.filename]
        if not uploads:
            raise ExtractError("choose at least one file", 400)
        added, skipped = [], []
        for upload in uploads:
            try:
                extracted = extract_text(upload.filename, upload.read(MAX_UPLOAD_BYTES + 1), MAX_FILE_CHARACTERS)
                added.append(index_text(library, upload.filename, extracted.text, worker.embed, settings.voice_max_chunks_per_file))
            except (ExtractError, VoiceError) as error:
                skipped.append({"name": upload.filename, "reason": str(error)})
        return jsonify({"added": added, "skipped": skipped, "totals": library.totals()})

    @app.delete("/api/voice/files/<int:source_id>")
    def voice_remove_file(source_id: int):
        if not library.remove_source(source_id):
            return _error("no such file in the voice library", 404)
        return jsonify({"totals": library.totals()})

    @app.post("/api/voice/rescan")
    def voice_rescan():
        if settings.voice_folder is None:
            return _error("no voice folder is set: choose one in the app's settings file, or upload files instead", 409)
        summary = rescan_folder(library, settings.voice_folder, worker.embed, settings.voice_max_chunks_per_file)
        return jsonify({**summary, "totals": library.totals()})

    @app.post("/api/humanise")
    def humanise_start():
        text = _text_field(_json_body(), settings.max_check_characters)
        job_id = runner.start(lambda progress, cancelled: humanise(text, library, worker, progress, cancelled))
        return jsonify({"job": job_id}), 202

    @app.get("/api/jobs/<job_id>")
    def job_state(job_id: str):
        view = runner.view(job_id)
        if view is None:
            return _error("that rewrite is not running any more (results are kept only until you fetch them)", 404)
        return jsonify(view)

    @app.delete("/api/jobs/<job_id>")
    def job_stop(job_id: str):
        if not runner.stop(job_id):
            return _error("that rewrite is not running any more", 404)
        return "", 204

    return app
