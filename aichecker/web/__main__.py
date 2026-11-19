import logging
import sys

from waitress import serve

from ..calibration import Calibration
from .app import create_app, load_calibration
from .auth import LocalAuth, ProxyAuth, SecretKeyError, load_or_create_secret
from .config import AppSettings, ConfigError, load_settings
from .signin import OWNER_FILE_NAME, OwnerAccount, PasswordAuth
from .upstream import WorkerClient
from .voice import VoiceLibrary

THREADS = 8
CHANNEL_TIMEOUT_SECONDS = 600
SESSION_COOKIE = "ai_checker_session"
MIN_WORKER_SECRET_BYTES = 32

log = logging.getLogger("aichecker.web")


def build_auth(settings: AppSettings, secret: bytes):
    if settings.auth_mode == "local":
        return LocalAuth(settings.local_user, settings.port)
    if settings.auth_mode == "proxy":
        return ProxyAuth(settings.trusted_proxies, settings.user_header, settings.allowed_users)
    return PasswordAuth(OwnerAccount(settings.data_dir / OWNER_FILE_NAME), secret, SESSION_COOKIE, settings.secure_cookies,
                        settings.allowed_hosts, trusted_proxies=settings.trusted_proxies)


def waitress_options(settings: AppSettings) -> dict:
    keeps_forwarded_for = settings.auth_mode == "builtin" and bool(settings.trusted_proxies)
    return {"clear_untrusted_proxy_headers": not keeps_forwarded_for}


def read_worker_secret(settings: AppSettings) -> bytes:
    if settings.worker_secret_file is not None:
        try:
            secret = settings.worker_secret_file.read_bytes().strip()
        except OSError as error:
            raise ConfigError(f"cannot read WORKER_SECRET_FILE {settings.worker_secret_file}: {error.strerror}") from error
    else:
        secret = settings.worker_secret.encode()
    if not secret:
        raise ConfigError("the worker secret is missing: set WORKER_SECRET (or WORKER_SECRET_FILE) to the same value the worker uses")
    if len(secret) < MIN_WORKER_SECRET_BYTES:
        raise ConfigError(f"the worker secret is shorter than {MIN_WORKER_SECRET_BYTES} bytes")
    return secret


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    try:
        settings = load_settings()
        worker_secret = read_worker_secret(settings)
        secret = load_or_create_secret(settings.data_dir, settings.secret_key)
    except (ConfigError, SecretKeyError) as error:
        log.error("configuration problem: %s", error)
        return 2
    auth = build_auth(settings, secret)
    if isinstance(auth, PasswordAuth) and auth.needs_setup():
        auth.setup_code()
    library = VoiceLibrary(settings.data_dir / "voice.db")
    worker = WorkerClient(settings.worker_url, worker_secret)
    calibration: Calibration | None = load_calibration(settings.calibration_file)
    app = create_app(settings, auth, secret, library, worker, calibration)
    log.info("AI Checker on http://%s:%d, data in %s, sign-in %s", settings.bind_host, settings.port, settings.data_dir,
             settings.auth_mode)
    serve(app, host=settings.bind_host, port=settings.port, threads=THREADS, channel_timeout=CHANNEL_TIMEOUT_SECONDS,
          ident="ai-checker", **waitress_options(settings))
    return 0


if __name__ == "__main__":
    sys.exit(main())
