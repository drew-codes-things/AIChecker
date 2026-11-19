import os
from dataclasses import dataclass
from pathlib import Path

ENV_FILE = Path(".worker.env")
DEFAULT_PORT = 8120
DEFAULT_BIND_HOST = "127.0.0.1"
DEFAULT_ALLOWED_PEERS = "127.0.0.1,::1"
DEFAULT_MODEL_DIR = "./models"
DEFAULT_CALIBRATION_FILE = Path(__file__).resolve().parent.parent / "data" / "calibration.json"
DEFAULT_LOCK_FILE = "./gpu.lock"
DEFAULT_ENGINE_URL = "http://127.0.0.1:11438"
MIN_SECRET_BYTES = 32
WILDCARD_HOSTS = ("0.0.0.0", "::")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class WorkerSettings:
    bind_host: str
    port: int
    allowed_peers: tuple[str, ...]
    secret: bytes
    model_dir: Path
    observer_name: str
    performer_name: str
    calibration_file: Path
    lock_file: Path
    controller_url: str
    engine_url: str
    idle_unload_seconds: float
    quiet_seconds: float
    gate_wait_seconds: float
    min_free_gpu_mb: int
    max_request_bytes: int
    max_text_chars: int
    rewrite_model: str
    embed_model: str


def _text(name: str, default: str) -> str:
    value = os.environ.get(name)
    return (default if value is None else value).strip()


def _number(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = _text(name, str(default))
    try:
        value = float(raw)
    except ValueError as error:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from error
    if not minimum <= value <= maximum:
        raise ConfigError(f"{name} must be between {minimum} and {maximum}, got {value}")
    return value


def read_env_file(path: Path = ENV_FILE) -> None:
    if not path.is_file():
        return
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        if not separator or not name.strip():
            raise ConfigError(f"{path}:{number} should look like NAME=value")
        os.environ.setdefault(name.strip(), value.strip().strip('"').strip("'"))


def read_secret(path: Path) -> bytes:
    try:
        secret = Path(path).read_bytes().strip()
    except OSError as error:
        raise ConfigError(f"cannot read the shared secret at {path}: {error.strerror}") from error
    return _checked_secret(secret, f"the shared secret at {path}")


def _checked_secret(secret: bytes, label: str) -> bytes:
    if len(secret) < MIN_SECRET_BYTES:
        raise ConfigError(f"{label} is shorter than {MIN_SECRET_BYTES} bytes")
    return secret


def load_secret() -> bytes:
    inline = _text("SECRET", "")
    path = _text("SECRET_FILE", "")
    if inline and path:
        raise ConfigError("set SECRET or SECRET_FILE, not both")
    if path:
        return read_secret(Path(path))
    if not inline:
        raise ConfigError("the shared secret is missing: set SECRET (or SECRET_FILE) to the same value the web app uses as WORKER_SECRET")
    return _checked_secret(inline.encode(), "SECRET")


def load_settings() -> WorkerSettings:
    read_env_file()
    peers = tuple(part.strip() for part in _text("ALLOWED_PEERS", DEFAULT_ALLOWED_PEERS).split(",") if part.strip())
    if not peers:
        raise ConfigError("ALLOWED_PEERS must name the address the web app connects from, for example 192.0.2.16")
    bind_host = _text("BIND_HOST", DEFAULT_BIND_HOST)
    if not bind_host or bind_host in WILDCARD_HOSTS:
        raise ConfigError("BIND_HOST must be one address of this machine (127.0.0.1 by default), not a wildcard")
    return WorkerSettings(
        bind_host=bind_host,
        port=int(_number("PORT", DEFAULT_PORT, 1, 65535)),
        allowed_peers=peers,
        secret=load_secret(),
        model_dir=Path(_text("MODEL_DIR", DEFAULT_MODEL_DIR)).resolve(),
        observer_name=_text("OBSERVER_NAME", "Qwen2.5-1.5B"),
        performer_name=_text("PERFORMER_NAME", "Qwen2.5-1.5B-Instruct"),
        calibration_file=Path(_text("CALIBRATION_FILE", "") or DEFAULT_CALIBRATION_FILE),
        lock_file=Path(_text("GPU_LOCK_FILE", DEFAULT_LOCK_FILE)).resolve(),
        controller_url=_text("CONTROLLER_URL", "").rstrip("/"),
        engine_url=(_text("ENGINE_URL", "") or DEFAULT_ENGINE_URL).rstrip("/"),
        idle_unload_seconds=_number("IDLE_UNLOAD_SECONDS", 45, 5, 3600),
        quiet_seconds=_number("QUIET_SECONDS", 20, 0, 600),
        gate_wait_seconds=_number("GATE_WAIT_SECONDS", 90, 1, 1800),
        min_free_gpu_mb=int(_number("MIN_FREE_GPU_MB", 9000, 1000, 24000)),
        max_request_bytes=int(_number("MAX_REQUEST_BYTES", 400_000, 1000, 10_000_000)),
        max_text_chars=int(_number("MAX_TEXT_CHARS", 60_000, 100, 2_000_000)),
        rewrite_model=_text("REWRITE_MODEL", "qwen3:14b"),
        embed_model=_text("EMBED_MODEL", "nomic-embed-text"),
    )
