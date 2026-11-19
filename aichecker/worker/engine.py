import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass

log = logging.getLogger("aichecker.engine")

CONTROLLER_TIMEOUT_SECONDS = 5
CONTROLLER_BUSY_TIMEOUT_SECONDS = 200
ENGINE_QUICK_TIMEOUT_SECONDS = 5
ENGINE_UNLOAD_TIMEOUT_SECONDS = 120
ENGINE_EMBED_TIMEOUT_SECONDS = 180
ENGINE_CHAT_TIMEOUT_SECONDS = 600
PARTIAL_OFFLOAD_SHARE = 0.9
IN_FLIGHT_KEY = "arty_requests_in_flight"
SECONDS_SINCE_USED_KEY = "seconds_since_arty_was_used"


class EngineError(RuntimeError):
    pass


@dataclass(frozen=True)
class ControllerStatus:
    in_flight: int
    seconds_since_used: float


def _call(method: str, url: str, payload: dict | None, timeout: float) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    return json.loads(body) if body else {}


class EngineLink:
    def __init__(self, controller_url: str, engine_url: str):
        self.controller_url = controller_url.rstrip("/")
        self.engine_url = engine_url.rstrip("/")

    def controller_status(self) -> ControllerStatus | None:
        if not self.controller_url:
            return None
        try:
            body = _call("GET", f"{self.controller_url}/status", None, CONTROLLER_TIMEOUT_SECONDS)
            return ControllerStatus(int(body[IN_FLIGHT_KEY]), float(body[SECONDS_SINCE_USED_KEY]))
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as error:
            log.warning("the trainer controller did not give a status: %s", error)
            return None

    def controller_busy(self, request_id: str) -> bool:
        return self._controller_post("busy", request_id, CONTROLLER_BUSY_TIMEOUT_SECONDS)

    def controller_done(self, request_id: str) -> bool:
        return self._controller_post("done", request_id, CONTROLLER_TIMEOUT_SECONDS)

    def _controller_post(self, path: str, request_id: str, timeout: float) -> bool:
        if not self.controller_url:
            return False
        try:
            _call("POST", f"{self.controller_url}/{path}", {"id": request_id}, timeout)
            return True
        except (urllib.error.URLError, OSError, ValueError) as error:
            log.warning("the trainer controller did not take /%s: %s", path, error)
            return False

    def engine_models(self) -> list[str] | None:
        try:
            body = _call("GET", f"{self.engine_url}/api/ps", None, ENGINE_QUICK_TIMEOUT_SECONDS)
            return [entry["name"] for entry in body.get("models", [])]
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as error:
            log.warning("the engine did not list its loaded models: %s", error)
            return None

    def engine_partial_models(self) -> list[str]:
        try:
            body = _call("GET", f"{self.engine_url}/api/ps", None, ENGINE_QUICK_TIMEOUT_SECONDS)
            return [entry["name"] for entry in body.get("models", [])
                    if entry.get("size") and entry.get("size_vram") is not None
                    and entry["size_vram"] < entry["size"] * PARTIAL_OFFLOAD_SHARE]
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as error:
            log.warning("the engine did not list its loaded models: %s", error)
            return []

    def unload_named_engine_models(self, names: list[str]) -> None:
        for name in names:
            try:
                _call("POST", f"{self.engine_url}/api/generate", {"model": name, "keep_alive": 0}, ENGINE_UNLOAD_TIMEOUT_SECONDS)
            except (urllib.error.URLError, OSError, ValueError) as error:
                log.warning("could not unload %s: %s", name, error)

    def unload_engine_models(self) -> list[str]:
        loaded = self.engine_models() or []
        self.unload_named_engine_models(loaded)
        return loaded

    def load_engine_model(self, name: str) -> bool:
        try:
            _call("POST", f"{self.engine_url}/api/generate", {"model": name}, ENGINE_CHAT_TIMEOUT_SECONDS)
            return True
        except (urllib.error.URLError, OSError, ValueError) as error:
            log.warning("could not put %s back: %s", name, error)
            return False

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        try:
            body = _call("POST", f"{self.engine_url}/api/embed", {"model": model, "input": texts}, ENGINE_EMBED_TIMEOUT_SECONDS)
            vectors = body["embeddings"]
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as error:
            raise EngineError(f"the engine could not embed with {model}: {error}") from error
        if len(vectors) != len(texts):
            raise EngineError(f"the engine returned {len(vectors)} vectors for {len(texts)} texts")
        return vectors

    def chat(self, model: str, messages: list[dict], options: dict) -> str:
        payload = {"model": model, "messages": messages, "stream": False, "think": False, "options": options}
        try:
            body = _call("POST", f"{self.engine_url}/api/chat", payload, ENGINE_CHAT_TIMEOUT_SECONDS)
            return body["message"]["content"]
        except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError) as error:
            raise EngineError(f"the engine could not answer with {model}: {error}") from error
