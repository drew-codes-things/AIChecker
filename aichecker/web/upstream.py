import json
import urllib.error
import urllib.request

SECRET_HEADER = "X-Checker-Secret"
SCORE_TIMEOUT_SECONDS = 240
EMBED_TIMEOUT_SECONDS = 240
REWRITE_TIMEOUT_SECONDS = 720
QUICK_TIMEOUT_SECONDS = 10
BAD_GATEWAY = 502
SERVICE_UNAVAILABLE = 503


class UpstreamError(RuntimeError):
    def __init__(self, message: str, status: int = BAD_GATEWAY, busy: bool = False, retry_after: int | None = None):
        super().__init__(message)
        self.status = status
        self.busy = busy
        self.retry_after = retry_after


class WorkerClient:
    def __init__(self, base_url: str, secret: bytes):
        self.base_url = base_url.rstrip("/")
        self.secret = secret.decode().strip()

    def _call(self, method: str, path: str, payload: dict | None, timeout: float) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        headers = {SECRET_HEADER: self.secret}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(f"{self.base_url}{path}", data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            raise self._refusal(error) from error
        except (urllib.error.URLError, OSError, TimeoutError) as error:
            raise UpstreamError("the scoring worker is not reachable right now") from error
        except ValueError as error:
            raise UpstreamError("the scoring worker sent an answer that could not be read") from error

    @staticmethod
    def _refusal(error: urllib.error.HTTPError) -> UpstreamError:
        try:
            body = json.loads(error.read())
            message = str(body.get("error") or f"the worker answered {error.code}")
            busy = bool(body.get("busy"))
        except (ValueError, AttributeError):
            message, busy = f"the worker answered {error.code}", False
        retry = error.headers.get("Retry-After") if error.headers else None
        retry_after = int(retry) if retry and retry.isdigit() else None
        if error.code == SERVICE_UNAVAILABLE:
            return UpstreamError(message, SERVICE_UNAVAILABLE, True, retry_after)
        status = error.code if 400 <= error.code < 500 and error.code not in (401, 403) else BAD_GATEWAY
        return UpstreamError(message, status, busy, retry_after)

    def health(self) -> dict:
        return self._call("GET", "/v1/health", None, QUICK_TIMEOUT_SECONDS)

    def score(self, text: str) -> dict:
        return self._call("POST", "/v1/score", {"text": text}, SCORE_TIMEOUT_SECONDS)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._call("POST", "/v1/embed", {"texts": texts}, EMBED_TIMEOUT_SECONDS)["vectors"]

    def rewrite(self, text: str, samples: list[str], profile: dict | None) -> dict:
        return self._call("POST", "/v1/rewrite", {"text": text, "samples": samples, "profile": profile}, REWRITE_TIMEOUT_SECONDS)
