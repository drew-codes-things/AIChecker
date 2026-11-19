import json

import pytest

from aichecker.web.upstream import SECRET_HEADER, UpstreamError, WorkerClient
from fakes import RecordingServer

SECRET = b"s" * 40


class HeaderServer(RecordingServer):
    pass


def test_requests_carry_the_shared_secret_and_json():
    seen = {}

    def route(payload):
        return 200, {"ok": True, "echo": payload}

    with RecordingServer({("POST", "/v1/score"): route}) as server:
        original = server.httpd.RequestHandlerClass._serve

        def capture(self):
            seen["secret"] = self.headers.get(SECRET_HEADER)
            seen["type"] = self.headers.get("Content-Type")
            return original(self)

        server.httpd.RequestHandlerClass.do_POST = capture
        answer = WorkerClient(server.url, SECRET).score("hello")
    assert answer["echo"] == {"text": "hello"} and seen["secret"] == SECRET.decode() and seen["type"] == "application/json"


def test_busy_answers_keep_their_retry_after_and_flag():
    def route(payload):
        return 503, {"error": "the engine is answering", "busy": True}

    with RecordingServer({("POST", "/v1/score"): route}) as server:
        with pytest.raises(UpstreamError) as error:
            WorkerClient(server.url, SECRET).score("x")
    assert error.value.status == 503 and error.value.busy and "engine" in str(error.value)


def test_client_errors_pass_through_but_auth_failures_become_bad_gateway():
    with RecordingServer({("POST", "/v1/score"): lambda p: (413, {"error": "too long"})}) as server:
        with pytest.raises(UpstreamError) as too_long:
            WorkerClient(server.url, SECRET).score("x")
    assert too_long.value.status == 413
    with RecordingServer({("POST", "/v1/score"): lambda p: (401, {"error": "missing or wrong shared secret"})}) as server:
        with pytest.raises(UpstreamError) as unauthorised:
            WorkerClient(server.url, SECRET).score("x")
    assert unauthorised.value.status == 502


def test_an_unreachable_worker_is_a_clear_bad_gateway():
    with pytest.raises(UpstreamError, match="not reachable") as error:
        WorkerClient("http://127.0.0.1:1", SECRET).health()
    assert error.value.status == 502


def test_embed_and_rewrite_use_their_routes():
    routes = {("POST", "/v1/embed"): {"vectors": [[1.0]]}, ("POST", "/v1/rewrite"): {"text": "new"}}
    with RecordingServer(routes) as server:
        client = WorkerClient(server.url, SECRET)
        assert client.embed(["a"]) == [[1.0]]
        assert client.rewrite("t", ["s"], {"words": 1})["text"] == "new"
    assert json.dumps(server.requests[1][2], sort_keys=True) == json.dumps({"profile": {"words": 1}, "samples": ["s"], "text": "t"}, sort_keys=True)
