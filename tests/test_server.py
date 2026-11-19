import http.client
import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from aichecker.worker.auth import SECRET_HEADER
from aichecker.worker.config import WorkerSettings
from aichecker.worker.gate import GateBusy
from aichecker.worker.humanise import Rewriter
from aichecker.worker.records import DocumentScore, ScoringInterrupted, SentenceStat
from aichecker.worker.server import Services, make_handler
from aichecker.worker.session import WorkerBusy
from fakes import FakeLink, FakeSession
from support import AI_LIKE, make_calibration

SECRET = "k" * 40
TEXT = "This is the first sentence of the sample. This is the second one, a little longer than the first."


def make_settings(peers=("127.0.0.1",)):
    return WorkerSettings(
        bind_host="127.0.0.1", port=0, allowed_peers=peers, secret=SECRET.encode(), model_dir=Path("/none"),
        observer_name="a", performer_name="b", calibration_file=Path("/none"), lock_file=Path("/none"),
        controller_url="", engine_url="", idle_unload_seconds=45, quiet_seconds=20, gate_wait_seconds=5,
        min_free_gpu_mb=9000, max_request_bytes=5000, max_text_chars=2000, rewrite_model="qwen3:14b", embed_model="nomic-embed-text")


DOCUMENT = DocumentScore(tokens=100, scored_tokens=100, truncated=False, characters_scored=len(TEXT), features=dict(AI_LIKE),
                         sentences=[SentenceStat(0, 40, 9, 5.0, 8.0), SentenceStat(41, 98, 15, 12.0, 11.0)])


@pytest.fixture
def running(request):
    peers = getattr(request, "param", ("127.0.0.1",))
    link = FakeLink()
    session = FakeSession(DOCUMENT)
    services = Services(make_settings(peers), make_calibration(), session, link, Rewriter(link, "qwen3:14b"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(services))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1], services, link
    server.shutdown()
    server.server_close()


def call(port, method, path, body=None, headers=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    sent = {SECRET_HEADER: SECRET} if headers is None else headers
    payload = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    connection.request(method, path, body=payload, headers=sent)
    response = connection.getresponse()
    data = response.read()
    connection.close()
    return response.status, (json.loads(data) if data else {}), response


def test_a_request_without_the_secret_header_is_refused_with_401(running):
    port, _, _ = running
    status, body, _ = call(port, "GET", "/v1/health", headers={})
    assert status == 401 and "secret" in body["error"]


def test_a_wrong_secret_is_refused_and_nothing_is_scored(running):
    port, services, _ = running
    status, _, _ = call(port, "POST", "/v1/score", {"text": TEXT}, headers={SECRET_HEADER: "wrong" * 10})
    assert status == 401
    assert services.session.scorer.seen == []


@pytest.mark.parametrize("running", [("192.0.2.16",)], indirect=True)
def test_a_caller_from_the_wrong_address_is_refused_with_403_even_with_the_secret(running):
    port, services, _ = running
    status, body, _ = call(port, "POST", "/v1/score", {"text": TEXT})
    assert status == 403 and "address" in body["error"]
    assert services.session.scorer.seen == []


def test_health_and_calibration_are_served_to_the_app(running):
    port, _, _ = running
    status, body, _ = call(port, "GET", "/v1/health")
    assert status == 200 and body["ok"] is True and body["pair_loaded"] is False
    status, body, _ = call(port, "GET", "/v1/calibration")
    assert status == 200 and body["thresholds"]["fpr1"] == 2.0


def test_scoring_returns_assessment_features_and_marked_sentences(running):
    port, services, _ = running
    status, body, response = call(port, "POST", "/v1/score", {"text": TEXT})
    assert status == 200
    assert body["assessment"]["band"] == "strong_sign"
    assert body["features"]["binoculars"] == 0.62
    assert [sentence["level"] for sentence in body["sentences"]] == ["strong", "none"]
    assert response.getheader("Cache-Control") == "no-store"
    assert services.session.scorer.seen == [TEXT]


def test_text_validation_and_size_limits(running):
    port, _, _ = running
    assert call(port, "POST", "/v1/score", {"text": ""})[0] == 400
    assert call(port, "POST", "/v1/score", {"text": "   "})[0] == 400
    assert call(port, "POST", "/v1/score", {"text": 5})[0] == 400
    assert call(port, "POST", "/v1/score", {})[0] == 400
    assert call(port, "POST", "/v1/score", {"text": "x" * 2001})[0] == 413


def test_body_limits_and_malformed_json(running):
    port, _, _ = running
    assert call(port, "POST", "/v1/score", raw=b"x" * 6000)[0] == 413
    assert call(port, "POST", "/v1/score", raw=b"{not json")[0] == 400
    assert call(port, "POST", "/v1/score", raw=b"[1,2]")[0] == 400
    assert call(port, "POST", "/v1/nowhere", {"text": "x"})[0] == 404
    assert call(port, "GET", "/v1/nowhere")[0] == 404


def test_the_gate_being_busy_answers_503_with_retry_after_and_does_not_crash(running):
    port, services, _ = running
    services.session.error = GateBusy("the model engine is answering or was used a moment ago", retry_after=15)
    status, body, response = call(port, "POST", "/v1/score", {"text": TEXT})
    assert status == 503 and body["busy"] is True and response.getheader("Retry-After") == "15"
    services.session.error = WorkerBusy("another request is using the checker")
    assert call(port, "POST", "/v1/score", {"text": TEXT})[0] == 503
    services.session.error = ScoringInterrupted("the model engine started answering")
    assert call(port, "POST", "/v1/score", {"text": TEXT})[0] == 503
    services.session.error = None
    assert call(port, "POST", "/v1/score", {"text": TEXT})[0] == 200


def test_embedding_validates_input_and_goes_through_engine_work(running):
    port, services, link = running
    status, body, _ = call(port, "POST", "/v1/embed", {"texts": ["alpha", "beta"]})
    assert status == 200 and len(body["vectors"]) == 2 and body["model"] == "nomic-embed-text"
    assert services.session.engine_calls == 1
    assert link.calls[-1][2] == ("clustering: alpha", "clustering: beta")
    assert call(port, "POST", "/v1/embed", {"texts": []})[0] == 400
    assert call(port, "POST", "/v1/embed", {"texts": ["x"] * 65})[0] == 400
    assert call(port, "POST", "/v1/embed", {"texts": ["", "a"]})[0] == 400
    assert call(port, "POST", "/v1/embed", {"texts": [1]})[0] == 400


def test_rewrite_validates_input_and_returns_the_engine_text(running):
    port, _, link = running
    link.reply = lambda messages: "A rewritten sentence that is long enough to pass the length guard."
    body = {"text": "An ordinary sentence that is long enough to rewrite here.", "samples": ["I write like this."]}
    status, answer, _ = call(port, "POST", "/v1/rewrite", body)
    assert status == 200 and answer["text"].startswith("A rewritten") and answer["model"] == "qwen3:14b"
    assert call(port, "POST", "/v1/rewrite", {"text": "x", "samples": []})[0] == 400
    assert call(port, "POST", "/v1/rewrite", {"text": "x", "samples": [""]})[0] == 400
    assert call(port, "POST", "/v1/rewrite", {"text": "x", "samples": ["s"], "profile": "nope"})[0] == 400
    assert call(port, "POST", "/v1/rewrite", {"text": "x" * 2500, "samples": ["s"]})[0] == 413
