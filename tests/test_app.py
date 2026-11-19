import io
import json

import pytest

from aichecker.web.app import create_app
from aichecker.web.auth import ProxyAuth, csrf_token
from aichecker.web.jobs import JobRunner
from aichecker.web.upstream import UpstreamError
from aichecker.web.voice import VoiceLibrary
from fakes_web import TEST_USER, FakeWorker, make_settings
from support import make_calibration, make_docx

SECRET = b"c" * 40
USER = TEST_USER
PARAGRAPH = "I like to write plainly about ordinary days, and I never use a long word where a short one will do. " * 3


@pytest.fixture
def parts(tmp_path):
    worker = FakeWorker()
    library = VoiceLibrary(tmp_path / "voice.db")
    app = create_app(make_settings(tmp_path), ProxyAuth(("127.0.0.1",), "X-Forwarded-User", (USER,)), SECRET, library, worker,
                     make_calibration(), JobRunner(run_in_thread=False))
    return app, worker, library


def headers(token=True, user=USER):
    result = {}
    if user:
        result["X-Forwarded-User"] = user
    if token:
        result["X-CSRF-Token"] = csrf_token(SECRET, user or USER)
    return result


def post_json(client, path, body, **kwargs):
    return client.post(path, data=json.dumps(body), content_type="application/json", headers=headers(**kwargs))


def test_a_request_without_the_signed_in_user_header_is_refused(parts):
    client = parts[0].test_client()
    assert client.get("/api/session").status_code == 401
    assert client.get("/").status_code == 401
    assert client.get("/api/session", headers={"X-Forwarded-User": "someone-else"}).status_code == 401


def test_the_header_is_ignored_unless_it_comes_from_the_trusted_proxy(parts):
    client = parts[0].test_client()
    response = client.get("/api/session", headers=headers(), environ_overrides={"REMOTE_ADDR": "192.0.2.50"})
    assert response.status_code == 401


def test_health_is_public_and_says_nothing_else(parts):
    response = parts[0].test_client().get("/api/health")
    assert response.status_code == 200 and response.get_json() == {"ok": True}


def test_session_gives_a_csrf_token_and_the_limits(parts):
    body = parts[0].test_client().get("/api/session", headers=headers(token=False)).get_json()
    assert body["csrfToken"] == csrf_token(SECRET, USER) and ".docx" in body["accepts"] and body["maxCheckCharacters"] == 5000


def test_writes_need_the_csrf_token_and_a_same_origin_request(parts):
    client = parts[0].test_client()
    body = json.dumps({"text": "hello there"})
    assert client.post("/api/check", data=body, content_type="application/json", headers=headers(token=False)).status_code == 403
    bad = headers()
    bad["X-CSRF-Token"] = "0" * 64
    assert client.post("/api/check", data=body, content_type="application/json", headers=bad).status_code == 403
    cross = headers()
    cross["Origin"] = "https://evil.example"
    assert client.post("/api/check", data=body, content_type="application/json", headers=cross).status_code == 403
    assert post_json(client, "/api/check", {"text": "hello there"}).status_code == 200


def test_check_passes_the_text_to_the_worker_and_enforces_limits(parts):
    app, worker, _ = parts
    client = app.test_client()
    assert post_json(client, "/api/check", {"text": "A sentence to check."}).get_json()["assessment"]["band"] == "little_sign"
    assert worker.scored == ["A sentence to check."]
    assert post_json(client, "/api/check", {"text": ""}).status_code == 400
    assert post_json(client, "/api/check", {"text": 4}).status_code == 400
    assert post_json(client, "/api/check", {"text": "x" * 5001}).status_code == 413
    assert client.post("/api/check", data="[]", content_type="application/json", headers=headers()).status_code == 400


def test_a_busy_worker_gives_503_with_retry_after_and_a_down_worker_502(parts):
    app, worker, _ = parts
    client = app.test_client()
    worker.down = True
    assert post_json(client, "/api/check", {"text": "hello"}).status_code == 502
    assert client.get("/api/status", headers=headers(token=False)).get_json()["worker"] == "down"
    worker.down = False
    worker.score = lambda text: (_ for _ in ()).throw(UpstreamError("the engine is answering", 503, True, 15))
    response = post_json(client, "/api/check", {"text": "hello"})
    assert response.status_code == 503 and response.headers["Retry-After"] == "15" and response.get_json()["busy"] is True


def test_extract_reads_uploaded_documents_and_reports_the_original_length(parts):
    client = parts[0].test_client()
    data = {"file": (io.BytesIO(make_docx(["First paragraph.", "Second paragraph."])), "essay.docx")}
    response = client.post("/api/extract", data=data, headers=headers(), content_type="multipart/form-data")
    assert response.status_code == 200
    assert response.get_json()["text"] == "First paragraph.\n\nSecond paragraph." and response.get_json()["truncated"] is False


def test_extract_refuses_missing_unsupported_and_oversized_files(parts):
    client = parts[0].test_client()
    assert client.post("/api/extract", data={}, headers=headers(), content_type="multipart/form-data").status_code == 400
    png = {"file": (io.BytesIO(b"abc"), "photo.png")}
    assert client.post("/api/extract", data=png, headers=headers(), content_type="multipart/form-data").status_code == 415
    big = {"file": (io.BytesIO(b"x" * (9 * 1024 * 1024)), "big.txt")}
    assert client.post("/api/extract", data=big, headers=headers(), content_type="multipart/form-data").status_code == 413


def test_the_whole_request_size_is_capped(parts):
    client = parts[0].test_client()
    huge = {"file": (io.BytesIO(b"x" * (41 * 1024 * 1024)), "huge.txt")}
    assert client.post("/api/extract", data=huge, headers=headers(), content_type="multipart/form-data").status_code == 413


def test_voice_files_can_be_added_listed_and_removed(parts):
    app, _, library = parts
    client = app.test_client()
    files = [(io.BytesIO(("\n\n".join([PARAGRAPH] * 4)).encode()), "mine.txt"), (io.BytesIO(b"abc"), "bad.png")]
    response = client.post("/api/voice/files", data={"files": files}, headers=headers(), content_type="multipart/form-data")
    body = response.get_json()
    assert response.status_code == 200 and len(body["added"]) == 1 and body["skipped"][0]["name"] == "bad.png"
    listing = client.get("/api/voice", headers=headers(token=False)).get_json()
    assert listing["totals"]["sources"] == 1 and listing["folderConfigured"] is False
    source_id = listing["sources"][0]["id"]
    assert client.delete(f"/api/voice/files/{source_id}", headers=headers()).status_code == 200
    assert client.delete(f"/api/voice/files/{source_id}", headers=headers()).status_code == 404
    assert library.totals()["sources"] == 0


def test_rescan_without_a_folder_says_how_to_set_one(parts):
    response = parts[0].test_client().post("/api/voice/rescan", headers=headers())
    assert response.status_code == 409 and "folder" in response.get_json()["error"]


def test_humanise_runs_as_a_job_and_returns_before_and_after(parts):
    app, worker, library = parts
    client = app.test_client()
    voice = "\n\n".join([PARAGRAPH] * 8)
    client.post("/api/voice/files", data={"files": (io.BytesIO(voice.encode()), "mine.txt")}, headers=headers(), content_type="multipart/form-data")
    started = post_json(client, "/api/humanise", {"text": "Some text to rewrite for the test."})
    assert started.status_code == 202
    state = client.get(f"/api/jobs/{started.get_json()['job']}", headers=headers(token=False)).get_json()
    assert state["status"] == "done" and state["result"]["rewritten"].endswith("(rewritten)")
    assert state["result"]["before"]["assessment"]["band"] == "little_sign"
    assert client.get(f"/api/jobs/{started.get_json()['job']}", headers=headers(token=False)).status_code == 404


def test_humanise_without_a_voice_fails_the_job_with_a_clear_message(parts):
    client = parts[0].test_client()
    started = post_json(client, "/api/humanise", {"text": "Some text to rewrite for the test."})
    state = client.get(f"/api/jobs/{started.get_json()['job']}", headers=headers(token=False)).get_json()
    assert state["status"] == "failed" and "own writing" in state["error"]


def test_accuracy_comes_from_the_calibration_and_is_404_without_one(parts, tmp_path):
    app, worker, library = parts
    body = app.test_client().get("/api/accuracy", headers=headers(token=False)).get_json()
    assert body["accuracy"]["overall"]["auroc"] == 0.9 and body["created"] == "2026-11-18"
    bare = create_app(make_settings(tmp_path), ProxyAuth(("127.0.0.1",), "X-Forwarded-User", (USER,)), SECRET, library, worker, None)
    assert bare.test_client().get("/api/accuracy", headers=headers(token=False)).status_code == 404


def test_responses_carry_security_headers_and_api_answers_are_not_cached(parts):
    response = parts[0].test_client().get("/api/session", headers=headers(token=False))
    assert "default-src 'self'" in response.headers["Content-Security-Policy"]
    assert response.headers["Cache-Control"] == "no-store" and response.headers["X-Frame-Options"] == "DENY"
