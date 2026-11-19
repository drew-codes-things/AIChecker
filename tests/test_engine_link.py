from aichecker.worker.engine import EngineLink, EngineError
from fakes import RecordingServer
import pytest

STATUS = {"arty_requests_in_flight": 1, "seconds_since_arty_was_used": 7, "training": True}


def test_status_is_read_from_the_controller():
    with RecordingServer({("GET", "/status"): STATUS}) as controller:
        status = EngineLink(controller.url, "http://127.0.0.1:1").controller_status()
    assert (status.in_flight, status.seconds_since_used) == (1, 7.0)


def test_status_is_none_when_the_controller_is_down_or_empty():
    assert EngineLink("http://127.0.0.1:1", "http://127.0.0.1:1").controller_status() is None
    assert EngineLink("", "http://127.0.0.1:1").controller_status() is None
    with RecordingServer({("GET", "/status"): {"unexpected": 1}}) as controller:
        assert EngineLink(controller.url, "http://127.0.0.1:1").controller_status() is None


def test_busy_and_done_post_the_request_id():
    with RecordingServer({("POST", "/busy"): {"ok": True}, ("POST", "/done"): {"ok": True}}) as controller:
        link = EngineLink(controller.url, "http://127.0.0.1:1")
        assert link.controller_busy("abc") and link.controller_done("abc")
    assert controller.requests == [("POST", "/busy", {"id": "abc"}), ("POST", "/done", {"id": "abc"})]


def test_busy_reports_failure_instead_of_raising_when_unreachable():
    assert EngineLink("http://127.0.0.1:1", "http://127.0.0.1:1").controller_busy("abc") is False


def test_unload_sends_keep_alive_zero_for_each_loaded_model():
    routes = {("GET", "/api/ps"): {"models": [{"name": "qwen3:14b"}, {"name": "nomic-embed-text:latest"}]},
              ("POST", "/api/generate"): {}}
    with RecordingServer(routes) as engine:
        gone = EngineLink("", engine.url).unload_engine_models()
    assert gone == ["qwen3:14b", "nomic-embed-text:latest"]
    posts = [payload for method, path, payload in engine.requests if method == "POST"]
    assert posts == [{"model": "qwen3:14b", "keep_alive": 0}, {"model": "nomic-embed-text:latest", "keep_alive": 0}]


def test_engine_models_is_none_when_the_engine_is_down():
    assert EngineLink("", "http://127.0.0.1:1").engine_models() is None
    assert EngineLink("", "http://127.0.0.1:1").unload_engine_models() == []


def test_embed_sends_the_model_and_returns_vectors():
    with RecordingServer({("POST", "/api/embed"): {"embeddings": [[1.0, 2.0], [3.0, 4.0]]}}) as engine:
        vectors = EngineLink("", engine.url).embed("nomic-embed-text", ["a", "b"])
    assert vectors == [[1.0, 2.0], [3.0, 4.0]]
    assert engine.requests[0][2] == {"model": "nomic-embed-text", "input": ["a", "b"]}


def test_embed_refuses_a_wrong_number_of_vectors_and_an_unreachable_engine():
    with RecordingServer({("POST", "/api/embed"): {"embeddings": [[1.0]]}}) as engine:
        with pytest.raises(EngineError, match="1 vectors for 2"):
            EngineLink("", engine.url).embed("m", ["a", "b"])
    with pytest.raises(EngineError):
        EngineLink("", "http://127.0.0.1:1").embed("m", ["a"])


def test_chat_turns_thinking_off_and_returns_the_content():
    with RecordingServer({("POST", "/api/chat"): {"message": {"content": "hello"}}}) as engine:
        reply = EngineLink("", engine.url).chat("qwen3:14b", [{"role": "user", "content": "hi"}], {"temperature": 0.5})
    assert reply == "hello"
    payload = engine.requests[0][2]
    assert payload["think"] is False and payload["stream"] is False and payload["options"] == {"temperature": 0.5}


def test_chat_reports_a_malformed_answer_as_an_engine_error():
    with RecordingServer({("POST", "/api/chat"): {"nothing": True}}) as engine:
        with pytest.raises(EngineError):
            EngineLink("", engine.url).chat("m", [], {})


def test_only_models_mostly_off_the_gpu_count_as_partial_and_can_be_unloaded_by_name():
    listing = {"models": [{"name": "qwen3:14b", "size": 12_000, "size_vram": 3_400},
                          {"name": "nomic-embed-text", "size": 600, "size_vram": 600},
                          {"name": "no-vram-field", "size": 500}]}
    routes = {("GET", "/api/ps"): listing, ("POST", "/api/generate"): {}}
    with RecordingServer(routes) as engine:
        link = EngineLink("", engine.url)
        assert link.engine_partial_models() == ["qwen3:14b"]
        link.unload_named_engine_models(["qwen3:14b"])
    assert ("POST", "/api/generate", {"model": "qwen3:14b", "keep_alive": 0}) in engine.requests


def test_partial_models_are_none_found_when_the_engine_is_down():
    assert EngineLink("", "http://127.0.0.1:1").engine_partial_models() == []
