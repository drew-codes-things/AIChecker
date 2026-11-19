import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from aichecker.worker.engine import ControllerStatus


class FakeLink:
    def __init__(self, status=None, loaded=None):
        self.status = status
        self.loaded = list(loaded or [])
        self.calls = []
        self.unreachable = False

    def controller_status(self):
        return None if self.unreachable else self.status

    def controller_busy(self, request_id):
        self.calls.append(("busy", request_id))
        return True

    def controller_done(self, request_id):
        self.calls.append(("done", request_id))
        return True

    def engine_models(self):
        return list(self.loaded)

    def engine_partial_models(self):
        return [name for name in self.loaded if name in getattr(self, "partial", [])]

    def unload_named_engine_models(self, names):
        self.calls.append(("unload-named", tuple(names)))
        self.loaded = [name for name in self.loaded if name not in names]

    def unload_engine_models(self):
        gone, self.loaded = self.loaded, []
        self.calls.append(("unload", tuple(gone)))
        return gone

    def load_engine_model(self, name):
        self.calls.append(("load", name))
        return True

    def embed(self, model, texts):
        self.calls.append(("embed", model, tuple(texts)))
        return [[float(len(text)), 1.0] for text in texts]

    def chat(self, model, messages, options):
        self.calls.append(("chat", model))
        return self.reply(messages) if hasattr(self, "reply") else "rewritten text of reasonable length here"


QUIET = ControllerStatus(in_flight=0, seconds_since_used=600.0)
ENGINE_ANSWERING = ControllerStatus(in_flight=1, seconds_since_used=0.0)


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class RecordingServer:
    def __init__(self, routes):
        self.routes = routes
        self.requests = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                return

            def _serve(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length)) if length else None
                owner.requests.append((self.command, self.path, payload))
                route = owner.routes.get((self.command, self.path))
                status, body = route(payload) if callable(route) else (200, route if route is not None else {})
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _serve

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


class FakeGate:
    holding = False


class FakeScorerForServer:
    loaded = False

    def __init__(self, document):
        self.document = document
        self.seen = []

    def score(self, text, should_yield=None):
        self.seen.append(text)
        return self.document


class FakeSession:
    def __init__(self, document, error=None):
        self.scorer = FakeScorerForServer(document)
        self.gate = FakeGate()
        self.error = error
        self.engine_calls = 0

    def should_yield(self):
        return False

    def scoring(self):
        from contextlib import contextmanager

        @contextmanager
        def manager():
            if self.error:
                raise self.error
            yield self.scorer

        return manager()

    def engine_work(self):
        from contextlib import contextmanager

        @contextmanager
        def manager():
            if self.error:
                raise self.error
            self.engine_calls += 1
            yield

        return manager()
