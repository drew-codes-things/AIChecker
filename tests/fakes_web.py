from aichecker.web.config import AppSettings
from aichecker.web.upstream import UpstreamError

TEST_USER = "demo-user"


class FakeWorker:
    def __init__(self):
        self.scored = []
        self.down = False

    def _check(self):
        if self.down:
            raise UpstreamError("the scoring worker is not reachable right now")

    def health(self):
        self._check()
        return {"ok": True, "pair_loaded": False, "gate_held": False}

    def score(self, text):
        self._check()
        self.scored.append(text)
        return {"assessment": {"band": "little_sign"}, "sentences": []}

    def embed(self, texts):
        self._check()
        return [[float(len(text) % 5 + 1), 1.0, 2.0] for text in texts]

    def rewrite(self, text, samples, profile):
        return {"text": text + " (rewritten)", "warnings": [], "kept_original": False}


def make_settings(tmp_path, folder=None, **overrides) -> AppSettings:
    values = {"data_dir": tmp_path, "port": 0, "trusted_proxies": ("127.0.0.1",), "allowed_users": (TEST_USER,),
              "worker_url": "http://x", "worker_secret_file": tmp_path / "s", "calibration_file": tmp_path / "c",
              "voice_folder": folder, "max_check_characters": 5000, "voice_max_chunks_per_file": 300, "auth_mode": "proxy"}
    values.update(overrides)
    return AppSettings(**values)
