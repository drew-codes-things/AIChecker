import pytest

from aichecker.worker.config import ConfigError, load_settings

SETTING_NAMES = ("ALLOWED_PEERS", "BIND_HOST", "PORT", "SECRET", "SECRET_FILE", "MODEL_DIR", "CALIBRATION_FILE", "GPU_LOCK_FILE",
                 "CONTROLLER_URL", "ENGINE_URL", "REWRITE_MODEL", "EMBED_MODEL", "OBSERVER_NAME", "PERFORMER_NAME")
SECRET = "w" * 40


def settings_with(monkeypatch, tmp_path, **values):
    monkeypatch.chdir(tmp_path)
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return load_settings()


def test_the_defaults_listen_on_loopback_for_loopback_and_name_no_machine(monkeypatch, tmp_path):
    settings = settings_with(monkeypatch, tmp_path, SECRET=SECRET)
    assert settings.bind_host == "127.0.0.1"
    assert settings.allowed_peers == ("127.0.0.1", "::1")
    assert settings.controller_url == ""
    assert settings.engine_url == "http://127.0.0.1:11438"
    assert settings.model_dir == (tmp_path / "models").resolve()
    assert settings.lock_file.name == "gpu.lock"
    assert settings.calibration_file.name == "calibration.json" and settings.calibration_file.is_file()
    assert settings.secret == SECRET.encode()


def test_the_secret_can_come_from_a_file(monkeypatch, tmp_path):
    (tmp_path / "worker.secret").write_text(SECRET + "\n")
    settings = settings_with(monkeypatch, tmp_path, SECRET_FILE=str(tmp_path / "worker.secret"))
    assert settings.secret == SECRET.encode()


def test_a_missing_or_short_secret_stops_the_worker(monkeypatch, tmp_path):
    with pytest.raises(ConfigError, match="SECRET"):
        settings_with(monkeypatch, tmp_path)
    with pytest.raises(ConfigError, match="shorter"):
        settings_with(monkeypatch, tmp_path, SECRET="short")


def test_a_secret_and_a_secret_file_together_is_ambiguous(monkeypatch, tmp_path):
    with pytest.raises(ConfigError, match="SECRET_FILE"):
        settings_with(monkeypatch, tmp_path, SECRET=SECRET, SECRET_FILE=str(tmp_path / "x"))


@pytest.mark.parametrize("host", ["0.0.0.0", "::"])
def test_a_wildcard_bind_address_is_refused(monkeypatch, tmp_path, host):
    with pytest.raises(ConfigError, match="BIND_HOST"):
        settings_with(monkeypatch, tmp_path, SECRET=SECRET, BIND_HOST=host)


def test_the_peers_and_the_engine_come_from_the_environment(monkeypatch, tmp_path):
    settings = settings_with(monkeypatch, tmp_path, SECRET=SECRET, BIND_HOST="192.0.2.13", ALLOWED_PEERS="192.0.2.16",
                             ENGINE_URL="http://192.0.2.20:9000/", REWRITE_MODEL="some-model:7b", CONTROLLER_URL="http://192.0.2.30:8790/")
    assert settings.allowed_peers == ("192.0.2.16",)
    assert settings.engine_url == "http://192.0.2.20:9000"
    assert settings.rewrite_model == "some-model:7b"
    assert settings.controller_url == "http://192.0.2.30:8790"


def test_the_worker_env_file_is_read(monkeypatch, tmp_path):
    (tmp_path / ".worker.env").write_text("SECRET=" + SECRET + "\nPORT=9120\n")
    monkeypatch.delenv("SECRET", raising=False)
    monkeypatch.delenv("PORT", raising=False)
    assert settings_with(monkeypatch, tmp_path).port == 9120


def test_main_stops_with_the_reason_when_the_configuration_is_wrong(monkeypatch, tmp_path):
    from aichecker.worker import server
    monkeypatch.chdir(tmp_path)
    for name in SETTING_NAMES:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit) as stopped:
        server.main()
    assert "configuration problem" in str(stopped.value) and "SECRET" in str(stopped.value)


def test_the_worker_env_example_loads_once_the_secret_is_filled_in(monkeypatch, tmp_path):
    from pathlib import Path
    example = (Path(__file__).resolve().parent.parent / ".worker.env.example").read_text()
    (tmp_path / ".worker.env").write_text(example.replace("SECRET=\n", f"SECRET={SECRET}\n", 1))
    settings = settings_with(monkeypatch, tmp_path)
    assert settings.secret == SECRET.encode() and settings.port == 8120 and settings.bind_host == "127.0.0.1"
