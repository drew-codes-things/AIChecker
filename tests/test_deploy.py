import os
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DEPLOY = REPO / "deploy"
SCRIPTS = (REPO / "deploy.sh", DEPLOY / "ct-app.sh", DEPLOY / "ct-worker.sh")
PLACEHOLDER = re.compile(r"__[A-Z_]+__")


def placeholders_in(path: Path) -> set[str]:
    return set(PLACEHOLDER.findall(path.read_text()))


REFUSING_NETWORK_TOOL = "#!/bin/sh\necho network tool called >&2\nexit 99\n"


@pytest.fixture
def no_network_tools(tmp_path, monkeypatch):
    folder = tmp_path / "stub-bin"
    folder.mkdir()
    for name in ("ssh", "scp"):
        tool = folder / name
        tool.write_text(REFUSING_NETWORK_TOOL)
        tool.chmod(0o755)
    monkeypatch.setenv("PATH", f"{folder}:{os.environ['PATH']}")


def run_bash(script: str):
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_the_deploy_scripts_are_valid_bash(script):
    assert run_bash(f"bash -n {script}").returncode == 0


@pytest.mark.parametrize("template, filler", [("app.env", "ct-app.sh"), ("worker.env", "ct-worker.sh"),
                                              ("ai-checker-worker.service", "ct-worker.sh"), ("ai-checker.nft", "ct-worker.sh"),
                                              ("Caddyfile.ai-checker", "ct-app.sh")])
def test_every_placeholder_in_a_template_is_filled_by_the_script_that_installs_it(template, filler):
    filled = placeholders_in(DEPLOY / filler)
    assert placeholders_in(DEPLOY / template) <= filled


def test_the_app_env_keeps_the_proxy_gate():
    lines = (DEPLOY / "app.env").read_text().splitlines()
    assert "AUTH_MODE=proxy" in lines
    assert any(line.startswith("TRUSTED_PROXIES=127.0.0.1") for line in lines)


def pin_function() -> str:
    text = (DEPLOY / "ct-app.sh").read_text()
    start = text.index("pin_proxy_mode_in_older_env() {")
    return text[start:text.index("\n}\n", start) + 3]


def test_an_older_env_file_without_a_sign_in_mode_is_pinned_to_the_proxy(tmp_path):
    env_file = tmp_path / "app.env"
    env_file.write_text("PORT=8113\nALLOWED_USERS=admin\n")
    result = run_bash(f'ENV_FILE={env_file}\n{pin_function()}\npin_proxy_mode_in_older_env')
    assert result.returncode == 0 and "AUTH_MODE=proxy was added" in result.stdout
    assert env_file.read_text().splitlines() == ["PORT=8113", "ALLOWED_USERS=admin", "AUTH_MODE=proxy"]


@pytest.mark.parametrize("existing", ["AUTH_MODE=builtin", "AUTH_MODE=proxy"])
def test_an_env_file_that_chose_a_mode_is_left_alone(tmp_path, existing):
    env_file = tmp_path / "app.env"
    env_file.write_text(f"PORT=8113\n{existing}\n")
    result = run_bash(f'ENV_FILE={env_file}\n{pin_function()}\npin_proxy_mode_in_older_env')
    assert result.returncode == 0 and result.stdout == ""
    assert env_file.read_text() == f"PORT=8113\n{existing}\n"


def site_file(tmp_path, **overrides) -> Path:
    values = dict(line.split("=", 1) for line in (DEPLOY / "site.env.example").read_text().splitlines())
    values.update({"WORKER_CT": "1", "APP_CT": "2"})
    values.update(overrides)
    path = tmp_path / "site.env"
    path.write_text("".join(f"{name}={value}\n" for name, value in values.items() if value is not None))
    return path


@pytest.mark.parametrize("overrides, complaint", [
    ({"WORKER_DATA_ROOT": ""}, "WORKER_DATA_ROOT"),
    ({"WORKER_DATA_ROOT": "relative/path"}, "WORKER_DATA_ROOT"),
    ({"WORKER_DATA_ROOT": "/srv/x; rm -rf /"}, "WORKER_DATA_ROOT"),
    ({"GPU_LOCK_FILE": "no-slash"}, "GPU_LOCK_FILE"),
    ({"CONTROLLER_URL": "ftp://host"}, "CONTROLLER_URL"),
    ({"CONTROLLER_URL": "http://host:1/$(id)"}, "CONTROLLER_URL"),
    ({"WORKER_HOST": "host name"}, "WORKER_HOST"),
])
def test_deploy_refuses_bad_site_values_before_touching_any_server(tmp_path, no_network_tools, overrides, complaint):
    result = run_bash(f"SITE_ENV={site_file(tmp_path, **overrides)} bash {REPO / 'deploy.sh'} backup")
    assert result.returncode == 1
    assert "refusing" in result.stdout and complaint in result.stdout
    assert "network tool called" not in result.stderr
