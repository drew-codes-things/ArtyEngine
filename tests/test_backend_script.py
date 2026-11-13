import os
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "host" / "arty-backend.sh"
SCRIPT_TIMEOUT_SECONDS = 30
SITE_WITH_CONTROLLER = """ARTY_CT=201
ARTY_CONTROLLER_URL=http://192.0.2.10:8790
ARTY_CONTROLLER_SETTINGS=/opt/trainer/settings.json
ARTY_CONTROLLER_PYTHON=/opt/trainer/venv/bin/python
ARTY_CONTROLLER_CODE=/opt/trainer/app/controller.py
ARTY_CONTROLLER_UNIT=trainer-controller
ARTY_CONTROLLER_MODULE=trainer.controller
ARTY_ENGINE_ALLOWED_HOSTS="192.0.2.13 arty.lan"
"""


def fake_pct(folder: Path) -> Path:
    bin_dir = folder / "bin"
    bin_dir.mkdir()
    for tool in ("pct", "systemctl"):
        script = bin_dir / tool
        script.write_text(f'#!/bin/sh\necho "{tool} $@" >> {folder / "calls.log"}\nexit 0\n')
        script.chmod(0o755)
    return bin_dir


def run_backend(folder: Path, site_text: str | None, *arguments):
    site = folder / "site.env"
    if site_text is not None:
        site.write_text(site_text)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("ARTY_")}
    environment.update(ARTY_ENGINE_SITE=str(site), PATH=f"{fake_pct(folder)}:{os.environ['PATH']}")
    return subprocess.run(["bash", str(SCRIPT), *arguments], env=environment, capture_output=True, text=True,
                          timeout=SCRIPT_TIMEOUT_SECONDS)


def calls(folder: Path) -> str:
    log = folder / "calls.log"
    return log.read_text() if log.exists() else ""


def test_the_settings_come_from_the_site_file(tmp_path):
    finished = run_backend(tmp_path, SITE_WITH_CONTROLLER, "settings")
    assert finished.returncode == 0, finished.stderr
    lines = finished.stdout.splitlines()
    assert "container: 201" in lines
    assert "controller: http://192.0.2.10:8790" in lines
    assert "controller settings: /opt/trainer/settings.json" in lines
    assert "controller python: /opt/trainer/venv/bin/python" in lines
    assert "controller code: /opt/trainer/app/controller.py" in lines
    assert "engine trainer_url: http://192.0.2.10:8790" in lines
    assert "controller unit: trainer-controller" in lines
    assert "controller module: trainer.controller" in lines
    assert "engine allowed hosts: 192.0.2.13 arty.lan" in lines
    assert "engine allowed origins: (none)" in lines
    assert calls(tmp_path) == ""


def test_no_controller_leaves_the_engine_without_a_trainer(tmp_path):
    finished = run_backend(tmp_path, "ARTY_CT=201\nARTY_CONTROLLER_URL=none\n", "settings")
    assert finished.returncode == 0, finished.stderr
    assert "engine trainer_url: (none)" in finished.stdout.splitlines()


def test_nothing_runs_without_a_container(tmp_path):
    finished = run_backend(tmp_path, None, "engine")
    assert finished.returncode != 0
    assert "ARTY_CT" in finished.stderr
    assert calls(tmp_path) == ""


def test_a_controller_needs_its_files_named(tmp_path):
    finished = run_backend(tmp_path, "ARTY_CT=201\nARTY_CONTROLLER_URL=http://192.0.2.10:8790\n", "ollama")
    assert finished.returncode != 0
    assert "ARTY_CONTROLLER_SETTINGS" in finished.stderr
    assert calls(tmp_path) == ""


def test_a_controller_needs_its_unit_and_module_named(tmp_path):
    site = SITE_WITH_CONTROLLER.replace("ARTY_CONTROLLER_UNIT=trainer-controller\n", "")
    finished = run_backend(tmp_path, site, "engine")
    assert finished.returncode != 0
    assert "ARTY_CONTROLLER_UNIT" in finished.stderr
    assert calls(tmp_path) == ""


def test_the_usage_names_the_parts_by_what_they_do(tmp_path):
    finished = run_backend(tmp_path, None)
    assert finished.returncode == 2
    assert "the web chat's route and the trainer controller" in finished.stderr


def test_the_engine_config_is_written_with_the_site_values(tmp_path):
    finished = run_backend(tmp_path, SITE_WITH_CONTROLLER, "engine-config")
    assert finished.returncode == 0, finished.stderr
    written = calls(tmp_path)
    assert "--trainer-url 'http://192.0.2.10:8790'" in written
    assert "--allowed-host '192.0.2.13' --allowed-host 'arty.lan'" in written
    assert "--allowed-origin" not in written
