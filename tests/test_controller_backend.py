import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "host"))

import controller_backend

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "host" / "controller_backend.py"
PYTHON = "/opt/trainer/venv/bin/python"
UNIT = "trainer-controller"
MODULE = "trainer.controller"
RUN_FOLDER = "/opt/trainer/runs/first"


def switch(tmp_path, backend):
    settings, drop_in = tmp_path / "controller-settings.json", tmp_path / "settings.conf"
    finished = subprocess.run([sys.executable, str(SCRIPT), backend, "--settings", str(settings),
                               "--drop-in", str(drop_in), "--python", PYTHON,
                               "--unit", UNIT, "--module", MODULE], capture_output=True, text=True, timeout=30)
    return finished, settings, drop_in


def test_the_engine_switch_points_the_controller_at_the_engine(tmp_path):
    finished, settings, drop_in = switch(tmp_path, "engine")
    assert finished.returncode == 0, finished.stderr
    assert json.loads(settings.read_text()) == {"ollama_url": "http://127.0.0.1:11438"}
    assert drop_in.read_text() == f"[Service]\nExecStart=\nExecStart={PYTHON} -m {MODULE} --settings {settings}\n"
    assert f"systemctl restart {UNIT}" in finished.stdout


def test_the_switch_keeps_the_other_controller_settings(tmp_path):
    (tmp_path / "controller-settings.json").write_text(json.dumps({"run_folder": RUN_FOLDER}))
    finished, settings, _ = switch(tmp_path, "engine")
    assert finished.returncode == 0, finished.stderr
    assert json.loads(settings.read_text()) == {"run_folder": RUN_FOLDER,
                                                "ollama_url": "http://127.0.0.1:11438"}


def test_the_controller_can_no_longer_be_pointed_at_ollama(tmp_path):
    (tmp_path / "controller-settings.json").write_text(json.dumps({"ollama_url": "http://127.0.0.1:11438"}))
    finished, settings, drop_in = switch(tmp_path, "ollama")
    assert finished.returncode != 0
    assert json.loads(settings.read_text()) == {"ollama_url": "http://127.0.0.1:11438"}
    assert not drop_in.exists()


def test_a_drop_in_that_reads_another_settings_file_is_not_overwritten(tmp_path):
    other = "[Service]\nExecStart=\nExecStart=/usr/bin/python3 -m trainer.controller --settings /elsewhere.json\n"
    (tmp_path / "settings.conf").write_text(other)
    finished, settings, drop_in = switch(tmp_path, "engine")
    assert finished.returncode != 0
    assert "/elsewhere.json" in finished.stderr
    assert drop_in.read_text() == other and not settings.exists()


def test_a_damaged_settings_file_is_left_alone(tmp_path):
    (tmp_path / "controller-settings.json").write_text('{"run_folder": ')
    finished, settings, _ = switch(tmp_path, "engine")
    assert finished.returncode != 0
    assert "controller-settings.json" in finished.stderr
    assert settings.read_text() == '{"run_folder": '


def test_the_controller_files_must_be_named(tmp_path):
    finished = subprocess.run([sys.executable, str(SCRIPT), "engine", "--drop-in", str(tmp_path / "settings.conf")],
                              capture_output=True, text=True, timeout=30)
    assert finished.returncode != 0
    for option in ("--settings", "--python", "--unit", "--module"):
        assert option in finished.stderr
    assert not (tmp_path / "settings.conf").exists()


def test_the_drop_in_belongs_to_the_named_unit():
    assert controller_backend.drop_in_path(UNIT) == Path("/etc/systemd/system/trainer-controller.service.d/settings.conf")


def test_a_unit_name_that_leaves_the_systemd_folder_is_refused(tmp_path):
    finished = subprocess.run([sys.executable, str(SCRIPT), "engine", "--settings", str(tmp_path / "s.json"), "--python", PYTHON,
                               "--unit", "../../tmp/evil", "--module", MODULE], capture_output=True, text=True, timeout=30)
    assert finished.returncode != 0
    assert "unit" in finished.stderr
    assert not (tmp_path / "s.json").exists()
