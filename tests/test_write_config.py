import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "write_config.py"
SHIPPED_CONFIG = REPO / "deploy" / "engine.json"
TRAINER = "http://192.0.2.10:8790"


def write(source, target, *extra):
    return subprocess.run([sys.executable, str(SCRIPT), str(source), str(target), *extra],
                          capture_output=True, text=True, timeout=30)


def test_the_shipped_config_names_no_trainer():
    assert json.loads(SHIPPED_CONFIG.read_text())["trainer_url"] == ""


def test_the_shipped_config_reads_only_the_engines_own_store():
    assert json.loads(SHIPPED_CONFIG.read_text())["model_roots"] == []


def test_the_site_model_roots_go_into_the_installed_config(tmp_path):
    target = tmp_path / "engine.json"
    finished = write(SHIPPED_CONFIG, target, "--trainer-url", "", "--model-root", "/srv/other/models", "--model-root", "/mnt/more")
    assert finished.returncode == 0, finished.stderr
    assert json.loads(target.read_text())["model_roots"] == ["/srv/other/models", "/mnt/more"]


def test_the_site_trainer_goes_into_the_installed_config(tmp_path):
    target = tmp_path / "engine.json"
    finished = write(SHIPPED_CONFIG, target, "--trainer-url", TRAINER)
    assert finished.returncode == 0, finished.stderr
    assert json.loads(target.read_text()) == {**json.loads(SHIPPED_CONFIG.read_text()), "trainer_url": TRAINER}
    assert target.stat().st_mode & 0o777 == 0o644


def test_no_trainer_leaves_the_setting_empty(tmp_path):
    target = tmp_path / "engine.json"
    finished = write(SHIPPED_CONFIG, target, "--trainer-url", "")
    assert finished.returncode == 0, finished.stderr
    assert json.loads(target.read_text())["trainer_url"] == ""


def test_a_config_the_engine_would_refuse_is_not_installed(tmp_path):
    source, target = tmp_path / "source.json", tmp_path / "engine.json"
    source.write_text(json.dumps({"listen": "127.0.0.1:11438", "no_such_setting": 1}))
    target.write_text("{}")
    finished = write(source, target, "--trainer-url", TRAINER)
    assert finished.returncode != 0
    assert "no_such_setting" in finished.stderr
    assert target.read_text() == "{}"


def test_the_site_hosts_and_origins_go_into_the_installed_config(tmp_path):
    target = tmp_path / "engine.json"
    finished = write(SHIPPED_CONFIG, target, "--trainer-url", TRAINER, "--allowed-host", "192.0.2.13",
                     "--allowed-host", "arty.lan", "--allowed-origin", "chat.example.org")
    assert finished.returncode == 0, finished.stderr
    installed = json.loads(target.read_text())
    assert installed["allowed_hosts"] == ["192.0.2.13", "arty.lan"]
    assert installed["allowed_origins"] == ["chat.example.org"]


def test_without_hosts_or_origins_the_lists_stay_empty(tmp_path):
    target = tmp_path / "engine.json"
    finished = write(SHIPPED_CONFIG, target, "--trainer-url", "")
    assert finished.returncode == 0, finished.stderr
    installed = json.loads(target.read_text())
    assert installed.get("allowed_hosts", []) == [] and installed.get("allowed_origins", []) == []
