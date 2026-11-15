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
ARTY_ENGINE_MODEL_ROOTS="/srv/other/models /mnt/more"
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
    assert "engine model roots besides its own store: /srv/other/models /mnt/more" in lines
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
    finished = run_backend(tmp_path, "ARTY_CT=201\nARTY_CONTROLLER_URL=http://192.0.2.10:8790\n", "engine")
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
    assert "--model-root '/srv/other/models' --model-root '/mnt/more'" in written


def test_without_model_roots_the_engine_reads_only_its_own_store(tmp_path):
    finished = run_backend(tmp_path, "ARTY_CT=201\nARTY_CONTROLLER_URL=none\n", "engine-config")
    assert finished.returncode == 0, finished.stderr
    assert "--model-root" not in calls(tmp_path)


def test_an_unknown_command_changes_nothing(tmp_path):
    finished = run_backend(tmp_path, SITE_WITH_CONTROLLER, "elsewhere")
    assert finished.returncode == 2
    assert "usage" in finished.stderr
    assert calls(tmp_path) == ""


def test_the_status_reads_the_engine_url_setting(tmp_path):
    finished = run_backend(tmp_path, "ARTY_CT=201\nARTY_CONTROLLER_URL=none\n", "status")
    assert finished.returncode == 0, finished.stderr
    assert "ARTY_ENGINE_URL=" in calls(tmp_path) and "11434" not in calls(tmp_path)


def run_engine_switch(folder: Path, unit_state: int, extra_site: str = "", tool_state: int = 0, ct_refuses: str = ""):
    code = folder / "controller.py"
    code.write_text("def reload_arty(): pass\n")
    site = folder / "site.env"
    site.write_text(f"ARTY_CT=201\nARTY_CONTROLLER_URL=http://127.0.0.1:9\nARTY_CONTROLLER_SETTINGS={folder / 'settings.json'}\n"
                    f"ARTY_CONTROLLER_PYTHON=/opt/trainer/venv/bin/python\nARTY_CONTROLLER_CODE={code}\n"
                    f"ARTY_CONTROLLER_UNIT=trainer-controller\nARTY_CONTROLLER_MODULE=trainer.controller\n{extra_site}")
    bin_dir = fake_pct(folder)
    (bin_dir / "systemctl").write_text(f'#!/bin/sh\necho "systemctl $@" >> {folder / "calls.log"}\n'
                                       f'[ "$1" = is-active ] && exit {unit_state}\nexit 0\n')
    (bin_dir / "python3").write_text(f'#!/bin/sh\necho "python3 $@" >> {folder / "calls.log"}\nexit {tool_state}\n')
    if ct_refuses:
        (bin_dir / "pct").write_text(f'#!/bin/sh\necho "pct $@" >> {folder / "calls.log"}\n'
                                     f'case "$*" in *{ct_refuses}*) exit 1;; esac\nexit 0\n')
    (bin_dir / "python3").chmod(0o755)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("ARTY_")}
    environment.update(ARTY_ENGINE_SITE=str(site), ARTY_CONTROLLER_SETTLE_SECONDS="0", PATH=f"{bin_dir}:{os.environ['PATH']}")
    return subprocess.run(["bash", str(SCRIPT), "engine"], env=environment, capture_output=True, text=True,
                          timeout=SCRIPT_TIMEOUT_SECONDS)


def test_the_switch_fails_when_the_controller_does_not_come_back(tmp_path):
    finished = run_engine_switch(tmp_path, 3)
    assert finished.returncode != 0
    assert "trainer-controller is not running" in finished.stderr
    assert "systemctl is-active" in calls(tmp_path)


def test_the_switch_goes_on_when_the_controller_comes_back(tmp_path):
    finished = run_engine_switch(tmp_path, 0)
    assert finished.returncode == 0, finished.stderr


def test_the_switch_drops_the_older_settings_the_site_names(tmp_path):
    finished = run_engine_switch(tmp_path, 0, 'ARTY_CONTROLLER_OLDER_SETTINGS="first_old second_old"\n')
    assert finished.returncode == 0, finished.stderr
    assert "--drop-setting first_old --drop-setting second_old" in calls(tmp_path)


def test_the_switch_drops_nothing_when_the_site_names_nothing(tmp_path):
    finished = run_engine_switch(tmp_path, 0)
    assert finished.returncode == 0, finished.stderr
    assert "--drop-setting" not in calls(tmp_path)


def test_the_switch_moves_arty_over_before_it_fails_on_the_controller(tmp_path):
    finished = run_engine_switch(tmp_path, 3)
    assert finished.returncode != 0
    assert "brain-through-engine.conf" in calls(tmp_path) and "keep_alive" in calls(tmp_path)
    assert "trainer-controller is not running" in finished.stderr.rstrip().splitlines()[-1]


def test_a_settings_tool_failure_says_the_engine_started_and_arty_was_not_moved(tmp_path):
    finished = run_engine_switch(tmp_path, 0, tool_state=1)
    assert finished.returncode != 0
    assert "Arty was not moved" in finished.stderr
    assert "brain-through-engine.conf" not in calls(tmp_path)


def test_the_controller_line_comes_last_even_when_caddy_refuses(tmp_path):
    finished = run_engine_switch(tmp_path, 3, ct_refuses="Caddyfile")
    assert finished.returncode != 0
    assert "trainer-controller is not running" in finished.stderr.rstrip().splitlines()[-1]
