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
    finished = run_engine_switch(tmp_path, 3, "ARTY_MAIN_MODEL=big-main:35b\n")
    assert finished.returncode != 0
    assert "brain-through-engine.conf" in calls(tmp_path) and "keep_alive" in calls(tmp_path)
    assert "trainer-controller is not running" in finished.stderr.rstrip().splitlines()[-1]


def test_a_settings_tool_failure_says_the_engine_started_and_arty_was_not_moved(tmp_path):
    finished = run_engine_switch(tmp_path, 0, tool_state=1)
    assert finished.returncode != 0
    assert "this run did not move Arty" in finished.stderr
    assert "brain-through-engine.conf" not in calls(tmp_path)


def test_the_controller_line_comes_last_even_when_caddy_refuses(tmp_path):
    finished = run_engine_switch(tmp_path, 3, ct_refuses="Caddyfile")
    assert finished.returncode != 0
    assert "trainer-controller is not running" in finished.stderr.rstrip().splitlines()[-1]


def last_line(finished) -> str:
    lines = finished.stderr.rstrip().splitlines()
    return lines[-1] if lines else ""


def test_a_refused_move_with_the_controller_down_never_says_arty_moved(tmp_path):
    finished = run_engine_switch(tmp_path, 3, ct_refuses="Caddyfile")
    assert finished.returncode != 0
    assert "Arty now uses the engine" not in finished.stderr
    assert "the move to it did not finish" in last_line(finished) and "trainer-controller" in last_line(finished)


def test_an_engine_that_does_not_answer_ends_with_where_arty_stands(tmp_path):
    finished = run_engine_switch(tmp_path, 0, ct_refuses="api/version")
    assert finished.returncode != 0
    assert "did not start" in last_line(finished) and "this run did not move Arty" in last_line(finished)
    assert "brain-through-engine.conf" not in calls(tmp_path)


def test_a_main_model_that_does_not_load_after_the_move_is_named_last(tmp_path):
    finished = run_engine_switch(tmp_path, 0, "ARTY_MAIN_MODEL=big-main:35b\n", ct_refuses="keep_alive")
    assert finished.returncode != 0
    assert "Arty now uses the engine" in last_line(finished) and "big-main:35b" in last_line(finished)


def test_without_a_main_model_setting_the_switch_loads_no_model_at_all(tmp_path):
    finished = run_engine_switch(tmp_path, 0)
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert "keep_alive" not in calls(tmp_path) and "qwen3:14b" not in finished.stdout + finished.stderr
    assert "no ARTY_MAIN_MODEL" in finished.stdout


def test_a_config_that_cannot_be_written_says_nothing_was_started_or_moved(tmp_path):
    finished = run_engine_switch(tmp_path, 0, ct_refuses="write_config.py")
    assert finished.returncode != 0
    assert "config could not be written" in last_line(finished) and "started nothing and moved nothing" in last_line(finished)
    assert "journalctl -u arty-engine" not in finished.stderr
    assert "enable --now arty-engine" not in calls(tmp_path)


def test_a_restart_that_fails_after_the_route_moved_says_the_move_did_not_finish(tmp_path):
    finished = run_engine_switch(tmp_path, 0, ct_refuses="restart?tool-hub?brain")
    assert finished.returncode != 0
    assert "the move to it did not finish" in last_line(finished)
    assert "not moved" not in finished.stderr and "did not move Arty" not in finished.stderr
    assert "where things point now:" in finished.stdout and "systemctl show brain" in calls(tmp_path)


def test_an_engine_that_does_not_start_shows_where_arty_points(tmp_path):
    finished = run_engine_switch(tmp_path, 0, ct_refuses="api/version")
    assert "still uses its old backend" not in finished.stderr
    assert "where things point now:" in finished.stdout and "systemctl show brain" in calls(tmp_path)


def test_a_unit_file_that_cannot_be_installed_says_nothing_was_started_or_moved(tmp_path):
    finished = run_engine_switch(tmp_path, 0, ct_refuses="arty-engine.service")
    assert finished.returncode != 0
    assert "unit file could not be installed" in last_line(finished)
    assert "started nothing and moved nothing" in last_line(finished)
    assert "did not start" not in finished.stderr and "journalctl -u arty-engine" not in finished.stderr
    assert "enable --now arty-engine" not in calls(tmp_path)


def test_the_sites_remote_models_file_reaches_the_engine_config(tmp_path):
    site = "ARTY_CT=201\nARTY_CONTROLLER_URL=none\nARTY_ENGINE_REMOTE_MODELS=/srv/ai/arty-engine/remote-models.json\n"
    finished = run_backend(tmp_path, site, "engine-config")
    assert finished.returncode == 0, finished.stderr
    assert "--remote-models '/srv/ai/arty-engine/remote-models.json'" in calls(tmp_path)


def test_without_remote_models_the_engine_config_names_none(tmp_path):
    finished = run_backend(tmp_path, "ARTY_CT=201\nARTY_CONTROLLER_URL=none\n", "engine-config")
    assert finished.returncode == 0, finished.stderr
    assert "--remote-models" not in calls(tmp_path)
