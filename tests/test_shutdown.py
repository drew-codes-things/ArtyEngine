import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from fake_store import build_store

REPO = Path(__file__).resolve().parents[1]
START_TIMEOUT_SECONDS = 20


def test_sigterm_stops_the_engine_through_its_cleanup(tmp_path):
    config = tmp_path / "engine.json"
    config.write_text(json.dumps({"listen": "127.0.0.1:0", "store": str(tmp_path / "own"),
                                  "model_roots": [str(build_store(tmp_path / "other"))],
                                  "runner_binary": "/nonexistent/server", "state_dir": str(tmp_path / "state")}))
    engine = subprocess.Popen([sys.executable, "-m", "arty_engine", "--config", str(config)], cwd=REPO,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        while "arty-engine" not in engine.stdout.readline():
            assert time.monotonic() < deadline and engine.poll() is None, "engine did not start"
        engine.send_signal(signal.SIGTERM)
        output, _ = engine.communicate(timeout=START_TIMEOUT_SECONDS)
    finally:
        if engine.poll() is None:
            engine.kill()
    assert engine.returncode == 0
    assert "stopping" in output


def test_an_engine_with_no_models_says_so_and_points_at_the_migration_tool(tmp_path):
    config = tmp_path / "engine.json"
    config.write_text(json.dumps({"listen": "127.0.0.1:0", "store": str(tmp_path / "own"), "model_roots": [str(tmp_path / "old")],
                                  "runner_binary": "/nonexistent/server", "state_dir": str(tmp_path / "state")}))
    engine = subprocess.Popen([sys.executable, "-m", "arty_engine", "--config", str(config)], cwd=REPO,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        startup = [engine.stdout.readline()]
        while "VRAM budget" not in startup[-1]:
            assert time.monotonic() < deadline and engine.poll() is None, "engine did not start"
            startup.append(engine.stdout.readline())
        engine.send_signal(signal.SIGTERM)
        engine.communicate(timeout=START_TIMEOUT_SECONDS)
    finally:
        if engine.poll() is None:
            engine.kill()
    warning = "".join(line for line in startup if "no models" in line)
    assert "tools/migrate_store.py" in warning and str(tmp_path / "old") in warning


def test_an_old_layout_root_beside_a_store_with_models_is_named_at_startup(tmp_path):
    old = tmp_path / "old"
    (old / "manifests-v2" / "host" / "library" / "m").mkdir(parents=True)
    (old / "blobs").mkdir()
    config = tmp_path / "engine.json"
    config.write_text(json.dumps({"listen": "127.0.0.1:0", "store": str(build_store(tmp_path / "own")), "model_roots": [str(old)],
                                  "runner_binary": "/nonexistent/server", "state_dir": str(tmp_path / "state")}))
    engine = subprocess.Popen([sys.executable, "-m", "arty_engine", "--config", str(config)], cwd=REPO,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + START_TIMEOUT_SECONDS
        startup = [engine.stdout.readline()]
        while "VRAM budget" not in startup[-1]:
            assert time.monotonic() < deadline and engine.poll() is None, "engine did not start"
            startup.append(engine.stdout.readline())
        engine.send_signal(signal.SIGTERM)
        engine.communicate(timeout=START_TIMEOUT_SECONDS)
    finally:
        if engine.poll() is None:
            engine.kill()
    warning = "".join(line for line in startup if "older layout" in line)
    assert str(old) in warning and "tools/migrate_store.py" in warning
