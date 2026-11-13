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
                                  "model_roots": [str(build_store(tmp_path / "ollama"))],
                                  "llama_server": "/nonexistent/llama-server", "state_dir": str(tmp_path / "state")}))
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
