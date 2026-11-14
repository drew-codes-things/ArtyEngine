import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SLICE_SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "container" / "slice.sh"
SCRIPT_TIMEOUT_SECONDS = 60
NOT_CONFIGURED = 5


class Recorder:
    def __init__(self):
        self.events = []
        self.lock = threading.Lock()
        self.controller_status = 200

    def add(self, role, method, path, body):
        with self.lock:
            self.events.append((role, method, path, body))

    def calls(self, role=None):
        return [(method, path, body) for event_role, method, path, body in self.events if role in (None, event_role)]


def serve(recorder, role):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self, status, body):
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            recorder.add(role, "GET", self.path, None)
            self._answer(200, {"models": [{"name": "qwen3:14b"}]})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            recorder.add(role, "POST", self.path, body)
            status = recorder.controller_status if role == "controller" else 200
            self._answer(status, {"ok": status == 200})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.fixture
def slice_world(tmp_path):
    recorder = Recorder()
    servers = {role: serve(recorder, role) for role in ("ollama", "engine", "controller")}
    environment = {**os.environ,
                   "SLICE_LOCK": str(tmp_path / "gpu-lock"),
                   "SLICE_OLLAMA_PORT": str(servers["ollama"].server_port),
                   "SLICE_ENGINE_PORT": str(servers["engine"].server_port),
                   "SLICE_CONTROLLER": f"http://127.0.0.1:{servers['controller'].server_port}",
                   "SLICE_WORK_DIR": str(tmp_path),
                   "ARTY_ENGINE_SITE": str(tmp_path / "site.env")}
    yield recorder, environment, tmp_path
    for server in servers.values():
        server.shutdown()


def run_slice(environment, *command):
    return subprocess.run(["bash", str(SLICE_SCRIPT), *command], env=environment, capture_output=True, text=True,
                          timeout=SCRIPT_TIMEOUT_SECONDS)


def marker_command(environment):
    controller = environment["SLICE_CONTROLLER"]
    return ["curl", "-s", "-o", "/dev/null", "-d", "{}", f"{controller}/bench-ran"]


def test_a_slice_holds_training_off_from_before_the_unload_until_after_the_reload(slice_world):
    recorder, environment, folder = slice_world
    finished = run_slice(environment, *marker_command(environment))
    assert finished.returncode == 0, finished.stdout + finished.stderr
    controller_paths = [path for _, path, _ in recorder.calls("controller")]
    assert controller_paths == ["/busy", "/bench-ran", "/done"]
    busy, _, done = recorder.calls("controller")
    assert busy[2]["id"] == done[2]["id"] and busy[2]["id"].startswith("slice-")
    events = recorder.events
    first_unload = next(index for index, event in enumerate(events) if event[3] and event[3].get("keep_alive") == 0)
    reload = next(index for index, event in enumerate(events)
                  if event[0] == "engine" and event[3] and "keep_alive" not in event[3])
    assert events.index(("controller", "POST", "/busy", busy[2])) < first_unload
    assert reload < events.index(("controller", "POST", "/done", done[2]))
    assert not (folder / "gpu-lock").exists()


def test_a_slice_does_not_run_when_the_controller_does_not_hand_over_the_gpu(slice_world):
    recorder, environment, folder = slice_world
    recorder.controller_status = 503
    finished = run_slice(environment, *marker_command(environment))
    assert finished.returncode == 4
    assert recorder.calls("ollama") == [] and recorder.calls("engine") == []
    assert [path for _, path, _ in recorder.calls("controller")] == ["/busy"]
    assert not (folder / "gpu-lock").exists()


def test_a_slice_can_skip_the_controller_on_purpose(slice_world):
    recorder, environment, folder = slice_world
    environment["SLICE_CONTROLLER"] = "none"
    finished = run_slice(environment, "true")
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert recorder.calls("controller") == []
    assert not (folder / "gpu-lock").exists()


def fake_systemctl(folder):
    bin_dir = folder / "bin"
    bin_dir.mkdir()
    script = bin_dir / "systemctl"
    script.write_text(f'#!/bin/sh\necho "$@" >> {folder / "systemctl.log"}\nexit 0\n')
    script.chmod(0o755)
    return bin_dir


def test_a_slice_stops_the_bench_engine_it_leaves_behind(slice_world):
    recorder, environment, folder = slice_world
    environment["PATH"] = f"{fake_systemctl(folder)}:{environment['PATH']}"
    finished = run_slice(environment, "true")
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert "stop arty-engine-test" in (folder / "systemctl.log").read_text().splitlines()


def test_a_slice_says_nothing_about_a_port_nobody_listens_on(slice_world):
    recorder, environment, folder = slice_world
    environment["SLICE_ENGINE_PORT"] = "9"
    finished = run_slice(environment, "true")
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert "Traceback" not in finished.stdout + finished.stderr


def test_a_slice_reads_the_controller_from_the_site_file(slice_world):
    recorder, environment, folder = slice_world
    (folder / "site.env").write_text(f"ARTY_CONTROLLER_URL={environment.pop('SLICE_CONTROLLER')}\n")
    finished = run_slice(environment, "true")
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert [path for _, path, _ in recorder.calls("controller")] == ["/busy", "/done"]


def test_a_slice_with_no_controller_setting_runs_nothing(slice_world):
    recorder, environment, folder = slice_world
    del environment["SLICE_CONTROLLER"]
    finished = run_slice(environment, "true")
    assert finished.returncode == NOT_CONFIGURED
    assert "ARTY_CONTROLLER_URL" in finished.stderr
    assert recorder.events == []
    assert not (folder / "gpu-lock").exists()


def test_the_settings_are_shown_without_taking_the_gpu(slice_world):
    recorder, environment, folder = slice_world
    finished = run_slice(environment, "--show-settings")
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert f"controller: {environment['SLICE_CONTROLLER']}" in finished.stdout.splitlines()
    assert f"lock: {folder / 'gpu-lock'}" in finished.stdout.splitlines()
    assert recorder.events == []
    assert not (folder / "gpu-lock").exists()


def test_a_slice_puts_back_the_main_model_the_site_names(slice_world):
    recorder, environment, folder = slice_world
    (folder / "site.env").write_text("ARTY_MAIN_MODEL=mimo-v2.6:9b\n")
    finished = run_slice(environment, "true")
    assert finished.returncode == 0, finished.stdout + finished.stderr
    reloads = [body for method, path, body in recorder.calls("engine") if body and "keep_alive" not in body]
    assert reloads == [{"model": "mimo-v2.6:9b"}]


def test_without_an_ollama_port_a_slice_leaves_ollama_alone(slice_world):
    recorder, environment, folder = slice_world
    del environment["SLICE_OLLAMA_PORT"]
    finished = run_slice(environment, "true")
    assert finished.returncode == 0, finished.stdout + finished.stderr
    assert recorder.calls("ollama") == []
    shown = run_slice(environment, "--show-settings")
    assert "ollama port: (none)" in shown.stdout.splitlines()
    assert f"reload port: {environment['SLICE_ENGINE_PORT']}" in shown.stdout.splitlines()
