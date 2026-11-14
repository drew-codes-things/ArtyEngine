import json
import os
import socket
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

CONTAINER = Path(__file__).resolve().parents[1] / "tools" / "container"
RESTART_SCRIPT = CONTAINER / "engine-restart.sh"
SLICE_ENGINE_SCRIPT = CONTAINER / "slice-engine.sh"
SCRIPT_TIMEOUT_SECONDS = 60
BENCH_CONFIG = Path(__file__).resolve().parents[1] / "deploy" / "engine-test.json"
LIVE_CONFIG = Path(__file__).resolve().parents[1] / "deploy" / "engine.json"


def answering_server():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            data = json.dumps({"version": "0.1.0"}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def write_fake(folder: Path, name: str, body: str) -> None:
    path = folder / name
    path.write_text("#!/bin/bash\n" + body + "\n")
    path.chmod(0o755)


@pytest.fixture
def bench_world(tmp_path):
    fakes = tmp_path / "bin"
    fakes.mkdir()
    write_fake(fakes, "systemctl", f'echo "systemctl $*" >> {tmp_path}/calls\n[ "$1" = is-active ] && exit "$(cat {tmp_path}/unit-state)"\nexit 0')
    write_fake(fakes, "systemd-run", f'echo "systemd-run $*" >> {tmp_path}/calls\nexit 0')
    write_fake(fakes, "sleep", "exit 0")
    (tmp_path / "unit-state").write_text("0")
    environment = {**os.environ, "PATH": f"{fakes}:{os.environ['PATH']}", "BENCH_HOME": str(tmp_path)}
    return tmp_path, environment


def config_on(folder: Path, port: int) -> Path:
    config = folder / "engine-test.json"
    config.write_text(json.dumps({"listen": f"127.0.0.1:{port}", "store": str(folder / "models")}))
    return config


def run(script: Path, arguments: list[str], environment: dict) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(script), *arguments], env=environment, capture_output=True, text=True,
                          timeout=SCRIPT_TIMEOUT_SECONDS)


def test_the_bench_engine_never_shares_the_live_engine_port_or_state():
    bench, live = json.loads(BENCH_CONFIG.read_text()), json.loads(LIVE_CONFIG.read_text())
    assert bench["listen"] != live["listen"] and bench["state_dir"] != live["state_dir"]


def test_restart_refuses_a_port_another_engine_already_answers_on(bench_world):
    folder, environment = bench_world
    held = answering_server()
    try:
        result = run(RESTART_SCRIPT, [str(config_on(folder, held.server_address[1]))], environment)
    finally:
        held.shutdown()
    assert result.returncode != 0
    assert f"port {held.server_address[1]}" in result.stderr and "already answers" in result.stderr
    assert "systemd-run" not in (folder / "calls").read_text()


def test_restart_fails_when_the_bench_unit_is_not_running(bench_world):
    folder, environment = bench_world
    (folder / "unit-state").write_text("3")
    result = run(RESTART_SCRIPT, [str(config_on(folder, free_port()))], environment)
    assert result.returncode != 0 and "arty-engine-test is not running" in result.stderr


def test_restart_names_the_bench_url_once_its_own_unit_answers(bench_world):
    folder, environment = bench_world
    port = free_port()
    write_fake(folder / "bin", "systemd-run", f'echo "systemd-run $*" >> {folder}/calls\n'
               f'python3 -m http.server {port} --bind 127.0.0.1 --directory {folder}/site >/dev/null 2>&1 &\n'
               f'echo $! > {folder}/server.pid')
    (folder / "site" / "api").mkdir(parents=True)
    (folder / "site" / "api" / "version").write_text("{}")
    try:
        result = run(RESTART_SCRIPT, [str(config_on(folder, port))], environment)
    finally:
        pid = folder / "server.pid"
        if pid.exists():
            os.kill(int(pid.read_text()), 15)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == f"http://127.0.0.1:{port}"
    assert "--unit=arty-engine-test" in (folder / "calls").read_text()


def test_a_slice_runs_no_bench_when_the_bench_engine_did_not_start(bench_world):
    folder, environment = bench_world
    write_fake(folder, "engine-restart.sh", 'echo "port taken" >&2\nexit 1')
    (folder / "bench-args.sh").write_text("BENCH_ARGS=\n")
    write_fake(folder / "bin", "timeout", f'echo "bench $*" >> {folder}/calls')
    result = run(SLICE_ENGINE_SCRIPT, ["speed", str(config_on(folder, free_port())), "label"], environment)
    assert result.returncode != 0
    assert not (folder / "calls").exists()


def test_a_slice_benches_the_url_the_bench_engine_answers_on(bench_world):
    folder, environment = bench_world
    write_fake(folder, "engine-restart.sh", 'echo "http://127.0.0.1:12345"')
    (folder / "bench-args.sh").write_text("BENCH_ARGS=\n")
    write_fake(folder / "bin", "timeout", f'echo "bench $*" >> {folder}/calls')
    result = run(SLICE_ENGINE_SCRIPT, ["speed", str(config_on(folder, 12345)), "label"], environment)
    assert result.returncode == 0, result.stderr
    assert "bench/bench.py http://127.0.0.1:12345 label" in (folder / "calls").read_text()


def live_engine_holding(models: list[str]):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            data = json.dumps({"models": [{"name": name} for name in models]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


@pytest.mark.parametrize("held, busy", [([], False), (["qwen3:14b"], True)])
def test_a_slice_records_whether_the_live_engine_held_a_model_during_the_bench(bench_world, held, busy):
    folder, environment = bench_world
    write_fake(folder, "engine-restart.sh", 'echo "http://127.0.0.1:12345"')
    (folder / "bench-args.sh").write_text("BENCH_ARGS=\n")
    write_fake(folder / "bin", "timeout", 'while [ "$1" != --output ]; do shift; done; echo \'{"label": "x"}\' > "$2"')
    live = live_engine_holding(held)
    try:
        environment = {**environment, "LIVE_ENGINE_URL": f"http://127.0.0.1:{live.server_address[1]}"}
        result = run(SLICE_ENGINE_SCRIPT, ["speed", str(config_on(folder, 12345)), "label"], environment)
    finally:
        live.shutdown()
    assert result.returncode == 0, result.stderr
    recorded = json.loads((folder / "runs" / "bench-label.json").read_text())
    assert recorded["label"] == "x" and recorded["live_engine"]["busy"] is busy
    assert recorded["live_engine"]["before"] == held and recorded["live_engine"]["after"] == held
