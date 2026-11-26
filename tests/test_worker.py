import argparse
import json
import logging
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from arty_engine import worker
from arty_engine.worker import CACHE_VARIABLE, WorkerConfig, is_allowed, load_worker_config, networks, run, worker_arguments

REPOSITORY = Path(__file__).resolve().parents[1]
RESET_ROUNDS = 40
UNREACHABLE_HOST = "192.0.2.1"
DEAD_PEER_SECONDS = 120
FAKE_WORKER = f"""#!{sys.executable}
import json, os, socket, sys, threading
arguments = sys.argv[1:]
host, port = arguments[arguments.index("-H") + 1], int(arguments[arguments.index("-p") + 1])
with open(os.environ["FAKE_WORKER_LOG"], "w") as log:
    json.dump({{"arguments": arguments, "cache": os.environ.get({CACHE_VARIABLE!r}, "")}}, log)
listener = socket.socket()
listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind((host, port))
listener.listen(8)

def echo(client):
    while True:
        data = client.recv(65536)
        if not data:
            break
        client.sendall(data)
    client.close()

while True:
    client, _ = listener.accept()
    threading.Thread(target=echo, args=(client,), daemon=True).start()
"""


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    binary = tmp_path / "runtime" / "arty-runtime-worker"
    binary.parent.mkdir()
    binary.write_text(FAKE_WORKER)
    binary.chmod(0o755)
    monkeypatch.setenv("FAKE_WORKER_LOG", str(tmp_path / "worker-log.json"))
    return binary


class RunningWorker:
    def __init__(self, config: WorkerConfig):
        self.processes: list[subprocess.Popen] = []
        self.gate_ready = threading.Event()
        self.gate = None
        self.result = None

        def launcher(arguments, **kwargs):
            process = subprocess.Popen(arguments, **kwargs)
            self.processes.append(process)
            return process

        def ready(gate):
            self.gate = gate
            self.gate_ready.set()

        def go():
            self.result = run(config, launcher=launcher, ready=ready)
        self.thread = threading.Thread(target=go, daemon=True)
        self.thread.start()
        assert self.gate_ready.wait(20), "the worker never got ready"

    def stop(self):
        for process in self.processes:
            process.terminate()
        self.thread.join(20)


def round_trip(port: int, payload: bytes) -> bytes:
    received = b""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=5) as connection:
            connection.sendall(payload)
            connection.shutdown(socket.SHUT_WR)
            while chunk := connection.recv(65536):
                received += chunk
    except OSError as error:
        if isinstance(error, (TimeoutError, ConnectionRefusedError)):
            raise
    return received


def test_an_allowed_client_reaches_the_worker_through_the_gate(tmp_path, fake_worker):
    port = free_port()
    running = RunningWorker(WorkerConfig(listen=f"127.0.0.1:{port}", allowed_clients=["127.0.0.1"], devices=["CUDA0", "CPU"],
                                         cache_dir=str(tmp_path / "cache"), worker_binary=str(fake_worker.parent), threads=3))
    try:
        payload = bytes(range(256)) * 4096
        assert round_trip(port, payload) == payload
        logged = json.loads((tmp_path / "worker-log.json").read_text())
        arguments = logged["arguments"]
        assert arguments[arguments.index("-H") + 1] == "127.0.0.1" and int(arguments[arguments.index("-p") + 1]) != port
        assert arguments[arguments.index("-d") + 1] == "CUDA0,CPU" and arguments[arguments.index("-t") + 1] == "3"
        assert "-c" in arguments and logged["cache"] == str(tmp_path / "cache") and (tmp_path / "cache").is_dir()
    finally:
        running.stop()
    assert running.result is not None


def test_a_client_outside_the_list_is_refused(tmp_path, fake_worker, caplog):
    port = free_port()
    with caplog.at_level(logging.WARNING):
        running = RunningWorker(WorkerConfig(listen=f"127.0.0.1:{port}", allowed_clients=["192.0.2.16"],
                                             worker_binary=str(fake_worker)))
        try:
            assert round_trip(port, b"hello") == b""
            deadline = time.monotonic() + 5
            while running.gate.refused == 0 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert running.gate.refused == 1
        finally:
            running.stop()
    assert "refused a connection from 127.0.0.1: not in allowed_clients" in caplog.text


def test_without_a_list_the_worker_binds_directly(tmp_path, fake_worker):
    port = free_port()
    running = RunningWorker(WorkerConfig(listen=f"127.0.0.1:{port}", worker_binary=str(fake_worker)))
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                assert round_trip(port, b"direct") == b"direct"
                break
            except OSError:
                time.sleep(0.1)
        arguments = json.loads((tmp_path / "worker-log.json").read_text())["arguments"]
        assert arguments[arguments.index("-p") + 1] == str(port) and running.gate is None and "-c" not in arguments
    finally:
        running.stop()


def test_an_open_worker_with_no_list_warns_loudly(caplog):
    with caplog.at_level(logging.WARNING):
        worker.warn_when_open(WorkerConfig(listen="0.0.0.0:50052"))
    warnings = [record for record in caplog.records if "has no authentication" in record.message]
    assert len(warnings) == worker.LOUD_REPEATS


def test_a_private_address_with_no_list_still_warns(caplog):
    with caplog.at_level(logging.WARNING):
        worker.warn_when_open(WorkerConfig(listen="192.0.2.196:50052"))
        worker.warn_when_open(WorkerConfig(listen="0.0.0.0:50052", allowed_clients=["192.0.2.16"]))
        worker.warn_when_open(WorkerConfig(listen="127.0.0.1:50052"))
    assert len([record for record in caplog.records if "has no authentication" in record.message]) == 1


@pytest.mark.parametrize("address, allowed", [
    ("192.0.2.16", True), ("192.0.2.17", False), ("10.0.0.5", True), ("::ffff:192.0.2.16", True), ("fe80::1%eth0", False),
    ("not an address", False),
])
def test_allowed_clients_take_addresses_and_networks(address, allowed):
    assert is_allowed(address, networks(["192.0.2.16", "10.0.0.0/8"])) is allowed


@pytest.mark.parametrize("settings, problem", [
    ({"listen": "0.0.0.0"}, "host:port"),
    ({"allowed_clients": ["192.0.2.300"]}, "allowed_clients"),
    ({"devices": "CUDA0"}, "devices"),
    ({"threads": -1}, "threads"),
])
def test_bad_worker_settings_are_refused(settings, problem):
    with pytest.raises(ValueError, match=problem):
        WorkerConfig(**settings)


def test_the_worker_config_file_and_flags_combine(tmp_path):
    path = tmp_path / "worker.json"
    path.write_text(json.dumps({"listen": "0.0.0.0:50052", "allowed_clients": ["192.0.2.16"], "devices": ["CUDA0"]}))
    config = load_worker_config(str(path))
    assert config.allowed_clients == ["192.0.2.16"]
    path.write_text(json.dumps({"listen": "0.0.0.0:50052", "password": "x"}))
    with pytest.raises(ValueError, match="unknown settings password"):
        load_worker_config(str(path))


def test_arguments_without_devices_or_cache_stay_minimal():
    assert worker_arguments("/w", "127.0.0.1", 5000, WorkerConfig()) == ["/w", "-H", "127.0.0.1", "-p", "5000"]


def test_a_missing_worker_binary_is_a_clear_error(tmp_path):
    with pytest.raises(RuntimeError, match="no runtime worker"):
        run(WorkerConfig(listen="127.0.0.1:1", worker_binary=str(tmp_path / "nothing")))


def test_the_worker_command_refuses_bad_settings_before_starting():
    finished = subprocess.run([sys.executable, "-m", "arty_engine", "worker", "--listen", "nowhere"], cwd=REPOSITORY,
                              capture_output=True, text=True, timeout=30)
    assert finished.returncode == 2 and "worker not started" in finished.stderr


class EchoServer:
    def __init__(self):
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(16)
        self.port = self.listener.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self):
        while True:
            try:
                client, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(target=self._echo, args=(client,), daemon=True).start()

    @staticmethod
    def _echo(client):
        with client:
            while data := client.recv(65536):
                client.sendall(data)

    def close(self):
        self.listener.close()


class AcceptFailsOnce:
    def __init__(self, listener):
        self.listener = listener
        self.failed = False

    def accept(self):
        if not self.failed:
            self.failed = True
            raise OSError(24, "Too many open files")
        return self.listener.accept()

    def fileno(self):
        return self.listener.fileno()

    def close(self):
        self.listener.close()


@pytest.fixture
def echo_server():
    server = EchoServer()
    yield server
    server.close()


def open_gate(inner_port: int, **settings) -> worker.Gate:
    gate = worker.Gate("127.0.0.1", 0, inner_port, networks(["127.0.0.1"]), **settings)
    gate.listen_port = gate.open()
    return gate


def test_a_worker_on_a_network_address_without_a_list_is_refused_unless_open(tmp_path, fake_worker):
    with pytest.raises(RuntimeError, match="--open"):
        run(WorkerConfig(listen=f"{UNREACHABLE_HOST}:{free_port()}", worker_binary=str(fake_worker)))
    with pytest.raises(RuntimeError, match="--open"):
        worker.refuse_unless_meant(WorkerConfig(listen="0.0.0.0:50052"))
    worker.refuse_unless_meant(WorkerConfig(listen="0.0.0.0:50052", open=True))
    worker.refuse_unless_meant(WorkerConfig(listen="0.0.0.0:50052", allowed_clients=["192.0.2.16"]))
    worker.refuse_unless_meant(WorkerConfig(listen="127.0.0.1:50052"))
    assert not (tmp_path / "worker-log.json").exists()


def test_the_worker_command_refuses_an_open_listen_and_takes_the_open_flag(tmp_path, fake_worker):
    finished = subprocess.run([sys.executable, "-m", "arty_engine", "worker", "--worker-binary", str(fake_worker), "--listen",
                               f"{UNREACHABLE_HOST}:{free_port()}"], cwd=REPOSITORY, capture_output=True, text=True, timeout=30)
    assert finished.returncode == 1 and "--open" in finished.stdout + finished.stderr
    parsed = worker.with_flags(WorkerConfig(), argparse.Namespace(listen="", allow=[], devices="", cache_dir="", worker_binary="",
                                                                   threads=None, open=True))
    assert parsed.open is True


def test_the_gate_keeps_taking_connections_after_an_accept_error(echo_server, caplog):
    gate = open_gate(echo_server.port)
    gate.listener = AcceptFailsOnce(gate.listener)
    caplog.set_level(logging.WARNING)
    threading.Thread(target=gate.serve, daemon=True).start()
    try:
        assert round_trip(gate.listen_port, b"still here") == b"still here"
        assert "couldn't take a connection" in caplog.text and not gate.stopped.is_set()
    finally:
        gate.close()


def test_the_gate_caps_the_relays_it_runs_at_once(echo_server, caplog):
    gate = open_gate(echo_server.port, most_relays=1)
    caplog.set_level(logging.WARNING)
    threading.Thread(target=gate.serve, daemon=True).start()
    held = socket.create_connection(("127.0.0.1", gate.listen_port), timeout=5)
    try:
        held.sendall(b"ping")
        assert held.recv(4) == b"ping"
        assert round_trip(gate.listen_port, b"second") == b""
        assert "already relaying 1 connections" in caplog.text
    finally:
        held.close()
        gate.close()
    deadline = time.monotonic() + 5
    while gate.relays._value != 1 and time.monotonic() < deadline:
        time.sleep(0.05)
    assert gate.relays._value == 1


def test_relayed_connections_use_keepalive(echo_server):
    gate = open_gate(echo_server.port)
    threading.Thread(target=gate.serve, daemon=True).start()
    seen = []
    original = worker.tune

    def recording_tune(end):
        original(end)
        seen.append(end.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE))
    try:
        worker.tune = recording_tune
        assert round_trip(gate.listen_port, b"x") == b"x"
    finally:
        worker.tune = original
        gate.close()
    assert seen == [1, 1]


def test_relayed_connections_find_a_dead_peer_within_a_couple_of_minutes():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as end:
        worker.tune(end)
        assert end.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == worker.KEEPALIVE_IDLE_SECONDS
        assert end.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == worker.KEEPALIVE_INTERVAL_SECONDS
        assert end.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT) == worker.KEEPALIVE_PROBES
    assert worker.KEEPALIVE_IDLE_SECONDS + worker.KEEPALIVE_INTERVAL_SECONDS * worker.KEEPALIVE_PROBES <= DEAD_PEER_SECONDS


def test_the_worker_exits_non_zero_when_its_gate_stops(tmp_path, fake_worker):
    port = free_port()
    running = RunningWorker(WorkerConfig(listen=f"127.0.0.1:{port}", allowed_clients=["127.0.0.1"],
                                         worker_binary=str(fake_worker)))
    running.gate.listener.shutdown(socket.SHUT_RDWR)
    running.gate.listener.close()
    running.thread.join(15)
    assert running.result == worker.GATE_STOPPED_EXIT
    assert all(process.poll() is not None for process in running.processes)


def test_round_trip_tolerates_a_peer_that_already_hung_up():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(64)
    port = listener.getsockname()[1]

    def reset_everyone():
        while True:
            try:
                client, _ = listener.accept()
            except OSError:
                return
            client.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            client.close()
    threading.Thread(target=reset_everyone, daemon=True).start()
    try:
        for _ in range(RESET_ROUNDS):
            assert round_trip(port, b"") == b""
            time.sleep(0.005)
    finally:
        listener.close()
