import argparse
import ipaddress
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from .rpc import check_address, split_address
from .runtime import WORKER, locate_binary
from .scrub import LIBRARY_WORD

log = logging.getLogger("arty-engine.worker")

DEFAULT_LISTEN = "0.0.0.0:50052"
LOOPBACK = "127.0.0.1"
WILDCARD_HOSTS = ("0.0.0.0", "::", "")
CACHE_VARIABLE = f"{LIBRARY_WORD.upper()}_CACHE"
RELAY_CHUNK_BYTES = 1024 * 1024
BACKLOG = 16
START_SECONDS = 30
START_POLL_SECONDS = 0.2
STOP_GRACE_SECONDS = 10
LOUD_REPEATS = 3
MOST_RELAYS = 16
CONNECT_TIMEOUT_SECONDS = 10
ACCEPT_RETRY_SECONDS = 1
GATE_STOPPED_EXIT = 3
GATE_CHECK_SECONDS = 1
KEEPALIVE_IDLE_SECONDS = 30
KEEPALIVE_INTERVAL_SECONDS = 10
KEEPALIVE_PROBES = 3
KEEPALIVE_TIMING = (("TCP_KEEPIDLE", KEEPALIVE_IDLE_SECONDS), ("TCP_KEEPINTVL", KEEPALIVE_INTERVAL_SECONDS),
                    ("TCP_KEEPCNT", KEEPALIVE_PROBES))
OPEN_REFUSAL = (
    "the rpc worker would listen on {listen} with no allowed_clients list, so anyone who can reach that port could use "
    "it; set allowed_clients (--allow) to the engine's address, listen on 127.0.0.1, or pass --open to accept that")
NO_AUTHENTICATION_WARNING = (
    "the rpc worker at {listen} has no authentication and no allowed_clients list: anyone who can reach this port can "
    "send it work and data and use its GPU and memory. Set allowed_clients to the engine's address, or bind it to a "
    "private address behind a firewall")


@dataclass
class WorkerConfig:
    listen: str = DEFAULT_LISTEN
    allowed_clients: list = field(default_factory=list)
    devices: list = field(default_factory=list)
    cache_dir: str = ""
    worker_binary: str = "/srv/ai/arty-engine/runtime/worker"
    library_path: list = field(default_factory=list)
    threads: int = 0
    open: bool = False

    def __post_init__(self):
        check_address(self.listen)
        networks(self.allowed_clients)
        if not isinstance(self.devices, list) or not all(isinstance(device, str) and device and "," not in device
                                                         for device in self.devices):
            raise ValueError("devices must be a list of device names such as CUDA0 and CPU")
        if not isinstance(self.cache_dir, str) or not isinstance(self.worker_binary, str) or not self.worker_binary:
            raise ValueError("cache_dir must be a folder (or empty) and worker_binary the runtime worker's path")
        if not isinstance(self.library_path, list) or not all(isinstance(folder, str) for folder in self.library_path):
            raise ValueError("library_path must be a list of folders")
        if isinstance(self.threads, bool) or not isinstance(self.threads, int) or self.threads < 0:
            raise ValueError("threads must be a whole number, 0 for the runtime's default")
        if not isinstance(self.open, bool):
            raise ValueError("open must be true or false")

    @property
    def host(self) -> str:
        return split_address(self.listen)[0]

    @property
    def port(self) -> int:
        return split_address(self.listen)[1]

    @property
    def open_to_everyone(self) -> bool:
        return self.host in WILDCARD_HOSTS and not self.allowed_clients

    @property
    def open_to_the_network(self) -> bool:
        return not self.allowed_clients and not self.host.startswith("127.") and self.host not in ("localhost", "::1")


def networks(entries) -> list:
    if not isinstance(entries, list):
        raise ValueError("allowed_clients must be a list of addresses or networks such as 192.168.1.16 or 192.168.1.0/24")
    try:
        return [ipaddress.ip_network(entry, strict=False) for entry in entries]
    except (TypeError, ValueError) as error:
        raise ValueError(f"allowed_clients: {error}") from error


def is_allowed(address: str, allowed: list) -> bool:
    try:
        client = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    if client.version == 6 and client.ipv4_mapped:
        client = client.ipv4_mapped
    return any(client in network for network in allowed)


def load_worker_config(path: str | None) -> WorkerConfig:
    if not path:
        return WorkerConfig()
    with open(path) as handle:
        raw = json.load(handle)
    known = {item.name for item in fields(WorkerConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown settings {', '.join(unknown)}")
    try:
        return WorkerConfig(**raw)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: {error}") from error


def worker_arguments(binary: str, host: str, port: int, config: WorkerConfig) -> list[str]:
    arguments = [binary, "-H", host, "-p", str(port)]
    if config.devices:
        arguments += ["-d", ",".join(config.devices)]
    if config.threads:
        arguments += ["-t", str(config.threads)]
    if config.cache_dir:
        arguments += ["-c"]
    return arguments


def worker_environment(binary: str, config: WorkerConfig) -> dict:
    environment = dict(os.environ)
    folders = [*config.library_path, str(Path(binary).resolve().parent)]
    if os.name == "nt":
        environment["PATH"] = os.pathsep.join([*folders, environment.get("PATH", "")])
    else:
        environment["LD_LIBRARY_PATH"] = os.pathsep.join([*folders, *filter(None, [environment.get("LD_LIBRARY_PATH")])])
    if config.cache_dir:
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        environment[CACHE_VARIABLE] = config.cache_dir
    return environment


def free_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind((LOOPBACK, 0))
        return probe.getsockname()[1]


def pump(source: socket.socket, target: socket.socket) -> None:
    buffer = bytearray(RELAY_CHUNK_BYTES)
    view = memoryview(buffer)
    try:
        while count := source.recv_into(buffer):
            target.sendall(view[:count])
        target.shutdown(socket.SHUT_WR)
    except OSError as error:
        log.debug("a relayed connection ended: %s", error)
        shut_both(source, target)


def shut_both(*ends: socket.socket) -> None:
    for end in ends:
        try:
            end.shutdown(socket.SHUT_RDWR)
        except OSError as error:
            log.debug("closing a relayed connection: %s", error)


def tune(end: socket.socket) -> None:
    end.settimeout(None)
    end.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    end.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    for option, value in KEEPALIVE_TIMING:
        if hasattr(socket, option):
            end.setsockopt(socket.IPPROTO_TCP, getattr(socket, option), value)


def relay_pair(client: socket.socket, inner_port: int) -> None:
    try:
        inner = socket.create_connection((LOOPBACK, inner_port), timeout=CONNECT_TIMEOUT_SECONDS)
    except OSError as error:
        log.warning("the runtime worker did not take a connection: %s", error)
        client.close()
        return
    for end in (client, inner):
        tune(end)
    backwards = threading.Thread(target=pump, args=(inner, client), daemon=True)
    backwards.start()
    pump(client, inner)
    backwards.join()
    client.close()
    inner.close()


@dataclass
class Gate:
    listen_host: str
    listen_port: int
    inner_port: int
    allowed: list
    listener: socket.socket | None = None
    refused: int = 0
    stopping: bool = False
    most_relays: int = MOST_RELAYS
    relays: threading.BoundedSemaphore | None = None
    stopped: threading.Event = field(default_factory=threading.Event)

    def open(self) -> int:
        family = socket.AF_INET6 if ":" in self.listen_host else socket.AF_INET
        self.listener = socket.socket(family, socket.SOCK_STREAM)
        if os.name != "nt":
            self.listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.listener.bind((self.listen_host or "0.0.0.0", self.listen_port))
        self.listener.listen(BACKLOG)
        return self.listener.getsockname()[1]

    def serve(self) -> None:
        try:
            self._accept_until_stopped()
        except Exception:
            log.exception("the rpc gate stopped taking connections")
        finally:
            self.stopped.set()

    def _accept_until_stopped(self) -> None:
        self.relays = threading.BoundedSemaphore(self.most_relays)
        while not self.stopping:
            try:
                client, address = self.listener.accept()
            except OSError as error:
                if self.stopping:
                    return
                if self.listener.fileno() < 0:
                    raise
                log.warning("couldn't take a connection: %s", error)
                time.sleep(ACCEPT_RETRY_SECONDS)
                continue
            self._hand_over(client, address[0])

    def _hand_over(self, client: socket.socket, address: str) -> None:
        if not is_allowed(address, self.allowed):
            self.refused += 1
            log.warning("refused a connection from %s: not in allowed_clients", address)
            client.close()
            return
        if not self.relays.acquire(blocking=False):
            log.warning("refused a connection from %s: already relaying %d connections", address, self.most_relays)
            client.close()
            return
        log.debug("connection from %s", address)
        threading.Thread(target=self._relay, args=(client,), daemon=True).start()

    def _relay(self, client: socket.socket) -> None:
        try:
            relay_pair(client, self.inner_port)
        finally:
            self.relays.release()

    def close(self) -> None:
        self.stopping = True
        if self.listener:
            self.listener.close()


def wait_for_port(port: int, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + START_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the runtime worker exited with {process.returncode} while starting")
        try:
            with socket.create_connection((LOOPBACK, port), timeout=1):
                return
        except OSError:
            time.sleep(START_POLL_SECONDS)
    raise RuntimeError(f"the runtime worker did not open port {port} within {START_SECONDS} s")


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=STOP_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def refuse_unless_meant(config: WorkerConfig) -> None:
    if config.open_to_the_network and not config.open:
        raise RuntimeError(OPEN_REFUSAL.format(listen=config.listen))


def warn_when_open(config: WorkerConfig) -> None:
    if not config.open_to_the_network:
        return
    for _ in range(LOUD_REPEATS if config.open_to_everyone else 1):
        log.warning(NO_AUTHENTICATION_WARNING.format(listen=config.listen))


def run(config: WorkerConfig, launcher=subprocess.Popen, ready=None) -> int:
    binary = locate_binary(config.worker_binary, WORKER)
    if not Path(binary).is_file():
        raise RuntimeError(f"no runtime worker at {config.worker_binary}; install the runtime (arty-engine runtime install) "
                           "or set worker_binary")
    refuse_unless_meant(config)
    warn_when_open(config)
    gated = bool(config.allowed_clients)
    host, port = (LOOPBACK, free_loopback_port()) if gated else (config.host, config.port)
    process = launcher(worker_arguments(binary, host, port, config), env=worker_environment(binary, config))
    gate = None
    try:
        if gated:
            wait_for_port(port, process)
            gate = Gate(config.host, config.port, port, networks(config.allowed_clients))
            bound = gate.open()
            threading.Thread(target=gate.serve, name="rpc-gate", daemon=True).start()
            log.info("rpc worker on %s:%d for %s only (devices %s)", config.host, bound, ", ".join(config.allowed_clients),
                     ",".join(config.devices) or "all")
        else:
            log.info("rpc worker on %s (devices %s)", config.listen, ",".join(config.devices) or "all")
        if ready:
            ready(gate)
        return wait_for_either(process, gate)
    finally:
        if gate:
            gate.close()
        stop(process)


def wait_for_either(process: subprocess.Popen, gate: Gate | None) -> int:
    while gate is None or not gate.stopped.is_set():
        try:
            return process.wait(timeout=None if gate is None else GATE_CHECK_SECONDS)
        except subprocess.TimeoutExpired:
            continue
    log.error("the rpc gate stopped, so the worker stops too")
    return GATE_STOPPED_EXIT


def with_flags(config: WorkerConfig, arguments: argparse.Namespace) -> WorkerConfig:
    changes = {}
    if arguments.listen:
        changes["listen"] = arguments.listen
    if arguments.allow:
        changes["allowed_clients"] = arguments.allow
    if arguments.devices:
        changes["devices"] = [device.strip() for device in arguments.devices.split(",") if device.strip()]
    if arguments.cache_dir:
        changes["cache_dir"] = arguments.cache_dir
    if arguments.worker_binary:
        changes["worker_binary"] = arguments.worker_binary
    if arguments.threads is not None:
        changes["threads"] = arguments.threads
    if arguments.open:
        changes["open"] = True
    return replace(config, **changes)


def exit_on_terminate(*_) -> None:
    raise SystemExit(0)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="arty-engine worker",
                                     description="Share this machine's GPU and memory with an ArtyEngine over RPC")
    parser.add_argument("--config", default=None, help="worker JSON settings (listen, allowed_clients, devices, cache_dir, "
                                                       "worker_binary, library_path, threads)")
    parser.add_argument("--listen", default="", help=f"host:port to listen on (default {DEFAULT_LISTEN})")
    parser.add_argument("--allow", action="append", default=[], help="a client address or network allowed in (repeat for more)")
    parser.add_argument("--devices", default="", help="comma-separated devices to share, such as CUDA0,CPU")
    parser.add_argument("--cache-dir", default="", help="keep a file cache of model weights here, so reloads send less")
    parser.add_argument("--worker-binary", default="", help="the runtime worker binary, or the runtime folder")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads for the CPU device")
    parser.add_argument("--open", action="store_true", help="allow listening on a network address with no --allow list")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if arguments.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout)
    try:
        config = with_flags(load_worker_config(arguments.config), arguments)
    except (OSError, ValueError) as error:
        print(f"worker not started: {error}", file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, exit_on_terminate)
    try:
        return run(config)
    except RuntimeError as error:
        log.error("%s", error)
        return 1
