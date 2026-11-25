import logging
import socket
import threading
import time
from dataclasses import dataclass, field

log = logging.getLogger("arty-engine.rpc")

CHECK_TIMEOUT_SECONDS = 2.0
LOWEST_PORT, HIGHEST_PORT = 1, 65535
WORKER_SETTINGS = ("address", "label", "memory_mb")
UNSAFE_ADDRESS_CHARACTERS = set(", \t\r\n")


@dataclass(frozen=True)
class RpcWorker:
    address: str
    label: str
    memory_mb: int = 0

    @property
    def host(self) -> str:
        return split_address(self.address)[0]

    @property
    def port(self) -> int:
        return split_address(self.address)[1]


def split_address(address: str) -> tuple[str, int]:
    host, separator, port = address.rpartition(":")
    if not separator or not host or not port.isdigit():
        raise ValueError(f"rpc worker address {address!r} must be host:port")
    return host.strip("[]"), int(port)


def check_address(address) -> str:
    if not isinstance(address, str) or not address or UNSAFE_ADDRESS_CHARACTERS & set(address):
        raise ValueError(f"rpc worker address {address!r} must be host:port with no spaces or commas")
    _, port = split_address(address)
    if not LOWEST_PORT <= port <= HIGHEST_PORT:
        raise ValueError(f"rpc worker address {address!r} has a port outside {LOWEST_PORT} to {HIGHEST_PORT}")
    return address


def worker_from_setting(entry) -> RpcWorker:
    if isinstance(entry, str):
        return RpcWorker(address=check_address(entry), label=entry)
    if not isinstance(entry, dict):
        raise ValueError("each rpc_workers entry must be \"host:port\" or an object with address, label and memory_mb")
    unknown = sorted(set(entry) - set(WORKER_SETTINGS))
    if unknown:
        raise ValueError(f"rpc_workers entry {entry.get('address')!r}: unknown settings {', '.join(unknown)}")
    address = check_address(entry.get("address"))
    label = entry.get("label", address)
    if not isinstance(label, str) or not label:
        raise ValueError(f"rpc_workers entry {address}: label must be a non-empty string")
    memory = entry.get("memory_mb", 0)
    if isinstance(memory, bool) or not isinstance(memory, int) or memory < 0:
        raise ValueError(f"rpc_workers entry {address}: memory_mb must be a whole number of MiB, 0 or more")
    return RpcWorker(address=address, label=label, memory_mb=memory)


def parse_workers(value) -> list[RpcWorker]:
    if not isinstance(value, list):
        raise ValueError("rpc_workers must be a list of \"host:port\" strings or objects")
    workers = [worker_from_setting(entry) for entry in value]
    for kind, names in (("label", [worker.label for worker in workers]), ("address", [worker.address for worker in workers])):
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"rpc_workers: the {kind} {', '.join(repeated)} is listed more than once")
    return workers


def probe(worker: RpcWorker, timeout: float = CHECK_TIMEOUT_SECONDS) -> str:
    try:
        with socket.create_connection((worker.host, worker.port), timeout=timeout):
            return ""
    except OSError as error:
        return str(error) or error.__class__.__name__


@dataclass
class WorkerState:
    up: bool | None = None
    checked_at: float | None = None
    latency_ms: float | None = None
    error: str = ""
    down_in_a_row: int = 0


@dataclass
class WorkerHealth:
    workers: list[RpcWorker]
    timeout: float = CHECK_TIMEOUT_SECONDS
    prober: object = probe
    states: dict[str, WorkerState] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def named(self, names: list[str] | None) -> list[RpcWorker]:
        if not names:
            return list(self.workers)
        return [worker for worker in self.workers if worker.label in names or worker.address in names]

    def check(self, workers: list[RpcWorker]) -> list[RpcWorker]:
        answering = []
        for worker in workers:
            started = time.monotonic()
            problem = self.prober(worker, self.timeout)
            self._record(worker, problem, (time.monotonic() - started) * 1000)
            if not problem:
                answering.append(worker)
        return answering

    def _record(self, worker: RpcWorker, problem: str, latency_ms: float) -> None:
        with self.lock:
            state = self.states.setdefault(worker.address, WorkerState())
            was_up = state.up
            state.up, state.checked_at, state.error = not problem, time.time(), problem
            state.latency_ms = None if problem else round(latency_ms, 1)
            state.down_in_a_row = state.down_in_a_row + 1 if problem else 0
        if problem and was_up is not False:
            log.warning("rpc worker %s (%s) is not answering: %s", worker.label, worker.address, problem)
        if not problem and was_up is False:
            log.info("rpc worker %s (%s) is answering again", worker.label, worker.address)

    def down_in_a_row(self, worker: RpcWorker) -> int:
        with self.lock:
            return self.states.get(worker.address, WorkerState()).down_in_a_row

    def describe(self, used_by: dict[str, list[str]]) -> list[dict]:
        with self.lock:
            states = {address: WorkerState(**vars(state)) for address, state in self.states.items()}
        described = []
        for worker in self.workers:
            state = states.get(worker.address, WorkerState())
            described.append({"label": worker.label, "address": worker.address, "memory_mb": worker.memory_mb,
                              "up": state.up, "checked_at": state.checked_at, "latency_ms": state.latency_ms,
                              "error": state.error, "models": sorted(used_by.get(worker.address, []))})
        return described
