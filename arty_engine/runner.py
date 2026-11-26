import json
import logging
import math
import os
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

from . import gpu, rpc
from .scrub import scrub, start_scrubbed_log
from .backend import Backend, resolve_backend
from .config import Config, SpeculativeSettings
from .gguf import whole_number
from .model_settings import (FAIL, RPC_AUTO, RPC_NEVER, layout_arguments, layout_for, run_arguments,
                             run_settings)
from .runtime import SERVER, locate_binary
from .store import Model, ModelStore, parse_name
from .translate import THINKING_SWITCH

log = logging.getLogger("arty-engine.runner")

HEALTH_POLL_SECONDS = 0.25
HEALTH_TIMEOUT_SECONDS = 2
STOP_GRACE_SECONDS = 10
REAPER_INTERVAL_SECONDS = 1.0
WAIT_FOR_IDLE_SECONDS = 600
WORKER_RECHECK_SECONDS = 5.0
WORKER_WATCH_SECONDS = 15.0
RANK_POOLING = 4
SLOTS_FOLDER = "slots"
KEYS_FOLDER = "keys"
KEY_BYTES = 32
PRIVATE_FILE_MODE = 0o600
PRIVATE_FOLDER_MODE = 0o700
LOG_DRAIN_SECONDS = 2
DEATH_WAIT_SECONDS = 3.0
DEATH_POLL_SECONDS = 0.1
KV_BYTES_PER_ELEMENT = {"f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625, "q5_0": 0.6875, "q5_1": 0.75,
                        "q4_0": 0.5625, "q4_1": 0.625, "iq4_nl": 0.5625}
DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
DURATION_UNITS = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
FOREVER = None
LONGEST_KEEP_ALIVE_SECONDS = 100 * 365 * 24 * 3600
TOOL_TO_JSON = re.compile(r"\btool\s*\|\s*tojson(?:\(((?:[^()]|\([^()]*\))*)\))?")
COMPACT_SEPARATORS = 'separators=(",", ":")'


class LoadError(RuntimeError):
    pass


class RunnerGone(Exception):
    pass


class WorkersDown(LoadError):
    pass


class WorkersBusy(WorkersDown):
    pass


def parse_keep_alive(value, default) -> float | None:
    if value is None or value == "":
        return parse_keep_alive(default, 300)
    if isinstance(value, bool):
        raise ValueError(f"keep_alive must be a duration, got {value!r}")
    if isinstance(value, (int, float)):
        return seconds_or_forever(value)
    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        number = None
    if number is not None:
        return seconds_or_forever(number)
    if text.startswith("-"):
        return FOREVER
    parts = DURATION_PART.findall(text)
    if not parts or "".join(amount + unit for amount, unit in parts) != text:
        raise ValueError(f"keep_alive {value!r} is not a duration like 10m, 30s or -1")
    return seconds_or_forever(sum(float(amount) * DURATION_UNITS[unit] for amount, unit in parts))


def seconds_or_forever(seconds) -> float | None:
    if isinstance(seconds, float) and math.isnan(seconds):
        raise ValueError("keep_alive can't be NaN")
    if seconds < 0 or seconds > LONGEST_KEEP_ALIVE_SECONDS:
        return FOREVER
    return float(seconds)


def _compact_tool_filter(match: re.Match) -> str:
    existing = (match.group(1) or "").strip()
    if "separators" in existing:
        return match.group(0)
    arguments = f"{existing}, {COMPACT_SEPARATORS}" if existing else COMPACT_SEPARATORS
    return f"tool | tojson({arguments})"


def compact_tool_definitions(template: str) -> str:
    return TOOL_TO_JSON.sub(_compact_tool_filter, template)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def largest_count(value) -> int:
    if isinstance(value, list):
        return max((whole_number(item) for item in value), default=0)
    return whole_number(value)


def kv_cache_mb(model: Model, ctx: int, cache_type: str) -> int:
    info = model.gguf
    layers = info.arch_number("block_count")
    heads = largest_count(info.arch_value("attention.head_count"))
    kv_heads = largest_count(info.arch_value("attention.head_count_kv")) or heads
    embedding = info.arch_number("embedding_length")
    head_dim = embedding // heads if heads else 0
    key_length = info.arch_number("attention.key_length") or head_dim
    value_length = info.arch_number("attention.value_length") or head_dim
    elements = layers * kv_heads * (key_length + value_length) * ctx
    return int(elements * KV_BYTES_PER_ELEMENT.get(cache_type, 2.0) / gpu.MIB)


@dataclass
class LaunchPlan:
    model: Model
    ctx: int
    speculative: SpeculativeSettings
    draft_path: Path | None = None
    template_file: Path | None = None
    cpu_only: bool = False
    may_leak_thinking: bool = False
    named_thinking_leaker: bool = False
    settings: dict = field(default_factory=dict)
    load_options: dict = field(default_factory=dict)
    workers: tuple = ()
    slot_dir: Path | None = None

    @property
    def key(self) -> tuple:
        return (self.model.load_digest, self.ctx, self.speculative.draft, self.speculative.spec_type, self.cpu_only,
                json.dumps(self.load_options, sort_keys=True))

    @property
    def rpc_mode(self) -> str:
        if self.cpu_only:
            return RPC_NEVER
        return self.settings.get("rpc", RPC_AUTO)

    @property
    def run(self) -> dict:
        return {**run_settings(self.settings), **self.load_options}


def warn_about_unknown_names(names: set[str], store: ModelStore, setting: str) -> None:
    for name in sorted(names - set(store.names())):
        log.warning("%s in %s matches no installed model, so it has no effect until a model of that name "
                    "is installed", name, setting)


def may_leak_thinking(model: Model, template_source: Model, leaker_names: set[str]) -> bool:
    if "thinking" not in model.capabilities:
        return False
    return model.name in leaker_names or THINKING_SWITCH not in template_source.gguf.chat_template


@dataclass
class Runner:
    plan: LaunchPlan
    port: int
    process: subprocess.Popen
    estimate_mb: int
    load_seconds: float = 0.0
    in_flight: int = 0
    last_used: float = field(default_factory=time.monotonic)
    expires_at: float | None = None
    expires_wall: float | None = None
    stopping: bool = False
    unload_requested: bool = False
    latest_keep_alive: float | None = None
    latest_keep_alive_noted: bool = False
    process_memory: Callable[[], dict[int, int]] = gpu.memory_by_process
    api_key: str = ""
    log_thread: threading.Thread | None = None
    key_file: Path | None = None

    @property
    def name(self) -> str:
        return self.plan.model.name

    @property
    def headers(self) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def alive(self) -> bool:
        return self.process.poll() is None and not self.stopping

    def vram_mb(self) -> int:
        return self.process_memory().get(self.process.pid, 0)


def is_reranker(model: Model) -> bool:
    return model.is_embedding and whole_number(model.gguf.arch_value("pooling_type")) == RANK_POOLING


def placement_arguments(plan: LaunchPlan) -> list[str]:
    if plan.cpu_only:
        return ["-ngl", "0", "--device", "none"]
    arguments = []
    if plan.workers:
        arguments += ["--rpc", ",".join(worker.address for worker in plan.workers)]
    layout = layout_for(plan.settings, with_workers=bool(plan.workers))
    if "gpu_layers" in plan.load_options:
        layout["gpu_layers"] = plan.load_options["gpu_layers"]
    if "main_gpu" in plan.load_options:
        layout["main_gpu"] = plan.load_options["main_gpu"]
    return arguments + layout_arguments(layout)


def build_arguments(config: Config, plan: LaunchPlan, port: int, key_file: Path | None = None) -> list[str]:
    model = plan.model
    run = plan.run
    arguments = [config.runner_binary, "--model", str(model.model_path), "--host", "127.0.0.1", "--port", str(port),
                 "--no-webui", "--offline", "-c", str(plan.ctx), "-np", str(run.get("parallel", config.parallel)), "-a", model.name,
                 "--fit-target", str(config.fit_target_mb), "--no-warmup", "--metrics"]
    if key_file:
        arguments += ["--api-key-file", str(key_file)]
    if plan.slot_dir:
        arguments += ["--slot-save-path", str(plan.slot_dir)]
    arguments += placement_arguments(plan) + run_arguments(run)
    if model.is_embedding:
        mode = "--rerank" if is_reranker(model) else "--embedding"
        return arguments + [mode, "-b", str(plan.ctx), "-ub", str(plan.ctx)]
    kv_cache_type = run.get("kv_cache_type", config.kv_cache_type)
    arguments += ["--jinja", "--reasoning-format", "deepseek", "-fa", run.get("flash_attn", config.flash_attn),
                  "-ctk", kv_cache_type, "-ctv", kv_cache_type, "-b", str(run.get("batch", config.batch)),
                  "-ub", str(run.get("ubatch", config.ubatch)), "--cache-ram", str(config.cache_ram_mb)]
    if model.projector_path:
        arguments += ["--mmproj", str(model.projector_path)]
    for adapter in model.adapter_paths:
        arguments += ["--lora", str(adapter)]
    if plan.template_file:
        arguments += ["--chat-template-file", str(plan.template_file)]
    arguments += speculative_arguments(config, plan, kv_cache_type)
    return arguments + [str(item) for item in config.extra_args.get(model.name, [])]


def speculative_arguments(config: Config, plan: LaunchPlan, kv_cache_type: str) -> list[str]:
    settings = plan.speculative
    arguments = []
    if plan.draft_path:
        arguments += ["--model-draft", str(plan.draft_path), "-ngld", "all", "--spec-draft-n-max", str(settings.draft_max),
                      "--spec-draft-n-min", str(settings.draft_min), "--spec-draft-p-min", str(settings.draft_p_min),
                      "-ctkd", kv_cache_type, "-ctvd", kv_cache_type]
    if settings.spec_type:
        arguments += ["--spec-type", settings.spec_type]
        if settings.ngram_min:
            arguments += ["--spec-ngram-mod-n-min", str(settings.ngram_min)]
        if settings.ngram_max:
            arguments += ["--spec-ngram-mod-n-max", str(settings.ngram_max)]
    return arguments


class RunnerPool:
    def __init__(self, config: Config, store: ModelStore, launcher=subprocess.Popen, memory=None, backend: Backend | None = None,
                 prober=rpc.probe):
        self.backend = backend or resolve_backend(config)
        self.config = replace(config, runner_binary=locate_binary(self.backend.runner_binary, SERVER),
                              library_path=list(self.backend.library_path))
        self.health = rpc.WorkerHealth(config.workers, prober=prober)
        self.store = store
        self.leaker_names = {parse_name(name).display for name in config.thinking_leakers}
        warn_about_unknown_names(self.leaker_names, store, "thinking_leakers")
        warn_about_unknown_names({parse_name(name).display for name in config.model_settings}, store, "model_settings")
        self.launcher = launcher
        self.memory = memory or partial(gpu.card_memory, self.backend.name)
        self.process_memory = partial(gpu.memory_by_process, self.backend.name)
        self.runners: dict[str, Runner] = {}
        self.loading: dict[int, Runner] = {}
        self.state = threading.Condition()
        self.load_lock = threading.Lock()
        self.state_dir = Path(config.state_dir)
        self._remove_stale_key_files()
        self._reaper = threading.Thread(target=self._reap_forever, name="keep-alive-reaper", daemon=True)
        self._watcher = threading.Thread(target=self._watch_workers_forever, name="rpc-worker-watch", daemon=True)

    def start_reaper(self) -> None:
        self._reaper.start()
        if self.health.workers:
            self._watcher.start()

    def plan(self, model: Model, ctx: int | None, cpu_only: bool = False, load_options: dict | None = None) -> LaunchPlan:
        cpu_only = cpu_only or self.backend.runs_everything_on_the_cpu
        if model.is_embedding:
            ctx = model.context_length or self.config.default_ctx
        else:
            ctx = int(ctx or model.params.get("num_ctx") or self.config.default_ctx)
            if model.context_length:
                ctx = min(ctx, model.context_length)
        speculative = SpeculativeSettings() if model.is_embedding or cpu_only else self.config.speculative_for(model.name)
        load_options = dict(load_options or {})
        if load_options.get("batch", 0) > ctx:
            load_options["batch"] = ctx
        plan = LaunchPlan(model=model, ctx=ctx, speculative=speculative, cpu_only=cpu_only,
                          settings=self.config.settings_for(model.name), load_options=load_options,
                          slot_dir=self.state_dir / SLOTS_FOLDER / safe_file_name(model.name))
        if speculative.draft:
            plan.draft_path = self.store.get(speculative.draft).model_path
        donor = self.store.template_donor(model)
        plan.template_file = self._launch_template(model, donor)
        plan.may_leak_thinking = may_leak_thinking(model, donor or model, self.leaker_names)
        plan.named_thinking_leaker = model.name in self.leaker_names
        return plan

    def _launch_template(self, model: Model, donor: Model | None) -> Path | None:
        source = donor or model
        template = source.gguf.chat_template
        if self.config.compact_tool_json:
            compact = compact_tool_definitions(template)
            if compact != template:
                return self._template_file(f"{source.digest}-compact-tools", compact)
        return self._template_file(donor.digest, template) if donor else None

    def _template_file(self, name: str, template: str) -> Path:
        folder = self.state_dir / "templates"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{name}.jinja"
        if not path.exists():
            path.write_text(template)
        return path

    def full_estimate_mb(self, plan: LaunchPlan) -> int:
        files = plan.model.model_path.stat().st_size
        if plan.model.projector_path:
            files += plan.model.projector_path.stat().st_size
        if plan.draft_path:
            files += plan.draft_path.stat().st_size
        cache_type = plan.run.get("kv_cache_type", self.config.kv_cache_type)
        kv = 0 if plan.model.is_embedding else kv_cache_mb(plan.model, plan.ctx, cache_type)
        return int(files / gpu.MIB) + kv + self.config.load_overhead_mb

    def estimate_mb(self, plan: LaunchPlan) -> int:
        if plan.cpu_only:
            return 0
        full = self.full_estimate_mb(plan)
        if not plan.workers:
            return full
        on_workers = sum(worker.memory_mb for worker in plan.workers)
        return max(self.config.load_overhead_mb, min(full - on_workers, self.config.vram_budget_mb))

    def acquire(self, model: Model, ctx: int | None, cpu_only: bool = False,
                load_options: dict | None = None) -> tuple[Runner, float]:
        plan = self.plan(model, ctx, cpu_only, load_options)
        runner = self._reuse(plan)
        if runner:
            return runner, 0.0
        with self.load_lock:
            runner = self._reuse(plan)
            if runner:
                return runner, 0.0
            self._retire(model.name)
            runner = self._load(plan)
            with self.state:
                runner.in_flight = 1
                self.runners[model.name] = runner
            return runner, runner.load_seconds

    def _load(self, plan: LaunchPlan) -> Runner:
        try:
            workers = self._workers_for(plan)
        except WorkersBusy as error:
            if plan.settings.get("rpc_when_down") == FAIL or not self._fits_here_now(plan):
                raise
            log.warning("%s; loading %s on this machine instead", error, plan.model.name)
            return self._load_as_planned(plan)
        if not workers:
            return self._load_as_planned(plan)
        try:
            return self._load_as_planned(replace(plan, workers=tuple(workers)))
        except WorkersDown as error:
            if plan.settings.get("rpc_when_down") == FAIL:
                raise
            log.warning("%s; loading %s on this machine instead", error, plan.model.name)
        return self._load_as_planned(plan)

    def _load_as_planned(self, plan: LaunchPlan) -> Runner:
        estimate = self.estimate_mb(plan)
        self._make_room(estimate, plan.model.name)
        return self._launch(plan, estimate)

    def _workers_for(self, plan: LaunchPlan) -> list[rpc.RpcWorker]:
        mode = plan.rpc_mode
        if mode == RPC_NEVER or not self.health.workers:
            return []
        if mode == RPC_AUTO and self.full_estimate_mb(plan) <= self.config.vram_budget_mb:
            return []
        wanted = self.health.named(plan.settings.get("rpc_workers"))
        self._free_workers_for(plan.model.name, wanted)
        answering = self._reachable_workers(wanted)
        if len(answering) == len(wanted):
            return answering
        silent = ", ".join(f"{worker.label} ({worker.address})" for worker in wanted if worker not in answering)
        reason = f"{plan.model.name} is set to use rpc workers, and {silent} did not answer"
        if plan.settings.get("rpc_when_down") == FAIL:
            raise WorkersDown(f"{reason}; start the worker, or set rpc_when_down to local to load it on this machine")
        log.warning("%s; loading it on this machine instead", reason)
        return []

    def _free_workers_for(self, name: str, wanted: list[rpc.RpcWorker]) -> None:
        addresses = {worker.address for worker in wanted}
        with self.state:
            holders = [runner for runner in self.runners.values() if runner.alive and runner.name != name
                       and addresses & {worker.address for worker in runner.plan.workers}]
            busy = [runner for runner in holders if runner.in_flight]
            if busy:
                held = ", ".join(sorted({worker.label for runner in busy for worker in runner.plan.workers
                                         if worker.address in addresses}))
                raise WorkersBusy(f"rpc worker {held} is serving {', '.join(runner.name for runner in busy)}, which is answering "
                                  f"a request, and a worker serves one model at a time; try {name} again when it is done")
            for runner in holders:
                runner.stopping = True
                del self.runners[runner.name]
        for runner in holders:
            log.info("unloading %s to free rpc worker %s for %s", runner.name,
                     ", ".join(worker.label for worker in runner.plan.workers), name)
            self._stop(runner)

    def _fits_here_now(self, plan: LaunchPlan) -> bool:
        needed = self.estimate_mb(plan)
        with self.state:
            busy_mb = sum(runner.estimate_mb for runner in self.runners.values() if runner.in_flight)
            idle_mb = sum(runner.estimate_mb for runner in self.runners.values() if not runner.in_flight)
        card = self.memory()
        return busy_mb + needed <= self.config.vram_budget_mb and (card is None or card.free_mb + idle_mb >= needed)

    def drop_if_dead(self, runner: Runner) -> bool:
        deadline = time.monotonic() + DEATH_WAIT_SECONDS
        while runner.process.poll() is None and time.monotonic() < deadline:
            time.sleep(DEATH_POLL_SECONDS)
        if runner.process.poll() is None:
            return False
        with self.state:
            runner.stopping = True
            if self.runners.get(runner.name) is runner:
                del self.runners[runner.name]
        log.warning("the model runner for %s died with %s while answering a request", runner.name, runner.process.returncode)
        return True

    def workers_status(self) -> list[dict]:
        used_by: dict[str, list[str]] = {}
        for runner in self.loaded():
            for worker in runner.plan.workers:
                used_by.setdefault(worker.address, []).append(runner.name)
        return self.health.describe(used_by)

    def _reuse(self, plan: LaunchPlan) -> Runner | None:
        with self.state:
            runner = self.runners.get(plan.model.name)
            if runner and runner.alive and runner.plan.key == plan.key:
                runner.plan.model = plan.model
                runner.in_flight += 1
                runner.expires_at = None
                runner.expires_wall = None
                runner.unload_requested = False
                return runner
        return None

    def note_keep_alive(self, runner: Runner, keep_alive: float | None) -> None:
        with self.state:
            runner.latest_keep_alive = keep_alive
            runner.latest_keep_alive_noted = True

    def _settle_expiry(self, runner: Runner, keep_alive: float | None) -> bool:
        if runner.latest_keep_alive_noted:
            keep_alive = runner.latest_keep_alive
            runner.latest_keep_alive_noted = False
        if keep_alive is FOREVER:
            runner.expires_at = runner.expires_wall = None
        else:
            runner.expires_at = runner.last_used + keep_alive
            runner.expires_wall = time.time() + keep_alive
        return keep_alive == 0 or runner.unload_requested

    def release(self, runner: Runner, keep_alive: float | None) -> None:
        stop_now = False
        with self.state:
            runner.in_flight -= 1
            runner.last_used = time.monotonic()
            if runner.in_flight == 0:
                stop_now = self._settle_expiry(runner, keep_alive)
            self.state.notify_all()
        if stop_now:
            self.unload_runner(runner)

    def borrow(self, name: str) -> Runner | None:
        with self.state:
            runner = self.runners.get(name)
            if not runner or not runner.alive or runner.unload_requested:
                return None
            runner.in_flight += 1
            return runner

    def give_back(self, runner: Runner) -> None:
        with self.state:
            runner.in_flight -= 1
            stop_now = False
            if runner.in_flight == 0 and runner.latest_keep_alive_noted:
                runner.last_used = time.monotonic()
                stop_now = self._settle_expiry(runner, None)
            elif runner.in_flight == 0:
                stop_now = runner.unload_requested
            self.state.notify_all()
        if stop_now:
            self.unload_runner(runner)

    def touch(self, model: Model, keep_alive: float | None) -> bool:
        with self.state:
            runner = self.runners.get(model.name)
            if not runner or not runner.alive:
                return False
            runner.in_flight += 1
        self.note_keep_alive(runner, keep_alive)
        self.release(runner, keep_alive)
        return True

    def holds(self, name: str) -> bool:
        with self.state:
            runner = self.runners.get(name)
            return bool(runner and runner.alive)

    def refresh_record(self, model: Model) -> None:
        with self.state:
            runner = self.runners.get(model.name)
            if runner and runner.plan.model.load_digest == model.load_digest:
                runner.plan.model = model

    def unload(self, name: str) -> bool:
        with self.state:
            runner = self.runners.get(name)
        if not runner:
            return False
        self.unload_runner(runner)
        return True

    def unload_runner(self, runner: Runner) -> None:
        with self.state:
            if runner.in_flight:
                runner.expires_at = time.monotonic()
                runner.expires_wall = time.time()
                runner.unload_requested = True
                return
            runner.stopping = True
            runner.unload_requested = True
            if self.runners.get(runner.name) is runner:
                del self.runners[runner.name]
            self.state.notify_all()
        self._stop(runner)

    def _retire(self, name: str) -> None:
        with self.state:
            runner = self.runners.get(name)
            if not runner:
                return
            deadline = time.monotonic() + WAIT_FOR_IDLE_SECONDS
            while runner.in_flight and runner.alive and time.monotonic() < deadline:
                self.state.wait(timeout=1)
            if runner.in_flight and runner.alive:
                raise LoadError(f"{name} is still answering another request with different settings; try again when it is done")
            runner.stopping = True
            if self.runners.get(name) is runner:
                del self.runners[name]
        self._stop(runner)

    def _make_room(self, needed_mb: int, for_name: str) -> None:
        if needed_mb == 0:
            return
        deadline = time.monotonic() + WAIT_FOR_IDLE_SECONDS
        while True:
            with self.state:
                ours = sum(runner.estimate_mb for runner in self.runners.values())
                within_budget = ours + needed_mb <= self.config.vram_budget_mb
                card = self.memory()
                fits_card = card is None or card.free_mb >= needed_mb
                if within_budget and fits_card:
                    return
                on_card = [runner for runner in self.runners.values() if runner.estimate_mb > 0]
                idle = sorted((runner for runner in on_card if runner.in_flight == 0), key=lambda runner: runner.last_used)
                if not idle and not on_card:
                    if not fits_card:
                        log.warning("%s needs ~%d MiB but only %d MiB is free on the card (used by other programs); "
                                    "the model runner will fit what it can", for_name, needed_mb, card.free_mb)
                    return
                if not idle:
                    if time.monotonic() > deadline:
                        raise LoadError(f"no room on the GPU for {for_name}: every loaded model is busy")
                    self.state.wait(timeout=1)
                    continue
                victim = idle[0]
                victim.stopping = True
                del self.runners[victim.name]
            log.info("unloading %s to make room for %s", victim.name, for_name)
            self._stop(victim)

    def _launch(self, plan: LaunchPlan, estimate: int) -> Runner:
        port = free_port()
        if plan.slot_dir:
            plan.slot_dir.mkdir(parents=True, exist_ok=True)
        logs = self.state_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{safe_file_name(plan.model.name)}.log"
        environment = {**os.environ, "LD_LIBRARY_PATH": ":".join(self.config.library_path + [str(Path(self.config.runner_binary).resolve().parent)])}
        if plan.cpu_only:
            environment.update(self.backend.cpu_only_environment)
        started = time.monotonic()
        on_workers = f", with rpc workers {', '.join(worker.label for worker in plan.workers)}" if plan.workers else ""
        api_key = secrets.token_urlsafe(KEY_BYTES)
        key_file = self._write_key_file(plan.model.name, api_key)
        arguments = build_arguments(self.config, plan, port, key_file)
        log.info("loading %s (ctx %d, ~%d MiB here%s): %s", plan.model.name, plan.ctx, estimate, on_workers, " ".join(arguments[1:]))
        runner = None
        try:
            process = self.launcher(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=environment,
                                    start_new_session=True)
            runner = Runner(plan=plan, port=port, process=process, estimate_mb=estimate, process_memory=self.process_memory,
                            api_key=api_key, log_thread=start_scrubbed_log(process, log_path), key_file=key_file)
            with self.state:
                self.loading[id(runner)] = runner
            try:
                self._wait_until_ready(runner, log_path)
            except LoadError:
                self._stop(runner)
                raise
        finally:
            if runner is not None:
                with self.state:
                    self.loading.pop(id(runner), None)
            key_file.unlink(missing_ok=True)
        runner.load_seconds = time.monotonic() - started
        log.info("loaded %s in %.1fs on port %d", plan.model.name, runner.load_seconds, port)
        return runner

    def _remove_stale_key_files(self) -> None:
        folder = self.state_dir / KEYS_FOLDER
        if not folder.is_dir():
            return
        for path in folder.iterdir():
            if path.is_file():
                path.unlink(missing_ok=True)

    def _write_key_file(self, name: str, api_key: str) -> Path:
        folder = self.state_dir / KEYS_FOLDER
        folder.mkdir(parents=True, exist_ok=True)
        folder.chmod(PRIVATE_FOLDER_MODE)
        path = folder / f"{safe_file_name(name)}.{secrets.token_hex(4)}.key"
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "w") as output:
            output.write(api_key + "\n")
        return path

    def _attached_addresses(self) -> set[str]:
        with self.state:
            holders = [runner for runner in self.runners.values() if runner.alive] + list(self.loading.values())
            return {worker.address for runner in holders for worker in runner.plan.workers}

    def _reachable_workers(self, workers: list[rpc.RpcWorker]) -> list[rpc.RpcWorker]:
        attached = self._attached_addresses()
        reachable = self.health.check([worker for worker in workers if worker.address not in attached], silence_is_down=True)
        return [worker for worker in workers if worker.address in attached or worker in reachable]

    def _silent_workers(self, runner: Runner) -> list[rpc.RpcWorker]:
        reachable = self.health.check(list(runner.plan.workers))
        return [worker for worker in runner.plan.workers if worker not in reachable]

    def _check_workers_while_loading(self, runner: Runner) -> None:
        silent = self._silent_workers(runner)
        if silent:
            names = ", ".join(f"{worker.label} ({worker.address})" for worker in silent)
            raise WorkersDown(f"rpc worker {names} stopped answering while {runner.name} was loading")

    def _blame_unconfirmed_workers(self, runner: Runner) -> None:
        self._check_workers_while_loading(runner)
        unconfirmed = [worker for worker in runner.plan.workers if not self.health.answered_last(worker)]
        if unconfirmed:
            names = ", ".join(f"{worker.label} ({worker.address})" for worker in unconfirmed)
            raise WorkersDown(f"the model runner for {runner.name} exited while loading, and rpc worker {names} did not answer")

    def _wait_until_ready(self, runner: Runner, log_path: Path) -> None:
        deadline = time.monotonic() + self.config.load_timeout_seconds
        next_worker_check = time.monotonic() + WORKER_RECHECK_SECONDS
        while time.monotonic() < deadline:
            if runner.process.poll() is not None:
                if runner.log_thread:
                    runner.log_thread.join(timeout=LOG_DRAIN_SECONDS)
                log.error("the model runner for %s exited with %s while loading; its last lines: %s", runner.name,
                          runner.process.returncode, scrub(last_lines(log_path)))
                if runner.plan.workers:
                    self._blame_unconfirmed_workers(runner)
                raise LoadError(f"the model runner for {runner.name} exited with {runner.process.returncode} while loading; "
                                "the engine log has the details")
            if runner.plan.workers and time.monotonic() >= next_worker_check:
                self._check_workers_while_loading(runner)
                next_worker_check = time.monotonic() + WORKER_RECHECK_SECONDS
            try:
                request = urllib.request.Request(f"{runner.base_url}/health", headers=runner.headers)
                with urllib.request.urlopen(request, timeout=HEALTH_TIMEOUT_SECONDS) as response:
                    if response.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(HEALTH_POLL_SECONDS)
        raise LoadError(f"{runner.name} did not finish loading within {self.config.load_timeout_seconds}s")

    @staticmethod
    def _stop(runner: Runner) -> None:
        if runner.process.poll() is not None:
            return
        runner.process.terminate()
        try:
            runner.process.wait(timeout=STOP_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            runner.process.kill()
            runner.process.wait()
        log.info("unloaded %s", runner.name)

    def reap_once(self) -> None:
        now = time.monotonic()
        with self.state:
            expired = [runner for runner in self.runners.values()
                       if runner.in_flight == 0 and runner.expires_at is not None and runner.expires_at <= now]
            dead = [runner for runner in self.runners.values() if runner.process.poll() is not None]
            for runner in expired + dead:
                runner.stopping = True
                if self.runners.get(runner.name) is runner:
                    del self.runners[runner.name]
        for runner in dead:
            on_workers = f" (it used rpc worker {', '.join(worker.label for worker in runner.plan.workers)})" if runner.plan.workers else ""
            log.warning("the model runner for %s died with %s%s; the next request loads it again", runner.name,
                        runner.process.returncode, on_workers)
        for runner in expired:
            self._stop(runner)

    def watch_workers_once(self) -> None:
        self._reachable_workers(self.health.workers)

    def _watch_workers_forever(self) -> None:
        while True:
            time.sleep(WORKER_WATCH_SECONDS)
            try:
                self.watch_workers_once()
            except Exception:
                log.exception("rpc worker watch failed")

    def _reap_forever(self) -> None:
        while True:
            time.sleep(REAPER_INTERVAL_SECONDS)
            try:
                self.reap_once()
            except Exception:
                log.exception("keep-alive reaper failed")

    def loaded(self) -> list[Runner]:
        with self.state:
            return [runner for runner in self.runners.values() if runner.alive]

    def shutdown(self) -> None:
        with self.state:
            runners = list(self.runners.values()) + list(self.loading.values())
            self.runners.clear()
            self.loading.clear()
        for runner in runners:
            runner.stopping = True
            self._stop(runner)
            if runner.key_file:
                runner.key_file.unlink(missing_ok=True)


def safe_file_name(name: str) -> str:
    return name.replace("/", "_").replace(":", "_")


def last_lines(path: Path, count: int = 8) -> str:
    try:
        return " | ".join(path.read_text(errors="replace").strip().splitlines()[-count:])
    except OSError:
        return "(no log)"
