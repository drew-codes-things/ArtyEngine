import logging
import math
import os
import re
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

from . import gpu
from .backend import Backend, resolve_backend
from .config import Config, SpeculativeSettings
from .store import Model, ModelStore, parse_name
from .translate import THINKING_SWITCH

log = logging.getLogger("arty-engine.runner")

HEALTH_POLL_SECONDS = 0.25
HEALTH_TIMEOUT_SECONDS = 2
STOP_GRACE_SECONDS = 10
REAPER_INTERVAL_SECONDS = 1.0
WAIT_FOR_IDLE_SECONDS = 600
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


def kv_cache_mb(model: Model, ctx: int, cache_type: str) -> int:
    info = model.gguf
    layers = int(info.arch_value("block_count", 0) or 0)
    heads = info.arch_value("attention.head_count", 0) or 0
    kv_heads = info.arch_value("attention.head_count_kv", heads) or heads
    if isinstance(kv_heads, list):
        kv_heads = max(kv_heads) if kv_heads else 0
    if isinstance(heads, list):
        heads = max(heads) if heads else 0
    embedding = int(info.arch_value("embedding_length", 0) or 0)
    head_dim = embedding // heads if heads else 0
    key_length = int(info.arch_value("attention.key_length", head_dim) or head_dim)
    value_length = int(info.arch_value("attention.value_length", head_dim) or head_dim)
    elements = layers * int(kv_heads) * (key_length + value_length) * ctx
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

    @property
    def key(self) -> tuple:
        return (self.model.digest, self.ctx, self.speculative.draft, self.speculative.spec_type, self.cpu_only)


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
    process_memory: Callable[[], dict[int, int]] = gpu.memory_by_process

    @property
    def name(self) -> str:
        return self.plan.model.name

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def alive(self) -> bool:
        return self.process.poll() is None and not self.stopping

    def vram_mb(self) -> int:
        return self.process_memory().get(self.process.pid, 0)


def build_arguments(config: Config, plan: LaunchPlan, port: int) -> list[str]:
    model = plan.model
    arguments = [config.runner_binary, "--model", str(model.model_path), "--host", "127.0.0.1", "--port", str(port),
                 "--no-webui", "--offline", "-c", str(plan.ctx), "-np", str(config.parallel), "-a", model.name,
                 "--fit-target", str(config.fit_target_mb), "--no-warmup"]
    if plan.cpu_only:
        arguments += ["-ngl", "0", "--device", "none"]
    if model.is_embedding:
        return arguments + ["--embedding", "-b", str(plan.ctx), "-ub", str(plan.ctx)]
    arguments += ["--jinja", "--reasoning-format", "deepseek", "-fa", config.flash_attn,
                  "-ctk", config.kv_cache_type, "-ctv", config.kv_cache_type, "-b", str(config.batch), "-ub", str(config.ubatch),
                  "--cache-ram", str(config.cache_ram_mb)]
    if model.projector_path:
        arguments += ["--mmproj", str(model.projector_path)]
    if plan.template_file:
        arguments += ["--chat-template-file", str(plan.template_file)]
    arguments += speculative_arguments(config, plan)
    return arguments + [str(item) for item in config.extra_args.get(model.name, [])]


def speculative_arguments(config: Config, plan: LaunchPlan) -> list[str]:
    settings = plan.speculative
    arguments = []
    if plan.draft_path:
        arguments += ["--model-draft", str(plan.draft_path), "-ngld", "all", "--spec-draft-n-max", str(settings.draft_max),
                      "--spec-draft-n-min", str(settings.draft_min), "--spec-draft-p-min", str(settings.draft_p_min),
                      "-ctkd", config.kv_cache_type, "-ctvd", config.kv_cache_type]
    if settings.spec_type:
        arguments += ["--spec-type", settings.spec_type]
        if settings.ngram_min:
            arguments += ["--spec-ngram-mod-n-min", str(settings.ngram_min)]
        if settings.ngram_max:
            arguments += ["--spec-ngram-mod-n-max", str(settings.ngram_max)]
    return arguments


class RunnerPool:
    def __init__(self, config: Config, store: ModelStore, launcher=subprocess.Popen, memory=None, backend: Backend | None = None):
        self.backend = backend or resolve_backend(config)
        self.config = replace(config, runner_binary=self.backend.runner_binary, library_path=list(self.backend.library_path))
        self.store = store
        self.leaker_names = {parse_name(name).display for name in config.thinking_leakers}
        self.launcher = launcher
        self.memory = memory or partial(gpu.card_memory, self.backend.name)
        self.process_memory = partial(gpu.memory_by_process, self.backend.name)
        self.runners: dict[str, Runner] = {}
        self.state = threading.Condition()
        self.load_lock = threading.Lock()
        self.state_dir = Path(config.state_dir)
        self._reaper = threading.Thread(target=self._reap_forever, name="keep-alive-reaper", daemon=True)

    def start_reaper(self) -> None:
        self._reaper.start()

    def plan(self, model: Model, ctx: int | None, cpu_only: bool = False) -> LaunchPlan:
        cpu_only = cpu_only or self.backend.runs_everything_on_the_cpu
        if model.is_embedding:
            ctx = model.context_length or self.config.default_ctx
        else:
            ctx = int(ctx or model.params.get("num_ctx") or self.config.default_ctx)
            if model.context_length:
                ctx = min(ctx, model.context_length)
        speculative = SpeculativeSettings() if model.is_embedding or cpu_only else self.config.speculative_for(model.name)
        plan = LaunchPlan(model=model, ctx=ctx, speculative=speculative, cpu_only=cpu_only)
        if speculative.draft:
            plan.draft_path = self.store.get(speculative.draft).model_path
        donor = self.store.template_donor(model)
        plan.template_file = self._launch_template(model, donor)
        plan.may_leak_thinking = may_leak_thinking(model, donor or model, self.leaker_names)
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

    def estimate_mb(self, plan: LaunchPlan) -> int:
        if plan.cpu_only:
            return 0
        files = plan.model.model_path.stat().st_size
        if plan.model.projector_path:
            files += plan.model.projector_path.stat().st_size
        if plan.draft_path:
            files += plan.draft_path.stat().st_size
        kv = 0 if plan.model.is_embedding else kv_cache_mb(plan.model, plan.ctx, self.config.kv_cache_type)
        return int(files / gpu.MIB) + kv + self.config.load_overhead_mb

    def acquire(self, model: Model, ctx: int | None, cpu_only: bool = False) -> tuple[Runner, float]:
        plan = self.plan(model, ctx, cpu_only)
        runner = self._reuse(plan)
        if runner:
            return runner, 0.0
        with self.load_lock:
            runner = self._reuse(plan)
            if runner:
                return runner, 0.0
            self._retire(model.name)
            estimate = self.estimate_mb(plan)
            self._make_room(estimate, model.name)
            runner = self._launch(plan, estimate)
            with self.state:
                runner.in_flight = 1
                self.runners[model.name] = runner
            return runner, runner.load_seconds

    def _reuse(self, plan: LaunchPlan) -> Runner | None:
        with self.state:
            runner = self.runners.get(plan.model.name)
            if runner and runner.alive and runner.plan.key == plan.key:
                runner.in_flight += 1
                runner.expires_at = None
                runner.expires_wall = None
                runner.unload_requested = False
                return runner
        return None

    def release(self, runner: Runner, keep_alive: float | None) -> None:
        stop_now = False
        with self.state:
            runner.in_flight -= 1
            runner.last_used = time.monotonic()
            if runner.in_flight == 0:
                if keep_alive is FOREVER:
                    runner.expires_at = runner.expires_wall = None
                else:
                    runner.expires_at = runner.last_used + keep_alive
                    runner.expires_wall = time.time() + keep_alive
                stop_now = keep_alive == 0 or runner.unload_requested
            self.state.notify_all()
        if stop_now:
            self.unload(runner.name)

    def touch(self, model: Model, keep_alive: float | None) -> bool:
        with self.state:
            runner = self.runners.get(model.name)
            if not runner or not runner.alive:
                return False
            runner.in_flight += 1
        self.release(runner, keep_alive)
        return True

    def unload(self, name: str) -> bool:
        with self.state:
            runner = self.runners.get(name)
            if not runner:
                return False
            if runner.in_flight:
                runner.expires_at = time.monotonic()
                runner.expires_wall = time.time()
                runner.unload_requested = True
                return True
            runner.stopping = True
            del self.runners[name]
        self._stop(runner)
        return True

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
            self.runners.pop(name, None)
        self._stop(runner)

    def _make_room(self, needed_mb: int, for_name: str) -> None:
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
        arguments = build_arguments(self.config, plan, port)
        logs = self.state_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log_path = logs / f"{plan.model.name.replace('/', '_').replace(':', '_')}.log"
        environment = {**os.environ, "LD_LIBRARY_PATH": ":".join(self.config.library_path + [str(Path(self.config.runner_binary).resolve().parent)])}
        if plan.cpu_only:
            environment.update(self.backend.cpu_only_environment)
        started = time.monotonic()
        log.info("loading %s (ctx %d, ~%d MiB): %s", plan.model.name, plan.ctx, estimate, " ".join(arguments[1:]))
        with open(log_path, "ab") as output:
            process = self.launcher(arguments, stdout=output, stderr=subprocess.STDOUT, env=environment, start_new_session=True)
        runner = Runner(plan=plan, port=port, process=process, estimate_mb=estimate, process_memory=self.process_memory)
        try:
            self._wait_until_ready(runner, log_path)
        except LoadError:
            self._stop(runner)
            raise
        runner.load_seconds = time.monotonic() - started
        log.info("loaded %s in %.1fs on port %d", plan.model.name, runner.load_seconds, port)
        return runner

    def _wait_until_ready(self, runner: Runner, log_path: Path) -> None:
        deadline = time.monotonic() + self.config.load_timeout_seconds
        while time.monotonic() < deadline:
            if runner.process.poll() is not None:
                raise LoadError(f"the model runner for {runner.name} exited with {runner.process.returncode} while loading: "
                                f"{last_lines(log_path)}")
            try:
                with urllib.request.urlopen(f"{runner.base_url}/health", timeout=HEALTH_TIMEOUT_SECONDS) as response:
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
                self.runners.pop(runner.name, None)
        for runner in dead:
            log.warning("the model runner for %s died with %s", runner.name, runner.process.returncode)
        for runner in expired:
            self._stop(runner)

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
            runners = list(self.runners.values())
            self.runners.clear()
        for runner in runners:
            self._stop(runner)


def last_lines(path: Path, count: int = 8) -> str:
    try:
        return " | ".join(path.read_text(errors="replace").strip().splitlines()[-count:])
    except OSError:
        return "(no log)"
