import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

CONFIG_ENV = "ARTY_ENGINE_CONFIG"
AUTO_BACKEND = "auto"
RUNTIME_BACKENDS = ("cuda", "rocm", "vulkan", "cpu")
GPU_BACKENDS = (AUTO_BACKEND, *RUNTIME_BACKENDS)
RUNTIME_SETTINGS = ("runner_binary", "library_path")


@dataclass
class SpeculativeSettings:
    draft: str = ""
    spec_type: str = ""
    draft_max: int = 16
    draft_min: int = 0
    draft_p_min: float = 0.75
    draft_ctx: int = 0
    ngram_min: int = 0
    ngram_max: int = 0

    @property
    def enabled(self) -> bool:
        return bool(self.draft or self.spec_type)


@dataclass(frozen=True)
class Runtime:
    runner_binary: str
    library_path: list


@dataclass
class Config:
    listen: str = "127.0.0.1:11438"
    store: str = "/srv/ai/arty-engine/models"
    model_roots: list = field(default_factory=list)
    runner_binary: str = "/srv/ai/arty-engine/runtime/server"
    library_path: list = field(default_factory=lambda: ["/srv/ai/arty-engine/cuda-12.8/lib"])
    state_dir: str = "/srv/ai/arty-engine/state"
    vram_budget_mb: int = 11500
    load_overhead_mb: int = 700
    default_ctx: int = 12288
    kv_cache_type: str = "q8_0"
    flash_attn: str = "auto"
    batch: int = 1024
    ubatch: int = 512
    parallel: int = 1
    cache_ram_mb: int = 1024
    fit_target_mb: int = 256
    compact_tool_json: bool = True
    default_keep_alive: str = "5m"
    load_timeout_seconds: int = 600
    trainer_url: str = ""
    pull_rate_bytes: int = 4_000_000
    pull_keep_free_mb: int = 2048
    speculative: dict = field(default_factory=dict)
    internal_models: list = field(default_factory=list)
    thinking_leakers: list = field(default_factory=list)
    extra_args: dict = field(default_factory=dict)
    allowed_hosts: list = field(default_factory=list)
    allowed_origins: list = field(default_factory=list)
    gpu_backend: str = AUTO_BACKEND
    runtimes: dict = field(default_factory=dict)

    def __post_init__(self):
        check_gpu_settings(self.gpu_backend, self.runtimes)
        if not isinstance(self.thinking_leakers, list) or not all(isinstance(name, str) for name in self.thinking_leakers):
            raise ValueError("thinking_leakers must be a list of model names")

    @property
    def roots(self) -> list[Path]:
        return [Path(self.store)] + [Path(root) for root in self.model_roots]

    def speculative_for(self, model_name: str) -> SpeculativeSettings:
        settings = self.speculative.get(model_name)
        if not settings:
            return SpeculativeSettings()
        known = {item.name for item in fields(SpeculativeSettings)}
        return SpeculativeSettings(**{key: value for key, value in settings.items() if key in known})

    def runtime_for(self, backend: str) -> Runtime:
        listed = self.runtimes.get(backend)
        if not listed:
            return Runtime(runner_binary=self.runner_binary, library_path=list(self.library_path))
        return Runtime(runner_binary=listed["runner_binary"], library_path=list(listed.get("library_path", [])))


def check_gpu_settings(backend, runtimes) -> None:
    if backend not in GPU_BACKENDS:
        raise ValueError(f"gpu_backend must be one of {', '.join(GPU_BACKENDS)}, not {backend!r}")
    if not isinstance(runtimes, dict):
        raise ValueError("runtimes must be an object of backend name to runtime settings")
    for name, settings in runtimes.items():
        check_runtime(name, settings)


def check_runtime(name, settings) -> None:
    if name not in RUNTIME_BACKENDS:
        raise ValueError(f"runtimes: {name!r} is not a backend ({', '.join(RUNTIME_BACKENDS)})")
    if not isinstance(settings, dict):
        raise ValueError(f"runtimes.{name} must be an object with runner_binary and library_path")
    unknown = sorted(set(settings) - set(RUNTIME_SETTINGS))
    if unknown:
        raise ValueError(f"runtimes.{name}: unknown settings {', '.join(unknown)}")
    if not isinstance(settings.get("runner_binary"), str) or not settings["runner_binary"]:
        raise ValueError(f"runtimes.{name} needs runner_binary, the path of that build's model server")
    folders = settings.get("library_path", [])
    if not isinstance(folders, list) or not all(isinstance(folder, str) for folder in folders):
        raise ValueError(f"runtimes.{name}.library_path must be a list of folders")


def load_config(path: str | None = None) -> Config:
    path = path or os.environ.get(CONFIG_ENV, "")
    if not path:
        return Config()
    with open(path) as handle:
        raw = json.load(handle)
    known = {item.name for item in fields(Config)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown settings {', '.join(unknown)}")
    try:
        return Config(**raw)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
