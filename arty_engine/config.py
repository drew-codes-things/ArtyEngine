import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path

CONFIG_ENV = "ARTY_ENGINE_CONFIG"


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


@dataclass
class Config:
    listen: str = "127.0.0.1:11438"
    store: str = "/srv/ai/arty-engine/models"
    model_roots: list = field(default_factory=lambda: ["/srv/ai/ollama"])
    llama_server: str = "/srv/ai/arty-engine/llama.cpp/current/llama-server"
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
    speculative: dict = field(default_factory=dict)
    internal_models: list = field(default_factory=list)
    extra_args: dict = field(default_factory=dict)
    allowed_hosts: list = field(default_factory=list)
    allowed_origins: list = field(default_factory=list)

    @property
    def roots(self) -> list[Path]:
        return [Path(self.store)] + [Path(root) for root in self.model_roots]

    def speculative_for(self, model_name: str) -> SpeculativeSettings:
        settings = self.speculative.get(model_name)
        if not settings:
            return SpeculativeSettings()
        known = {item.name for item in fields(SpeculativeSettings)}
        return SpeculativeSettings(**{key: value for key, value in settings.items() if key in known})


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
    return Config(**raw)
