import math

from .store import ModelNotFound, parse_name

RPC_AUTO, RPC_ALWAYS, RPC_NEVER = "auto", "always", "never"
RPC_MODES = (RPC_AUTO, RPC_ALWAYS, RPC_NEVER)
LOAD_LOCALLY, FAIL = "local", "fail"
WHEN_WORKERS_DOWN = (LOAD_LOCALLY, FAIL)
SPLIT_MODES = ("none", "layer", "row", "tensor")
SWITCH_WORDS = ("on", "off", "auto")
LOAD_MODES = ("auto", "none", "mmap", "mlock", "mmap+mlock", "dio")
ROPE_SCALING = ("none", "linear", "yarn")
GPU_LAYER_WORDS = ("auto", "all")
KV_CACHE_TYPES = ("f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1")
NUMA_MODES = ("distribute", "isolate", "numactl")
WITH_WORKERS = "with_workers"
LIST_SEPARATOR = ","


def whole(lowest: int = 0):
    def check(value, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < lowest:
            raise ValueError(f"{name} must be a whole number of at least {lowest}")
        return value
    return check


def positive_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a number above 0")
    return value


def one_of(choices: tuple):
    def check(value, name: str) -> str:
        if value not in choices:
            raise ValueError(f"{name} must be one of {', '.join(choices)}, not {value!r}")
        return value
    return check


def switch(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def plain_texts(value, name: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if any(LIST_SEPARATOR in item or item != item.strip() for item in value):
        raise ValueError(f"{name}: an entry can't hold a comma or start or end with a space")
    return value


def split_shares(value, name: str) -> list:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a list of numbers, one share per device")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
            raise ValueError(f"{name} must hold numbers of 0 or more")
    return value


def tensor_overrides(value, name: str) -> list[str]:
    plain_texts(value, name)
    for item in value:
        pattern, separator, place = item.partition("=")
        if not separator or not pattern or not place or "=" in place:
            raise ValueError(f"{name}: {item!r} must be <tensor name pattern>=<buffer type>, such as blk\\.1[0-9]\\.ffn_.*_exps\\.=CPU")
    return value


def gpu_layers(value, name: str):
    if isinstance(value, str):
        return one_of(GPU_LAYER_WORDS)(value, name)
    return whole(0)(value, name)


def adapters(value, name: str) -> list[dict]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a list of {{\"path\": ..., \"scale\": ...}} objects")
    for adapter in value:
        if not isinstance(adapter, dict) or set(adapter) - {"path", "scale"}:
            raise ValueError(f"{name}: each adapter is an object with path and an optional scale")
        plain_texts([adapter.get("path")], f"{name} path")
        if "scale" in adapter and (isinstance(adapter["scale"], bool) or not isinstance(adapter["scale"], (int, float))
                                   or not math.isfinite(adapter["scale"])):
            raise ValueError(f"{name}: scale must be a number")
    return value


LAYOUT_CHECKS = {
    "gpu_layers": gpu_layers, "devices": plain_texts, "tensor_split": split_shares, "split_mode": one_of(SPLIT_MODES),
    "main_gpu": whole(0), "tensor_overrides": tensor_overrides, "cpu_moe": switch, "cpu_moe_layers": whole(0),
    "fit": switch,
}
RUN_CHECKS = {
    "parallel": whole(1), "kv_cache_type": one_of(KV_CACHE_TYPES), "flash_attn": one_of(SWITCH_WORDS),
    "batch": whole(1), "ubatch": whole(1), "threads": whole(1), "cache_reuse": whole(0), "load_mode": one_of(LOAD_MODES),
    "kv_offload": switch, "numa": one_of(NUMA_MODES), "rope_scaling": one_of(ROPE_SCALING), "rope_scale": positive_number,
    "rope_freq_base": positive_number, "rope_freq_scale": positive_number, "yarn_orig_ctx": whole(0), "lora": adapters,
}
RPC_CHECKS = {"rpc": one_of(RPC_MODES), "rpc_workers": plain_texts, "rpc_when_down": one_of(WHEN_WORKERS_DOWN)}


def check_layout(name: str, layout) -> None:
    if not isinstance(layout, dict):
        raise ValueError(f"{name} must be an object")
    for key, value in layout.items():
        if key not in LAYOUT_CHECKS:
            raise ValueError(f"{name}: unknown setting {key} (layout settings: {', '.join(LAYOUT_CHECKS)})")
        LAYOUT_CHECKS[key](value, f"{name}.{key}")


def check_one_model(model: str, settings, worker_names: set[str]) -> None:
    prefix = f"model_settings.{model}"
    if not isinstance(settings, dict):
        raise ValueError(f"{prefix} must be an object")
    for key, value in settings.items():
        if key == WITH_WORKERS:
            check_layout(f"{prefix}.{WITH_WORKERS}", value)
            continue
        check = LAYOUT_CHECKS.get(key) or RUN_CHECKS.get(key) or RPC_CHECKS.get(key)
        if check is None:
            raise ValueError(f"{prefix}: unknown setting {key}")
        check(value, f"{prefix}.{key}")
    unknown_workers = sorted(set(settings.get("rpc_workers", [])) - worker_names)
    if unknown_workers:
        raise ValueError(f"{prefix}.rpc_workers names {', '.join(unknown_workers)}, which rpc_workers does not list")
    if settings.get("rpc") == RPC_ALWAYS and not worker_names:
        raise ValueError(f"{prefix}.rpc is always, but rpc_workers lists no workers")


def check_model_settings(value, worker_names: set[str]) -> None:
    if not isinstance(value, dict):
        raise ValueError("model_settings must be an object of model name to settings")
    seen: dict[str, str] = {}
    for model, settings in value.items():
        try:
            display = parse_name(model).display
        except ModelNotFound as error:
            raise ValueError(f"model_settings: {error}") from error
        if display in seen:
            raise ValueError(f"model_settings names {seen[display]} and {model}, which are the same model ({display})")
        seen[display] = model
        check_one_model(model, settings, worker_names)


def number_text(value) -> str:
    return str(int(value)) if float(value).is_integer() else repr(float(value))


def layout_arguments(layout: dict) -> list[str]:
    arguments = []
    if "gpu_layers" in layout:
        arguments += ["-ngl", str(layout["gpu_layers"])]
    if "devices" in layout:
        arguments += ["-dev", LIST_SEPARATOR.join(layout["devices"])]
    if "split_mode" in layout:
        arguments += ["-sm", layout["split_mode"]]
    if "tensor_split" in layout:
        arguments += ["-ts", LIST_SEPARATOR.join(number_text(share) for share in layout["tensor_split"])]
    if "main_gpu" in layout:
        arguments += ["-mg", str(layout["main_gpu"])]
    if "tensor_overrides" in layout:
        arguments += ["-ot", LIST_SEPARATOR.join(layout["tensor_overrides"])]
    if layout.get("cpu_moe"):
        arguments += ["-cmoe"]
    if "cpu_moe_layers" in layout:
        arguments += ["-ncmoe", str(layout["cpu_moe_layers"])]
    if "fit" in layout:
        arguments += ["--fit", "on" if layout["fit"] else "off"]
    return arguments


def run_arguments(settings: dict) -> list[str]:
    arguments = []
    if "threads" in settings:
        arguments += ["-t", str(settings["threads"])]
    if "load_mode" in settings:
        arguments += ["-lm", settings["load_mode"]]
    if settings.get("kv_offload") is False:
        arguments += ["-nkvo"]
    if "numa" in settings:
        arguments += ["--numa", settings["numa"]]
    if "cache_reuse" in settings:
        arguments += ["--cache-reuse", str(settings["cache_reuse"])]
    if "rope_scaling" in settings:
        arguments += ["--rope-scaling", settings["rope_scaling"]]
    for key, flag in (("rope_scale", "--rope-scale"), ("rope_freq_base", "--rope-freq-base"),
                      ("rope_freq_scale", "--rope-freq-scale")):
        if key in settings:
            arguments += [flag, number_text(settings[key])]
    if "yarn_orig_ctx" in settings:
        arguments += ["--yarn-orig-ctx", str(settings["yarn_orig_ctx"])]
    if "lora" in settings:
        arguments += ["--lora-scaled", LIST_SEPARATOR.join(f"{adapter['path']}:{number_text(adapter.get('scale', 1))}"
                                                           for adapter in settings["lora"])]
    return arguments


def layout_for(settings: dict, with_workers: bool) -> dict:
    layout = {key: value for key, value in settings.items() if key in LAYOUT_CHECKS}
    if with_workers:
        layout.update(settings.get(WITH_WORKERS) or {})
    return layout


def run_settings(settings: dict) -> dict:
    return {key: value for key, value in settings.items() if key in RUN_CHECKS}
