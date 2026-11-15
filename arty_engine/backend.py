from dataclasses import dataclass
from pathlib import Path

from .config import AUTO_BACKEND, Config
from .gpu import AMD_DRIVER, driver_name

CUDA, ROCM, VULKAN, CPU = "cuda", "rocm", "vulkan", "cpu"
HIDDEN_CUDA_DEVICES = {"CUDA_VISIBLE_DEVICES": ""}
HIDDEN_ROCM_DEVICES = {"ROCR_VISIBLE_DEVICES": ""}
HIDDEN_DEVICES = {CUDA: HIDDEN_CUDA_DEVICES, ROCM: HIDDEN_ROCM_DEVICES, VULKAN: {},
                  CPU: {**HIDDEN_CUDA_DEVICES, **HIDDEN_ROCM_DEVICES}}
NVIDIA_DEVICE = "dev/nvidia0"
AMD_COMPUTE_DEVICE = "dev/kfd"
RENDER_NODES = "dev/dri"
DRM_SYSFS = "sys/class/drm"
KFD_NODES = "sys/class/kfd/kfd/topology/nodes"
KFD_TARGET_KEY = "gfx_target_version"
ROCM_BUILD_TARGETS = frozenset({
    "gfx908", "gfx90a", "gfx942", "gfx950", "gfx1010", "gfx1011", "gfx1012", "gfx1030", "gfx1031", "gfx1032", "gfx1033",
    "gfx1034", "gfx1035", "gfx1036", "gfx1100", "gfx1101", "gfx1102", "gfx1103", "gfx1150", "gfx1151", "gfx1152",
    "gfx1200", "gfx1201",
})


@dataclass(frozen=True)
class Backend:
    name: str
    reason: str
    runner_binary: str
    library_path: tuple

    @property
    def cpu_only_environment(self) -> dict[str, str]:
        return dict(HIDDEN_DEVICES[self.name])

    @property
    def runs_everything_on_the_cpu(self) -> bool:
        return self.name == CPU


def gfx_target(version: int) -> str:
    major, minor, stepping = version // 10000, version // 100 % 100, version % 100
    return f"gfx{major}{minor:x}{stepping:x}"


def kfd_target_version(properties: Path) -> int:
    try:
        lines = properties.read_text().splitlines()
    except OSError:
        return 0
    for line in lines:
        key, _, value = line.partition(" ")
        if key == KFD_TARGET_KEY:
            return int(value) if value.strip().isdigit() else 0
    return 0


@dataclass(frozen=True)
class Machine:
    root: Path = Path("/")

    def has_nvidia(self) -> bool:
        return (self.root / NVIDIA_DEVICE).exists()

    def gfx_targets(self) -> list[str]:
        try:
            nodes = sorted((self.root / KFD_NODES).glob("*/properties"))
        except OSError:
            return []
        versions = [kfd_target_version(properties) for properties in nodes]
        return [gfx_target(version) for version in versions if version]

    def has_amd_compute(self) -> bool:
        return (self.root / AMD_COMPUTE_DEVICE).exists()

    def render_nodes(self) -> list[str]:
        try:
            return sorted(node.name for node in (self.root / RENDER_NODES).glob("renderD*"))
        except OSError:
            return []

    def amd_render_nodes(self) -> list[str]:
        return [node for node in self.render_nodes() if driver_name(self.root / DRM_SYSFS / node / "device") == AMD_DRIVER]


def choose_backend(config: Config, machine: Machine) -> tuple[str, str]:
    if config.gpu_backend != AUTO_BACKEND:
        return config.gpu_backend, "set in the config"
    if machine.has_nvidia():
        return CUDA, f"auto: /{NVIDIA_DEVICE} found"
    amd_nodes = machine.amd_render_nodes()
    rocm_note = ""
    if amd_nodes and machine.has_amd_compute() and ROCM in config.runtimes:
        targets = machine.gfx_targets()
        built_for = [target for target in targets if target in ROCM_BUILD_TARGETS]
        if built_for:
            return ROCM, (f"auto: /{AMD_COMPUTE_DEVICE} and amdgpu render node {amd_nodes[0]} found, card {built_for[0]} "
                          f"is a ROCm build target, ROCm runtime configured")
        rocm_note = f"; not ROCm: the card's target ({', '.join(targets) or 'not readable'}) is not one the ROCm build carries"
    nodes = amd_nodes or machine.render_nodes()
    if nodes and VULKAN in config.runtimes:
        return VULKAN, f"auto: render node {nodes[0]} found, Vulkan runtime configured{rocm_note}"
    return CPU, f"auto: {no_gpu_reason(config, amd_nodes, nodes)}{rocm_note}"


def no_gpu_reason(config: Config, amd_nodes: list[str], nodes: list[str]) -> str:
    if amd_nodes:
        return (f"amdgpu render node {amd_nodes[0]} found, but no usable runtime for it "
                f"(ROCm needs /{AMD_COMPUTE_DEVICE} and runtimes.rocm, Vulkan needs runtimes.vulkan)")
    if nodes:
        return f"render node {nodes[0]} found, but no runtimes.vulkan in the config"
    return "no NVIDIA card and no render node found"


def resolve_backend(config: Config, machine: Machine | None = None) -> Backend:
    name, reason = choose_backend(config, machine or Machine())
    runtime = config.runtime_for(name)
    return Backend(name=name, reason=reason, runner_binary=runtime.runner_binary, library_path=tuple(runtime.library_path))
