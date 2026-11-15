import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .config import AUTO_BACKEND, Config
from .gpu import AMD_DRIVER, driver_name

CUDA, ROCM, VULKAN, CPU = "cuda", "rocm", "vulkan", "cpu"
HIDDEN_DEVICES = {CUDA: {"CUDA_VISIBLE_DEVICES": ""}, ROCM: {"HIP_VISIBLE_DEVICES": ""}, VULKAN: {}, CPU: {}}
NVIDIA_TOOL = "nvidia-smi"
NVIDIA_DEVICE = "dev/nvidia0"
AMD_COMPUTE_DEVICE = "dev/kfd"
RENDER_NODES = "dev/dri"
DRM_SYSFS = "sys/class/drm"


@dataclass(frozen=True)
class Backend:
    name: str
    reason: str
    runner_binary: str
    library_path: tuple

    @property
    def cpu_only_environment(self) -> dict[str, str]:
        return dict(HIDDEN_DEVICES[self.name])


@dataclass(frozen=True)
class Machine:
    root: Path = Path("/")
    which: Callable[[str], str | None] = shutil.which

    def has_nvidia(self) -> bool:
        return bool(self.which(NVIDIA_TOOL)) or (self.root / NVIDIA_DEVICE).exists()

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
        return CUDA, f"auto: {NVIDIA_TOOL} or /{NVIDIA_DEVICE} found"
    amd_nodes = machine.amd_render_nodes()
    if amd_nodes and machine.has_amd_compute() and ROCM in config.runtimes:
        return ROCM, f"auto: /{AMD_COMPUTE_DEVICE} and amdgpu render node {amd_nodes[0]} found, ROCm runtime configured"
    nodes = amd_nodes or machine.render_nodes()
    if nodes and VULKAN in config.runtimes:
        return VULKAN, f"auto: render node {nodes[0]} found, Vulkan runtime configured"
    return CPU, f"auto: {no_gpu_reason(config, amd_nodes, nodes)}"


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
