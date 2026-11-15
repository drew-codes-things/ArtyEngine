import json
import os
from pathlib import Path

import pytest

from arty_engine.backend import ROCM_BUILD_TARGETS, Machine, gfx_target, resolve_backend
from arty_engine.config import Config, load_config

REPO = Path(__file__).resolve().parents[1]
SHIPPED_CONFIGS = [REPO / "deploy" / "engine.json", REPO / "deploy" / "engine-test.json"]
SHIPPED_RUNNER = "/srv/ai/arty-engine/runtime/server"
SHIPPED_LIBRARIES = ("/srv/ai/arty-engine/cuda-12.8/lib",)
ROCM_RUNTIME = {"runner_binary": "/opt/rocm-build/server", "library_path": ["/opt/rocm-libs/lib"]}
VULKAN_RUNTIME = {"runner_binary": "/opt/vulkan-build/server"}
RX_7900_XTX = 110000
RX_570 = 80003
CPU_NODE = 0
README = REPO / "README.md"


class FakeMachine:
    def __init__(self, root: Path):
        self.root = root

    def machine(self) -> Machine:
        return Machine(root=self.root)

    def kfd_node(self, number: int, version: int) -> None:
        node = self.root / "sys" / "class" / "kfd" / "kfd" / "topology" / "nodes" / str(number)
        node.mkdir(parents=True)
        (node / "properties").write_text(f"cpu_cores_count 0\ngfx_target_version {version}\nsimd_count 192\n")

    def amd_card(self, version: int = RX_7900_XTX) -> None:
        self.render_node("renderD128", "amdgpu")
        self.device("dev/kfd")
        self.kfd_node(0, CPU_NODE)
        self.kfd_node(1, version)

    def device(self, relative: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def render_node(self, name: str, driver: str) -> None:
        self.device(f"dev/dri/{name}")
        driver_folder = self.root / "sys" / "bus" / "pci" / "drivers" / driver
        driver_folder.mkdir(parents=True, exist_ok=True)
        device = self.root / "sys" / "class" / "drm" / name / "device"
        device.mkdir(parents=True)
        (device / "driver").symlink_to(driver_folder)


@pytest.fixture
def host(tmp_path):
    return FakeMachine(tmp_path)


def resolved(host, **settings):
    return resolve_backend(Config(**settings), host.machine())


def test_an_nvidia_device_node_means_cuda(host):
    host.device("dev/nvidia0")
    host.render_node("renderD128", "amdgpu")
    host.device("dev/kfd")
    backend = resolved(host, runtimes={"rocm": ROCM_RUNTIME})
    assert backend.name == "cuda" and "nvidia" in backend.reason


def test_a_leftover_nvidia_tool_without_its_device_does_not_mean_cuda(host, tmp_path, monkeypatch):
    tools = tmp_path / "tools"
    tools.mkdir()
    (tools / "nvidia-smi").write_text("#!/bin/sh\nexit 9\n")
    (tools / "nvidia-smi").chmod(0o755)
    monkeypatch.setenv("PATH", f"{tools}:{os.environ['PATH']}")
    host.amd_card()
    assert resolved(host, runtimes={"rocm": ROCM_RUNTIME}).name == "rocm"
    assert resolved(host).name == "cpu"


def test_an_amd_card_with_kfd_and_a_rocm_runtime_means_rocm(host):
    host.amd_card()
    backend = resolved(host, runtimes={"rocm": ROCM_RUNTIME, "vulkan": VULKAN_RUNTIME})
    assert backend.name == "rocm" and "renderD128" in backend.reason and "gfx1100" in backend.reason
    assert backend.runner_binary == "/opt/rocm-build/server" and backend.library_path == ("/opt/rocm-libs/lib",)
    assert backend.cpu_only_environment == {"ROCR_VISIBLE_DEVICES": ""}


def test_a_card_the_rocm_build_carries_no_code_for_uses_vulkan(host):
    host.amd_card(RX_570)
    backend = resolved(host, runtimes={"rocm": ROCM_RUNTIME, "vulkan": VULKAN_RUNTIME})
    assert backend.name == "vulkan" and "gfx803" in backend.reason and "not one the ROCm build carries" in backend.reason


def test_a_card_target_that_cannot_be_read_never_means_rocm(host):
    host.render_node("renderD128", "amdgpu")
    host.device("dev/kfd")
    backend = resolved(host, runtimes={"rocm": ROCM_RUNTIME, "vulkan": VULKAN_RUNTIME})
    assert backend.name == "vulkan" and "not readable" in backend.reason
    assert resolved(host, runtimes={"rocm": ROCM_RUNTIME}).name == "cpu"


@pytest.mark.parametrize("version, target", [(110000, "gfx1100"), (90010, "gfx90a"), (100300, "gfx1030"),
                                             (80003, "gfx803"), (120001, "gfx1201"), (90402, "gfx942")])
def test_kfd_target_versions_become_gfx_names(version, target):
    assert gfx_target(version) == target


def test_the_rocm_build_targets_match_the_readme():
    listed = " ".join(README.read_text().split("The build carries GPU code for ", 1)[1].split(".", 1)[0].split())
    named = set()
    for part in listed.replace(" and ", ", ").split(", "):
        first, _, last = part.partition(" to ")
        named.add(first)
        if last:
            prefix = first[:-1]
            named |= {f"{prefix}{digit}" for digit in range(int(first[-1]), int(last[-1]) + 1)}
    assert named == ROCM_BUILD_TARGETS


def test_an_amd_card_without_kfd_falls_back_to_vulkan(host):
    host.render_node("renderD129", "amdgpu")
    backend = resolved(host, runtimes={"rocm": ROCM_RUNTIME, "vulkan": VULKAN_RUNTIME})
    assert backend.name == "vulkan" and "renderD129" in backend.reason
    assert backend.runner_binary == "/opt/vulkan-build/server" and backend.library_path == ()
    assert backend.cpu_only_environment == {}


def test_an_amd_card_with_kfd_but_no_rocm_runtime_uses_vulkan(host):
    host.amd_card()
    assert resolved(host, runtimes={"vulkan": VULKAN_RUNTIME}).name == "vulkan"


def test_an_amd_render_node_is_preferred_over_another_one_for_vulkan(host):
    host.render_node("renderD128", "i915")
    host.render_node("renderD129", "amdgpu")
    assert "renderD129" in resolved(host, runtimes={"vulkan": VULKAN_RUNTIME}).reason


def test_another_vendors_render_node_uses_vulkan_when_configured(host):
    host.render_node("renderD128", "i915")
    assert resolved(host, runtimes={"vulkan": VULKAN_RUNTIME}).name == "vulkan"
    assert resolved(host, runtimes={"rocm": ROCM_RUNTIME}).name == "cpu"


def test_an_amd_card_with_no_runtime_configured_runs_on_the_cpu_and_says_why(host):
    host.render_node("renderD128", "amdgpu")
    host.device("dev/kfd")
    backend = resolved(host)
    assert backend.name == "cpu" and "amdgpu render node renderD128" in backend.reason
    assert backend.runner_binary == Config().runner_binary


def test_no_gpu_at_all_runs_on_the_cpu(host):
    backend = resolved(host, runtimes={"rocm": ROCM_RUNTIME, "vulkan": VULKAN_RUNTIME})
    assert backend.name == "cpu" and "no NVIDIA card and no render node" in backend.reason
    assert backend.cpu_only_environment == {"CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}
    assert backend.runs_everything_on_the_cpu


def test_a_backend_set_in_the_config_is_used_whatever_is_fitted(host):
    host.device("dev/nvidia0")
    backend = resolved(host, gpu_backend="vulkan", runtimes={"vulkan": VULKAN_RUNTIME})
    assert (backend.name, backend.reason, backend.runner_binary) == ("vulkan", "set in the config", "/opt/vulkan-build/server")


def test_a_backend_without_its_own_runtime_uses_the_top_level_runtime(host):
    backend = resolved(host, gpu_backend="rocm", runner_binary="/opt/one/server", library_path=["/opt/one/lib"])
    assert (backend.runner_binary, backend.library_path) == ("/opt/one/server", ("/opt/one/lib",))


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
@pytest.mark.parametrize("nvidia", [True, False], ids=["nvidia fitted", "nvidia missing at start"])
def test_the_shipped_configs_resolve_exactly_as_before(host, path, nvidia):
    if nvidia:
        host.device("dev/nvidia0")
    host.render_node("renderD128", "amdgpu")
    backend = resolve_backend(load_config(str(path)), host.machine())
    assert (backend.name, backend.runner_binary, backend.library_path) == ("cuda", SHIPPED_RUNNER, SHIPPED_LIBRARIES)
    assert backend.cpu_only_environment == {"CUDA_VISIBLE_DEVICES": ""}


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda path: path.name)
def test_an_installed_config_from_before_resolves_as_before_on_the_nvidia_server(host, tmp_path, path):
    older = json.loads(path.read_text())
    older.pop("gpu_backend", None)
    file = tmp_path / "older.json"
    file.write_text(json.dumps(older))
    host.device("dev/nvidia0")
    backend = resolve_backend(load_config(str(file)), host.machine())
    assert (backend.name, backend.runner_binary, backend.library_path) == ("cuda", SHIPPED_RUNNER, SHIPPED_LIBRARIES)
    assert backend.cpu_only_environment == {"CUDA_VISIBLE_DEVICES": ""}


@pytest.mark.parametrize("settings, message", [
    ({"gpu_backend": "metal"}, "gpu_backend must be one of"),
    ({"runtimes": {"opencl": {"runner_binary": "/x"}}}, "'opencl' is not a backend"),
    ({"runtimes": {"rocm": {"library_path": []}}}, "runtimes.rocm needs runner_binary"),
    ({"runtimes": {"rocm": {"runner_binary": "/x", "binary": "/y"}}}, "runtimes.rocm: unknown settings binary"),
    ({"runtimes": {"vulkan": {"runner_binary": "/x", "library_path": "/lib"}}}, "library_path must be a list"),
    ({"runtimes": ["rocm"]}, "runtimes must be an object"),
])
def test_bad_gpu_settings_stop_the_engine_with_the_file_named(tmp_path, settings, message):
    file = tmp_path / "engine.json"
    file.write_text(json.dumps(settings))
    with pytest.raises(ValueError) as refused:
        load_config(str(file))
    assert message in str(refused.value) and str(file) in str(refused.value)
