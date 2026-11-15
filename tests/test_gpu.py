import json
import subprocess

import pytest

from arty_engine import gpu
from arty_engine.gpu import GpuMemory

NVIDIA_QUERY = ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"]
GIB = 1024 * gpu.MIB


class FakeTools:
    def __init__(self, outputs: dict):
        self.outputs = outputs
        self.calls = []

    def __call__(self, command, **settings):
        self.calls.append((command, settings))
        answer = self.outputs.get(command[0], FileNotFoundError(command[0]))
        if isinstance(answer, BaseException):
            raise answer
        return subprocess.CompletedProcess(command, 0, stdout=answer, stderr="")


@pytest.fixture
def tools(monkeypatch):
    def install(outputs: dict) -> FakeTools:
        fake = FakeTools(outputs)
        monkeypatch.setattr(gpu.subprocess, "run", fake)
        return fake
    return install


def amd_card(drm, name, total_bytes, used_bytes, driver="amdgpu"):
    driver_folder = drm.parent / "drivers" / driver
    driver_folder.mkdir(parents=True, exist_ok=True)
    device = drm / name / "device"
    device.mkdir(parents=True)
    (device / "driver").symlink_to(driver_folder)
    (device / "mem_info_vram_total").write_text(f"{total_bytes}\n")
    (device / "mem_info_vram_used").write_text(f"{used_bytes}\n")


def test_the_nvidia_reading_runs_the_same_query_as_before(tools):
    fake = tools({"nvidia-smi": "12288, 700\n"})
    assert gpu.card_memory() == GpuMemory(total_mb=12288, used_mb=700, source="nvidia-smi")
    assert gpu.card_memory("cuda").free_mb == 11588
    command, settings = fake.calls[0]
    assert command == NVIDIA_QUERY
    assert settings["timeout"] == gpu.SMI_TIMEOUT_SECONDS == 10 and settings["check"] is True


def test_no_nvidia_tool_reads_as_unknown(tools):
    tools({})
    assert gpu.card_memory("cuda") is None and gpu.memory_by_process("cuda") == {}


def test_amd_smi_json_gives_the_card_memory(tools):
    usage = {"total_vram": {"value": 16368, "unit": "MB"}, "used_vram": {"value": 1024, "unit": "MB"}}
    tools({"amd-smi": json.dumps([{"gpu": 0, "mem_usage": usage}])})
    assert gpu.card_memory("rocm") == GpuMemory(total_mb=16368, used_mb=1024, source="amd-smi")


def test_amd_smi_wrapped_output_picks_the_card_with_the_most_memory(tools):
    small = {"total_vram": {"value": 512, "unit": "MB"}, "used_vram": {"value": 100, "unit": "MB"}}
    large = {"total_vram": {"value": 8192, "unit": "MB"}, "used_vram": {"value": 300, "unit": "MB"}}
    tools({"amd-smi": json.dumps({"gpu_data": [{"gpu": 0, "mem_usage": small}, {"gpu": 1, "mem_usage": large}]})})
    assert gpu.card_memory("vulkan") == GpuMemory(total_mb=8192, used_mb=300, source="amd-smi")


def test_rocm_smi_is_read_when_amd_smi_is_missing(tools):
    cards = {"card0": {"VRAM Total Memory (B)": str(16 * GIB), "VRAM Total Used Memory (B)": str(2 * GIB)}}
    fake = tools({"rocm-smi": json.dumps(cards)})
    assert gpu.card_memory("rocm") == GpuMemory(total_mb=16384, used_mb=2048, source="rocm-smi")
    assert [command for command, _ in fake.calls] == [gpu.AMD_SMI_MEMORY, gpu.ROCM_SMI_MEMORY]
    assert all(settings["timeout"] == gpu.SMI_TIMEOUT_SECONDS for _, settings in fake.calls)


def test_sysfs_is_read_when_neither_tool_answers(tools, tmp_path, monkeypatch):
    drm = tmp_path / "class" / "drm"
    amd_card(drm, "card1", 8 * GIB, GIB)
    amd_card(drm, "card0", 0, 0, driver="i915")
    monkeypatch.setattr(gpu, "DRM_FOLDER", drm)
    tools({"amd-smi": subprocess.TimeoutExpired("amd-smi", 10), "rocm-smi": subprocess.CalledProcessError(2, "rocm-smi")})
    assert gpu.card_memory("rocm") == GpuMemory(total_mb=8192, used_mb=1024, source="sysfs card1")


def test_unreadable_tool_output_falls_through_to_sysfs(tools, tmp_path, monkeypatch):
    drm = tmp_path / "class" / "drm"
    amd_card(drm, "card0", 4 * GIB, 0)
    monkeypatch.setattr(gpu, "DRM_FOLDER", drm)
    tools({"amd-smi": "not json", "rocm-smi": json.dumps({"card0": {"VRAM Total Memory (B)": "N/A"}})})
    assert gpu.card_memory("vulkan") == GpuMemory(total_mb=4096, used_mb=0, source="sysfs card0")


def test_no_amd_card_reads_as_unknown(tools, tmp_path, monkeypatch):
    drm = tmp_path / "class" / "drm"
    amd_card(drm, "card0", GIB, 0, driver="i915")
    monkeypatch.setattr(gpu, "DRM_FOLDER", drm)
    tools({})
    assert gpu.card_memory("rocm") is None


def test_the_cpu_backend_and_amd_processes_ask_no_tool(tools):
    fake = tools({"nvidia-smi": "1, 1\n"})
    assert gpu.card_memory("cpu") is None
    assert gpu.memory_by_process("rocm") == {} and gpu.memory_by_process("vulkan") == {}
    assert fake.calls == []


def test_any_card_memory_tries_nvidia_then_amd(tools):
    usage = {"total_vram": "2048 MB", "used_vram": "48 MB"}
    tools({"amd-smi": json.dumps([{"mem_usage": usage}])})
    assert gpu.any_card_memory() == GpuMemory(total_mb=2048, used_mb=48, source="amd-smi")
