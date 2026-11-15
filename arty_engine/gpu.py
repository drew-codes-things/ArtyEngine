import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

SMI_TIMEOUT_SECONDS = 10
MIB = 1024 * 1024
DRM_FOLDER = Path("/sys/class/drm")
AMD_DRIVER = "amdgpu"
AMD_SMI_MEMORY = ["amd-smi", "metric", "--mem-usage", "--json"]
ROCM_SMI_MEMORY = ["rocm-smi", "--showmeminfo", "vram", "--json"]
ROCM_SMI_TOTAL = "VRAM Total Memory (B)"
ROCM_SMI_USED = "VRAM Total Used Memory (B)"
AMD_SMI_UNITS = ("MB", "MiB")
NVIDIA_BACKENDS = ("cuda",)
AMD_BACKENDS = ("rocm", "vulkan")


@dataclass
class GpuMemory:
    total_mb: int
    used_mb: int
    source: str = ""

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb


def _output(command: list[str]) -> str | None:
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=SMI_TIMEOUT_SECONDS, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None


def _query(arguments: list[str]) -> list[list[str]]:
    output = _output(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"])
    if output is None:
        return []
    return [[cell.strip() for cell in line.split(",")] for line in output.splitlines() if line.strip()]


def nvidia_card_memory() -> GpuMemory | None:
    rows = _query(["--query-gpu=memory.total,memory.used"])
    if not rows:
        return None
    try:
        return GpuMemory(total_mb=int(rows[0][0]), used_mb=int(rows[0][1]), source="nvidia-smi")
    except (ValueError, IndexError):
        return None


def _json_output(command: list[str]):
    output = _output(command)
    if output is None:
        return None
    try:
        return json.loads(output)
    except ValueError:
        return None


def _memory_usages(data) -> list[dict]:
    if isinstance(data, list):
        return [usage for item in data for usage in _memory_usages(item)]
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("mem_usage"), dict):
        return [data["mem_usage"]]
    return [usage for value in data.values() for usage in _memory_usages(value)]


def _mebibytes(value) -> int | None:
    if isinstance(value, dict):
        if value.get("unit") not in AMD_SMI_UNITS:
            return None
        value = value.get("value")
    if isinstance(value, str):
        number, _, unit = value.strip().partition(" ")
        if unit and unit not in AMD_SMI_UNITS:
            return None
        value = number
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _largest(cards: list[GpuMemory]) -> GpuMemory | None:
    return max(cards, key=lambda card: card.total_mb, default=None)


def amd_smi_memory() -> GpuMemory | None:
    cards = []
    for usage in _memory_usages(_json_output(AMD_SMI_MEMORY)):
        total, used = _mebibytes(usage.get("total_vram")), _mebibytes(usage.get("used_vram"))
        if total and used is not None:
            cards.append(GpuMemory(total_mb=total, used_mb=used, source="amd-smi"))
    return _largest(cards)


def _bytes_in_mib(value) -> int | None:
    try:
        return int(str(value).strip()) // MIB
    except ValueError:
        return None


def rocm_smi_memory() -> GpuMemory | None:
    data = _json_output(ROCM_SMI_MEMORY)
    if not isinstance(data, dict):
        return None
    cards = []
    for card in data.values():
        if not isinstance(card, dict) or ROCM_SMI_TOTAL not in card or ROCM_SMI_USED not in card:
            continue
        total, used = _bytes_in_mib(card[ROCM_SMI_TOTAL]), _bytes_in_mib(card[ROCM_SMI_USED])
        if total and used is not None:
            cards.append(GpuMemory(total_mb=total, used_mb=used, source="rocm-smi"))
    return _largest(cards)


def driver_name(device: Path) -> str:
    try:
        return (device / "driver").resolve(strict=True).name
    except OSError:
        return ""


def amd_cards(drm: Path | None = None) -> list[Path]:
    folder = drm or DRM_FOLDER
    try:
        cards = sorted(path for path in folder.glob("card*") if path.name[len("card"):].isdigit())
    except OSError:
        return []
    return [card for card in cards if driver_name(card / "device") == AMD_DRIVER]


def _sysfs_number(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def sysfs_memory(drm: Path | None = None) -> GpuMemory | None:
    cards = []
    for card in amd_cards(drm):
        total = _sysfs_number(card / "device" / "mem_info_vram_total")
        used = _sysfs_number(card / "device" / "mem_info_vram_used")
        if total and used is not None:
            cards.append(GpuMemory(total_mb=total // MIB, used_mb=used // MIB, source=f"sysfs {card.name}"))
    return _largest(cards)


def amd_card_memory(drm: Path | None = None) -> GpuMemory | None:
    return amd_smi_memory() or rocm_smi_memory() or sysfs_memory(drm)


def card_memory(backend: str = "cuda") -> GpuMemory | None:
    if backend in NVIDIA_BACKENDS:
        return nvidia_card_memory()
    if backend in AMD_BACKENDS:
        return amd_card_memory()
    return None


def any_card_memory() -> GpuMemory | None:
    return nvidia_card_memory() or amd_card_memory()


def memory_by_process(backend: str = "cuda") -> dict[int, int]:
    if backend not in NVIDIA_BACKENDS:
        return {}
    usage = {}
    for row in _query(["--query-compute-apps=pid,used_memory"]):
        try:
            usage[int(row[0])] = int(row[1])
        except (ValueError, IndexError):
            continue
    return usage
