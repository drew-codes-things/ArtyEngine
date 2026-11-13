import subprocess
from dataclasses import dataclass

NVIDIA_SMI_TIMEOUT_SECONDS = 10
MIB = 1024 * 1024


@dataclass
class GpuMemory:
    total_mb: int
    used_mb: int

    @property
    def free_mb(self) -> int:
        return self.total_mb - self.used_mb


def _query(arguments: list[str]) -> list[list[str]]:
    try:
        output = subprocess.run(["nvidia-smi", *arguments, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                timeout=NVIDIA_SMI_TIMEOUT_SECONDS, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [[cell.strip() for cell in line.split(",")] for line in output.splitlines() if line.strip()]


def card_memory() -> GpuMemory | None:
    rows = _query(["--query-gpu=memory.total,memory.used"])
    if not rows:
        return None
    try:
        return GpuMemory(total_mb=int(rows[0][0]), used_mb=int(rows[0][1]))
    except (ValueError, IndexError):
        return None


def memory_by_process() -> dict[int, int]:
    usage = {}
    for row in _query(["--query-compute-apps=pid,used_memory"]):
        try:
            usage[int(row[0])] = int(row[1])
        except (ValueError, IndexError):
            continue
    return usage
