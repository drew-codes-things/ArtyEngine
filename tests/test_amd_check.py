import base64
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "amd-check.sh"
KIT_FILES = [SCRIPT, REPO / "tools" / "amd_check_steps.py", REPO / "docs" / "amd-check.md"]
SCRIPT_TIMEOUT_SECONDS = 120
EXIT_FAILED, EXIT_NO_AMD = 1, 2
AMD_PCI = "03:00.0 VGA compatible controller [0300]: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [1002:744c]"
INTEL_PCI = "00:02.0 VGA compatible controller [0300]: Intel Corporation TigerLake-LP GT2 [8086:9a49]"
ROCM_DEVICES = "Available devices:\n  ROCm0: Fake Radeon (16368 MiB, 16000 MiB free)"
VULKAN_DEVICES = "Available devices:\n  Vulkan0: Fake Radeon (RADV) (8192 MiB, 8000 MiB free)"
FAKE_CURL = """#!/bin/bash
out=""
while [ $# -gt 1 ]; do
  [ "$1" = -o ] && { out=$2; shift; }
  shift
done
echo "$1" >> "$FAKE_ROOT/curl-calls"
[ -z "${FAKE_CURL_SLOW:-}" ] || sleep "$FAKE_CURL_SLOW"
case "$1" in
  *api.github.com*) [ -f "$FAKE_ROOT/release.json" ] || exit 22; cp "$FAKE_ROOT/release.json" "$out" ;;
  *) cp "$FAKE_ROOT/runtime.tar.gz" "$out" ;;
esac
"""
FAKE_LDD = """#!/bin/bash
if [ -n "${FAKE_LDD_MISSING:-}" ]; then
  [[ ":${LD_LIBRARY_PATH:-}:" == *"/_rocm_sdk_core/lib:"* ]] || printf '\\tlibamdhip64.so.7 => not found\\n'
  [[ ":${LD_LIBRARY_PATH:-}:" == *"/_rocm_sdk_libraries/lib:"* ]] || printf '\\tlibhipblas.so.3 => not found\\n\\tlibrocblas.so.5 => not found\\n'
fi
printf '\\tlibc.so.6 => /lib/libc.so.6\\n'
"""
FAKE_PYTHON = f"""#!/bin/bash
if [ "${{1:-}}" = -m ] && [ "${{2:-}}" = venv ]; then
  [ -z "${{FAKE_NO_VENV:-}}" ] || exit 1
  mkdir -p "$3/bin" "$3/lib/python3.13/site-packages"
  cp "$FAKE_ROOT/fake-pip" "$3/bin/pip"
  exit 0
fi
exec {sys.executable} "$@"
"""
FAKE_PIP = """#!/bin/bash
echo "$* PIP_CACHE_DIR=${PIP_CACHE_DIR:-}" >> "$FAKE_ROOT/pip-calls"
[ -z "${FAKE_PIP_FAILS:-}" ] || exit 1
packages="$(dirname "$0")/../lib/python3.13/site-packages"
mkdir -p "$packages/_rocm_sdk_core/lib" "$packages/_rocm_sdk_libraries/lib"
touch "$packages/_rocm_sdk_core/lib/libamdhip64.so.7" "$packages/_rocm_sdk_libraries/lib/libhipblas.so.3" \\
  "$packages/_rocm_sdk_libraries/lib/librocblas.so.5"
target=$(echo "$*" | grep -o 'device-gfx[0-9a-f]*' | sed 's/device-//')
[ -z "$target" ] || mkdir -p "$packages/rocm_sdk_device_$target-10.0.0.dist-info"
"""
FAKE_AMD_SMI = """#!/bin/bash
[ -z "${FAKE_NO_CARD_MEMORY:-}" ] || exit 1
echo '[{"gpu": 0, "mem_usage": {"total_vram": {"value": 16368, "unit": "MB"}, "used_vram": {"value": 368, "unit": "MB"}}}]'
"""
RX_7900_XTX, RX_570 = 110000, 80003
INTEL_VULKAN_DEVICES = "Available devices:\n  Vulkan0: Intel(R) UHD Graphics 630 (CFL GT2) (8192 MiB, 8000 MiB free)"
ALLOWED_HOSTS = {"github.com", "api.github.com", "huggingface.co", "hf.co", "stable.repo.amd.com", "rocm.docs.amd.com", "127.0.0.1"}
FAKE_SERVER = """#!/bin/bash
if [ "$1" = --list-devices ]; then printf '%b\\n' "$FAKE_DEVICES"; exit 0; fi
exit 1
"""


def script_value(name: str) -> str:
    return re.search(rf"^{name}=(.*)$", SCRIPT.read_text(), re.M).group(1)


RELEASE = script_value("RELEASE")
ASSET_PREFIX = script_value("ASSET_URL_PREFIX").replace("$RELEASE", RELEASE).rsplit("/", 1)[1]
ROCM_ASSET = f"{ASSET_PREFIX}{script_value('ROCM_ASSET_KIND')}.tar.gz"
VULKAN_ASSET = f"{ASSET_PREFIX}{script_value('VULKAN_ASSET_KIND')}.tar.gz"
RUNTIME_PROJECT_WORDS = {part.split("-")[0].split(".")[0] for part in script_value("RELEASE_API").split("/repos/")[1].split("/")[:2]}


def program(folder: Path, name: str, text: str) -> None:
    path = folder / name
    path.write_text(text)
    path.chmod(0o755)


class Rig:
    def __init__(self, root: Path):
        self.root = root
        self.bin = root / "bin"
        self.bin.mkdir()
        (root / "tmp").mkdir()
        self.report = root / "report.txt"
        self.environment = {**os.environ, "PATH": f"{self.bin}:{os.environ['PATH']}", "FAKE_ROOT": str(root),
                            "AMD_CHECK_DRM": str(root / "sys" / "class" / "drm"), "AMD_CHECK_DEV": str(root / "dev"),
                            "AMD_CHECK_KFD": str(root / "kfd"), "TMPDIR": str(root / "tmp"), "FAKE_DEVICES": ROCM_DEVICES}
        self.runtime_digest = self.build_runtime()
        self.environment["AMD_CHECK_RUNTIME_SHA256"] = self.runtime_digest
        self.environment.pop("HSA_OVERRIDE_GFX_VERSION", None)
        program(self.bin, "curl", FAKE_CURL)
        program(self.bin, "ldd", FAKE_LDD)
        program(self.bin, "python3", FAKE_PYTHON)
        program(self.bin, "amd-smi", FAKE_AMD_SMI)
        program(root, "fake-pip", FAKE_PIP)
        self.lspci(AMD_PCI)

    def lspci(self, line: str) -> None:
        program(self.bin, "lspci", f"#!/bin/bash\necho '{line}'\n")

    def render_node(self, name: str = "renderD128", driver: str = "amdgpu", in_dev: bool = True) -> None:
        driver_folder = self.root / "sys" / "bus" / "pci" / "drivers" / driver
        driver_folder.mkdir(parents=True, exist_ok=True)
        device = self.root / "sys" / "class" / "drm" / name / "device"
        device.mkdir(parents=True)
        (device / "driver").symlink_to(driver_folder)
        if in_dev:
            self.device(f"dri/{name}")

    def kfd_node(self, number: int, version: int) -> None:
        node = self.root / "kfd" / str(number)
        node.mkdir(parents=True)
        (node / "properties").write_text(f"cpu_cores_count 0\ngfx_target_version {version}\n")

    def pip_calls(self) -> list[str]:
        calls = self.root / "pip-calls"
        return calls.read_text().splitlines() if calls.exists() else []

    def device(self, relative: str) -> None:
        path = self.root / "dev" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def build_runtime(self) -> str:
        package = self.root / "package" / "pkg-b1"
        package.mkdir(parents=True)
        program(package, "fake-server", FAKE_SERVER)
        program(package, "fake-rpc-server", "#!/bin/bash\nexit 1\n")
        for library in ("libfake-hip.so", "libfake-vulkan.so"):
            (package / library).write_text("")
        archive = self.root / "runtime.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(package, arcname="pkg-b1")
        return hashlib.sha256(archive.read_bytes()).hexdigest()

    def release_listing(self, digest: str) -> None:
        assets = [{"name": name, "digest": f"sha256:{digest}"} for name in (ROCM_ASSET, VULKAN_ASSET)]
        (self.root / "release.json").write_text(json.dumps({"assets": assets}))

    def start(self, *arguments: str, **extra) -> subprocess.Popen:
        return subprocess.Popen(["bash", str(SCRIPT), "--report", str(self.report), *arguments], cwd=self.root,
                                env={**self.environment, **extra}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, start_new_session=True)

    def run(self, *arguments: str, script: Path = SCRIPT, **extra) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(script), "--report", str(self.report), *arguments], cwd=self.root,
                              env={**self.environment, **extra}, capture_output=True, text=True, timeout=SCRIPT_TIMEOUT_SECONDS)

    def kit_pinned_to_this_runtime(self, constant: str) -> Path:
        tools = self.root / "kit" / "tools"
        tools.mkdir(parents=True)
        text = re.sub(rf"^{constant}=.*$", f"{constant}={self.runtime_digest}", SCRIPT.read_text(), count=1, flags=re.M)
        (tools / SCRIPT.name).write_text(text)
        shutil.copy(SCRIPT.with_name("amd_check_steps.py"), tools)
        shutil.copytree(SCRIPT.parents[1] / "arty_engine", tools.parent / "arty_engine",
                        ignore=shutil.ignore_patterns("__pycache__"))
        self.environment.pop("AMD_CHECK_RUNTIME_SHA256")
        return tools / SCRIPT.name

    def curl_calls(self) -> list[str]:
        calls = self.root / "curl-calls"
        return calls.read_text().splitlines() if calls.exists() else []

    def leftover_work(self) -> list[Path]:
        return list((self.root / "tmp").iterdir())

    def steps(self) -> dict[str, str]:
        found = {}
        for line in self.report.read_text().splitlines():
            matched = re.match(r"  (PASS|FAIL|CHECK|INFO) ([\w-]+): (.*)", line)
            if matched:
                found[matched.group(2)] = f"{matched.group(1)} {matched.group(3)}"
        return found


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


def amd_with_rocm(rig: Rig, version: int = RX_7900_XTX) -> None:
    rig.render_node()
    rig.device("kfd")
    rig.kfd_node(0, 0)
    rig.kfd_node(1, version)
    rig.release_listing(rig.runtime_digest)


def result_line(rig: Rig) -> str:
    return next(line for line in rig.report.read_text().splitlines() if line.startswith("result: "))


def load_steps():
    spec = importlib.util.spec_from_file_location("amd_check_steps", REPO / "tools" / "amd_check_steps.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STEPS = load_steps()


def port_answers(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(1)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def test_a_dry_run_prints_the_plan_and_touches_nothing(rig):
    finished = rig.run("--dry-run")
    assert finished.returncode == 0, finished.stderr
    assert "1. detect the GPU" in finished.stdout and "9. stop the engine" in finished.stdout
    assert "ubuntu-rocm-10.0-x64" in finished.stdout and "Qwen3-VL-2B" in finished.stdout
    assert rig.curl_calls() == [] and not rig.report.exists() and rig.leftover_work() == []


def test_no_amd_card_stops_at_detection_with_a_report(rig):
    rig.lspci(INTEL_PCI)
    rig.render_node(driver="i915")
    finished = rig.run()
    assert finished.returncode == EXIT_NO_AMD, finished.stdout
    assert rig.steps()["detect"].startswith("FAIL no AMD GPU found")
    assert "result: FAIL" in rig.report.read_text()
    assert rig.curl_calls() == [] and rig.leftover_work() == []


def test_an_amd_card_without_its_driver_says_to_install_it(rig):
    finished = rig.run()
    assert finished.returncode == EXIT_NO_AMD
    assert "install the driver" in rig.steps()["detect"]


def test_a_render_node_missing_from_dev_is_named(rig):
    rig.render_node(in_dev=False)
    finished = rig.run()
    assert finished.returncode == EXIT_NO_AMD and "not in" in rig.steps()["detect"]


def test_rocm_runs_through_download_config_and_engine_start_then_cleans_up(rig):
    amd_with_rocm(rig)
    finished = rig.run("--stop-after", "start")
    assert finished.returncode == 0, finished.stdout + finished.stderr
    steps = rig.steps()
    assert steps["detect"].startswith("PASS backend rocm.") and "/dev/kfd: yes" in steps["detect"]
    assert "card target: gfx1100" in steps["detect"]
    assert steps["download"] == f"PASS release {RELEASE}, ubuntu-rocm-10.0-x64 build, sha256 {rig.runtime_digest} matches the pinned digest (the release listing agrees)"
    assert steps["libraries"].startswith("PASS every library of the rocm backend loads (ROCm libraries: system)")
    assert steps["devices"] == "PASS   ROCm0: Fake Radeon (16368 MiB, 16000 MiB free)"
    assert steps["start"].startswith("PASS version") and "GPU backend rocm (set in the config)" in steps["start"]
    assert "16000 MiB free of 16368 MiB (amd-smi)" in steps["start"]
    assert steps["cleanup"] == "PASS engine stopped, temporary folder removed"
    port = int(re.search(r"listen 127\.0\.0\.1:(\d+)", steps["config"]).group(1))
    assert not port_answers(port) and rig.leftover_work() == []
    assert any("rocm-10.0" in call for call in rig.curl_calls())
    assert result_line(rig) == "result: INCOMPLETE (stopped after start as asked; every step that ran passed)"
    assert f"runtime: release {RELEASE}, ubuntu-rocm-10.0-x64 build" in rig.report.read_text()


def test_the_written_config_names_the_backend_and_its_runtime(rig):
    amd_with_rocm(rig)
    finished = rig.run("--stop-after", "config", "--keep")
    assert finished.returncode == 0, finished.stdout
    work = rig.leftover_work()[0]
    config = json.loads((work / "engine.json").read_text())
    assert config["gpu_backend"] == "rocm" and config["listen"].startswith("127.0.0.1:")
    assert config["runtimes"] == {"rocm": {"runner_binary": str(work / "runtime" / "fake-server"), "library_path": []}}
    assert config["store"] == str(work / "models") and config["pull_keep_free_mb"] == 512


def test_no_kfd_means_the_vulkan_build(rig):
    rig.render_node()
    rig.release_listing(rig.runtime_digest)
    finished = rig.run("--stop-after", "download", FAKE_DEVICES=VULKAN_DEVICES)
    assert finished.returncode == 0, finished.stdout
    steps = rig.steps()
    assert steps["detect"].startswith("PASS backend vulkan (no /dev/kfd)")
    assert "ubuntu-vulkan-x64 build" in steps["download"] and "Vulkan0" in steps["devices"]
    assert steps["libraries"].startswith("PASS every library of the vulkan backend loads (ROCm libraries: not used)")


def test_a_digest_that_does_not_match_the_release_stops_the_check(rig):
    amd_with_rocm(rig)
    rig.release_listing("0" * 64)
    finished = rig.run()
    assert finished.returncode == EXIT_FAILED
    assert rig.steps()["download"] == (f"FAIL release {RELEASE}, ubuntu-rocm-10.0-x64 build has sha256 {rig.runtime_digest}, "
                                       f"but the release lists {'0' * 64}")
    assert "config" not in rig.steps() and rig.leftover_work() == []


def test_an_unreachable_release_listing_still_runs_a_runtime_that_matches_its_pinned_digest(rig):
    amd_with_rocm(rig)
    (rig.root / "release.json").unlink()
    rig.run("--stop-after", "download")
    download = rig.steps()["download"]
    assert download.startswith(f"PASS release {RELEASE}, ubuntu-rocm-10.0-x64 build, sha256 {rig.runtime_digest} matches")
    assert "pinned" in download and "listing could not be read" in download


def test_a_runtime_that_does_not_match_its_pinned_digest_stops_the_check_even_when_the_listing_agrees(rig):
    amd_with_rocm(rig)
    finished = rig.run(AMD_CHECK_RUNTIME_SHA256="1" * 64)
    assert finished.returncode == EXIT_FAILED, finished.stdout
    download = rig.steps()["download"]
    assert download.startswith(f"FAIL release {RELEASE}, ubuntu-rocm-10.0-x64 build has sha256 {rig.runtime_digest}")
    assert "1" * 64 in download and "libraries" not in rig.steps() and rig.leftover_work() == []


def test_the_kit_s_pinned_digests_match_the_readme():
    readme = (SCRIPT.parents[1] / "README.md").read_text()
    published = dict(re.findall(r"`([0-9a-f]{64})` \((ROCm|Vulkan)\)", readme.replace("\n", " ")))
    assert {kind: digest for digest, kind in published.items()} == {"ROCm": script_value("ROCM_RUNTIME_SHA256"),
                                                                    "Vulkan": script_value("VULKAN_RUNTIME_SHA256")}


def test_a_runtime_that_sees_no_gpu_fails(rig):
    amd_with_rocm(rig)
    finished = rig.run(FAKE_DEVICES="Available devices:\n  (none)")
    assert finished.returncode == EXIT_FAILED
    assert rig.steps()["devices"] == "FAIL the runtime lists no ROCm device, so models would run on the CPU"


def test_missing_rocm_libraries_without_amd_s_packages_fail_with_their_names(rig):
    amd_with_rocm(rig)
    finished = rig.run("--no-rocm-wheels", FAKE_LDD_MISSING="1")
    assert finished.returncode == EXIT_FAILED
    assert rig.steps()["libraries"].startswith("FAIL the rocm build cannot load: libamdhip64.so.7 libhipblas.so.3 librocblas.so.5")
    assert rig.pip_calls() == []


def test_amd_s_packages_that_do_not_install_fail_the_library_step(rig):
    amd_with_rocm(rig)
    finished = rig.run(FAKE_LDD_MISSING="1", FAKE_PIP_FAILS="1")
    assert finished.returncode == EXIT_FAILED
    assert rig.steps()["libraries"] == ("FAIL missing libamdhip64.so.7 libhipblas.so.3 librocblas.so.5, and AMD's ROCm "
                                        "packages did not install")


def test_amd_s_packages_bring_both_library_folders_and_the_card_s_kernels(rig):
    amd_with_rocm(rig)
    finished = rig.run("--stop-after", "config", "--keep", FAKE_LDD_MISSING="1")
    assert finished.returncode == 0, finished.stdout
    work = rig.leftover_work()[0]
    [call] = rig.pip_calls()
    assert "rocm[libraries,device-gfx1100]==10.0.0" in call and "--index-url https://stable.repo.amd.com/rocm/whl-next/" in call
    assert call.endswith(f"PIP_CACHE_DIR={work}/pip-cache")
    folders = json.loads((work / "engine.json").read_text())["runtimes"]["rocm"]["library_path"]
    packages = work / "rocm-libs" / "lib" / "python3.13" / "site-packages"
    assert folders == [str(packages / "_rocm_sdk_core" / "lib"), str(packages / "_rocm_sdk_libraries" / "lib")]
    assert "installed: rocm-sdk-device-gfx1100" in rig.steps()["libraries"]
    assert "rocm-sdk-device-gfx1100" in rig.report.read_text().split("ROCm libraries: ", 1)[1].splitlines()[0]


def test_a_card_target_given_by_override_picks_its_kernel_package(rig):
    amd_with_rocm(rig, RX_570)
    finished = rig.run("--backend", "rocm", "--stop-after", "download", FAKE_LDD_MISSING="1", HSA_OVERRIDE_GFX_VERSION="10.3.0")
    assert finished.returncode == 0, finished.stdout
    assert "rocm[libraries,device-gfx1030]==10.0.0" in rig.pip_calls()[0]


def test_no_kernel_package_is_guessed_for_a_card_target_that_cannot_be_read(rig):
    rig.render_node()
    rig.device("kfd")
    rig.release_listing(rig.runtime_digest)
    finished = rig.run("--backend", "rocm", FAKE_LDD_MISSING="1")
    assert finished.returncode == EXIT_FAILED
    assert "no kernel package for card target not readable" in rig.steps()["libraries"] and rig.pip_calls() == []


def test_a_card_the_rocm_build_has_no_code_for_is_checked_on_vulkan(rig):
    amd_with_rocm(rig, RX_570)
    finished = rig.run("--stop-after", "detect")
    assert finished.returncode == 0, finished.stdout
    assert rig.steps()["detect"].startswith("PASS backend vulkan (the ROCm build carries no code for card target gfx803)")


def test_a_missing_venv_module_fails_before_anything_is_downloaded(rig):
    amd_with_rocm(rig)
    finished = rig.run(FAKE_NO_VENV="1")
    assert finished.returncode == EXIT_FAILED
    assert "python3-venv" in rig.steps()["tools"] and rig.curl_calls() == []


def test_unreadable_card_memory_fails_the_start(rig):
    amd_with_rocm(rig)
    finished = rig.run("--stop-after", "start", FAKE_NO_CARD_MEMORY="1")
    assert finished.returncode == EXIT_FAILED
    assert "card memory not readable" in rig.steps()["start"] and rig.steps()["start"].startswith("FAIL")


def test_only_an_intel_vulkan_device_fails_the_device_step(rig):
    rig.render_node()
    rig.release_listing(rig.runtime_digest)
    finished = rig.run(FAKE_DEVICES=INTEL_VULKAN_DEVICES)
    assert finished.returncode == EXIT_FAILED
    assert rig.steps()["devices"].startswith("FAIL the runtime lists only other vendors' Vulkan devices")


def test_an_interrupted_check_is_never_a_pass(rig):
    amd_with_rocm(rig)
    check = rig.start(FAKE_CURL_SLOW="20")
    deadline = time.monotonic() + SCRIPT_TIMEOUT_SECONDS
    while not rig.curl_calls() and time.monotonic() < deadline:
        time.sleep(0.1)
    os.killpg(check.pid, signal.SIGINT)
    check.communicate(timeout=SCRIPT_TIMEOUT_SECONDS)
    assert check.returncode == 130
    assert result_line(rig) == "result: INCOMPLETE (interrupted; only the steps below ran)" and rig.leftover_work() == []


def test_the_temporary_folder_defaults_to_a_disk_backed_place(rig):
    finished = rig.run("--dry-run", TMPDIR="")
    assert "temporary folder under /var/tmp" in finished.stdout


def test_the_kit_s_own_output_names_the_runtime_only_in_url_lines(rig):
    amd_with_rocm(rig)
    finished = rig.run("--stop-after", "start", FAKE_LDD_MISSING="1")
    assert finished.returncode == 0, finished.stdout
    for line in finished.stdout.splitlines() + rig.report.read_text().splitlines():
        if any(word in line.lower() for word in RUNTIME_PROJECT_WORDS):
            assert "https://" in line, line
    assert ROCM_ASSET not in finished.stdout + rig.report.read_text()
    assert "libfake-hip.so" not in finished.stdout


@pytest.mark.parametrize("log, expect, passes", [
    ("load_tensors: offloaded 29/29 layers to GPU\n", "card", True),
    ("load_tensors: offloaded 0/29 layers to GPU\n", "card", False),
    ("load_tensors: offloaded 0/29 layers to GPU\n", "cpu", True),
    ("load_tensors: offloaded 29/29 layers to GPU\n", "cpu", False),
    ("nothing about layers\n", "card", False),
    ("nothing about layers\n", "cpu", False),
])
def test_the_offload_step_reads_only_what_the_step_logged(tmp_path, log, expect, passes):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "model.log").write_text("load_tensors: offloaded 29/29 layers to GPU\nload_tensors: offloaded 0/29 layers to GPU\n")
    since = tmp_path / "sizes.json"
    since.write_text(STEPS.log_sizes(logs))
    with open(logs / "model.log", "a") as appended:
        appended.write(log)
    if passes:
        assert "layers on the card" in STEPS.offload(logs, since, expect)
    else:
        with pytest.raises(STEPS.CheckFailed):
            STEPS.offload(logs, since, expect)


@pytest.mark.parametrize("answer", ['"ROCK 42"', "**ROCK 42**", "Rock 42!", "The text reads: ROCK 42."])
def test_the_picture_check_reads_past_punctuation(tmp_path, monkeypatch, answer):
    monkeypatch.setattr(STEPS, "call", lambda *args, **kwargs: {"message": {"content": answer}})
    assert STEPS.picture("http://127.0.0.1:1", "m", tmp_path / "p.png").startswith("m read")


def test_the_picture_check_still_fails_a_wrong_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(STEPS, "call", lambda *args, **kwargs: {"message": {"content": "ROCKY 4"}})
    with pytest.raises(STEPS.CheckFailed):
        STEPS.picture("http://127.0.0.1:1", "m", tmp_path / "p.png")


def test_the_kit_s_rocm_targets_match_the_engine_s():
    from arty_engine.backend import ROCM_BUILD_TARGETS
    assert set(script_value("ROCM_BUILD_TARGETS").strip('"').split()) == ROCM_BUILD_TARGETS


class FakeEngine:
    def __init__(self, chat_answer: str = "Paris."):
        self.requests = []
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                engine.requests.append(("GET", self.path, None))
                self.answer({"models": [{"name": "m", "size_vram": 512 * 1024 * 1024}]} if self.path == "/api/ps" else {"version": "test"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                engine.requests.append(("POST", self.path, body))
                if self.path == "/api/pull":
                    self.answer({"status": "success"})
                elif self.path == "/api/embed":
                    self.answer({"embeddings": [[0.0] * 768]})
                elif body["messages"][0].get("images"):
                    self.answer({"message": {"content": "ROCK 42"}})
                else:
                    self.answer({"message": {"content": chat_answer}})

            def answer(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"

    def close(self):
        self.http.shutdown()
        self.http.server_close()


@pytest.fixture
def fake_engine():
    engines = []

    def start(**settings):
        engine = FakeEngine(**settings)
        engines.append(engine)
        return engine
    yield start
    for engine in engines:
        engine.close()


def test_the_model_checks_against_a_running_engine(rig, fake_engine):
    amd_with_rocm(rig)
    engine = fake_engine()
    finished = rig.run("--engine-url", engine.url)
    assert finished.returncode == 0, finished.stdout
    steps = rig.steps()
    for step in ("pull-chat", "pull-embed", "pull-vision", "chat", "chat-cpu-only", "embed", "picture", "loaded"):
        assert steps[step].startswith("PASS"), (step, steps[step])
    assert steps["memory"].startswith("INFO") and "download" not in steps
    assert result_line(rig).startswith("result: INCOMPLETE (not checked: GPU offload")
    posts = [body for method, path, body in engine.requests if method == "POST"]
    assert [body["model"] for body in posts if "messages" not in body and "input" not in body] == [
        "hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0", "nomic-embed-text", "hf.co/Qwen/Qwen3-VL-2B-Instruct-GGUF:Q4_K_M"]
    chats = [body for body in posts if "messages" in body]
    assert [body["options"].get("num_gpu") for body in chats[:2]] == [None, 0]
    image = base64.b64decode(chats[2]["messages"][0]["images"][0])
    assert image.startswith(b"\x89PNG\r\n\x1a\n")
    assert rig.leftover_work() == []


def test_a_wrong_answer_fails_its_check_and_the_rest_still_run(rig, fake_engine):
    amd_with_rocm(rig)
    finished = rig.run("--engine-url", fake_engine(chat_answer="London.").url)
    assert finished.returncode == EXIT_FAILED
    steps = rig.steps()
    assert steps["chat"].startswith("FAIL hf.co/Qwen/Qwen3-0.6B-GGUF:Q8_0 answered 'London.'")
    assert steps["embed"].startswith("PASS") and steps["picture"].startswith("PASS")


def test_the_kit_holds_no_home_values_and_the_script_needs_no_sudo():
    home_values = re.compile(r"\b(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})\b"
                             r"|\bCT ?\d{3}\b|/home/|PRIVATE KEY|\bhf_[A-Za-z0-9]{20}")
    for path in KIT_FILES:
        text = path.read_text()
        assert not home_values.search(text), path
        hosts = set(re.findall(r"https?://([^/\s:)]+)", text))
        assert hosts <= ALLOWED_HOSTS, (path.name, hosts - ALLOWED_HOSTS)
    assert "sudo" not in SCRIPT.read_text() and "sudo" not in KIT_FILES[1].read_text()


def test_the_runtime_project_s_name_appears_only_on_its_download_url_lines():
    assert len(RUNTIME_PROJECT_WORDS) == 2
    for path in KIT_FILES:
        for line in path.read_text().splitlines():
            if any(word in line.lower() for word in RUNTIME_PROJECT_WORDS):
                assert "https://" in line, (path.name, line)


def test_the_rocm_build_is_checked_against_the_rocm_pin(rig):
    amd_with_rocm(rig)
    kit = rig.kit_pinned_to_this_runtime("ROCM_RUNTIME_SHA256")
    finished = rig.run("--stop-after", "download", script=kit)
    assert finished.returncode == 0, finished.stdout
    assert rig.steps()["download"].startswith(f"PASS release {RELEASE}, ubuntu-rocm-10.0-x64 build, sha256 {rig.runtime_digest}")
    assert rig.steps()["download"].endswith("(the release listing agrees)")


def test_the_vulkan_build_is_checked_against_the_vulkan_pin(rig):
    rig.render_node()
    rig.release_listing(rig.runtime_digest)
    kit = rig.kit_pinned_to_this_runtime("VULKAN_RUNTIME_SHA256")
    finished = rig.run("--stop-after", "download", script=kit, FAKE_DEVICES=VULKAN_DEVICES)
    assert finished.returncode == 0, finished.stdout
    assert rig.steps()["download"].startswith(f"PASS release {RELEASE}, ubuntu-vulkan-x64 build, sha256 {rig.runtime_digest}")
    assert rig.steps()["download"].endswith("(the release listing agrees)")


def test_the_copied_kit_still_fails_a_runtime_the_release_listing_does_not_name(rig):
    amd_with_rocm(rig)
    rig.release_listing("0" * 64)
    kit = rig.kit_pinned_to_this_runtime("ROCM_RUNTIME_SHA256")
    finished = rig.run("--stop-after", "download", script=kit)
    assert finished.returncode == EXIT_FAILED, finished.stdout
    assert rig.steps()["download"].endswith(f"but the release lists {'0' * 64}")


def test_a_build_is_never_checked_against_the_other_build_s_pin(rig):
    rig.render_node()
    rig.release_listing(rig.runtime_digest)
    kit = rig.kit_pinned_to_this_runtime("ROCM_RUNTIME_SHA256")
    finished = rig.run("--stop-after", "download", script=kit, FAKE_DEVICES=VULKAN_DEVICES)
    assert finished.returncode == EXIT_FAILED
    assert f"this kit pins {script_value('VULKAN_RUNTIME_SHA256')}" in rig.steps()["download"]


def test_the_test_hook_that_replaces_the_pin_is_documented():
    guide = (SCRIPT.parents[1] / "docs" / "amd-check.md").read_text()
    assert "`AMD_CHECK_RUNTIME_SHA256`" in guide and "leave them unset on a real run" in guide
