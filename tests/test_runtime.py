import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from arty_engine import runtime
from arty_engine.runtime import EARLIER_WORKER_NAMES, SERVER, WORKER, ReleaseInfo, RuntimeInstallError, install, locate_binary
from test_no_runtime_names import UNWANTED_NAMES

EARLIER_WORKER = EARLIER_WORKER_NAMES[0]
UPSTREAM_WORD = "ll" + "ama"
CORE_LIBRARY = "libggml-base.so.0"
MAIN_LIBRARY = f"lib{UPSTREAM_WORD}.so.0"
SERVER_LIBRARY = f"lib{UPSTREAM_WORD}-server-impl.so"
UNUSED_LIBRARY = f"lib{UPSTREAM_WORD}-bench-impl.so"
PATCHELF = os.environ.get("PATCHELF") or shutil.which("patchelf") or ""
C_COMPILER = shutil.which("gcc") or shutil.which("cc") or ""
needs_elf_tools = pytest.mark.skipif(not (PATCHELF and C_COMPILER), reason="needs a C compiler and patchelf")
REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGER = REPOSITORY / "tools" / "package_runtime.py"
INFO = ReleaseInfo(release="runtime-b1", platforms=("linux-x64", "windows-x64"), sha256={})
LICENCE = "MIT License\n\nCopyright (c) 2023-2026 The ggml authors\n\nPermission is hereby granted...\n"


def runtime_zip(entries: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, text in entries.items():
            bundle.writestr(name, text)
    return buffer.getvalue()


GOOD_ZIP = runtime_zip({"arty-runtime-server": "#!/bin/sh\n", "arty-runtime-worker": "#!/bin/sh\n", "libx.so": "",
                        "LICENSE": LICENCE, "THIRD_PARTY_NOTICES": LICENCE})


class FakeWeb:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.asked: list[str] = []

    def __call__(self, url: str):
        self.asked.append(url)
        if url not in self.files:
            raise RuntimeInstallError(f"{url} answered 404")
        return io.BytesIO(self.files[url])


def asset_url(platform: str = "linux-x64", release: str = "runtime-b1") -> str:
    return runtime.release_url(release, runtime.asset_name(platform))


def test_the_release_lives_on_arty_engine_s_own_github():
    assert asset_url() == "https://github.com/drew-codes-things/ArtyEngine/releases/download/runtime-b1/arty-runtime-linux-x64.zip"
    shipped = ReleaseInfo.read()
    assert shipped.release == "runtime-b12110" and "linux-cuda-12.8-x64" in shipped.platforms


def test_install_checks_the_pinned_digest_unpacks_and_points_the_links(tmp_path):
    web = FakeWeb({asset_url(): GOOD_ZIP})
    info = ReleaseInfo(INFO.release, INFO.platforms, {"linux-x64": hashlib.sha256(GOOD_ZIP).hexdigest()})
    folder = install("linux-x64", tmp_path, info=info, opener=web)
    assert folder == tmp_path / "runtime-b1-linux-x64"
    assert os.access(folder / "arty-runtime-server", os.X_OK) and (folder / "LICENSE").read_text() == LICENCE
    assert (tmp_path / "server").resolve() == (folder / "arty-runtime-server").resolve()
    assert (tmp_path / "worker").resolve() == (folder / "arty-runtime-worker").resolve()
    assert web.asked == [asset_url()] and not list(tmp_path.glob("*.partial"))


def test_without_a_pin_the_install_is_refused_unless_asked(tmp_path):
    sums = f"{hashlib.sha256(GOOD_ZIP).hexdigest()}  arty-runtime-linux-x64.zip\n".encode()
    web = FakeWeb({asset_url(): GOOD_ZIP, runtime.release_url("runtime-b1", "SHA256SUMS"): sums})
    with pytest.raises(RuntimeInstallError, match="--allow-unpinned"):
        install("linux-x64", tmp_path, info=INFO, opener=web)
    assert web.asked == [] and not (tmp_path / "server").exists()
    install("linux-x64", tmp_path, info=INFO, opener=web, allow_unpinned=True)
    assert (tmp_path / "server").exists()


def test_the_runtime_command_offers_the_unpinned_choice():
    finished = subprocess.run([sys.executable, "-m", "arty_engine", "runtime", "install", "--help"], cwd=REPOSITORY,
                              capture_output=True, text=True, timeout=30)
    assert finished.returncode == 0 and "--allow-unpinned" in finished.stdout


def test_the_shipped_release_pins_nothing_it_has_not_checked():
    assert ReleaseInfo.read().sha256 == {}


def test_a_zip_whose_digest_differs_installs_nothing(tmp_path):
    web = FakeWeb({asset_url(): GOOD_ZIP})
    info = ReleaseInfo(INFO.release, INFO.platforms, {"linux-x64": "0" * 64})
    with pytest.raises(RuntimeInstallError, match="expected 0000"):
        install("linux-x64", tmp_path, info=info, opener=web)
    assert sorted(path.name for path in tmp_path.iterdir()) == []


def test_a_custom_runtime_url_needs_its_digest(tmp_path):
    custom = "https://mirror.example/runtime.zip"
    web = FakeWeb({custom: GOOD_ZIP})
    with pytest.raises(RuntimeInstallError, match="needs --sha256"):
        install("linux-x64", tmp_path, custom_url=custom, info=INFO, opener=web)
    install("linux-x64", tmp_path, custom_url=custom, sha256=hashlib.sha256(GOOD_ZIP).hexdigest(), info=INFO, opener=web)
    assert web.asked == [custom]


@pytest.mark.parametrize("bad_name", ["../escape", "/etc/passwd", "C:/windows/x", "lib/C:evil", "a/b:stream"])
def test_a_zip_with_unsafe_paths_is_refused(tmp_path, bad_name):
    bundle = runtime_zip({"arty-runtime-server": "", bad_name: "x"})
    info = ReleaseInfo(INFO.release, INFO.platforms, {"linux-x64": hashlib.sha256(bundle).hexdigest()})
    with pytest.raises(RuntimeInstallError, match="unsafe path"):
        install("linux-x64", tmp_path / "runtime", info=info, opener=FakeWeb({asset_url(): bundle}))
    assert not (tmp_path / "escape").exists() and not (tmp_path / "runtime" / "server").exists()


def test_a_zip_without_the_server_binary_is_refused(tmp_path):
    bundle = runtime_zip({"LICENSE": LICENCE})
    info = ReleaseInfo(INFO.release, INFO.platforms, {"linux-x64": hashlib.sha256(bundle).hexdigest()})
    with pytest.raises(RuntimeInstallError, match="holds no arty-runtime-server"):
        install("linux-x64", tmp_path, info=info, opener=FakeWeb({asset_url(): bundle}))
    assert not (tmp_path / "runtime-b1-linux-x64").exists()


def test_an_installed_release_is_kept_unless_replace_is_asked(tmp_path):
    info = ReleaseInfo(INFO.release, INFO.platforms, {"linux-x64": hashlib.sha256(GOOD_ZIP).hexdigest()})
    web = FakeWeb({asset_url(): GOOD_ZIP})
    install("linux-x64", tmp_path, info=info, opener=web)
    with pytest.raises(RuntimeInstallError, match="--replace"):
        install("linux-x64", tmp_path, info=info, opener=web)
    install("linux-x64", tmp_path, info=info, opener=web, replace=True)


def test_a_replace_that_fails_halfway_keeps_the_installed_release(tmp_path, monkeypatch):
    info = ReleaseInfo(INFO.release, INFO.platforms, {"linux-x64": hashlib.sha256(GOOD_ZIP).hexdigest()})
    web = FakeWeb({asset_url(): GOOD_ZIP})
    folder = install("linux-x64", tmp_path, info=info, opener=web)
    (folder / "marker").write_text("old")
    real_rename = Path.rename

    def failing_rename(self, target):
        if self.name.endswith(".partial") and Path(target) == folder:
            raise OSError("disk went away")
        return real_rename(self, target)
    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(OSError, match="disk went away"):
        install("linux-x64", tmp_path, info=info, opener=web, replace=True)
    assert (folder / "marker").read_text() == "old" and (folder / "arty-runtime-server").exists()
    assert sorted(path.name for path in tmp_path.iterdir() if "old-" in path.name) == []


def test_an_unknown_platform_is_refused_before_anything_is_downloaded(tmp_path):
    web = FakeWeb({})
    with pytest.raises(RuntimeInstallError, match="unknown platform"):
        install("amiga-68k", tmp_path, info=INFO, opener=web)
    assert web.asked == []


def executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n")
    path.chmod(0o755)
    return path


def test_the_new_binary_names_are_found_in_a_runtime_folder(tmp_path):
    executable(tmp_path / "arty-runtime-server")
    executable(tmp_path / "arty-runtime-worker")
    assert locate_binary(str(tmp_path), SERVER) == str(tmp_path / "arty-runtime-server")
    assert locate_binary(str(tmp_path / "server"), SERVER) == str(tmp_path / "arty-runtime-server")
    assert locate_binary(str(tmp_path), WORKER) == str(tmp_path / "arty-runtime-worker")


def test_an_earlier_unpacked_build_still_works(tmp_path):
    executable(tmp_path / EARLIER_WORKER)
    earlier_server = executable(tmp_path / "".join(["lla", "ma-server"]))
    executable(tmp_path / "fake-cli")
    assert locate_binary(str(tmp_path), SERVER) == str(earlier_server)
    assert locate_binary(str(tmp_path), WORKER) == str(tmp_path / EARLIER_WORKER)


def test_the_live_install_s_link_keeps_working(tmp_path):
    build = tmp_path / "earlier-build"
    build.mkdir()
    target = executable(build / "any-name-server")
    (tmp_path / "server").symlink_to(target)
    assert locate_binary(str(tmp_path / "server"), SERVER) == str(tmp_path / "server")


def test_a_path_that_finds_nothing_is_left_as_given(tmp_path):
    assert locate_binary("/opt/nowhere/server", SERVER) == "/opt/nowhere/server"
    assert locate_binary(str(tmp_path / "server"), SERVER) == str(tmp_path / "server")


def upstream_archive(tmp_path: Path, kind: str, licence: str = LICENCE, extra: dict | None = None) -> Path:
    folder = tmp_path / "upstream" / "build-b1"
    folder.mkdir(parents=True)
    files = {"tool-server": "server", EARLIER_WORKER: "worker", "tool-cli": "cli", "libtool.so.0.1": "lib",
             "libggml-rpc.so": "rpc lib", **({"LICENSE": licence} if licence else {}), **(extra or {})}
    for name, text in files.items():
        executable(folder / name).write_text(text)
    (folder / "libtool.so").symlink_to("libtool.so.0.1")
    archive = tmp_path / f"upstream.{kind}"
    if kind == "tar.gz":
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(folder, arcname="build-b1")
    else:
        with zipfile.ZipFile(archive, "w") as bundle:
            for path in sorted(folder.iterdir()):
                bundle.write(path, f"{path.name}")
    return archive


def package(tmp_path: Path, archive: Path, platform: str = "linux-x64", digest: str = "", *extra: str):
    digest = digest or hashlib.sha256(archive.read_bytes()).hexdigest()
    return subprocess.run([sys.executable, str(PACKAGER), "--platform", platform, "--source-file", str(archive), "--sha256",
                           digest, "--out", str(tmp_path / "out"), *extra], capture_output=True, text=True, timeout=60)


def compile_elf(folder: Path, name: str, source: str, needed: tuple = (), soname: str = "", shared: bool = True) -> None:
    source_file = folder.parent / f"{name}.c"
    source_file.write_text(source)
    command = [C_COMPILER, *(["-shared", "-fPIC"] if shared else []), "-o", str(folder / name), str(source_file),
               "-Wl,-rpath,$ORIGIN", "-L", str(folder), *[f"-l:{library}" for library in needed]]
    if soname:
        command.append(f"-Wl,-soname,{soname}")
    subprocess.run(command, check=True, capture_output=True, timeout=120)


def elf_archive(tmp_path: Path, kind: str, extra_server: bool = False) -> Path:
    folder = tmp_path / "upstream" / "build-b1"
    folder.mkdir(parents=True)
    compile_elf(folder, "libggml-base.so.0.1", "int base_value(void) { return 40; }", soname=CORE_LIBRARY)
    (folder / CORE_LIBRARY).symlink_to("libggml-base.so.0.1")
    compile_elf(folder, f"{MAIN_LIBRARY}.9", "int base_value(void); int core_value(void) { return base_value() + 1; }",
                (CORE_LIBRARY,), MAIN_LIBRARY)
    (folder / MAIN_LIBRARY).symlink_to(f"{MAIN_LIBRARY}.9")
    compile_elf(folder, SERVER_LIBRARY, "int core_value(void); int answer(void) { return core_value() + 1; }", (MAIN_LIBRARY,),
                SERVER_LIBRARY)
    compile_elf(folder, UNUSED_LIBRARY, "int unused(void) { return 0; }", soname=UNUSED_LIBRARY)
    compile_elf(folder, "libggml-cpu-x64.so", "int base_value(void); int module_value(void) { return base_value(); }",
                (CORE_LIBRARY,))
    server_source = '#include <stdio.h>\nint answer(void);\nint main(void) { printf("server %d\\n", answer()); return 0; }\n'
    compile_elf(folder, "tool-server", server_source, (SERVER_LIBRARY,), shared=False)
    if extra_server:
        compile_elf(folder, "other-server", server_source.replace("server %d", "other %d"), (SERVER_LIBRARY,), shared=False)
    compile_elf(folder, EARLIER_WORKER, '#include <stdio.h>\nint base_value(void);\n'
                'int main(void) { printf("worker %d\\n", base_value()); return 0; }\n', (CORE_LIBRARY,), shared=False)
    (folder / "LICENSE").write_text(LICENCE)
    archive = tmp_path / f"upstream.{kind}"
    if kind == "tar.gz":
        with tarfile.open(archive, "w:gz") as bundle:
            bundle.add(folder, arcname="build-b1")
    else:
        with zipfile.ZipFile(archive, "w") as bundle:
            for path in sorted(folder.iterdir()):
                bundle.write(path, path.name)
    return archive


def installed_asset(tmp_path: Path) -> Path:
    asset = (tmp_path / "out" / "arty-runtime-linux-x64.zip").read_bytes()
    info = ReleaseInfo(INFO.release, INFO.platforms, {"linux-x64": hashlib.sha256(asset).hexdigest()})
    return install("linux-x64", tmp_path / "runtime", info=info, opener=FakeWeb({asset_url(): asset}))


@needs_elf_tools
@pytest.mark.parametrize("kind", ["tar.gz", "zip"])
def test_the_packager_ships_only_linked_libraries_under_engine_names(tmp_path, kind):
    finished = package(tmp_path, elf_archive(tmp_path, kind), "linux-x64", "", "--patchelf", PATCHELF)
    assert finished.returncode == 0, finished.stderr
    asset = tmp_path / "out" / "arty-runtime-linux-x64.zip"
    with zipfile.ZipFile(asset) as bundle:
        names = sorted(bundle.namelist())
        assert names == ["LICENSE", "THIRD_PARTY_NOTICES", "arty-runtime-server", "arty-runtime-worker",
                         "libarty-runtime-server-impl.so", "libarty-runtime.so.0", CORE_LIBRARY, "libggml-cpu-x64.so"]
        assert not [name for name in names if UNWANTED_NAMES.search(name) or UPSTREAM_WORD in name.lower()]
        assert bundle.read("LICENSE").decode() == LICENCE
        assert "Copyright (c) 2023-2026 The ggml authors" in bundle.read("THIRD_PARTY_NOTICES").decode()
        assert bundle.getinfo("arty-runtime-server").external_attr >> 16 & 0o111
    sums = (tmp_path / "out" / "SHA256SUMS").read_text()
    assert sums == f"{hashlib.sha256(asset.read_bytes()).hexdigest()}  arty-runtime-linux-x64.zip\n"


@needs_elf_tools
def test_a_packaged_runtime_installs_and_runs_with_its_renamed_libraries(tmp_path):
    assert package(tmp_path, elf_archive(tmp_path, "tar.gz"), "linux-x64", "", "--patchelf", PATCHELF).returncode == 0
    folder = installed_asset(tmp_path)
    clean = {"PATH": os.environ.get("PATH", "")}
    server = subprocess.run([str(folder / "arty-runtime-server")], capture_output=True, text=True, timeout=30, env=clean)
    worker = subprocess.run([str(folder / "arty-runtime-worker")], capture_output=True, text=True, timeout=30, env=clean)
    assert server.stdout == "server 42\n" and worker.stdout == "worker 40\n", server.stderr + worker.stderr
    soname = subprocess.run([PATCHELF, "--print-soname", str(folder / "libarty-runtime.so.0")], capture_output=True, text=True)
    assert soname.stdout.strip() == "libarty-runtime.so.0"


def test_the_packager_refuses_a_linux_build_that_is_not_elf(tmp_path):
    finished = package(tmp_path, upstream_archive(tmp_path, "zip"), "linux-x64", "", "--patchelf", "/bin/true")
    assert finished.returncode == 1 and "is not a Linux (ELF) binary" in finished.stderr


def test_the_packager_refuses_an_archive_with_another_digest(tmp_path):
    finished = package(tmp_path, upstream_archive(tmp_path, "zip"), "linux-x64", "0" * 64)
    assert finished.returncode == 1 and "expected 0000" in finished.stderr
    assert not (tmp_path / "out" / "arty-runtime-linux-x64.zip").exists()


def test_the_packager_refuses_a_build_without_its_licence(tmp_path):
    finished = package(tmp_path, upstream_archive(tmp_path, "zip", licence=""))
    assert finished.returncode == 1 and "LICENSE" in finished.stderr


def test_the_packager_refuses_a_licence_without_the_copyright_line(tmp_path):
    finished = package(tmp_path, upstream_archive(tmp_path, "zip", licence="MIT License\n"))
    assert finished.returncode == 1 and "The ggml authors" in finished.stderr


@needs_elf_tools
def test_the_packager_asks_for_the_name_when_two_servers_could_be_meant(tmp_path):
    archive = elf_archive(tmp_path, "zip", extra_server=True)
    finished = package(tmp_path, archive, "linux-x64", "", "--patchelf", PATCHELF)
    assert finished.returncode == 1 and "--server-binary" in finished.stderr
    finished = package(tmp_path, archive, "linux-x64", "", "--patchelf", PATCHELF, "--server-binary", "other-server")
    assert finished.returncode == 0, finished.stderr
    folder = installed_asset(tmp_path)
    ran = subprocess.run([str(folder / "arty-runtime-server")], capture_output=True, text=True, timeout=30)
    assert ran.stdout == "other 42\n"


def test_windows_builds_keep_their_exe_suffix(tmp_path):
    archive = upstream_archive(tmp_path, "zip", extra={"tool-server.exe": "win server", EARLIER_WORKER + ".exe": "win worker",
                                                       "tool.dll": "dll"})
    finished = package(tmp_path, archive, "windows-x64", "", "--server-binary", "tool-server.exe", "--worker-binary",
                       EARLIER_WORKER + ".exe")
    assert finished.returncode == 0, finished.stderr
    with zipfile.ZipFile(tmp_path / "out" / "arty-runtime-windows-x64.zip") as bundle:
        assert bundle.read("arty-runtime-server.exe") == b"win server" and "tool.dll" in bundle.namelist()


def test_the_runtime_command_finds_binaries(tmp_path):
    executable(tmp_path / "arty-runtime-server")
    finished = subprocess.run([sys.executable, "-m", "arty_engine", "runtime", "find", "--dir", str(tmp_path)],
                              cwd=REPOSITORY, capture_output=True, text=True, timeout=30)
    assert json.loads(finished.stdout) == {"server": str(tmp_path / "arty-runtime-server"), "worker": str(tmp_path / "worker")}
