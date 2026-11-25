import argparse
import fnmatch
import hashlib
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arty_engine.runtime import SERVER_BINARY, WINDOWS_SUFFIX, WORKER_BINARY, ReleaseInfo, asset_name

REPOSITORY = Path(__file__).resolve().parents[1]
NOTICES = REPOSITORY / "THIRD_PARTY_NOTICES"
REQUIRED_COPYRIGHT = "Copyright (c) 2023-2026 The ggml authors"
LICENSE_NAMES = ("LICENSE", "LICENSE.txt", "LICENSE.md")
LIBRARY_PATTERNS = ("*.so", "*.so.*", "*.dll", "*.dylib")
SERVER_ENDING = "-server"
WORKER_ENDING = "rpc-server"
FIXED_TIME = (1980, 1, 1, 0, 0, 0)
EXECUTABLE_MODE = 0o755
PLAIN_MODE = 0o644
CHUNK_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 120
CHECKSUMS = "SHA256SUMS"
ELF_MAGIC = b"\x7fELF"
UPSTREAM_LIBRARY_PREFIX = "lib" + "ll" + "ama"
ENGINE_LIBRARY_PREFIX = "libarty-runtime"
BACKEND_MODULE = re.compile(r"^libggml-(?!base\.)[A-Za-z0-9_-]+\.so$")
PATCHELF_TIMEOUT_SECONDS = 60


class PackagingError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "arty-engine-packager/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response, open(target, "wb") as output:
            shutil.copyfileobj(response, output, CHUNK_BYTES)
    except (urllib.error.URLError, OSError) as error:
        raise PackagingError(f"couldn't download {url}: {error}") from error


def is_safe(name: str) -> bool:
    member = PurePosixPath(name.replace("\\", "/"))
    return not member.is_absolute() and ".." not in member.parts and bool(member.parts) and ":" not in member.parts[0]


def extract(archive: Path, folder: Path) -> None:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as bundle:
            unsafe = [name for name in bundle.namelist() if not is_safe(name)]
            if unsafe:
                raise PackagingError(f"{archive.name} holds unsafe paths such as {unsafe[0]!r}")
            bundle.extractall(folder)
        return
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as bundle:
            unsafe = [member.name for member in bundle.getmembers() if not is_safe(member.name)]
            if unsafe:
                raise PackagingError(f"{archive.name} holds unsafe paths such as {unsafe[0]!r}")
            bundle.extractall(folder, **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
        return
    raise PackagingError(f"{archive.name} is neither a zip nor a tar archive")


def stem(path: Path) -> str:
    return path.name.removesuffix(WINDOWS_SUFFIX)


def one_binary(folder: Path, chosen: str, matches, role: str) -> Path:
    if chosen:
        found = [path for path in folder.rglob(chosen) if path.is_file()]
    else:
        found = [path for path in folder.rglob("*") if path.is_file() and matches(stem(path))]
    if len(found) != 1:
        names = ", ".join(sorted(path.name for path in found)) or "none"
        raise PackagingError(f"expected one {role} binary in the archive, found {names}; name it with --{role}-binary")
    return found[0]


def is_server_name(name: str) -> bool:
    return name.endswith(SERVER_ENDING) and "rpc" not in name


def is_worker_name(name: str) -> bool:
    return name.endswith(WORKER_ENDING)


def license_file(folder: Path) -> Path:
    for name in LICENSE_NAMES:
        found = sorted(folder.rglob(name))
        if found:
            return found[0]
    raise PackagingError("the archive has no LICENSE file; the runtime's MIT notice must ship with it")


def libraries(folder: Path) -> list[Path]:
    return sorted({path for path in folder.iterdir() if path.is_file()
                   and any(fnmatch.fnmatch(path.name, pattern) for pattern in LIBRARY_PATTERNS)})


def is_elf(path: Path) -> bool:
    with open(path, "rb") as handle:
        return handle.read(len(ELF_MAGIC)) == ELF_MAGIC


def shipped_name(name: str) -> str:
    if name.startswith(UPSTREAM_LIBRARY_PREFIX):
        return ENGINE_LIBRARY_PREFIX + name[len(UPSTREAM_LIBRARY_PREFIX):]
    return name


def run_patchelf(patchelf: str, *arguments: str) -> str:
    try:
        finished = subprocess.run([patchelf, *arguments], capture_output=True, text=True, timeout=PATCHELF_TIMEOUT_SECONDS)
    except OSError as error:
        raise PackagingError(f"couldn't run {patchelf}: {error}") from error
    if finished.returncode != 0:
        raise PackagingError(f"patchelf {' '.join(arguments)} failed: {finished.stderr.strip()}")
    return finished.stdout


def needed_names(patchelf: str, path: Path) -> list[str]:
    return [line.strip() for line in run_patchelf(patchelf, "--print-needed", str(path)).splitlines() if line.strip()]


def backend_modules(folder: Path) -> list[Path]:
    return sorted(path for path in folder.iterdir() if path.is_file() and BACKEND_MODULE.match(path.name))


def linked_libraries(patchelf: str, folder: Path, roots: list[Path]) -> dict[str, Path]:
    shipped: dict[str, Path] = {}
    waiting = list(roots)
    while waiting:
        for name in needed_names(patchelf, waiting.pop()):
            candidate = folder / name
            if name in shipped or not candidate.is_file():
                continue
            shipped[name] = candidate.resolve()
            waiting.append(shipped[name])
    return shipped


def renamed_copy(patchelf: str, source: Path, staging: Path, name: str, renames: dict[str, str]) -> Path:
    target = staging / name
    shutil.copyfile(source, target)
    arguments = ["--set-soname", name] if name in renames.values() else []
    for needed in needed_names(patchelf, target):
        if needed in renames:
            arguments += ["--replace-needed", needed, renames[needed]]
    if arguments:
        run_patchelf(patchelf, *arguments, str(target))
    return target


def linux_files(patchelf: str, server: Path, worker: Path, staging: Path) -> list[tuple[Path, str, int]]:
    for binary in (server, worker):
        if not is_elf(binary):
            raise PackagingError(f"{binary.name} is not a Linux (ELF) binary")
    modules = backend_modules(server.parent)
    linked = linked_libraries(patchelf, server.parent, [server, worker, *modules])
    for module in modules:
        linked.setdefault(module.name, module)
    renames = {name: shipped_name(name) for name in linked if shipped_name(name) != name}
    files = [(renamed_copy(patchelf, server, staging, SERVER_BINARY, renames), SERVER_BINARY, EXECUTABLE_MODE),
             (renamed_copy(patchelf, worker, staging, WORKER_BINARY, renames), WORKER_BINARY, EXECUTABLE_MODE)]
    for name, source in sorted(linked.items()):
        shipped = renames.get(name, name)
        files.append((renamed_copy(patchelf, source, staging, shipped, renames), shipped, EXECUTABLE_MODE))
    return files


def windows_files(server: Path, worker: Path) -> list[tuple[Path, str, int]]:
    files = [(server, SERVER_BINARY + WINDOWS_SUFFIX, EXECUTABLE_MODE), (worker, WORKER_BINARY + WINDOWS_SUFFIX, EXECUTABLE_MODE)]
    return files + [(library, library.name, EXECUTABLE_MODE) for library in libraries(server.parent)]


def add_file(bundle: zipfile.ZipFile, source: Path, name: str, mode: int) -> None:
    entry = zipfile.ZipInfo(name, date_time=FIXED_TIME)
    entry.external_attr = (0o100000 | mode) << 16
    entry.compress_type = zipfile.ZIP_DEFLATED
    with open(source, "rb") as handle, bundle.open(entry, "w") as output:
        shutil.copyfileobj(handle, output, CHUNK_BYTES)


def check_licence(licence: Path) -> None:
    licence_text = licence.read_text(errors="replace")
    if REQUIRED_COPYRIGHT not in licence_text or REQUIRED_COPYRIGHT not in NOTICES.read_text():
        raise PackagingError(f"the runtime licence and THIRD_PARTY_NOTICES must both carry {REQUIRED_COPYRIGHT!r}")


def write_bundle(target: Path, files: list[tuple[Path, str, int]], licence: Path) -> None:
    with zipfile.ZipFile(target, "w") as bundle:
        for source, name, mode in files:
            add_file(bundle, source, name, mode)
        add_file(bundle, licence, "LICENSE", PLAIN_MODE)
        add_file(bundle, NOTICES, "THIRD_PARTY_NOTICES", PLAIN_MODE)


def record_checksum(out: Path, asset: str, digest: str) -> None:
    listing = out / CHECKSUMS
    lines = [line for line in (listing.read_text().splitlines() if listing.exists() else []) if not line.endswith(f"  {asset}")]
    listing.write_text("\n".join(sorted([*lines, f"{digest}  {asset}"])) + "\n")


def package(platform: str, source: Path, sha256: str, out: Path, server_name: str = "", worker_name: str = "",
            patchelf: str = "") -> Path:
    info = ReleaseInfo.read()
    if platform not in info.platforms:
        raise PackagingError(f"unknown platform {platform!r}; one of {', '.join(info.platforms)}")
    found = sha256_of(source)
    if found != sha256.lower():
        raise PackagingError(f"{source.name} has SHA-256 {found}, expected {sha256.lower()}; nothing was packaged")
    out.mkdir(parents=True, exist_ok=True)
    target = out / asset_name(platform)
    with tempfile.TemporaryDirectory(dir=out) as scratch:
        unpacked = Path(scratch) / "unpacked"
        extract(source, unpacked)
        server = one_binary(unpacked, server_name, is_server_name, "server")
        worker = one_binary(unpacked, worker_name, is_worker_name, "worker")
        licence = license_file(unpacked)
        check_licence(licence)
        if server.name.endswith(WINDOWS_SUFFIX):
            files = windows_files(server, worker)
        else:
            staging = Path(scratch) / "staging"
            staging.mkdir()
            files = linux_files(patchelf or find_patchelf(), server, worker, staging)
        partial = target.with_name(target.name + ".partial")
        write_bundle(partial, files, licence)
        partial.replace(target)
    record_checksum(out, target.name, sha256_of(target))
    return target


def find_patchelf() -> str:
    found = shutil.which("patchelf")
    if not found:
        raise PackagingError("patchelf is needed to give the Linux libraries their ArtyEngine names; install it "
                             "(apt install patchelf) or pass --patchelf")
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Repackage an official runtime build as an ArtyEngine runtime release asset")
    parser.add_argument("--platform", required=True, help="the asset's platform name, such as linux-cuda-12.8-x64")
    parser.add_argument("--source-url", default="", help="the official build archive to download")
    parser.add_argument("--source-file", type=Path, default=None, help="or an archive already downloaded")
    parser.add_argument("--sha256", required=True, help="the official archive's published SHA-256")
    parser.add_argument("--out", type=Path, required=True, help="folder for the asset and SHA256SUMS")
    parser.add_argument("--server-binary", default="", help="the server binary's name in the archive, when it can't be told")
    parser.add_argument("--worker-binary", default="", help="the rpc worker binary's name in the archive, when it can't be told")
    parser.add_argument("--patchelf", default="", help="the patchelf program to use (default: the one on PATH)")
    arguments = parser.parse_args()
    if bool(arguments.source_url) == bool(arguments.source_file):
        parser.error("give exactly one of --source-url and --source-file")
    try:
        source = arguments.source_file
        if arguments.source_url:
            arguments.out.mkdir(parents=True, exist_ok=True)
            source = arguments.out / f".{arguments.platform}.source.partial"
            fetch(arguments.source_url, source)
        target = package(arguments.platform, source, arguments.sha256, arguments.out, arguments.server_binary,
                         arguments.worker_binary, arguments.patchelf)
    except (PackagingError, OSError, zipfile.BadZipFile, tarfile.TarError) as error:
        print(f"not packaged: {error}", file=sys.stderr)
        return 1
    finally:
        if arguments.source_url:
            (arguments.out / f".{arguments.platform}.source.partial").unlink(missing_ok=True)
    print(f"{target}  sha256 {sha256_of(target)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
