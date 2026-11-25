import argparse
import hashlib
import json
import logging
import os
import shutil
import stat
import sys
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .config import load_config

log = logging.getLogger("arty-engine.runtime")

SERVER, WORKER = "server", "worker"
SERVER_BINARY, WORKER_BINARY = "arty-runtime-server", "arty-runtime-worker"
WINDOWS_SUFFIX = ".exe"
EARLIER_WORKER_NAMES = ("ggml-rpc-server", "rpc-server")
BINARY_NAMES = {
    SERVER: (SERVER_BINARY, SERVER_BINARY + WINDOWS_SUFFIX, SERVER),
    WORKER: (WORKER_BINARY, WORKER_BINARY + WINDOWS_SUFFIX, *EARLIER_WORKER_NAMES,
             *(name + WINDOWS_SUFFIX for name in EARLIER_WORKER_NAMES), WORKER),
}
EARLIER_SERVER_ENDINGS = ("-server", "-server" + WINDOWS_SUFFIX)
RELEASE_FILE = Path(__file__).with_name("runtime_release.json")
RELEASE_ASSET_URL = "https://github.com/drew-codes-things/ArtyEngine/releases/download/{release}/{asset}"
CHECKSUMS_ASSET = "SHA256SUMS"
DEFAULT_RUNTIME_DIR = "/srv/ai/arty-engine/runtime"
CHUNK_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 60
USER_AGENT = "arty-engine-runtime/0.1"
EXECUTABLE_BITS = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
PARTIAL_SUFFIX = ".partial"


class RuntimeInstallError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReleaseInfo:
    release: str
    platforms: tuple
    sha256: dict

    @staticmethod
    def read(path: Path = RELEASE_FILE) -> "ReleaseInfo":
        values = json.loads(path.read_text())
        return ReleaseInfo(release=values["release"], platforms=tuple(values["platforms"]), sha256=dict(values.get("sha256") or {}))


def asset_name(platform: str) -> str:
    return f"arty-runtime-{platform}.zip"


def release_url(release: str, asset: str) -> str:
    return RELEASE_ASSET_URL.format(release=release, asset=asset)


def is_earlier_server(path: Path) -> bool:
    return path.is_file() and path.name.endswith(EARLIER_SERVER_ENDINGS) and "rpc" not in path.name


def binary_in(folder: Path, role: str) -> Path | None:
    for name in BINARY_NAMES[role]:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    if role != SERVER:
        return None
    try:
        earlier = sorted(path for path in folder.iterdir() if is_earlier_server(path))
    except OSError:
        return None
    return earlier[0] if len(earlier) == 1 else None


def locate_binary(configured: str, role: str) -> str:
    path = Path(configured)
    if path.is_file():
        return configured
    folder = path if path.is_dir() else path.parent
    found = binary_in(folder, role) if folder.is_dir() else None
    return str(found) if found else configured


def safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts or not member.parts or ":" in member.parts[0]:
        raise RuntimeInstallError(f"the runtime zip holds an unsafe path {name!r}; nothing was installed")
    return member


def open_url(url: str):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as error:
        raise RuntimeInstallError(f"{url} answered {error.code}") from error
    except (urllib.error.URLError, OSError) as error:
        raise RuntimeInstallError(f"couldn't reach {url}: {error}") from error


def download(url: str, target: Path, opener=open_url) -> str:
    digest = hashlib.sha256()
    try:
        with opener(url) as response, open(target, "wb") as output:
            while chunk := response.read(CHUNK_BYTES):
                digest.update(chunk)
                output.write(chunk)
    except OSError as error:
        target.unlink(missing_ok=True)
        raise RuntimeInstallError(f"downloading {url} failed: {error}") from error
    return digest.hexdigest()


def listed_digest(text: str, asset: str) -> str:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].lstrip("*") == asset:
            return parts[0].lower()
    return ""


def expected_digest(info: ReleaseInfo, platform: str, given: str, custom_url: str, opener=open_url) -> str:
    if given:
        return given.lower()
    if info.sha256.get(platform):
        return info.sha256[platform].lower()
    if custom_url:
        raise RuntimeInstallError("a custom runtime_url needs --sha256, the SHA-256 of that zip")
    with opener(release_url(info.release, CHECKSUMS_ASSET)) as response:
        found = listed_digest(response.read().decode(errors="replace"), asset_name(platform))
    if not found:
        raise RuntimeInstallError(f"release {info.release} lists no SHA-256 for {asset_name(platform)}")
    log.warning("the SHA-256 of %s is not pinned in this copy of the engine; using the one release %s lists",
                asset_name(platform), info.release)
    return found


def unpack(archive: Path, folder: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        members = [(entry, safe_member(entry.filename)) for entry in bundle.infolist()]
        for entry, member in members:
            target = folder.joinpath(*member.parts)
            if entry.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, open(target, "wb") as output:
                shutil.copyfileobj(source, output, CHUNK_BYTES)
            if member.name.startswith("arty-runtime-") or entry.external_attr >> 16 & EXECUTABLE_BITS:
                target.chmod(target.stat().st_mode | EXECUTABLE_BITS)


def point_link(link: Path, target: Path) -> None:
    temporary = link.with_name(link.name + PARTIAL_SUFFIX)
    temporary.unlink(missing_ok=True)
    temporary.symlink_to(target)
    os.replace(temporary, link)


def install(platform: str, runtime_dir: Path, custom_url: str = "", sha256: str = "", replace: bool = False,
            info: ReleaseInfo | None = None, opener=open_url) -> Path:
    info = info or ReleaseInfo.read()
    if platform not in info.platforms:
        raise RuntimeInstallError(f"unknown platform {platform!r}; one of {', '.join(info.platforms)}")
    asset = asset_name(platform)
    url = custom_url or release_url(info.release, asset)
    folder = runtime_dir / f"{info.release}-{platform}"
    if folder.exists() and not replace:
        raise RuntimeInstallError(f"{folder} is already there; pass --replace to install it again")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    expected = expected_digest(info, platform, sha256, custom_url, opener)
    archive = runtime_dir / f"{asset}{PARTIAL_SUFFIX}"
    found = download(url, archive, opener)
    if found != expected:
        archive.unlink(missing_ok=True)
        raise RuntimeInstallError(f"{url} has SHA-256 {found}, expected {expected}; nothing was installed")
    staging = runtime_dir / f"{folder.name}{PARTIAL_SUFFIX}"
    shutil.rmtree(staging, ignore_errors=True)
    try:
        unpack(archive, staging)
    finally:
        archive.unlink(missing_ok=True)
    if binary_in(staging, SERVER) is None:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeInstallError(f"{url} holds no {SERVER_BINARY}; nothing was installed")
    if folder.exists():
        shutil.rmtree(folder)
    staging.rename(folder)
    if os.name != "nt":
        for role in (SERVER, WORKER):
            binary = binary_in(folder, role)
            if binary:
                point_link(runtime_dir / role, binary.relative_to(runtime_dir))
    return folder


def describe_binaries(runtime_dir: Path) -> dict:
    return {role: locate_binary(str(runtime_dir / role), role) for role in (SERVER, WORKER)}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="arty-engine runtime", description="Install or find the model runtime ArtyEngine starts")
    actions = parser.add_subparsers(dest="action", required=True)
    installing = actions.add_parser("install", help="download a runtime build from ArtyEngine's releases and point the links at it")
    installing.add_argument("--platform", required=True, help="such as linux-cuda-12.8-x64, linux-x64 or windows-cuda-12.4-x64")
    installing.add_argument("--dir", type=Path, default=Path(DEFAULT_RUNTIME_DIR), help="the runtime folder")
    installing.add_argument("--url", default="", help="a runtime zip at another address (runtime_url in engine.json)")
    installing.add_argument("--sha256", default="", help="the zip's SHA-256, needed with --url")
    installing.add_argument("--config", default=None, help="engine.json to read runtime_url from")
    installing.add_argument("--replace", action="store_true", help="install again over a folder of the same release")
    finding = actions.add_parser("find", help="print the server and worker binaries found in a runtime folder")
    finding.add_argument("--dir", type=Path, default=Path(DEFAULT_RUNTIME_DIR))
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stdout)
    if arguments.action == "find":
        print(json.dumps(describe_binaries(arguments.dir), indent=2))
        return 0
    custom_url = arguments.url
    if not custom_url and arguments.config:
        custom_url = load_config(arguments.config).runtime_url
    try:
        folder = install(arguments.platform, arguments.dir, custom_url, arguments.sha256, arguments.replace)
    except (RuntimeInstallError, OSError, zipfile.BadZipFile) as error:
        print(f"runtime not installed: {error}", file=sys.stderr)
        return 1
    print(f"installed {folder}")
    print(json.dumps(describe_binaries(arguments.dir), indent=2))
    return 0
