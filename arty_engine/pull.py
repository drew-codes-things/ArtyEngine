import hashlib
import http.client
import json
import logging
import os
import re
import shlex
import shutil
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from pathlib import Path

from .gguf import GgufError
from .store import (LOCAL_HOST, MODEL_LAYER, PARAMS_LAYER, PROJECTOR_LAYER, SYSTEM_LAYER, TEMPLATE_LAYER, ModelName,
                    ModelNotFound, ModelStore, blob_path, engine_manifest, is_digest, named_digests, parse_name,
                    write_manifest)

log = logging.getLogger("arty-engine.pull")
HF_HOSTS = ("hf.co", "huggingface.co")
WEIGHT_LAYERS = (MODEL_LAYER, PROJECTOR_LAYER)
SETTING_LAYER_WORDS = {TEMPLATE_LAYER: "template", SYSTEM_LAYER: "system prompt", PARAMS_LAYER: "parameters"}
CLEANUP_TOOL = Path(__file__).resolve().parents[1] / "tools" / "unused_blobs.py"
TOKEN_HOSTS = ("huggingface.co",)
HF_API = "https://huggingface.co/api/models/{repo}?blobs=true"
HF_FILE = "https://huggingface.co/{repo}/resolve/{revision}/{file}"
DEFAULT_REVISION = "main"
REVISION = re.compile(r"[0-9a-f]{40}")
SHA256_HEX = re.compile(r"[0-9a-f]{64}")
CHUNK_BYTES = 256 * 1024
REQUEST_TIMEOUT_SECONDS = 60
SPLIT_FILE = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
USER_AGENT = "arty-engine/0.1"
PROJECTOR_PREFERENCE = ("f16", "bf16", "q8_0", "")
HF_SOURCE = "Hugging Face"
HF_NAME_FORM = "hf.co/<owner>/<repo>:<quant>"
SHORT_NAMES_FILE = Path(__file__).with_name("short_names.json")
RANGE_NOT_SATISFIABLE = 416
DOWNLOAD_ATTEMPTS = 2
BYTES_PER_GIB = 1024 ** 3
BYTES_PER_MIB = 1024 ** 2


@dataclass(frozen=True)
class PullPlan:
    source: ModelName | None
    target: ModelName
    pins: "ShortName | None"
    replaces: bool = False
    installed_at: Path | None = None


@dataclass(frozen=True)
class ShortName:
    source: str
    model_sha256: str = ""
    projector_sha256: str = ""


class PullError(RuntimeError):
    def __init__(self, message: str, status: int = HTTPStatus.BAD_GATEWAY):
        super().__init__(message)
        self.status = int(status)


class RateLimiter:
    def __init__(self, bytes_per_second: int, clock=time.monotonic, sleep=time.sleep):
        self.rate = bytes_per_second
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()
        self.started = None
        self.sent = 0

    def take(self, count: int) -> None:
        if self.rate <= 0:
            return
        with self.lock:
            if self.started is None:
                self.started = self.clock()
            self.sent += count
            ahead = self.sent / self.rate - (self.clock() - self.started)
        if ahead > 0:
            self.sleep(ahead)


def _open(url: str, headers: dict | None = None):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    token = os.environ.get("HF_TOKEN", "")
    if token and urllib.parse.urlsplit(url).hostname in TOKEN_HOSTS:
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)


def _refusal_of(error: urllib.error.HTTPError, url: str, source: str) -> PullError:
    if error.code == HTTPStatus.NOT_FOUND:
        return PullError(f"{source} has no such model: {url} answered 404", HTTPStatus.NOT_FOUND)
    if error.code == HTTPStatus.UNAUTHORIZED and not os.environ.get("HF_TOKEN"):
        return PullError(f"{source} has no public model at {url} (it answered 401): the repo is missing, private or gated; "
                         "for a private or gated repo set HF_TOKEN for the engine", HTTPStatus.NOT_FOUND)
    if error.code == HTTPStatus.UNAUTHORIZED:
        return PullError(f"{source} refused the engine's HF_TOKEN for {url} (it answered 401)")
    return PullError(f"{source} answered {error.code} for {url}")


def _get_json(url: str, source: str):
    try:
        with _open(url) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise _refusal_of(error, url, source) from error
    except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
        raise PullError(f"couldn't read {url} from {source}: {error!r:.200}") from error
    except (ValueError, RecursionError) as error:
        raise PullError(f"{source} sent {url} as something other than JSON the engine can read") from error


def checked_digest(digest, source: str) -> str:
    if not is_digest(digest):
        raise PullError(f"{source} sent the blob digest {digest!r:.80}, which isn't sha256: and 64 hex digits; nothing was downloaded")
    return digest


def checked_size(size, source: str) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise PullError(f"{source} sent the blob size {size!r:.40}, which isn't a whole number of bytes")
    return size


def is_readable_sibling(entry) -> bool:
    return isinstance(entry, dict) and isinstance(entry.get("rfilename", ""), str) and isinstance(entry.get("lfs") or {}, dict)


def hf_siblings(info) -> list[dict]:
    if not isinstance(info, dict):
        raise PullError(f"{HF_SOURCE} sent a file list that isn't a JSON object")
    siblings = info.get("siblings") or []
    if not isinstance(siblings, list) or not all(is_readable_sibling(entry) for entry in siblings):
        raise PullError(f"{HF_SOURCE} sent a file list the engine can't read")
    return siblings


def choose_hf_files(siblings: list[dict], quant: str) -> tuple[dict, dict | None]:
    ggufs = [entry for entry in siblings if entry.get("rfilename", "").lower().endswith(".gguf")]
    models = [entry for entry in ggufs if "mmproj" not in entry["rfilename"].lower()]
    projectors = [entry for entry in ggufs if "mmproj" in entry["rfilename"].lower()]
    wanted = (quant or "Q4_K_M").lower()
    matches = [entry for entry in models if wanted in Path(entry["rfilename"]).name.lower()]
    if not matches:
        names = ", ".join(Path(entry["rfilename"]).name for entry in models) or "none"
        raise PullError(f"no GGUF for {quant or 'Q4_K_M'} in that repo (it has: {names})", HTTPStatus.NOT_FOUND)
    single = [entry for entry in matches if not SPLIT_FILE.search(entry["rfilename"])]
    if not single:
        raise PullError("that quant is split into several files, which the engine can't pull yet", HTTPStatus.NOT_IMPLEMENTED)
    chosen = min(single, key=lambda entry: len(entry["rfilename"]))
    projector = None
    for preference in PROJECTOR_PREFERENCE:
        projector = next((entry for entry in projectors if names_quant(entry["rfilename"], preference)), None)
        if projector:
            break
    return chosen, projector


def names_quant(file_name: str, quant: str) -> bool:
    if not quant:
        return True
    return re.search(rf"(?<![a-z0-9]){re.escape(quant)}(?![a-z0-9])", Path(file_name).name.lower()) is not None


def load_short_names(path: Path = SHORT_NAMES_FILE) -> dict[str, ShortName]:
    listed = json.loads(Path(path).read_text())
    names = {}
    for short, entry in listed.items():
        source = entry.get("source", "") if isinstance(entry, dict) else ""
        parsed_short, parsed_source = parse_name(short), parse_name(source)
        if parsed_short.host != LOCAL_HOST or parsed_source.host not in HF_HOSTS:
            raise ValueError(f"{path}: {short} must be a short name for a Hugging Face model ({HF_NAME_FORM}), not {source}")
        pins = [entry.get("model_sha256", ""), entry.get("projector_sha256", "")]
        if not SHA256_HEX.fullmatch(pins[0]) or (pins[1] and not SHA256_HEX.fullmatch(pins[1])):
            raise ValueError(f"{path}: {short} needs model_sha256 (and projector_sha256 if it has one) as 64 hex digits")
        names[parsed_short.display] = ShortName(parsed_source.display, *pins)
    return names


class Puller:
    def __init__(self, store_root: Path, rate_bytes: int, short_names: dict[str, ShortName] | None = None,
                 store: ModelStore | None = None, keep_free_bytes: int = 0):
        self.root = Path(store_root)
        self.limiter = RateLimiter(rate_bytes)
        self.short_names = load_short_names() if short_names is None else short_names
        self.store = store
        self.keep_free_bytes = keep_free_bytes

    def check(self, name: str, source: str | None = None) -> None:
        self._plan(name, source)

    def pull(self, name: str, progress, source: str | None = None) -> bool:
        plan = self._plan(name, source)
        if plan.source is None:
            if source is not None:
                progress({"status": f"{plan.target.display} is already installed from {source}"})
            progress({"status": "success"})
            return False
        old_manifest = installed_manifest(plan.installed_at, plan.target) if plan.replaces else None
        self._pull_hf(plan.source, plan.target, progress, plan.pins)
        replaced = plan.replaces and self._report_replace(plan, old_manifest, progress)
        progress({"status": "success"})
        return replaced

    def _report_replace(self, plan: PullPlan, old_manifest, progress) -> bool:
        target, source = plan.target.display, plan.source.display
        old_layers = manifest_layers(old_manifest)
        if plan.installed_at != self.root:
            log.warning("pulled %s from %s into %s; the copy in %s is no longer used", target, source, self.root, plan.installed_at)
            progress({"status": f"installed {target} from {source} in {self.root}; the copy in {plan.installed_at} is left "
                                f"as it is and no longer used"})
            return True
        if weight_digests(old_layers) == weight_digests(manifest_layers(installed_manifest(self.root, plan.target))):
            old_manifest["source"] = installed_manifest(self.root, plan.target)["source"]
            write_manifest(self.root, plan.target, old_manifest)
            progress({"status": f"{target} already had the files of {source}; only its recorded source changed"})
            return False
        old_source = installed_source_of(old_manifest) or "no recorded source"
        log.warning("replaced %s: it was %s, it is now %s", target, old_source, source)
        dropped = [word for media_type, word in SETTING_LAYER_WORDS.items() if any(layer.get("mediaType") == media_type
                                                                                   for layer in old_layers)]
        not_carried = f"; its {', '.join(dropped)} were not carried over" if dropped else ""
        kept_bytes = old_copy_bytes(old_layers, named_digests(self.root))
        progress({"status": f"replaced {target} with {source}{not_carried}; the old copy's files ({readable_size(kept_bytes)}) stay "
                            f"in the store. To free them, stop the engine, then run "
                            f"{cleanup_command(self.root)} (it removes every blob no model names, not only these; run it "
                            f"without --remove first to see the list)"})
        return True

    def _plan(self, name: str, source: str | None) -> PullPlan:
        if source is None:
            return PullPlan(*self._source_of(name))
        chosen = parse_name(source)
        target = parse_name(name)
        if chosen.host not in HF_HOSTS:
            raise PullError(f"a source must be a Hugging Face name ({HF_NAME_FORM}), not {source}", HTTPStatus.BAD_REQUEST)
        if target.host != LOCAL_HOST and target != chosen:
            raise PullError(f"{target.display} is a full name; install a source under a short name (such as qwen2.5vl:7b), "
                            f"or pull {chosen.display} itself", HTTPStatus.BAD_REQUEST)
        short = self.short_names.get(target.display)
        pins = short if short and same_source(short.source, chosen) else None
        installed_at = self._installed_root(target)
        if installed_at is None:
            return PullPlan(chosen, target, pins)
        if same_source(installed_source(installed_at, target), chosen):
            return PullPlan(None, target, None)
        return PullPlan(chosen, target, pins, replaces=True, installed_at=installed_at)

    def _installed_root(self, target: ModelName) -> Path | None:
        roots = self.store.roots if self.store is not None else [self.root]
        return next((root for root in roots if target.manifest_path(root).is_file()), None)

    def _source_of(self, name: str) -> tuple[ModelName | None, ModelName, ShortName | None]:
        target = parse_name(name)
        if target.host in HF_HOSTS:
            return target, target, None
        if target.host != LOCAL_HOST:
            raise PullError(f"the engine pulls from Hugging Face only: name the model as {HF_NAME_FORM}, not {name}",
                            HTTPStatus.BAD_REQUEST)
        short = self.short_names.get(target.display)
        if target.manifest_path(self.root).is_file():
            self._check_installed(target, short)
            return None, target, None
        if not short:
            known = ", ".join(sorted(self.short_names)) or "none"
            raise PullError(f"{target.display} is not one of the engine's short names ({known}); name the model as "
                            f"{HF_NAME_FORM}", HTTPStatus.BAD_REQUEST)
        return parse_name(short.source), target, short

    def _check_installed(self, target: ModelName, short: ShortName | None) -> None:
        if self.store is None:
            return
        try:
            model = self.store.get(target.display)
        except (ModelNotFound, GgufError, OSError, ValueError, RecursionError) as error:
            problem = f"can't be read ({error})"
        else:
            if model.runnable:
                return
            problem = f"is not supported ({model.unrunnable_reason})"
        again = f" to get {short.source}" if short else ""
        raise PullError(f"{target.display} is installed but {problem}. A pull doesn't replace an installed model: delete it "
                        f"first (DELETE /api/delete), then pull {target.display} again{again}", HTTPStatus.CONFLICT)

    def _pull_hf(self, parsed: ModelName, target: ModelName, progress, pins: ShortName | None = None) -> None:
        repo = f"{parsed.namespace}/{parsed.model}"
        quant = "" if parsed.tag == "latest" else parsed.tag
        progress({"status": "pulling manifest"})
        info = _get_json(HF_API.format(repo=repo), HF_SOURCE)
        chosen, projector = choose_hf_files(hf_siblings(info), quant)
        if pins and pins.model_sha256 and not pins.projector_sha256:
            projector = None
        revision = info.get("sha") if REVISION.fullmatch(str(info.get("sha", ""))) else DEFAULT_REVISION
        files = [self._listed_file(entry, media_type) for entry, media_type in ((chosen, MODEL_LAYER), (projector, PROJECTOR_LAYER))
                 if entry]
        if pins:
            check_pins(files, pins, target)
        self._check_free_space(files, target)
        layers = []
        for entry, media_type, digest, size in files:
            url = HF_FILE.format(repo=repo, revision=revision, file=urllib.parse.quote(entry["rfilename"]))
            self._download(url, digest, size, progress)
            layers.append({"mediaType": media_type, "digest": digest, "size": size, "from": entry["rfilename"]})
        progress({"status": "writing manifest"})
        manifest = engine_manifest(layers)
        manifest["source"] = {"name": parsed.display, "revision": revision}
        write_manifest(self.root, target, manifest)

    @staticmethod
    def _listed_file(entry: dict, media_type: str) -> tuple[dict, str, str, int]:
        lfs = entry.get("lfs") or {}
        if not lfs.get("sha256"):
            raise PullError(f"{entry['rfilename']} has no sha256 on Hugging Face, so it can't be verified")
        digest = checked_digest(f"sha256:{lfs['sha256']}", HF_SOURCE)
        size = checked_size(lfs.get("size") or 0, HF_SOURCE) or checked_size(entry.get("size") or 0, HF_SOURCE)
        if not size:
            raise PullError(f"{entry['rfilename']} is listed on Hugging Face without a size, so its download can't be limited "
                            f"or its space checked; nothing was downloaded")
        return entry, media_type, digest, size

    def _check_free_space(self, files: list, target: ModelName) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        needed = sum(size - partial_size(blob_path(self.root, digest)) for _, _, digest, size in files
                     if not blob_path(self.root, digest).exists())
        free = shutil.disk_usage(self.root).free
        if free - needed < self.keep_free_bytes:
            raise PullError(f"pulling {target.display} needs {needed / BYTES_PER_GIB:.1f} GiB and would leave less than "
                            f"{self.keep_free_bytes / BYTES_PER_GIB:.1f} GiB free on the store's disk "
                            f"({free / BYTES_PER_GIB:.1f} GiB free now); nothing was downloaded", HTTPStatus.INSUFFICIENT_STORAGE)

    def _download(self, url: str, digest: str, size: int, progress) -> None:
        if not is_digest(digest):
            raise PullError(f"refusing to download {digest!r}: a blob digest must be sha256: and 64 hex digits")
        target = blob_path(self.root, digest)
        if target.exists():
            progress({"status": f"pulling {digest[7:19]}", "digest": digest, "total": size, "completed": size})
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = partial_path(target)
        if size and partial.exists() and partial.stat().st_size >= size:
            if partial.stat().st_size == size and f"sha256:{sha256_of(partial)}" == digest:
                partial.rename(target)
                return
            partial.unlink()
        for attempt in range(DOWNLOAD_ATTEMPTS):
            try:
                hasher = self._fetch(url, digest, size, partial, progress)
                break
            except urllib.error.HTTPError as error:
                if error.code != RANGE_NOT_SATISFIABLE or attempt + 1 == DOWNLOAD_ATTEMPTS or not partial.exists():
                    raise _refusal_of(error, url, HF_SOURCE) from error
                partial.unlink()
        progress({"status": "verifying sha256 digest"})
        if f"sha256:{hasher.hexdigest()}" != digest:
            partial.unlink(missing_ok=True)
            raise PullError(f"{digest} failed its sha256 check; the partial file was removed")
        keep_download(partial, target, digest)

    def _fetch(self, url: str, digest: str, size: int, partial: Path, progress):
        have = partial.stat().st_size if partial.exists() else 0
        hasher = hashlib.sha256()
        if have:
            with open(partial, "rb") as existing:
                while chunk := existing.read(CHUNK_BYTES):
                    hasher.update(chunk)
        headers = {"Range": f"bytes={have}-"} if have else {}
        too_long = False
        try:
            with _open(url, headers) as response, open(partial, "ab" if have and response.status == 206 else "wb") as output:
                if have and response.status != 206:
                    hasher = hashlib.sha256()
                    have = 0
                last_report = 0.0
                while chunk := response.read(CHUNK_BYTES):
                    if size and have + len(chunk) > size:
                        too_long = True
                        break
                    self.limiter.take(len(chunk))
                    output.write(chunk)
                    hasher.update(chunk)
                    have += len(chunk)
                    if time.monotonic() - last_report > 1:
                        progress({"status": f"pulling {digest[7:19]}", "digest": digest, "total": size, "completed": have})
                        last_report = time.monotonic()
        except urllib.error.HTTPError:
            raise
        except (urllib.error.URLError, OSError, http.client.HTTPException) as error:
            raise PullError(f"download of {digest} stopped at {have} bytes: {error!r:.200}; pull again to resume") from error
        if too_long:
            partial.unlink()
            raise PullError(f"{HF_SOURCE} sent more than the {size} bytes it listed for {digest}; the partial file was removed")
        return hasher


def installed_manifest(root: Path, target: ModelName):
    try:
        return json.loads(target.manifest_path(root).read_text())
    except (OSError, ValueError, RecursionError):
        return None


def installed_source(root: Path, target: ModelName) -> str | None:
    return installed_source_of(installed_manifest(root, target))


def installed_source_of(manifest) -> str | None:
    source = manifest.get("source") if isinstance(manifest, dict) else None
    return source.get("name") if isinstance(source, dict) else None


def manifest_layers(manifest) -> list[dict]:
    layers = manifest.get("layers") if isinstance(manifest, dict) else None
    return [layer for layer in layers if isinstance(layer, dict)] if isinstance(layers, list) else []


def weight_digests(layers: list[dict]) -> set[str]:
    return {layer.get("digest") for layer in layers if layer.get("mediaType") in WEIGHT_LAYERS}


def old_copy_bytes(old_layers: list[dict], still_named: set[str]) -> int:
    unnamed = {layer.get("digest"): int(layer.get("size") or 0) for layer in old_layers if layer.get("digest") not in still_named}
    return sum(unnamed.values())


def readable_size(size: int) -> str:
    if size >= BYTES_PER_GIB:
        return f"{size / BYTES_PER_GIB:.1f} GiB"
    if size >= BYTES_PER_MIB:
        return f"{size / BYTES_PER_MIB:.1f} MiB"
    return f"{size} bytes"


def cleanup_command(root: Path) -> str:
    return f"python3 {shlex.quote(str(CLEANUP_TOOL))} --store {shlex.quote(str(root))} --remove --remove-unnamed-blobs"


def source_key(name: ModelName) -> tuple[str, str, str]:
    return name.namespace.lower(), name.model.lower(), name.tag.lower()


def same_source(recorded: str | None, chosen: ModelName) -> bool:
    if not recorded:
        return False
    try:
        parsed = parse_name(recorded)
    except ModelNotFound:
        return False
    return parsed.host in HF_HOSTS and source_key(parsed) == source_key(chosen)


def keep_download(partial: Path, target: Path, digest: str) -> None:
    try:
        partial.rename(target)
    except FileNotFoundError as error:
        raise PullError(f"the download of {digest} disappeared from the store before it could be kept (was the store "
                        f"cleaned during the pull?); pull again") from error


def partial_path(blob: Path) -> Path:
    return blob.with_name(blob.name + "-partial")


def partial_size(blob: Path) -> int:
    partial = partial_path(blob)
    return partial.stat().st_size if partial.exists() else 0


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def check_pins(files: list, pins: ShortName, target: ModelName) -> None:
    expected = {MODEL_LAYER: pins.model_sha256, PROJECTOR_LAYER: pins.projector_sha256}
    listed = {media_type: (entry["rfilename"], digest[7:]) for entry, media_type, digest, _ in files}
    for media_type, sha in expected.items():
        if not sha:
            continue
        name, found = listed.get(media_type, ("(no such file)", ""))
        if found != sha:
            raise PullError(f"{HF_SOURCE} changed {name} since the engine's short name {target.display} was checked "
                            f"(sha256 {found or 'missing'}, expected {sha}); nothing was downloaded")
