import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http import HTTPStatus
from pathlib import Path

from .store import DEFAULT_HOST, LEGACY_HOST, MODEL_LAYER, PROJECTOR_LAYER, ModelName, blob_path, is_digest, parse_name

HF_HOSTS = ("hf.co", "huggingface.co")
TOKEN_HOSTS = ("huggingface.co",)
HF_API = "https://huggingface.co/api/models/{repo}?blobs=true"
HF_FILE = "https://huggingface.co/{repo}/resolve/main/{file}"
OLLAMA_REGISTRY = "https://registry.ollama.ai/v2/{namespace}/{model}/{kind}/{reference}"
MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
CHUNK_BYTES = 256 * 1024
REQUEST_TIMEOUT_SECONDS = 60
SPLIT_FILE = re.compile(r"-\d{5}-of-\d{5}\.gguf$")
USER_AGENT = "arty-engine/0.1"
PROJECTOR_PREFERENCE = ("f16", "bf16", "q8_0", "")
OLLAMA_SOURCE = "the Ollama registry"
HF_SOURCE = "Hugging Face"


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


def _get_json(url: str, source: str, headers: dict | None = None):
    try:
        with _open(url, headers) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        if error.code == HTTPStatus.NOT_FOUND:
            raise PullError(f"{source} has no such model: {url} answered 404", HTTPStatus.NOT_FOUND) from error
        raise PullError(f"{source} answered {error.code} for {url}") from error
    except (urllib.error.URLError, OSError) as error:
        raise PullError(f"couldn't read {url} from {source}: {error}") from error
    except ValueError as error:
        raise PullError(f"{source} sent {url} as something other than JSON") from error


def checked_digest(digest, source: str) -> str:
    if not is_digest(digest):
        raise PullError(f"{source} sent the blob digest {digest!r:.80}, which isn't sha256: and 64 hex digits; nothing was downloaded")
    return digest


def checked_size(size, source: str) -> int:
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise PullError(f"{source} sent the blob size {size!r:.40}, which isn't a whole number of bytes")
    return size


def registry_blobs(manifest) -> list[tuple[str, int]]:
    if not isinstance(manifest, dict):
        raise PullError(f"{OLLAMA_SOURCE} sent a manifest that isn't a JSON object")
    layers, config = manifest.get("layers", []), manifest.get("config", {})
    if not isinstance(layers, list) or not isinstance(config, dict) or not all(isinstance(layer, dict) for layer in layers):
        raise PullError(f"{OLLAMA_SOURCE} sent a manifest whose layers aren't a list of objects")
    return [(checked_digest(layer["digest"], OLLAMA_SOURCE), checked_size(layer.get("size", 0), OLLAMA_SOURCE))
            for layer in [*layers, config] if layer.get("digest")]


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
        projector = next((entry for entry in projectors if preference in entry["rfilename"].lower()), None)
        if projector:
            break
    return chosen, projector


class Puller:
    def __init__(self, store_root: Path, rate_bytes: int):
        self.root = Path(store_root)
        self.limiter = RateLimiter(rate_bytes)

    def pull(self, name: str, progress) -> None:
        parsed = parse_name(name)
        if parsed.host in HF_HOSTS:
            self._pull_hf(parsed, progress)
        elif parsed.host in (DEFAULT_HOST, LEGACY_HOST):
            self._pull_ollama(parsed, progress)
        else:
            raise PullError(f"don't know how to pull from {parsed.host}", HTTPStatus.BAD_REQUEST)
        progress({"status": "success"})

    def _pull_hf(self, parsed: ModelName, progress) -> None:
        repo = f"{parsed.namespace}/{parsed.model}"
        quant = "" if parsed.tag == "latest" else parsed.tag
        progress({"status": "pulling manifest"})
        info = _get_json(HF_API.format(repo=repo), HF_SOURCE)
        chosen, projector = choose_hf_files(hf_siblings(info), quant)
        layers = []
        for entry, media_type in ((chosen, MODEL_LAYER), (projector, PROJECTOR_LAYER)):
            if not entry:
                continue
            lfs = entry.get("lfs") or {}
            if not lfs.get("sha256"):
                raise PullError(f"{entry['rfilename']} has no sha256 on Hugging Face, so it can't be verified")
            digest = checked_digest(f"sha256:{lfs['sha256']}", HF_SOURCE)
            listed_size = checked_size(lfs.get("size") or 0, HF_SOURCE)
            url = HF_FILE.format(repo=repo, file=urllib.parse.quote(entry["rfilename"]))
            self._download(url, digest, listed_size or checked_size(entry.get("size") or 0, HF_SOURCE), progress)
            layers.append({"mediaType": media_type, "digest": digest, "size": listed_size, "from": entry["rfilename"]})
        manifest = {"schemaVersion": 2, "mediaType": MANIFEST_ACCEPT, "config": {"mediaType": "application/vnd.docker.container.image.v1+json",
                    "digest": "", "size": 0}, "layers": layers, "runner": "llamacpp", "format": "gguf"}
        self._write_manifest(parsed, manifest, progress)

    def _pull_ollama(self, parsed: ModelName, progress) -> None:
        progress({"status": "pulling manifest"})
        url = OLLAMA_REGISTRY.format(namespace=parsed.namespace, model=parsed.model, kind="manifests", reference=parsed.tag)
        manifest = _get_json(url, OLLAMA_SOURCE, {"Accept": MANIFEST_ACCEPT})
        for digest, size in registry_blobs(manifest):
            blob_url = OLLAMA_REGISTRY.format(namespace=parsed.namespace, model=parsed.model, kind="blobs", reference=digest)
            self._download(blob_url, digest, size, progress)
        self._write_manifest(parsed, manifest, progress)

    def _download(self, url: str, digest: str, size: int, progress) -> None:
        if not is_digest(digest):
            raise PullError(f"refusing to download {digest!r}: a blob digest must be sha256: and 64 hex digits")
        target = blob_path(self.root, digest)
        if target.exists():
            progress({"status": f"pulling {digest[7:19]}", "digest": digest, "total": size, "completed": size})
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + "-partial")
        have = partial.stat().st_size if partial.exists() else 0
        hasher = hashlib.sha256()
        if have:
            with open(partial, "rb") as existing:
                while chunk := existing.read(CHUNK_BYTES):
                    hasher.update(chunk)
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with _open(url, headers) as response, open(partial, "ab" if have and response.status == 206 else "wb") as output:
                if have and response.status != 206:
                    hasher = hashlib.sha256()
                    have = 0
                last_report = 0.0
                while chunk := response.read(CHUNK_BYTES):
                    self.limiter.take(len(chunk))
                    output.write(chunk)
                    hasher.update(chunk)
                    have += len(chunk)
                    if time.monotonic() - last_report > 1:
                        progress({"status": f"pulling {digest[7:19]}", "digest": digest, "total": size, "completed": have})
                        last_report = time.monotonic()
        except urllib.error.HTTPError as error:
            raise PullError(f"{url} answered {error.code}") from error
        except (urllib.error.URLError, OSError) as error:
            raise PullError(f"download of {digest} stopped at {have} bytes: {error}; pull again to resume") from error
        progress({"status": "verifying sha256 digest"})
        if f"sha256:{hasher.hexdigest()}" != digest:
            partial.unlink()
            raise PullError(f"{digest} failed its sha256 check; the partial file was removed")
        partial.rename(target)

    def _write_manifest(self, parsed: ModelName, manifest: dict, progress) -> None:
        progress({"status": "writing manifest"})
        path = self.root / "manifests-v2" / parsed.host / parsed.namespace / parsed.model / parsed.tag
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(manifest))
        temporary.replace(path)
