import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .gguf import GgufError, GgufInfo, plain_metadata, read_gguf

DEFAULT_HOST = "ollama.com"
LEGACY_HOST = "registry.ollama.ai"
DEFAULT_NAMESPACE = "library"
DEFAULT_TAG = "latest"
MANIFEST_DIRS = ("manifests-v2", "manifests")
MODEL_LAYER = "application/vnd.ollama.image.model"
PROJECTOR_LAYER = "application/vnd.ollama.image.projector"
TEMPLATE_LAYER = "application/vnd.ollama.image.template"
SYSTEM_LAYER = "application/vnd.ollama.image.system"
PARAMS_LAYER = "application/vnd.ollama.image.params"
LICENSE_LAYER = "application/vnd.ollama.image.license"
MANIFEST_LIST = "application/vnd.ollama.manifest.list.v2+json"
PREFERRED_RUNNERS = ("llamacpp", "ggml", "")
EMBEDDING_ARCHITECTURES = {"bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3", "modern-bert", "neo-bert"}
OLLAMA_ONLY_ARCHITECTURES = {"qwen25vl"}
LLAMA_CPP_BUILDS = {"qwen25vl": "hf.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M"}
MILLION = 1_000_000
BILLION = 1_000_000_000
TEN_BILLION = 10 * BILLION
PATH_STEPS = {".", ".."}
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*(:[A-Za-z0-9._-]+)?$")
FILE_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
    13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS",
    23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS",
    31: "IQ1_M", 32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4",
}


class ModelNotFound(LookupError):
    pass


class NotOwned(PermissionError):
    pass


@dataclass(frozen=True)
class ModelName:
    host: str
    namespace: str
    model: str
    tag: str

    @property
    def display(self) -> str:
        if self.host in (DEFAULT_HOST, LEGACY_HOST) and self.namespace == DEFAULT_NAMESPACE:
            return f"{self.model}:{self.tag}"
        return f"{self.host}/{self.namespace}/{self.model}:{self.tag}"

    def manifest_paths(self, root: Path) -> list[Path]:
        hosts = (DEFAULT_HOST, LEGACY_HOST) if self.host in (DEFAULT_HOST, LEGACY_HOST) else (self.host,)
        return [root / folder / host / self.namespace / self.model / self.tag for folder in MANIFEST_DIRS for host in hosts]


def parse_name(name: str) -> ModelName:
    name = (name or "").strip()
    if not NAME_PATTERN.match(name):
        raise ModelNotFound(f"invalid model name {name!r}")
    base, _, tag = name.partition(":")
    parts = base.split("/")
    if any(part in PATH_STEPS for part in (*parts, tag)):
        raise ModelNotFound(f"invalid model name {name!r}")
    if len(parts) == 1:
        return ModelName(DEFAULT_HOST, DEFAULT_NAMESPACE, parts[0].lower(), tag or DEFAULT_TAG)
    if len(parts) == 2:
        return ModelName(DEFAULT_HOST, parts[0].lower(), parts[1].lower(), tag or DEFAULT_TAG)
    if len(parts) == 3:
        return ModelName(parts[0].lower(), parts[1], parts[2], tag or DEFAULT_TAG)
    raise ModelNotFound(f"invalid model name {name!r}")


@dataclass
class Model:
    name: str
    root: Path
    digest: str
    manifest: dict
    modified_at: datetime
    size: int
    model_path: Path
    projector_path: Path | None
    template_digest: str = ""
    gguf: GgufInfo | None = None
    params: dict = field(default_factory=dict)
    system: str = ""
    license: str = ""
    ollama_template: str = ""
    ollama_config: dict = field(default_factory=dict)

    @property
    def architecture(self) -> str:
        return self.gguf.architecture if self.gguf else ""

    @property
    def is_embedding(self) -> bool:
        return self.architecture in EMBEDDING_ARCHITECTURES or self.gguf.arch_value("pooling_type") is not None

    @property
    def runnable(self) -> bool:
        return self.architecture not in OLLAMA_ONLY_ARCHITECTURES and not (self.gguf.has_inline_vision and not self.projector_path)

    @property
    def unrunnable_reason(self) -> str:
        if self.runnable:
            return ""
        reason = (f"{self.name} is stored in Ollama's own engine format ({self.architecture} with the vision weights inside "
                  "the model file); llama.cpp needs the text model and a separate projector")
        build = LLAMA_CPP_BUILDS.get(self.architecture)
        return f"{reason}. Use a llama.cpp build with its projector instead, such as {build} (ai-pull {build})" if build else reason

    @property
    def capabilities(self) -> list[str]:
        if self.is_embedding:
            return ["embedding"]
        found = ["completion"]
        if self.projector_path or self.gguf.has_inline_vision:
            found.append("vision")
        template = self.gguf.chat_template or self.ollama_template
        if "tools" in template or ".Tools" in template:
            found.append("tools")
        if "think" in template or ".Think" in template:
            found.append("thinking")
        return found

    @property
    def context_length(self) -> int:
        return int(self.gguf.arch_value("context_length", 0) or 0)

    @property
    def details(self) -> dict:
        metadata = self.gguf.metadata
        file_type = metadata.get("general.file_type")
        model_layer = next((layer for layer in self.manifest.get("layers", []) if layer.get("mediaType") == MODEL_LAYER), {})
        return {
            "parent_model": model_layer.get("from", "") if "/" in str(model_layer.get("from", "")) else "",
            "format": "gguf",
            "family": self.architecture,
            "families": [self.architecture],
            "parameter_size": self.ollama_config.get("model_type") or parameter_size(self.gguf.parameter_count),
            "quantization_level": FILE_TYPE_NAMES.get(file_type, str(file_type) if file_type is not None else ""),
        }

    def describe(self) -> dict:
        return {
            "name": self.name, "model": self.name, "modified_at": iso_time(self.modified_at), "size": self.size,
            "digest": self.digest, "details": self.details, "capabilities": self.capabilities,
        }


def iso_time(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parameter_size(count: int) -> str:
    if count < BILLION:
        return f"{count / MILLION:.0f}M"
    if count < TEN_BILLION:
        return f"{count / BILLION:.2f}B"
    return f"{count / BILLION:.1f}B"


def _read_json(path: Path) -> dict:
    with open(path) as handle:
        return json.load(handle)


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_digest(value) -> bool:
    return isinstance(value, str) and DIGEST_PATTERN.fullmatch(value) is not None


def blob_path(root: Path, digest: str) -> Path:
    if not is_digest(digest):
        raise ModelNotFound(f"invalid blob digest {digest!r}")
    return root / "blobs" / digest.replace(":", "-")


class ModelStore:
    def __init__(self, roots: list[Path]):
        self.roots = [Path(root) for root in roots]
        self._gguf_cache: dict[tuple, GgufInfo] = {}

    def _gguf(self, path: Path) -> GgufInfo:
        stat = path.stat()
        key = (str(path), stat.st_size, stat.st_mtime_ns)
        if key not in self._gguf_cache:
            self._gguf_cache[key] = read_gguf(path)
        return self._gguf_cache[key]

    def _locate(self, name: ModelName) -> tuple[Path, Path] | None:
        for root in self.roots:
            for path in name.manifest_paths(root):
                if path.is_file():
                    return root, path
        return None

    def _resolve_manifest(self, root: Path, path: Path) -> tuple[dict, str]:
        manifest = _read_json(path)
        digest = path.resolve().name.removeprefix("sha256-") if path.is_symlink() else _sha256_of(path)
        if manifest.get("mediaType") != MANIFEST_LIST:
            return manifest, digest
        entries = [entry for entry in manifest.get("manifests", []) if entry.get("format") == "gguf"]
        entries.sort(key=lambda entry: PREFERRED_RUNNERS.index(entry.get("runner", "")) if entry.get("runner", "") in PREFERRED_RUNNERS
                     else len(PREFERRED_RUNNERS))
        if not entries:
            raise ModelNotFound(f"{path} lists no GGUF build")
        chosen = entries[0]["digest"]
        return _read_json(blob_path(root, chosen)), chosen.removeprefix("sha256:")

    def get(self, name: str) -> Model:
        parsed = parse_name(name)
        found = self._locate(parsed)
        if not found:
            raise ModelNotFound(f"model '{name}' not found")
        root, path = found
        manifest, digest = self._resolve_manifest(root, path)
        layers = {layer.get("mediaType"): layer for layer in manifest.get("layers", [])}
        if MODEL_LAYER not in layers:
            raise ModelNotFound(f"model '{name}' has no model layer")
        model_path = blob_path(root, layers[MODEL_LAYER]["digest"])
        projector = layers.get(PROJECTOR_LAYER)
        model = Model(
            name=parsed.display, root=root, digest=digest, manifest=manifest,
            modified_at=datetime.fromtimestamp(path.lstat().st_mtime, tz=timezone.utc),
            size=sum(int(layer.get("size", 0)) for layer in manifest.get("layers", []) if layer.get("mediaType", "").startswith("application/vnd.ollama.image"))
            + int(manifest.get("config", {}).get("size", 0)),
            model_path=model_path,
            projector_path=blob_path(root, projector["digest"]) if projector else None,
            template_digest=layers.get(TEMPLATE_LAYER, {}).get("digest", ""),
        )
        model.gguf = self._gguf(model_path)
        model.params = self._layer_json(root, layers.get(PARAMS_LAYER))
        model.system = self._layer_text(root, layers.get(SYSTEM_LAYER))
        model.license = self._layer_text(root, layers.get(LICENSE_LAYER))
        model.ollama_template = self._layer_text(root, layers.get(TEMPLATE_LAYER))
        model.ollama_config = self._config_json(root, manifest.get("config"))
        return model

    @staticmethod
    def _layer_text(root: Path, layer: dict | None) -> str:
        if not layer:
            return ""
        return blob_path(root, layer["digest"]).read_text(errors="replace")

    @classmethod
    def _layer_json(cls, root: Path, layer: dict | None) -> dict:
        text = cls._layer_text(root, layer)
        try:
            return json.loads(text) if text else {}
        except ValueError:
            return {}

    @classmethod
    def _config_json(cls, root: Path, config: dict | None) -> dict:
        has_blob = bool(config and config.get("digest")) and blob_path(root, config["digest"]).is_file()
        return cls._layer_json(root, config) if has_blob else {}

    def names(self) -> list[str]:
        found = []
        for root in self.roots:
            for folder in MANIFEST_DIRS:
                base = root / folder
                if not base.is_dir():
                    continue
                for path in sorted(base.glob("*/*/*/*")):
                    if path.is_file():
                        host, namespace, model, tag = path.relative_to(base).parts
                        display = ModelName(host, namespace, model, tag).display
                        if display not in found:
                            found.append(display)
        return found

    def models(self) -> list[Model]:
        listed = []
        for name in self.names():
            try:
                listed.append(self.get(name))
            except (ModelNotFound, GgufError, OSError, ValueError):
                continue
        return sorted(listed, key=lambda model: model.modified_at, reverse=True)

    def template_donor(self, model: Model) -> Model | None:
        if model.gguf.chat_template or not model.template_digest:
            return None
        for other in self.models():
            if other.template_digest == model.template_digest and other.gguf.chat_template:
                return other
        return None

    def gguf_metadata(self, model: Model) -> dict:
        return plain_metadata(model.gguf)

    @property
    def own_root(self) -> Path:
        return self.roots[0]

    def owned_manifest(self, name: str) -> Path:
        parsed = parse_name(name)
        found = self._locate(parsed)
        if not found:
            raise ModelNotFound(f"model '{name}' not found")
        root, path = found
        if root != self.own_root:
            raise NotOwned(f"{parsed.display} is in Ollama's store ({root}); the engine never changes it, use ollama rm with Ollama running")
        return path

    def remove_owned(self, name: str) -> list[Path]:
        root = self.own_root
        path = self.owned_manifest(name)
        candidates = self._blobs_of(root, path)
        path.unlink()
        self._prune_empty_folders(root, path.parent)
        still_used = set().union(*(self._blobs_of(root, other) for other in self._manifest_files(root)))
        removed = [blob for blob in sorted(candidates - still_used) if blob.is_file()]
        for blob in removed:
            blob.unlink()
        return removed

    @staticmethod
    def _manifest_files(root: Path) -> list[Path]:
        return [path for folder in MANIFEST_DIRS if (root / folder).is_dir() for path in (root / folder).glob("*/*/*/*") if path.is_file()]

    def _blobs_of(self, root: Path, path: Path) -> set[Path]:
        blobs = {path.resolve()} if path.is_symlink() else set()
        manifest = _read_json(path)
        for entry in manifest.get("manifests", []):
            if is_digest(entry.get("digest")) and (listed := blob_path(root, entry["digest"])).is_file():
                blobs |= {listed} | self._blobs_of(root, listed)
        for layer in manifest.get("layers", []) + [manifest.get("config") or {}]:
            if is_digest(layer.get("digest")):
                blobs.add(blob_path(root, layer["digest"]))
        return {blob.resolve() for blob in blobs}

    @staticmethod
    def _prune_empty_folders(root: Path, folder: Path) -> None:
        stops = {root / name for name in MANIFEST_DIRS}
        while folder not in stops and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            folder = folder.parent
