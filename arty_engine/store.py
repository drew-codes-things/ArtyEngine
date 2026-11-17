import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .gguf import GgufError, GgufInfo, plain_metadata, read_gguf

log = logging.getLogger("arty-engine.store")

LOCAL_HOST = "local"
DEFAULT_NAMESPACE = "library"
DEFAULT_TAG = "latest"
CATALOG_FOLDER = "catalog"
MANIFEST_SUFFIX = ".json"
PARTIAL_SUFFIX = ".partial"
OLDER_LAYOUT_FOLDERS = ("manifests-v2", "manifests")
SCHEMA_VERSION = 1
MEDIA_PREFIX = "application/vnd.arty-engine."
MANIFEST_MEDIA_TYPE = f"{MEDIA_PREFIX}manifest.v1+json"
CONFIG_MEDIA_TYPE = f"{MEDIA_PREFIX}config"
MODEL_LAYER = f"{MEDIA_PREFIX}model"
PROJECTOR_LAYER = f"{MEDIA_PREFIX}projector"
TEMPLATE_LAYER = f"{MEDIA_PREFIX}template"
SYSTEM_LAYER = f"{MEDIA_PREFIX}system"
PARAMS_LAYER = f"{MEDIA_PREFIX}params"
LICENSE_LAYER = f"{MEDIA_PREFIX}license"
SOURCE_FIELD = "source"
EMBEDDING_ARCHITECTURES = {"bert", "nomic-bert", "nomic-bert-moe", "jina-bert-v2", "jina-bert-v3", "modern-bert", "neo-bert"}
UNSUPPORTED_ARCHITECTURES = {"qwen25vl"}
MILLION = 1_000_000
BILLION = 1_000_000_000
TEN_BILLION = 10 * BILLION
PATH_STEPS = {".", ".."}
DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]*(:[A-Za-z0-9._-]+)?")
LONGEST_FILE_NAME = 255
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
        if self.host != LOCAL_HOST:
            return f"{self.host}/{self.namespace}/{self.model}:{self.tag}"
        if self.namespace == DEFAULT_NAMESPACE:
            return f"{self.model}:{self.tag}"
        return f"{self.namespace}/{self.model}:{self.tag}"

    def manifest_path(self, root: Path) -> Path:
        return root / CATALOG_FOLDER / self.host / self.namespace / self.model / f"{self.tag}{MANIFEST_SUFFIX}"


def parse_name(name: str) -> ModelName:
    if not isinstance(name, str) or not NAME_PATTERN.fullmatch(name):
        raise ModelNotFound(f"invalid model name {name!r}")
    base, _, tag = name.partition(":")
    parts = base.split("/")
    if "" in parts or any(part in PATH_STEPS or len(part) > LONGEST_FILE_NAME for part in (*parts, tag)):
        raise ModelNotFound(f"invalid model name {name!r}")
    if len(parts) == 1:
        return ModelName(LOCAL_HOST, DEFAULT_NAMESPACE, parts[0].lower(), tag or DEFAULT_TAG)
    if len(parts) == 2:
        return ModelName(LOCAL_HOST, parts[0].lower(), parts[1].lower(), tag or DEFAULT_TAG)
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
    stored_template: str = ""
    stored_config: dict = field(default_factory=dict)

    @property
    def load_digest(self) -> str:
        without_source = {key: value for key, value in self.manifest.items() if key != SOURCE_FIELD}
        return hashlib.sha256(json.dumps(without_source, sort_keys=True).encode()).hexdigest()

    @property
    def architecture(self) -> str:
        return self.gguf.architecture if self.gguf else ""

    @property
    def is_embedding(self) -> bool:
        return self.architecture in EMBEDDING_ARCHITECTURES or self.gguf.arch_value("pooling_type") is not None

    @property
    def runnable(self) -> bool:
        return self.architecture not in UNSUPPORTED_ARCHITECTURES and not (self.gguf.has_inline_vision and not self.projector_path)

    @property
    def unrunnable_reason(self) -> str:
        if self.runnable:
            return ""
        return (f"{self.name} is not supported by ArtyEngine: its model file ({self.architecture}) keeps the vision weights "
                "inside it, and the engine needs the text model and a separate projector file. Pull a build of the model "
                "that ships a projector file from Hugging Face instead")

    @property
    def capabilities(self) -> list[str]:
        if self.is_embedding:
            return ["embedding"]
        found = ["completion"]
        if self.projector_path or self.gguf.has_inline_vision:
            found.append("vision")
        template = self.gguf.chat_template or self.stored_template
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
            "parameter_size": self.stored_config.get("model_type") or parameter_size(self.gguf.parameter_count),
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
            path = name.manifest_path(root)
            if path.is_file():
                return root, path
        return None

    def get(self, name: str) -> Model:
        parsed = parse_name(name)
        found = self._locate(parsed)
        if not found:
            raise ModelNotFound(f"model '{name}' not found")
        root, path = found
        manifest = _read_json(path)
        if not is_engine_manifest(manifest):
            raise ModelNotFound(f"{path} is not an ArtyEngine manifest")
        layers = {layer["mediaType"]: layer for layer in manifest["layers"]}
        if MODEL_LAYER not in layers:
            raise ModelNotFound(f"model '{name}' has no model layer")
        model_path = blob_path(root, layers[MODEL_LAYER]["digest"])
        projector = layers.get(PROJECTOR_LAYER)
        config = manifest.get("config") or {}
        model = Model(
            name=parsed.display, root=root, digest=_sha256_of(path), manifest=manifest,
            modified_at=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
            size=sum(int(layer.get("size", 0)) for layer in manifest.get("layers", [])) + int(config.get("size", 0)),
            model_path=model_path,
            projector_path=blob_path(root, projector["digest"]) if projector else None,
            template_digest=layers.get(TEMPLATE_LAYER, {}).get("digest", ""),
        )
        model.gguf = self._gguf(model_path)
        model.params = self._layer_json(root, layers.get(PARAMS_LAYER))
        model.system = self._layer_text(root, layers.get(SYSTEM_LAYER))
        model.license = self._layer_text(root, layers.get(LICENSE_LAYER))
        model.stored_template = self._layer_text(root, layers.get(TEMPLATE_LAYER))
        model.stored_config = self._config_json(root, config)
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
            value = json.loads(text) if text else {}
        except (ValueError, RecursionError):
            return {}
        return value if isinstance(value, dict) else {}

    @classmethod
    def _config_json(cls, root: Path, config: dict | None) -> dict:
        has_blob = bool(config and config.get("digest")) and blob_path(root, config["digest"]).is_file()
        return cls._layer_json(root, config) if has_blob else {}

    def names(self) -> list[str]:
        found = []
        for root in self.roots:
            for path in _manifest_files(root):
                display = name_of_manifest(root, path).display
                if display not in found:
                    found.append(display)
        return found

    def models(self) -> list[Model]:
        listed = []
        for name in self.names():
            try:
                listed.append(self.get(name))
            except (ModelNotFound, GgufError, OSError, ValueError, RecursionError) as error:
                log.warning("left %s out of the model list: %s", name, error)
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
            raise NotOwned(f"{parsed.display} is read in place from {root} (model_roots); the engine never changes that store")
        return path

    def remove_owned(self, name: str) -> list[Path]:
        root = self.own_root
        path = self.owned_manifest(name)
        candidates = self._blobs_of(root, path)
        path.unlink()
        self._prune_empty_folders(root, path.parent)
        still_used = set().union(*(self._blobs_of(root, other) for other in _manifest_files(root)))
        removed = [blob for blob in sorted(candidates - still_used) if blob.is_file()]
        for blob in removed:
            blob.unlink()
        return removed

    def _blobs_of(self, root: Path, path: Path) -> set[Path]:
        try:
            manifest = _read_json(path)
        except (OSError, ValueError, RecursionError) as error:
            log.warning("couldn't read %s, so its blobs count as unused: %s", path, error)
            return set()
        digests = {entry.get("digest") for entry in _entries_of(manifest)}
        return {blob_path(root, digest).resolve() for digest in digests if is_digest(digest)}

    @staticmethod
    def _prune_empty_folders(root: Path, folder: Path) -> None:
        stop = root / CATALOG_FOLDER
        while folder != stop and folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()
            folder = folder.parent


def named_digests(root: Path) -> set[str]:
    named = set()
    for path in _manifest_files(root):
        try:
            named |= {entry.get("digest") for entry in _entries_of(_read_json(path))}
        except (OSError, ValueError, RecursionError) as error:
            log.warning("couldn't read %s while counting the blobs models name: %s", path, error)
    return named


def _entries_of(manifest) -> list[dict]:
    if not isinstance(manifest, dict):
        return []
    layers = manifest.get("layers")
    entries = layers if isinstance(layers, list) else []
    return [entry for entry in [*entries, manifest.get("config")] if isinstance(entry, dict)]


def is_engine_manifest(manifest) -> bool:
    if not isinstance(manifest, dict) or manifest.get("mediaType") != MANIFEST_MEDIA_TYPE:
        return False
    layers = manifest.get("layers")
    config = manifest.get("config", {})
    return (isinstance(layers, list) and all(_is_entry(layer) for layer in layers)
            and isinstance(config, dict) and (not config or _is_entry(config)))


def _is_entry(entry) -> bool:
    return (isinstance(entry, dict) and isinstance(entry.get("mediaType"), str) and is_digest(entry.get("digest"))
            and isinstance(entry.get("size", 0), int) and not isinstance(entry.get("size", 0), bool))


def roots_in_older_layout(roots: list[Path]) -> list[Path]:
    return [root for root in roots
            if not (root / CATALOG_FOLDER).is_dir() and any((root / folder).is_dir() for folder in OLDER_LAYOUT_FOLDERS)]


def _manifest_files(root: Path) -> list[Path]:
    catalog = root / CATALOG_FOLDER
    if not catalog.is_dir():
        return []
    return sorted(path for path in catalog.glob(f"*/*/*/*{MANIFEST_SUFFIX}") if path.is_file())


def name_of_manifest(root: Path, path: Path) -> ModelName:
    host, namespace, model, file_name = path.relative_to(root / CATALOG_FOLDER).parts
    return ModelName(host, namespace, model, file_name.removesuffix(MANIFEST_SUFFIX))


def engine_manifest(layers: list[dict], config: dict | None = None) -> dict:
    manifest = {"schemaVersion": SCHEMA_VERSION, "mediaType": MANIFEST_MEDIA_TYPE, "layers": layers}
    if config:
        manifest["config"] = config
    return manifest


def write_manifest(root: Path, name: ModelName, manifest: dict) -> Path:
    path = name.manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + PARTIAL_SUFFIX)
    with open(temporary, "w") as handle:
        handle.write(json.dumps(manifest))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    sync_folder(path.parent)
    return path


def sync_folder(folder: Path) -> None:
    descriptor = os.open(folder, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
