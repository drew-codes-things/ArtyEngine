import hashlib
import json
import os
from pathlib import Path

from fake_store import EMBED, GO_TEMPLATE, INLINE_VISION, QWEN3, VISION_TEXT, _gguf_blob

VENDOR = "application/vnd.example"
REGISTRY_HOST = "registry.example"
SHORT_HOST = "example.com"
DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
DOCKER_CONFIG = "application/vnd.docker.container.image.v1+json"
OLD_TIME_NS = 1_700_000_000_000_000_000


def _blob(root: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    (root / "blobs" / f"sha256-{digest}").write_bytes(data)
    return f"sha256:{digest}"


def _layer(root: Path, kind: str, text: str) -> dict:
    data = text.encode()
    return {"mediaType": f"{VENDOR}.image.{kind}", "digest": _blob(root, data), "size": len(data)}


def _old_manifest(root: Path, metadata: dict, tensors: dict, template: str = "", system: str = "",
                  projector: dict | None = None, config_fields: dict | None = None) -> dict:
    digest, size = _gguf_blob(root, metadata, tensors)
    layers = []
    if projector:
        projector_digest, projector_size = _gguf_blob(root, projector, {"v.blk.0.attn_q.weight": (2, 2)})
        layers.append({"mediaType": f"{VENDOR}.image.projector", "digest": projector_digest, "size": projector_size, "from": "mmproj.gguf"})
    layers.append({"mediaType": f"{VENDOR}.image.model", "digest": digest, "size": size, "from": "model.gguf"})
    if template:
        layers.append(_layer(root, "template", template))
    if system:
        layers.append(_layer(root, "system", system))
    layers.append(_layer(root, "params", json.dumps({"temperature": 0.6})))
    config = json.dumps({"model_format": "gguf", **(config_fields or {})}).encode()
    return {"schemaVersion": 2, "mediaType": DOCKER_MANIFEST, "config": {"mediaType": DOCKER_CONFIG, "digest": _blob(root, config),
            "size": len(config)}, "layers": layers}


def _link(root: Path, host: str, namespace: str, model: str, tag: str, digest: str) -> Path:
    link = root / "manifests-v2" / host / namespace / model / tag
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(f"../../../../blobs/{digest.replace(':', '-')}", link)
    os.utime(link, ns=(OLD_TIME_NS, OLD_TIME_NS), follow_symlinks=False)
    return link


def _file(root: Path, folder: str, host: str, namespace: str, model: str, tag: str, manifest: dict) -> Path:
    path = root / folder / host / namespace / model / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    os.utime(path, ns=(OLD_TIME_NS, OLD_TIME_NS))
    return path


def build_old_store(root: Path) -> Path:
    qwen3 = _old_manifest(root, QWEN3, {"blk.0.w": (100, 100)}, template=GO_TEMPLATE, config_fields={"model_type": "14.8B"})
    qwen3_digest = _blob(root, json.dumps(qwen3).encode())
    _link(root, SHORT_HOST, "library", "qwen3", "14b", qwen3_digest)
    older_copy = {**qwen3, "layers": qwen3["layers"] + [{"mediaType": DOCKER_MANIFEST, "digest": qwen3_digest, "size": 1}]}
    _file(root, "manifests", REGISTRY_HOST, "library", "qwen3", "14b", older_copy)
    looker = _old_manifest(root, VISION_TEXT, {"blk.0.w": (10, 10)}, projector={"general.architecture": "clip"})
    listing = {"schemaVersion": 2, "mediaType": f"{VENDOR}.manifest.list.v2+json", "manifests": [
        {"mediaType": DOCKER_MANIFEST, "digest": _blob(root, json.dumps(looker).encode()), "runner": "native", "format": "gguf"},
        {"mediaType": DOCKER_MANIFEST, "digest": "sha256:" + "0" * 64, "runner": "other", "format": "safetensors"}]}
    _link(root, SHORT_HOST, "library", "looker-v2", "9b", _blob(root, json.dumps(listing).encode()))
    vision = _old_manifest(root, INLINE_VISION, {"blk.0.w": (10, 10), "v.blk.0.w": (10, 10)}, system="You are helpful.")
    _link(root, SHORT_HOST, "library", "qwen2.5vl", "7b", _blob(root, json.dumps(vision).encode()))
    _file(root, "manifests", REGISTRY_HOST, "library", "nomic-embed-text", "latest", _old_manifest(root, EMBED, {"blk.0.w": (10, 10)}))
    pulled = _old_manifest(root, QWEN3, {"blk.0.w": (4, 4)})
    _file(root, "manifests-v2", "hf.co", "Someone", "Tiny-GGUF", "Q8_0", pulled)
    _file(root, "manifests-v2", SHORT_HOST, "someone", "custom", "1", _old_manifest(root, QWEN3, {"blk.0.w": (6, 6)}))
    (root / "metadata").mkdir()
    (root / "metadata" / "cache.json").write_text('{"kv": {}}')
    return root


EXPECTED_NAMES = {"qwen3:14b", "looker-v2:9b", "qwen2.5vl:7b", "nomic-embed-text:latest", "hf.co/Someone/Tiny-GGUF:Q8_0",
                  "someone/custom:1"}


def add_old_model(root: Path, folder: str, host: str, namespace: str, model: str, tag: str, rows: int = 8) -> dict:
    manifest = _old_manifest(root, QWEN3, {"blk.0.w": (rows, rows)})
    _file(root, folder, host, namespace, model, tag, manifest)
    return manifest


def write_old(root: Path, folder: str, host: str, namespace: str, model: str, tag: str, manifest) -> Path:
    path = root / folder / host / namespace / model / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))
    return path
