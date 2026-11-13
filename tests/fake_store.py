import hashlib
import json
import os
from pathlib import Path

from gguf_writer import write_gguf

QWEN3_TEMPLATE = "{%- if tools %}tools{%- for tool in tools %}{{- tool | tojson }}{%- endfor %}{%- endif %}{{ '<think>' if enable_thinking }}"
MIMO_TEMPLATE = "{%- if tools %}tools{%- for tool in tools %}{{- '\\n' ~ (tool | tojson(ensure_ascii=False)) -}}{%- endfor %}{%- endif %}<think>"
GO_TEMPLATE = "{{- if .Tools }}tools{{ end }}{{ if .Think }}think{{ end }}"


def _blob(root: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    (root / "blobs" / f"sha256-{digest}").write_bytes(data)
    return f"sha256:{digest}"


def _gguf_blob(root: Path, metadata: dict, tensors: dict) -> tuple[str, int]:
    root.mkdir(parents=True, exist_ok=True)
    scratch = root / "scratch.gguf"
    write_gguf(scratch, metadata, tensors)
    data = scratch.read_bytes()
    scratch.unlink()
    return _blob(root, data), len(data)


def _manifest(root: Path, layers: list[dict], config_fields: dict | None = None) -> dict:
    config = json.dumps({"model_format": "gguf", **(config_fields or {})}).encode()
    return {"schemaVersion": 2, "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {"mediaType": "application/vnd.docker.container.image.v1+json", "digest": _blob(root, config), "size": len(config)},
            "layers": layers}


def _text_layer(root: Path, media: str, text: str) -> dict:
    data = text.encode()
    return {"mediaType": f"application/vnd.ollama.image.{media}", "digest": _blob(root, data), "size": len(data)}


def add_v2_model(root: Path, model: str, tag: str, metadata: dict, tensors: dict, template: str = "", params: dict | None = None,
                 projector: dict | None = None, system: str = "", as_list: bool = False, config_fields: dict | None = None) -> None:
    digest, size = _gguf_blob(root, metadata, tensors)
    layers = []
    if projector:
        projector_digest, projector_size = _gguf_blob(root, projector, {"v.blk.0.attn_q.weight": (2, 2)})
        layers.append({"mediaType": "application/vnd.ollama.image.projector", "digest": projector_digest, "size": projector_size})
    layers.append({"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": size})
    if template:
        layers.append(_text_layer(root, "template", template))
    if system:
        layers.append(_text_layer(root, "system", system))
    if params is not None:
        layers.append(_text_layer(root, "params", json.dumps(params)))
    manifest = _manifest(root, layers, config_fields)
    manifest_digest = _blob(root, json.dumps(manifest).encode())
    target_digest = manifest_digest
    if as_list:
        listing = {"schemaVersion": 2, "mediaType": "application/vnd.ollama.manifest.list.v2+json", "manifests": [
            {"mediaType": "application/vnd.docker.distribution.manifest.v2+json", "digest": "sha256:" + "0" * 64, "runner": "mlx", "format": "safetensors"},
            {"mediaType": "application/vnd.docker.distribution.manifest.v2+json", "digest": manifest_digest, "runner": "llamacpp", "format": "gguf"}]}
        target_digest = _blob(root, json.dumps(listing).encode())
    link = root / "manifests-v2" / "ollama.com" / "library" / model / tag
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(f"../../../../blobs/{target_digest.replace(':', '-')}", link)


def add_v1_model(root: Path, model: str, tag: str, metadata: dict, tensors: dict) -> None:
    digest, size = _gguf_blob(root, metadata, tensors)
    manifest = _manifest(root, [{"mediaType": "application/vnd.ollama.image.model", "digest": digest, "size": size}])
    path = root / "manifests" / "registry.ollama.ai" / "library" / model / tag
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest))


QWEN3 = {"general.architecture": "qwen3", "qwen3.context_length": 40960, "qwen3.block_count": 40, "qwen3.attention.head_count": 40,
         "qwen3.attention.head_count_kv": 8, "qwen3.embedding_length": 5120, "qwen3.attention.key_length": 128,
         "qwen3.attention.value_length": 128, "general.file_type": 15, "tokenizer.chat_template": QWEN3_TEMPLATE}
EMBED = {"general.architecture": "nomic-bert", "nomic-bert.context_length": 2048, "nomic-bert.pooling_type": 1, "general.file_type": 1}
OLLAMA_ENGINE_VISION = {"general.architecture": "qwen25vl", "qwen25vl.context_length": 128000, "general.file_type": 15}
VISION_TEXT = {"general.architecture": "qwen35", "qwen35.context_length": 262144, "general.file_type": 15,
               "tokenizer.chat_template": MIMO_TEMPLATE}


def build_store(root: Path) -> Path:
    add_v2_model(root, "qwen3", "14b", QWEN3, {"blk.0.w": (1000, 1000)}, template=GO_TEMPLATE,
                 params={"temperature": 0.6, "stop": ["<|im_end|>"]})
    add_v2_model(root, "untemplated", "latest", {**QWEN3, "tokenizer.chat_template": ""}, {"blk.0.w": (10, 10)}, template=GO_TEMPLATE)
    add_v2_model(root, "mimo", "9b", VISION_TEXT, {"blk.0.w": (10, 10)}, projector={"general.architecture": "clip"}, as_list=True)
    add_v2_model(root, "qwen2.5vl", "7b", OLLAMA_ENGINE_VISION, {"blk.0.w": (10, 10), "v.blk.0.w": (10, 10)}, system="You are helpful.")
    add_v1_model(root, "nomic-embed-text", "latest", EMBED, {"blk.0.w": (10, 10)})
    return root
