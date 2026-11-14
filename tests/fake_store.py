import hashlib
import json
from pathlib import Path

from arty_engine.store import (CONFIG_MEDIA_TYPE, MEDIA_PREFIX, MODEL_LAYER, PROJECTOR_LAYER, engine_manifest, parse_name,
                               write_manifest)
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


def _config(root: Path, config_fields: dict | None) -> dict:
    data = json.dumps({"model_format": "gguf", **(config_fields or {})}).encode()
    return {"mediaType": CONFIG_MEDIA_TYPE, "digest": _blob(root, data), "size": len(data)}


def _text_layer(root: Path, media: str, text: str) -> dict:
    data = text.encode()
    return {"mediaType": f"{MEDIA_PREFIX}{media}", "digest": _blob(root, data), "size": len(data)}


def add_model(root: Path, model: str, tag: str, metadata: dict, tensors: dict, template: str = "", params: dict | None = None,
              projector: dict | None = None, system: str = "", config_fields: dict | None = None) -> Path:
    digest, size = _gguf_blob(root, metadata, tensors)
    layers = []
    if projector:
        projector_digest, projector_size = _gguf_blob(root, projector, {"v.blk.0.attn_q.weight": (2, 2)})
        layers.append({"mediaType": PROJECTOR_LAYER, "digest": projector_digest, "size": projector_size})
    layers.append({"mediaType": MODEL_LAYER, "digest": digest, "size": size})
    if template:
        layers.append(_text_layer(root, "template", template))
    if system:
        layers.append(_text_layer(root, "system", system))
    if params is not None:
        layers.append(_text_layer(root, "params", json.dumps(params)))
    return write_manifest(root, parse_name(f"{model}:{tag}"), engine_manifest(layers, _config(root, config_fields)))


QWEN3 = {"general.architecture": "qwen3", "qwen3.context_length": 40960, "qwen3.block_count": 40, "qwen3.attention.head_count": 40,
         "qwen3.attention.head_count_kv": 8, "qwen3.embedding_length": 5120, "qwen3.attention.key_length": 128,
         "qwen3.attention.value_length": 128, "general.file_type": 15, "tokenizer.chat_template": QWEN3_TEMPLATE}
EMBED = {"general.architecture": "nomic-bert", "nomic-bert.context_length": 2048, "nomic-bert.pooling_type": 1, "general.file_type": 1}
INLINE_VISION = {"general.architecture": "qwen25vl", "qwen25vl.context_length": 128000, "general.file_type": 15}
VISION_TEXT = {"general.architecture": "qwen35", "qwen35.context_length": 262144, "general.file_type": 15,
               "tokenizer.chat_template": MIMO_TEMPLATE}


def build_store(root: Path) -> Path:
    add_model(root, "qwen3", "14b", QWEN3, {"blk.0.w": (1000, 1000)}, template=GO_TEMPLATE,
              params={"temperature": 0.6, "stop": ["<|im_end|>"]})
    add_model(root, "untemplated", "latest", {**QWEN3, "tokenizer.chat_template": ""}, {"blk.0.w": (10, 10)}, template=GO_TEMPLATE)
    add_model(root, "mimo", "9b", VISION_TEXT, {"blk.0.w": (10, 10)}, projector={"general.architecture": "clip"})
    add_model(root, "qwen2.5vl", "7b", INLINE_VISION, {"blk.0.w": (10, 10), "v.blk.0.w": (10, 10)}, system="You are helpful.")
    add_model(root, "nomic-embed-text", "latest", EMBED, {"blk.0.w": (10, 10)})
    return root
