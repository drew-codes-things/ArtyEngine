import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path

from .store import (ADAPTER_LAYER, LICENSE_LAYER, MESSAGES_LAYER, MODEL_LAYER, PARAMS_LAYER, PARTIAL_SUFFIX, PROJECTOR_LAYER,
                    SYSTEM_LAYER, TEMPLATE_LAYER, Model, ModelNotFound, ModelStore, blob_path, engine_manifest, is_digest,
                    parse_name, write_manifest)

QUOTES = '"""'
LIST_PARAMETERS = {"stop"}
TEXT_LAYERS = {"template": TEMPLATE_LAYER, "system": SYSTEM_LAYER, "license": LICENSE_LAYER}
CARRIED_LAYERS = (MODEL_LAYER, PROJECTOR_LAYER, TEMPLATE_LAYER, SYSTEM_LAYER, LICENSE_LAYER, PARAMS_LAYER, MESSAGES_LAYER,
                  ADAPTER_LAYER)
MESSAGE_ROLES = ("system", "user", "assistant")
PROJECTOR_WORDS = ("mmproj", "projector")
CHUNK_BYTES = 1024 * 1024
GGUF_SUFFIX = ".gguf"


class RecipeError(ValueError):
    def __init__(self, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.status = int(status)


@dataclass
class Recipe:
    base: str = ""
    parameters: dict = field(default_factory=dict)
    texts: dict = field(default_factory=dict)
    adapters: list = field(default_factory=list)
    messages: list = field(default_factory=list)


def typed_value(text: str):
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    return text


def unquoted(text: str) -> str:
    text = text.strip()
    if len(text) >= 2 and text[0] == text[-1] == '"':
        return text[1:-1]
    return text


def add_parameter(parameters: dict, name: str, value) -> None:
    if name in LIST_PARAMETERS:
        parameters.setdefault(name, []).append(str(value))
    else:
        parameters[name] = value


def recipe_lines(text: str):
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        index += 1
        if not line or line.startswith("#"):
            continue
        command, _, rest = line.partition(" ")
        rest = rest.strip()
        if rest.startswith(QUOTES):
            body = rest[len(QUOTES):]
            while QUOTES not in body and index < len(lines):
                body += "\n" + lines[index]
                index += 1
            if QUOTES not in body:
                raise RecipeError(f"the {command} value that starts with {QUOTES} is never closed")
            rest = body[:body.index(QUOTES)]
        yield command.lower(), rest


def parse_recipe(text: str) -> Recipe:
    if not isinstance(text, str):
        raise RecipeError("the recipe must be text")
    recipe = Recipe()
    for command, value in recipe_lines(text):
        if command == "from":
            recipe.base = unquoted(value)
        elif command == "parameter":
            name, _, setting = value.partition(" ")
            if not setting.strip():
                raise RecipeError(f"PARAMETER {name} has no value")
            add_parameter(recipe.parameters, name, typed_value(unquoted(setting)))
        elif command in TEXT_LAYERS:
            recipe.texts[command] = value.removeprefix("\n").removesuffix("\n")
        elif command == "adapter":
            recipe.adapters.append(unquoted(value))
        elif command == "message":
            role, _, content = value.partition(" ")
            if role not in MESSAGE_ROLES:
                raise RecipeError(f"MESSAGE role must be one of {', '.join(MESSAGE_ROLES)}, not {role!r}")
            recipe.messages.append({"role": role, "content": unquoted(content)})
        else:
            raise RecipeError(f"unknown recipe command {command.upper()}")
    return recipe


def recipe_of_request(request: dict) -> Recipe:
    recipe = parse_recipe(request["modelfile"]) if request.get("modelfile") else Recipe()
    if request.get("from"):
        recipe.base = str(request["from"])
    for name in TEXT_LAYERS:
        if request.get(name) is not None:
            if not isinstance(request[name], str):
                raise RecipeError(f"{name} must be text")
            recipe.texts[name] = request[name]
    parameters = request.get("parameters")
    if parameters is not None:
        if not isinstance(parameters, dict):
            raise RecipeError("parameters must be an object")
        recipe.parameters.update(parameters)
    messages = request.get("messages")
    if messages is not None:
        if not isinstance(messages, list) or not all(isinstance(item, dict) and item.get("role") in MESSAGE_ROLES
                                                     and isinstance(item.get("content", ""), str) for item in messages):
            raise RecipeError("messages must be a list of {role, content} objects")
        recipe.messages = [{"role": item["role"], "content": item.get("content", "")} for item in messages]
    adapters = request.get("adapters")
    if adapters is not None:
        if not isinstance(adapters, dict):
            raise RecipeError("adapters must be an object of file name to sha256 digest")
        recipe.adapters += list(adapters.values())
    return recipe


def text_blob(root: Path, text: str) -> tuple[str, int]:
    data = text.encode()
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    target = blob_path(root, digest)
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + PARTIAL_SUFFIX)
        partial.write_bytes(data)
        partial.replace(target)
    return digest, len(data)


def text_layer(root: Path, media_type: str, text: str) -> dict:
    digest, size = text_blob(root, text)
    return {"mediaType": media_type, "digest": digest, "size": size}


def bring_blob(store: ModelStore, source_root: Path, digest: str) -> None:
    target = blob_path(store.own_root, digest)
    if target.is_file():
        return
    source = blob_path(source_root, digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    try:
        os.link(source, partial)
    except OSError:
        shutil.copyfile(source, partial)
    partial.replace(target)


def owned_blob(store: ModelStore, digest, what: str) -> dict:
    if not is_digest(digest):
        raise RecipeError(f"{what} must be a sha256 digest of a blob sent to /api/blobs first, not {digest!r}")
    path = blob_path(store.own_root, digest)
    if not path.is_file():
        raise RecipeError(f"{what} {digest} is not in the store; send it to /api/blobs/{digest} first")
    return {"digest": digest, "size": path.stat().st_size}


def base_layers(store: ModelStore, model: Model) -> list[dict]:
    layers = [dict(layer) for layer in model.manifest.get("layers", []) if layer.get("mediaType") in CARRIED_LAYERS]
    for layer in layers:
        bring_blob(store, model.root, layer["digest"])
    return layers


def file_layers(store: ModelStore, files) -> list[dict]:
    if not isinstance(files, dict) or not files:
        raise RecipeError("files must be an object of file name to sha256 digest")
    weights = {name: digest for name, digest in files.items() if str(name).lower().endswith(GGUF_SUFFIX)}
    projectors = {name: digest for name, digest in weights.items() if any(word in name.lower() for word in PROJECTOR_WORDS)}
    models = {name: digest for name, digest in weights.items() if name not in projectors}
    if len(models) != 1 or len(projectors) > 1:
        raise RecipeError("files must name exactly one model .gguf (and at most one projector .gguf); split models are not supported")
    layers = [{"mediaType": MODEL_LAYER, **owned_blob(store, digest, name), "from": name} for name, digest in models.items()]
    layers += [{"mediaType": PROJECTOR_LAYER, **owned_blob(store, digest, name), "from": name} for name, digest in projectors.items()]
    return layers


def starting_layers(store: ModelStore, recipe: Recipe, files) -> list[dict]:
    if files:
        return file_layers(store, files)
    if not recipe.base:
        raise RecipeError("a model is made from something: give from (an installed model or a blob digest) or files")
    if is_digest(recipe.base):
        return [{"mediaType": MODEL_LAYER, **owned_blob(store, recipe.base, "from")}]
    if recipe.base.startswith(("/", ".", "~")) or "\\" in recipe.base:
        raise RecipeError("from can't be a path on the engine's machine; send the file to /api/blobs and use its digest")
    try:
        return base_layers(store, store.get(recipe.base))
    except ModelNotFound as error:
        raise RecipeError(f"from: model '{recipe.base}' not found", HTTPStatus.NOT_FOUND) from error


def replaced(layers: list[dict], media_type: str, layer: dict | None) -> list[dict]:
    kept = [entry for entry in layers if entry["mediaType"] != media_type]
    return kept + [layer] if layer else kept


def recipe_layers(store: ModelStore, recipe: Recipe, layers: list[dict]) -> list[dict]:
    root = store.own_root
    for name, media_type in TEXT_LAYERS.items():
        if name in recipe.texts:
            layers = replaced(layers, media_type, text_layer(root, media_type, recipe.texts[name]))
    if recipe.parameters:
        inherited = next((entry for entry in layers if entry["mediaType"] == PARAMS_LAYER), None)
        merged = {**(stored_json(root, inherited) if inherited else {}), **recipe.parameters}
        layers = replaced(layers, PARAMS_LAYER, text_layer(root, PARAMS_LAYER, json.dumps(merged)))
    if recipe.messages:
        layers = replaced(layers, MESSAGES_LAYER, text_layer(root, MESSAGES_LAYER, json.dumps(recipe.messages)))
    for adapter in recipe.adapters:
        layers.append({"mediaType": ADAPTER_LAYER, **owned_blob(store, adapter, "adapter")})
    return layers


def stored_json(root: Path, layer: dict) -> dict:
    try:
        value = json.loads(blob_path(root, layer["digest"]).read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def create(store: ModelStore, request: dict, progress) -> str:
    name = request.get("model") or request.get("name")
    if not isinstance(name, str) or not name:
        raise RecipeError("model is required")
    target = parse_name(name)
    if request.get("quantize"):
        raise RecipeError("quantize isn't supported by the engine: pull a quantized build from Hugging Face instead",
                          HTTPStatus.NOT_IMPLEMENTED)
    progress({"status": "reading the recipe"})
    recipe = recipe_of_request(request)
    layers = starting_layers(store, recipe, request.get("files"))
    layers = recipe_layers(store, recipe, layers)
    progress({"status": "writing manifest"})
    manifest = engine_manifest(layers)
    if recipe.base:
        manifest["created_from"] = recipe.base
    write_manifest(store.own_root, target, manifest)
    return target.display


def copy(store: ModelStore, source: str, destination: str) -> str:
    if not isinstance(source, str) or not source or not isinstance(destination, str) or not destination:
        raise RecipeError("source and destination are required")
    target = parse_name(destination)
    model = store.get(source)
    layers = base_layers(store, model)
    config = model.manifest.get("config")
    if config:
        bring_blob(store, model.root, config["digest"])
    manifest = {**engine_manifest(layers, config), **{key: value for key, value in model.manifest.items()
                                                      if key not in ("layers", "config", "schemaVersion", "mediaType")}}
    write_manifest(store.own_root, target, manifest)
    return target.display


def has_blob(store: ModelStore, digest: str) -> bool:
    return is_digest(digest) and blob_path(store.own_root, digest).is_file()


def drain(source, length: int) -> None:
    left = length
    while left > 0:
        chunk = source.read(min(CHUNK_BYTES, left))
        if not chunk:
            return
        left -= len(chunk)


def receive_blob(store: ModelStore, digest: str, source, length: int, keep_free_bytes: int) -> bool:
    if not is_digest(digest):
        raise RecipeError(f"{digest!r} is not a sha256 digest")
    target = blob_path(store.own_root, digest)
    if target.is_file():
        drain(source, length)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(target.parent).free - length < keep_free_bytes:
        raise RecipeError("the blob would leave less than pull_keep_free_mb free on the store's disk",
                          HTTPStatus.INSUFFICIENT_STORAGE)
    partial = target.with_name(target.name + PARTIAL_SUFFIX)
    hasher = hashlib.sha256()
    left = length
    with open(partial, "wb") as output:
        while left > 0:
            chunk = source.read(min(CHUNK_BYTES, left))
            if not chunk:
                break
            hasher.update(chunk)
            output.write(chunk)
            left -= len(chunk)
    if left or f"sha256:{hasher.hexdigest()}" != digest:
        partial.unlink(missing_ok=True)
        raise RecipeError(f"the blob sent does not match {digest}" if not left else "the blob ended early")
    partial.replace(target)
    return True
