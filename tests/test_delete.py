import hashlib
import json
import urllib.request

import pytest

from arty_engine.store import MODEL_LAYER, PROJECTOR_LAYER, engine_manifest, parse_name, write_manifest
from engine_harness import RunningEngine, error_of, http_json
from fake_store import QWEN3, _gguf_blob

HF_NAME = "hf.co/Someone/Tiny-GGUF:Q8_0"
TWIN_NAME = "hf.co/Someone/Tiny-GGUF:Q4_K_M"
REPLIES = {"/v1/chat/completions": lambda body: {"json": {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}}}


def add_pulled_model(root, name: str, metadata: dict, shared_projector: str = "") -> str:
    digest, size = _gguf_blob(root, metadata, {"blk.0.w": (10, 10)})
    layers = [{"mediaType": MODEL_LAYER, "digest": digest, "size": size}]
    if shared_projector:
        layers.append({"mediaType": PROJECTOR_LAYER, "digest": shared_projector, "size": 1})
    write_manifest(root, parse_name(name), engine_manifest(layers))
    return digest


def tree_fingerprint(root) -> str:
    entries = sorted(f"{path.relative_to(root)} {path.lstat().st_size}" for path in root.rglob("*"))
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


def delete(url: str, name: str):
    request = urllib.request.Request(f"{url}/api/delete", data=json.dumps({"model": name}).encode(), method="DELETE")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status, response.read()


@pytest.fixture
def engine(tmp_path):
    own = tmp_path / "own"
    projector, _ = _gguf_blob(own, {"general.architecture": "clip"}, {"v.blk.0.w": (2, 2)})
    model_digest = add_pulled_model(own, HF_NAME, {**QWEN3, "general.name": "eight"}, projector)
    twin_digest = add_pulled_model(own, TWIN_NAME, {**QWEN3, "general.name": "four"}, projector)
    running = RunningEngine(tmp_path, REPLIES, internal_models=[TWIN_NAME])
    running.blobs = {"model": model_digest, "twin": twin_digest, "projector": projector}
    yield running
    running.close()


def blob(engine, key: str):
    return engine.config.roots[0] / "blobs" / engine.blobs[key].replace(":", "-")


def test_deleting_a_pulled_model_removes_it_and_only_its_own_blobs(engine):
    other_before = tree_fingerprint(engine.config.roots[1])
    assert delete(engine.url, HF_NAME) == (200, b"")
    assert HF_NAME not in [model["name"] for model in http_json(f"{engine.url}/api/tags")["models"]]
    assert not blob(engine, "model").exists()
    assert blob(engine, "projector").exists() and blob(engine, "twin").exists()
    assert tree_fingerprint(engine.config.roots[1]) == other_before


def test_models_read_in_place_are_left_alone(engine):
    other_before = tree_fingerprint(engine.config.roots[1])
    status, body = error_of(lambda: delete(engine.url, "qwen3:14b"))
    assert status == 405 and "read in place" in body["error"] and "model_roots" in body["error"]
    assert tree_fingerprint(engine.config.roots[1]) == other_before


def test_deleting_an_unknown_model_is_404(engine):
    status, body = error_of(lambda: delete(engine.url, "nope:1b"))
    assert status == 404 and body == {"error": "model 'nope:1b' not found"}


def test_internal_models_are_not_deleted(engine):
    status, body = error_of(lambda: delete(engine.url, TWIN_NAME))
    assert status == 409 and "internal_models" in body["error"]
    assert blob(engine, "twin").exists()


def test_delete_waits_for_no_pull(engine):
    engine.engine.pull_lock.acquire()
    try:
        status, body = error_of(lambda: delete(engine.url, HF_NAME))
    finally:
        engine.engine.pull_lock.release()
    assert status == 409 and "pull" in body["error"]
    assert blob(engine, "model").exists()


def test_a_loaded_model_is_unloaded_before_it_is_deleted(engine):
    http_json(f"{engine.url}/api/generate", {"model": HF_NAME, "keep_alive": -1})
    assert [model["name"] for model in http_json(f"{engine.url}/api/ps")["models"]] == [HF_NAME]
    delete(engine.url, HF_NAME)
    assert http_json(f"{engine.url}/api/ps")["models"] == []


def test_push_says_there_is_no_registry(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/push", {"model": HF_NAME}))
    assert status == 501 and "no registry" in body["error"]


def test_refusing_a_model_read_in_place_leaves_it_loaded(engine):
    http_json(f"{engine.url}/api/generate", {"model": "qwen3:14b", "keep_alive": -1})
    error_of(lambda: delete(engine.url, "qwen3:14b"))
    assert [model["name"] for model in http_json(f"{engine.url}/api/ps")["models"]] == ["qwen3:14b"]
