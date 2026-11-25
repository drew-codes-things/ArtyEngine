import hashlib
import http.client
import json
import os
import stat
import struct
import threading
import urllib.error
import urllib.request

import pytest

from arty_engine import recipes
from arty_engine.recipes import RecipeError, parse_recipe
from arty_engine.store import ModelStore, blob_path
from engine_harness import RunningEngine, error_of, http_json
from fake_store import QWEN3, build_store
from gguf_writer import write_gguf

GGUF_START = b"GGUF" + struct.pack("<IQQ", 3, 0, 0)

CHAT_REPLY = {"json": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]}}
RECIPE = '''# a recipe
FROM qwen3:14b
PARAMETER temperature 0.2
PARAMETER num_ctx 4096
PARAMETER stop "<|a|>"
PARAMETER stop "<|b|>"
SYSTEM """You are Arty.
Answer briefly."""
LICENSE MIT
MESSAGE user "Hello"
MESSAGE assistant "Hi there."
'''


@pytest.fixture
def running(tmp_path):
    engine = RunningEngine(tmp_path, {"/v1/chat/completions": CHAT_REPLY}, vram_budget_mb=100000)
    yield engine
    engine.close()


def send_blob(url, data: bytes, digest: str | None = None) -> int:
    digest = digest or f"sha256:{hashlib.sha256(data).hexdigest()}"
    request = urllib.request.Request(f"{url}/api/blobs/{digest}", data=data, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.status


def blob_known(url, digest: str) -> int:
    request = urllib.request.Request(f"{url}/api/blobs/{digest}", method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def test_a_recipe_is_read_into_its_parts():
    recipe = parse_recipe(RECIPE)
    assert recipe.base == "qwen3:14b"
    assert recipe.parameters == {"temperature": 0.2, "num_ctx": 4096, "stop": ["<|a|>", "<|b|>"]}
    assert recipe.texts == {"system": "You are Arty.\nAnswer briefly.", "license": "MIT"}
    assert recipe.messages == [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there."}]


@pytest.mark.parametrize("text, problem", [
    ("FROM a\nWIBBLE b", "unknown recipe command WIBBLE"),
    ('SYSTEM """never closed', "never closed"),
    ("PARAMETER temperature", "has no value"),
    ("MESSAGE robot hi", "MESSAGE role"),
])
def test_bad_recipes_are_refused(text, problem):
    with pytest.raises(RecipeError, match=problem):
        parse_recipe(text)


def test_a_model_made_from_a_recipe_carries_its_settings(running):
    lines = []
    request = urllib.request.Request(f"{running.url}/api/create", data=json.dumps({"model": "arty-small", "modelfile": RECIPE}).encode(),
                                     method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        lines = [json.loads(line) for line in response if line.strip()]
    assert lines[-1] == {"status": "success"}
    shown = http_json(f"{running.url}/api/show", {"model": "arty-small"})
    assert shown["system"] == "You are Arty.\nAnswer briefly." and shown["license"] == "MIT"
    assert "temperature" in shown["parameters"] and "<|b|>" in shown["parameters"]
    http_json(f"{running.url}/api/chat", {"model": "arty-small", "messages": [{"role": "user", "content": "yo"}], "stream": False})
    sent = running.upstream_requests()[-1]["body"]
    assert [message["content"] for message in sent["messages"]] == ["You are Arty.\nAnswer briefly.", "Hello", "Hi there.", "yo"]
    assert sent["temperature"] == 0.2
    names = [model["name"] for model in http_json(f"{running.url}/api/tags")["models"]]
    assert "arty-small:latest" in names and "qwen3:14b" in names


def test_a_model_is_made_from_sent_blobs(running, tmp_path):
    weights = tmp_path / "tiny.gguf"
    write_gguf(weights, QWEN3, {"blk.0.w": (4, 4)})
    data = weights.read_bytes()
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    assert blob_known(running.url, digest) == 404
    assert send_blob(running.url, data) == 201
    assert send_blob(running.url, data) == 200
    assert blob_known(running.url, digest) == 200
    http_json(f"{running.url}/api/create", {"model": "tiny", "files": {"tiny.gguf": digest}, "stream": False})
    shown = http_json(f"{running.url}/api/show", {"model": "tiny"})
    assert shown["details"]["family"] == "qwen3"


def test_a_blob_that_does_not_match_its_digest_is_refused(running):
    status, body = error_of(lambda: send_blob(running.url, b"abc", "sha256:" + "0" * 64))
    assert status == 400 and "does not match" in body["error"]
    assert blob_known(running.url, "sha256:" + "0" * 64) == 404


def test_an_adapter_becomes_a_lora_argument(running):
    data = GGUF_START + b"adapter weights"
    digest = f"sha256:{hashlib.sha256(data).hexdigest()}"
    send_blob(running.url, data)
    http_json(f"{running.url}/api/create", {"model": "tuned", "from": "qwen3:14b", "adapters": {"a.gguf": digest}, "stream": False})
    http_json(f"{running.url}/api/chat", {"model": "tuned", "messages": [{"role": "user", "content": "x"}], "stream": False})
    arguments = running.launcher.launched[-1].arguments
    assert arguments[arguments.index("--lora") + 1].endswith(digest.replace(":", "-"))


@pytest.mark.parametrize("request_body, status, problem", [
    ({"model": "x", "from": "missing:1b"}, 404, "not found"),
    ({"model": "x"}, 400, "made from something"),
    ({"model": "x", "from": "/etc/passwd"}, 400, "can't be a path"),
    ({"model": "x", "from": "qwen3:14b", "quantize": "q4_K_M"}, 501, "quantize"),
    ({"model": "x", "files": {"a.gguf": "sha256:" + "1" * 64}}, 400, "not in the store"),
    ({"model": "x", "from": "qwen3:14b", "adapters": {"a": "nonsense"}}, 400, "sha256 digest"),
])
def test_create_refuses_what_it_can_not_do(running, request_body, status, problem):
    found_status, body = error_of(lambda: http_json(f"{running.url}/api/create", {**request_body, "stream": False}))
    assert found_status == status and problem in body["error"]


def test_a_copy_is_a_new_name_for_the_same_model(running):
    http_json(f"{running.url}/api/copy", {"source": "qwen3:14b", "destination": "my-qwen"})
    original = http_json(f"{running.url}/api/show", {"model": "qwen3:14b"})
    copied = http_json(f"{running.url}/api/show", {"model": "my-qwen"})
    assert copied["parameters"] == original["parameters"] and copied["details"] == original["details"]
    status, _ = error_of(lambda: http_json(f"{running.url}/api/copy", {"source": "missing", "destination": "y"}))
    assert status == 404


def test_push_is_still_not_offered(running):
    status, body = error_of(lambda: http_json(f"{running.url}/api/push", {"model": "qwen3:14b"}))
    assert status == 501 and "no registry" in body["error"]


def sha_of(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def send_chunked(url, data: bytes, digest: str, pieces: int = 3) -> tuple[int, bytes]:
    host, port = url.split("//", 1)[1].split(":")
    connection = http.client.HTTPConnection(host, int(port), timeout=30)
    size = max(1, len(data) // pieces)
    connection.request("POST", f"/api/blobs/{digest}", body=iter([data[start:start + size] for start in range(0, len(data), size)]),
                       encode_chunked=True)
    response = connection.getresponse()
    answer = (response.status, response.read())
    connection.close()
    return answer


def test_a_blob_can_be_sent_chunked(running):
    data = os.urandom(300_000)
    status, _ = send_chunked(running.url, data, sha_of(data))
    assert status == 201 and blob_known(running.url, sha_of(data)) == 200
    assert blob_path(running.engine.store.own_root, sha_of(data)).read_bytes() == data


def test_a_chunked_blob_that_does_not_match_is_refused(running):
    status, body = send_chunked(running.url, b"abc" * 1000, "sha256:" + "0" * 64)
    assert status == 400 and b"does not match" in body


class PausingSource:
    def __init__(self, data: bytes, at_second_read):
        self.data = data
        self.position = 0
        self.reads = 0
        self.at_second_read = at_second_read

    def read(self, size: int) -> bytes:
        self.reads += 1
        if self.reads == 2:
            self.at_second_read()
        chunk = self.data[self.position:self.position + size]
        self.position += len(chunk)
        return chunk


class PlainSource(PausingSource):
    def __init__(self, data: bytes):
        super().__init__(data, lambda: None)


def test_two_uploads_of_the_same_blob_at_once_both_land_intact(tmp_path):
    store = ModelStore([tmp_path / "own"])
    data = os.urandom(recipes.CHUNK_BYTES * 2 + 12345)
    digest = sha_of(data)
    outcomes = []

    def second_upload():
        outcomes.append(recipes.receive_blob(store, digest, PlainSource(data), len(data), 0))
    first = recipes.receive_blob(store, digest, PausingSource(data, second_upload), len(data), 0)
    assert outcomes == [True] and first is True
    assert hashlib.sha256(blob_path(store.own_root, digest).read_bytes()).hexdigest() == digest.split(":", 1)[1]
    assert not [path for path in blob_path(store.own_root, digest).parent.iterdir() if path.name.endswith(".partial")]


def test_parallel_uploads_over_http_all_succeed(running):
    data = os.urandom(recipes.CHUNK_BYTES * 3)
    results = []
    threads = [threading.Thread(target=lambda: results.append(send_blob(running.url, data))) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(results)[-1] == 201 and set(results) <= {200, 201} and len(results) == 4
    assert blob_path(running.engine.store.own_root, sha_of(data)).read_bytes() == data


def test_a_model_can_not_be_made_from_a_blob_that_is_not_gguf(running):
    data = b"this is not a model"
    send_blob(running.url, data)
    http_json(f"{running.url}/api/create", {"model": "keeper", "from": "qwen3:14b", "stream": False})
    before = http_json(f"{running.url}/api/show", {"model": "keeper"})
    for body in ({"model": "keeper", "files": {"m.gguf": sha_of(data)}}, {"model": "keeper", "from": sha_of(data)},
                 {"model": "keeper", "from": "qwen3:14b", "adapters": {"a.gguf": sha_of(data)}}):
        status, answer = error_of(lambda: http_json(f"{running.url}/api/create", {**body, "stream": False}))
        assert status == 400 and "not a GGUF file" in answer["error"]
    assert http_json(f"{running.url}/api/show", {"model": "keeper"}) == before


@pytest.mark.parametrize("start", [
    b"GGUF" + struct.pack("<IQQ", 0x61646120, 0, 0),
    b"GGUF" + struct.pack("<IQQ", 1, 0, 0),
    b"GGUF" + struct.pack("<IQQ", 3, 2 ** 40, 0),
    b"GGUF" + struct.pack("<IQQ", 3, 0, 2 ** 40),
    b"GGUF" + struct.pack("<I", 3),
])
def test_a_blob_with_the_gguf_magic_but_no_sound_header_is_not_a_model(running, start):
    data = start + b" weights"
    send_blob(running.url, data)
    status, answer = error_of(lambda: http_json(f"{running.url}/api/create", {
        "model": "keeper", "from": "qwen3:14b", "adapters": {"a.gguf": sha_of(data)}, "stream": False}))
    assert status == 400 and "not a GGUF file" in answer["error"]


def test_a_sent_blob_is_readable_like_the_rest_of_the_store(running):
    data = GGUF_START + b"shared weights"
    send_blob(running.url, data)
    stored = blob_path(running.engine.store.own_root, sha_of(data))
    assert stat.S_IMODE(stored.stat().st_mode) == 0o644


def test_a_copy_that_would_fill_the_disk_is_refused(tmp_path, monkeypatch):
    other = tmp_path / "other"
    build_store(other)
    store = ModelStore([tmp_path / "own", other])
    monkeypatch.setattr(os, "link", lambda source, target: (_ for _ in ()).throw(OSError("cross-device link")))
    with pytest.raises(RecipeError) as raised:
        recipes.copy(store, "qwen3:14b", "mine", keep_free_bytes=10 ** 18)
    assert raised.value.status == 507 and "pull_keep_free_mb" in str(raised.value)
    with pytest.raises(RecipeError) as raised:
        recipes.create(store, {"model": "mine", "from": "qwen3:14b"}, lambda update: None, keep_free_bytes=10 ** 18)
    assert raised.value.status == 507
    assert not (tmp_path / "own" / "manifests").exists() or not list((tmp_path / "own" / "manifests").rglob("mine"))


def test_head_refuses_a_foreign_web_page_like_get_does(running):
    request = urllib.request.Request(f"{running.url}/api/blobs/sha256:{'0' * 64}", method="HEAD",
                                     headers={"Origin": "https://evil.example"})
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request, timeout=30)
    assert raised.value.code == 403
