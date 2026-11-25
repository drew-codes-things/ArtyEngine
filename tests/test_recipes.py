import hashlib
import json
import urllib.error
import urllib.request

import pytest

from arty_engine.recipes import RecipeError, parse_recipe
from engine_harness import RunningEngine, error_of, http_json
from fake_store import QWEN3
from gguf_writer import write_gguf

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
    data = b"adapter weights"
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
