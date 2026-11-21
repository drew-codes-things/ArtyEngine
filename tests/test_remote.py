import json
import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from arty_engine.config import Config, load_config
from engine_harness import RunningEngine, error_of, http_json, stream_chat
from fake_openai_server import (BROKEN_MODEL, ECHO_KEY_MODEL, FORBIDDEN_MODEL, GOOD_KEY, LIMITED_MODEL, REDIRECT_MODEL,
                                SLOW_MODEL,
                                FakeOpenAiServer)

REPO = Path(__file__).resolve().parents[1]
QWEN = "remote/friend/qwen3-14b-q4_k_m"
BIG = "remote/friend/friend-big-model-q8"
KEY_ENV = "ARTY_TEST_FRIEND_KEY"


@pytest.fixture
def upstream():
    server = FakeOpenAiServer()
    yield server
    server.close()


def key_file(tmp_path, key=GOOD_KEY, mode=0o600, file_name="friend.key") -> str:
    path = tmp_path / file_name
    path.write_text(key + "\n")
    path.chmod(mode)
    return str(path)


def remote_settings(upstream, tmp_path, **extra) -> dict:
    return {"friend": {"url": upstream.url, "key_file": key_file(tmp_path), **extra}}


@pytest.fixture
def engine(tmp_path, upstream):
    running = RunningEngine(tmp_path, {}, remote_models=remote_settings(upstream, tmp_path))
    yield running
    running.close()


def make_engine(tmp_path, remotes) -> RunningEngine:
    return RunningEngine(tmp_path, {}, remote_models=remotes)


def chat(engine, model, content="hello", **extra):
    return http_json(f"{engine.url}/api/chat", {"model": model, "messages": [{"role": "user", "content": content}],
                                                "stream": False, **extra})


def test_tags_list_the_remote_servers_models_under_a_remote_prefix(engine):
    names = [model["name"] for model in http_json(f"{engine.url}/api/tags")["models"]]
    assert QWEN in names and BIG in names
    assert any(not name.startswith("remote/") for name in names)


def test_remote_entries_say_they_are_remote_and_take_no_card_memory(engine):
    entry = next(model for model in http_json(f"{engine.url}/api/tags")["models"] if model["name"] == QWEN)
    assert entry["size"] == 0 and entry["remote"] is True
    assert entry["details"]["family"] == "remote"
    assert entry["capabilities"] == ["completion", "tools"]


def test_a_chat_goes_to_the_remote_server_with_its_key_and_its_own_model_name(engine, upstream):
    reply = chat(engine, QWEN, options={"num_ctx": 8192, "top_k": 20, "temperature": 0.5, "num_predict": 64})
    assert reply["message"]["content"] == "remote says: hello"
    assert reply["model"] == QWEN and reply["done"] is True
    sent = upstream.requests[-1]
    assert sent["authorization"] == f"Bearer {GOOD_KEY}"
    assert sent["body"]["model"] == "Qwen3-14B-Q4_K_M"
    assert sent["body"]["temperature"] == 0.5 and sent["body"]["max_tokens"] == 64
    assert "top_k" not in sent["body"] and "num_ctx" not in sent["body"]
    assert engine.launcher.launched == []


def test_a_remote_chat_returns_tool_calls_and_thinking_the_way_the_brain_reads_them(engine):
    tools = [{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}]
    reply = chat(engine, QWEN, tools=tools)
    assert reply["message"]["tool_calls"][0]["function"] == {"index": 0, "name": "get_time", "arguments": {"zone": "UTC"}}
    assert reply["message"]["thinking"] == "thought first"


def test_a_streamed_remote_chat_streams_text_thinking_and_tool_calls(engine):
    tools = [{"type": "function", "function": {"name": "get_time", "parameters": {"type": "object"}}}]
    texts = []
    final, lines = stream_chat(engine.url, {"model": QWEN, "messages": [{"role": "user", "content": "hi"}], "tools": tools},
                               texts.append)
    assert "".join(texts) == "remote stream"
    assert any(line["message"].get("thinking") == "thinking it over" for line in lines if "message" in line)
    assert final["message"]["tool_calls"][0]["function"]["arguments"] == {"zone": "UTC"}
    assert lines[-1]["done"] is True and lines[-1]["eval_count"] == 2


def test_generate_works_with_a_remote_model(engine):
    reply = http_json(f"{engine.url}/api/generate", {"model": QWEN, "prompt": "ping", "stream": False})
    assert reply["response"] == "remote says: ping"


def test_loading_or_unloading_a_remote_model_starts_nothing_and_answers(engine, upstream):
    loaded = http_json(f"{engine.url}/api/generate", {"model": QWEN, "keep_alive": "10m"})
    unloaded = http_json(f"{engine.url}/api/generate", {"model": QWEN, "keep_alive": 0})
    assert loaded["done_reason"] == "load" and unloaded["done_reason"] == "unload"
    assert engine.launcher.launched == [] and upstream.requests == []
    assert http_json(f"{engine.url}/api/ps")["models"] == []


def test_show_gives_capabilities_and_never_the_key_or_the_address(engine, upstream):
    shown = http_json(f"{engine.url}/api/show", {"model": QWEN})
    assert shown["capabilities"] == ["completion", "tools"]
    assert shown["remote"] == {"name": "friend", "model": "Qwen3-14B-Q4_K_M"}
    text = json.dumps(shown)
    assert GOOD_KEY not in text and upstream.url not in text


def test_a_wrong_key_is_a_readable_refusal_without_the_key(tmp_path, upstream):
    wrong = "sf_wrong_key_value"
    running = make_engine(tmp_path, {"friend": {"url": upstream.url, "key_file": key_file(tmp_path, wrong),
                                                "models": ["Qwen3-14B-Q4_K_M"]}})
    try:
        status, payload = error_of(lambda: chat(running, QWEN))
    finally:
        running.close()
    assert status == 502
    assert "refused the key (HTTP 401)" in payload["error"] and "friend.key" in payload["error"]
    assert wrong not in payload["error"]


@pytest.mark.parametrize("model, status, words", [
    (LIMITED_MODEL, 429, "is limiting requests (HTTP 429), try again in 7 s"),
    (FORBIDDEN_MODEL, 502, "refused access (HTTP 403)"),
    (BROKEN_MODEL, 502, "failed (HTTP 500): upstream crashed"),
])
def test_remote_refusals_become_readable_errors(tmp_path, upstream, model, status, words):
    running = make_engine(tmp_path, remote_settings(upstream, tmp_path, models=[model]))
    try:
        got_status, payload = error_of(lambda: chat(running, f"remote/friend/{model}"))
    finally:
        running.close()
    assert got_status == status and words in payload["error"]


def test_a_key_echoed_back_by_the_remote_server_is_hidden(tmp_path, upstream):
    running = make_engine(tmp_path, remote_settings(upstream, tmp_path, models=[ECHO_KEY_MODEL]))
    try:
        status, payload = error_of(lambda: chat(running, f"remote/friend/{ECHO_KEY_MODEL}"))
    finally:
        running.close()
    assert status == 502 and GOOD_KEY not in payload["error"] and "[key hidden]" in payload["error"]


def test_a_redirect_is_not_followed_so_the_key_goes_nowhere_else(tmp_path, upstream):
    running = make_engine(tmp_path, remote_settings(upstream, tmp_path, models=[REDIRECT_MODEL]))
    try:
        status, payload = error_of(lambda: chat(running, f"remote/friend/{REDIRECT_MODEL}"))
    finally:
        running.close()
    assert status == 502 and "redirect (HTTP 302)" in payload["error"]
    assert [request["path"] for request in upstream.requests] == ["/v1/chat/completions"]


def test_a_remote_server_that_does_not_answer_in_time_is_a_504(tmp_path, upstream):
    running = make_engine(tmp_path, remote_settings(upstream, tmp_path, models=[SLOW_MODEL], timeout_seconds=1))
    started = time.monotonic()
    try:
        status, payload = error_of(lambda: chat(running, f"remote/friend/{SLOW_MODEL}"))
    finally:
        running.close()
    assert status == 504 and "did not answer within 1 s" in payload["error"]
    assert time.monotonic() - started < 2.5


def test_an_unreachable_remote_server_is_left_out_of_the_list_and_a_chat_says_why(tmp_path, upstream):
    address = upstream.url
    upstream.close()
    running = make_engine(tmp_path, {"friend": {"url": address, "key_file": key_file(tmp_path)}})
    try:
        names = [model["name"] for model in http_json(f"{running.url}/api/tags")["models"]]
        status, payload = error_of(lambda: chat(running, QWEN))
    finally:
        running.close()
    assert not any(name.startswith("remote/") for name in names) and names
    assert status == 502 and "isn't reachable" in payload["error"]


def test_a_key_file_other_users_can_read_is_refused(tmp_path, upstream):
    running = make_engine(tmp_path, {"friend": {"url": upstream.url, "key_file": key_file(tmp_path, mode=0o644),
                                                "models": ["Qwen3-14B-Q4_K_M"]}})
    try:
        status, payload = error_of(lambda: chat(running, QWEN))
    finally:
        running.close()
    assert status == 500 and "chmod 600" in payload["error"] and GOOD_KEY not in payload["error"]
    assert upstream.requests == []


def test_the_key_can_come_from_an_environment_variable(tmp_path, upstream, monkeypatch):
    monkeypatch.setenv(KEY_ENV, GOOD_KEY)
    running = make_engine(tmp_path, {"friend": {"url": upstream.url, "key_env": KEY_ENV}})
    try:
        reply = chat(running, QWEN)
    finally:
        running.close()
    assert reply["message"]["content"] == "remote says: hello"


def test_a_missing_environment_variable_says_which(tmp_path, upstream, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    running = make_engine(tmp_path, {"friend": {"url": upstream.url, "key_env": KEY_ENV, "models": ["x"]}})
    try:
        status, payload = error_of(lambda: chat(running, "remote/friend/x"))
    finally:
        running.close()
    assert status == 500 and f"${KEY_ENV} is not set" in payload["error"]


def test_listed_models_in_the_config_skip_asking_the_server(tmp_path, upstream):
    running = make_engine(tmp_path, remote_settings(upstream, tmp_path, models=["Only-This"]))
    try:
        names = [model["name"] for model in http_json(f"{running.url}/api/tags")["models"] if model["name"].startswith("remote/")]
    finally:
        running.close()
    assert names == ["remote/friend/only-this"] and upstream.listings == 0


def test_the_model_list_is_asked_for_once_a_minute_not_on_every_listing(engine, upstream):
    for _ in range(3):
        http_json(f"{engine.url}/api/tags")
    chat(engine, QWEN)
    assert upstream.listings == 1


def test_thinking_switches_reach_the_remote_server_only_when_it_is_marked_as_thinking(tmp_path, upstream):
    plain = make_engine(tmp_path / "plain", remote_settings(upstream, tmp_path, models=["m"]))
    thinking = make_engine(tmp_path / "thinking", {"friend": {"url": upstream.url, "key_file": key_file(tmp_path),
                                                              "models": ["m"], "capabilities": ["tools", "thinking"]}})
    try:
        chat(plain, "remote/friend/m", think=False)
        without = upstream.requests[-1]["body"]
        chat(thinking, "remote/friend/m", think=True)
        with_switch = upstream.requests[-1]["body"]
        status, payload = error_of(lambda: chat(plain, "remote/friend/m", think=True))
    finally:
        plain.close()
        thinking.close()
    assert "chat_template_kwargs" not in without
    assert with_switch["chat_template_kwargs"] == {"enable_thinking": True}
    assert status == 400 and "does not support thinking" in payload["error"]


@pytest.mark.parametrize("model", ["remote/nobody/x", "remote/friend/not-there"])
def test_an_unknown_remote_model_is_a_404(engine, model):
    status, payload = error_of(lambda: chat(engine, model))
    assert status == 404 and "not found" in payload["error"]


def test_remote_models_are_not_embedded_rendered_pulled_or_deleted(engine):
    embed_status, _ = error_of(lambda: http_json(f"{engine.url}/api/embed", {"model": QWEN, "input": "x"}))
    raw_status, _ = error_of(lambda: http_json(f"{engine.url}/api/generate", {"model": QWEN, "prompt": "x", "raw": True,
                                                                             "stream": False}))
    pull_status, pull = error_of(lambda: http_json(f"{engine.url}/api/pull", {"model": QWEN, "stream": False}))
    delete_request = urllib.request.Request(f"{engine.url}/api/delete", data=json.dumps({"model": QWEN}).encode(), method="DELETE")
    delete_status, delete = error_of(lambda: urllib.request.urlopen(delete_request, timeout=10))
    assert embed_status == 400 and raw_status == 501
    assert pull_status == 400 and "remote" in pull["error"]
    assert delete_status == 405 and "remote_models" in delete["error"]


def test_the_key_never_reaches_the_engine_log(tmp_path, upstream, caplog):
    caplog.set_level(logging.DEBUG)
    running = make_engine(tmp_path, {"friend": {"url": upstream.url, "key_file": key_file(tmp_path, "sf_wrong_and_secret")},
                                     "other": {"url": upstream.url, "key_file": key_file(tmp_path, file_name="other.key")}})
    try:
        http_json(f"{running.url}/api/tags")
        error_of(lambda: chat(running, QWEN))
        chat(running, "remote/other/qwen3-14b-q4_k_m")
    finally:
        running.close()
    assert "refused the key" in caplog.text
    assert "sf_wrong_and_secret" not in caplog.text and GOOD_KEY not in caplog.text


@pytest.mark.parametrize("settings, words", [
    ({"friend": {"url": "https://x.example/v1", "key": "sf_inline"}}, "never put the key in the config"),
    ({"friend": {"url": "https://x.example/v1", "api_key": "sf_inline"}}, "never put the key in the config"),
    ({"Friend": {"url": "https://x.example/v1"}}, "lower case"),
    ({"friend": {"url": "ftp://x.example/v1"}}, "http:// or https://"),
    ({"friend": {"url": "https://user:pass@x.example/v1"}}, "no user, password"),
    ({"friend": {"key_file": "/k"}}, "url is required"),
    ({"friend": {"url": "https://x.example/v1", "key_file": "/k", "key_env": "K"}}, "not both"),
    ({"friend": {"url": "https://x.example/v1", "capabilities": ["embedding"]}}, "capabilities"),
    ({"friend": {"url": "https://x.example/v1", "timeout_seconds": 0}}, "timeout_seconds"),
    ({"friend": {"url": "https://x.example/v1", "colour": "red"}}, "unknown settings colour"),
    (["friend"], "remote_models must be an object"),
])
def test_bad_remote_settings_are_refused_when_the_config_loads(settings, words):
    with pytest.raises(ValueError, match=words):
        Config(remote_models=settings)


def test_a_config_file_with_remote_models_loads(tmp_path):
    path = tmp_path / "engine.json"
    path.write_text(json.dumps({"remote_models": {"friend": {"url": "https://x.example/v1", "key_file": "/srv/k"}}}))
    assert load_config(str(path)).remote_models["friend"]["key_file"] == "/srv/k"


def test_write_config_adds_the_sites_remote_models(tmp_path):
    remotes = tmp_path / "remote-models.json"
    remotes.write_text(json.dumps({"friend": {"url": "https://x.example/v1", "key_file": "/srv/k"}}))
    target = tmp_path / "engine.json"
    finished = subprocess.run([sys.executable, str(REPO / "tools" / "write_config.py"), str(REPO / "deploy" / "engine.json"),
                               str(target), "--trainer-url", "", "--remote-models", str(remotes)],
                              capture_output=True, text=True, timeout=60)
    assert finished.returncode == 0, finished.stderr
    assert json.loads(target.read_text())["remote_models"] == {"friend": {"url": "https://x.example/v1", "key_file": "/srv/k"}}


def test_write_config_refuses_remote_models_with_a_key_in_them(tmp_path):
    remotes = tmp_path / "remote-models.json"
    remotes.write_text(json.dumps({"friend": {"url": "https://x.example/v1", "key": "sf_inline"}}))
    target = tmp_path / "engine.json"
    finished = subprocess.run([sys.executable, str(REPO / "tools" / "write_config.py"), str(REPO / "deploy" / "engine.json"),
                               str(target), "--trainer-url", "", "--remote-models", str(remotes)],
                              capture_output=True, text=True, timeout=60)
    assert finished.returncode == 1 and "never put the key in the config" in finished.stderr
    assert not target.exists() and "sf_inline" not in finished.stderr
