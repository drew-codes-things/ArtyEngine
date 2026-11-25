import json
import urllib.request

import pytest

from arty_engine.translate import without_oldest_turn
from engine_harness import RunningEngine, error_of, http_json

OVERFLOW = "request (900 tokens) exceeds the available context size (512 tokens), try increasing it"
MAX_MESSAGES = 3


def chat_reply(body):
    if len(body["messages"]) > MAX_MESSAGES:
        return {"error": OVERFLOW, "status": 400}
    if body.get("stream"):
        return {"sse": ['data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}]}', "data: [DONE]"]}
    return {"json": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "ok"}}]}}


REPLIES = {
    "/v1/chat/completions": chat_reply,
    "/infill": {"json": {"content": "b = 2", "stop": True, "tokens_evaluated": 5, "tokens_predicted": 3}},
    "/completion": {"json": {"content": '{"a": 1}', "stop": True}},
}
LONG_CHAT = [{"role": "system", "content": "s"}, {"role": "user", "content": "1"}, {"role": "assistant", "content": "a"},
             {"role": "user", "content": "2"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "3"}]


@pytest.fixture
def running(tmp_path):
    engine = RunningEngine(tmp_path, REPLIES, vram_budget_mb=100000)
    yield engine
    engine.close()


def last_body(running, path):
    return [request["body"] for request in running.upstream_requests() if request["path"] == path][-1]


def test_a_long_chat_loses_its_oldest_turns_until_it_fits(running):
    answer = http_json(f"{running.url}/api/chat", {"model": "qwen3:14b", "messages": LONG_CHAT, "stream": False})
    assert answer["message"]["content"] == "ok"
    assert [message["content"] for message in last_body(running, "/v1/chat/completions")["messages"]] == ["s", "3"]


def test_a_streamed_chat_is_trimmed_before_it_starts(running):
    request = urllib.request.Request(f"{running.url}/api/chat", method="POST", data=json.dumps(
        {"model": "qwen3:14b", "messages": LONG_CHAT, "stream": True}).encode())
    with urllib.request.urlopen(request, timeout=30) as response:
        lines = [json.loads(line) for line in response if line.strip()]
    assert "".join(line["message"]["content"] for line in lines) == "ok" and lines[-1]["done"]


def test_truncate_false_keeps_the_error(running):
    status, body = error_of(lambda: http_json(f"{running.url}/api/chat", {"model": "qwen3:14b", "messages": LONG_CHAT,
                                                                           "stream": False, "truncate": False}))
    assert status == 400 and "exceeds the available context size" in body["error"]


def test_a_single_turn_that_is_too_long_still_fails():
    assert without_oldest_turn([{"role": "system", "content": "s"}, {"role": "user", "content": "x"}]) is None


def test_trimming_keeps_tool_results_with_their_turn():
    messages = [{"role": "user", "content": "1"}, {"role": "assistant", "content": "", "tool_calls": [{}]},
                {"role": "tool", "content": "r"}, {"role": "assistant", "content": "a"}, {"role": "user", "content": "2"}]
    assert without_oldest_turn(messages) == [{"role": "user", "content": "2"}]


def test_a_suffix_fills_in_the_middle(running):
    answer = http_json(f"{running.url}/api/generate", {"model": "qwen3:14b", "prompt": "a = 1\n", "suffix": "\nc = 3",
                                                       "stream": False})
    assert answer["response"] == "b = 2" and answer["done"]
    sent = last_body(running, "/infill")
    assert sent["input_prefix"] == "a = 1\n" and sent["input_suffix"] == "\nc = 3" and sent["temperature"] == 0.6


@pytest.mark.parametrize("format_value, schema", [("json", {"type": "object"}),
                                                  ({"type": "object", "properties": {"a": {"type": "integer"}}},
                                                   {"type": "object", "properties": {"a": {"type": "integer"}}})])
def test_a_raw_prompt_can_ask_for_json(running, format_value, schema):
    answer = http_json(f"{running.url}/api/generate", {"model": "qwen3:14b", "prompt": "<s>", "raw": True, "format": format_value,
                                                       "stream": False})
    assert answer["response"] == '{"a": 1}' and last_body(running, "/completion")["json_schema"] == schema


def test_a_template_override_is_refused_clearly(running):
    status, body = error_of(lambda: http_json(f"{running.url}/api/generate", {"model": "qwen3:14b", "prompt": "x",
                                                                               "template": "{{ .Prompt }}"}))
    assert status == 501 and "raw: true" in body["error"]


def test_a_suffix_for_a_model_without_fill_in_tokens_answers_501(tmp_path):
    engine = RunningEngine(tmp_path, {"/infill": {"error": "Infill is not supported by this model", "status": 501}},
                           vram_budget_mb=100000)
    try:
        for stream in (False, True):
            status, body = error_of(lambda: http_json(f"{engine.url}/api/generate", {
                "model": "qwen3:14b", "prompt": "a", "suffix": "c", "stream": stream}))
            assert status == 501 and "not supported" in body["error"]
    finally:
        engine.close()
