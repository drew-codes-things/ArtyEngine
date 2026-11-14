import json
from pathlib import Path

import pytest

from engine_harness import RunningEngine, http_json, stream_chat

FIXTURES = Path(__file__).resolve().parent / "fixtures"
API_RECORDED = json.loads((FIXTURES / "api_recorded.json").read_text())
UPSTREAM = json.loads((FIXTURES / "runner_recorded.json").read_text())


REQUEST_FIELDS = ("messages", "tools", "response_format", "chat_template_kwargs", "max_tokens", "stream")


def request_part(body: dict) -> dict:
    return {field: body.get(field) for field in REQUEST_FIELDS}


def replay(body):
    for recorded in UPSTREAM.values():
        if request_part(recorded["body"]) == request_part(body):
            return {"json": recorded["json"]} if "json" in recorded else {"sse": recorded["sse"]}
    return {"error": "no recording for this request", "status": 500}


@pytest.fixture
def engine(tmp_path):
    running = RunningEngine(tmp_path, {"/v1/chat/completions": replay})
    yield running
    running.close()


def shape(value):
    if isinstance(value, dict):
        return {key: shape(item) for key, item in value.items() if key not in ("id", "created_at")}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


@pytest.mark.parametrize("case", ["judge_json", "tool_round", "tool_result", "think_unset"])
def test_recorded_brain_requests_get_the_recorded_reply_shape(engine, case):
    recorded = API_RECORDED[case]
    reply = http_json(f"{engine.url}/api/chat", recorded["request"])
    assert shape(reply) == shape(recorded["reply"])
    sent = engine.upstream_requests()[-1]["body"]
    assert request_part(sent) == request_part(UPSTREAM[case]["body"])


def test_recorded_stream_matches_the_recorded_chunks(engine):
    recorded = API_RECORDED["no_think_stream"]
    pieces = []
    reply, lines = stream_chat(engine.url, recorded["request"], pieces.append)
    assert reply["message"]["content"] == "The train arrives at **17:15**."
    assert shape(lines[0]) == shape(recorded["stream"][0])
    assert shape(lines[-1]) == shape(recorded["stream"][-1])


def test_tool_call_arguments_come_back_as_objects(engine):
    reply = http_json(f"{engine.url}/api/chat", API_RECORDED["tool_round"]["request"])
    assert reply["message"]["tool_calls"][0]["function"]["arguments"] == {"expression": "12 * 15"}


def test_think_unset_returns_thinking_separately(engine):
    reply = http_json(f"{engine.url}/api/chat", API_RECORDED["think_unset"]["request"])
    assert reply["message"]["thinking"] and "<think>" not in reply["message"]["content"]
    assert "thinking" in API_RECORDED["think_unset"]["reply"]["message"]
