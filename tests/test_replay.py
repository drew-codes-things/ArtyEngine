import json
from pathlib import Path

import pytest

from engine_harness import RunningEngine, error_of, http_json

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RECORDED = json.loads((FIXTURES / "runner_raw_recorded.json").read_text())
RENDERED = RECORDED["apply_template"]["json"]["prompt"]
WRITTEN = RECORDED["completion"]["json"]["content"]
IMAGE_REFUSAL = json.loads(RECORDED["apply_template_image"]["error"])["error"]["message"]
TOOLS = RECORDED["apply_template"]["body"]["tools"]
BRAIN_BODY = {"model": "qwen3:14b", "messages": RECORDED["apply_template"]["body"]["messages"], "tools": TOOLS, "think": False,
              "options": {"temperature": 0, "num_predict": 40, "seed": 1, "num_ctx": 8192}, "keep_alive": -1}


def apply_template_reply(body):
    if body.get("model") == "untemplated:latest":
        return {"error": IMAGE_REFUSAL, "status": 500}
    return {"json": RECORDED["apply_template"]["json"]}


def completion_reply(body):
    case = RECORDED["completion_stream" if body.get("stream") else "completion"]
    if "sse" in case:
        return {"sse": case["sse"]}
    if body.get("max_tokens") == 3:
        return {"json": {**case["json"], "content": "<tool_call>\n{", "stop_type": "limit"}}
    return {"json": case["json"]}


@pytest.fixture
def engine(tmp_path):
    running = RunningEngine(tmp_path, {"/apply-template": apply_template_reply, "/completion": completion_reply}, default_keep_alive="-1")
    yield running
    running.close()


def written_reply(url: str, body: dict) -> str:
    shown = http_json(f"{url}/api/chat", {**body, "stream": False, "_debug_render_only": True})
    replay = {"model": body["model"], "prompt": shown["_debug_info"]["rendered_template"], "raw": True, "stream": False,
              "options": body.get("options") or {}}
    replay.update({key: body[key] for key in ("keep_alive", "shift") if key in body})
    return http_json(f"{url}/api/generate", replay).get("response") or ""


def upstream(engine, path):
    return [request["body"] for request in engine.upstream_requests() if request["path"] == path]


def test_render_only_returns_the_template_the_runner_rendered(engine):
    shown = http_json(f"{engine.url}/api/chat", {**BRAIN_BODY, "stream": False, "_debug_render_only": True})
    assert shown["_debug_info"] == {"rendered_template": RENDERED}
    assert shown["model"] == "qwen3:14b" and shown["done"] is True
    sent = upstream(engine, "/apply-template")[-1]
    assert sent["messages"] == BRAIN_BODY["messages"] and sent["tools"] == TOOLS
    assert sent["chat_template_kwargs"] == {"enable_thinking": False} and "stream" not in sent
    assert upstream(engine, "/v1/chat/completions") == []


def test_render_only_counts_images(engine):
    shown = http_json(f"{engine.url}/api/chat", {"model": "mimo:9b", "stream": False, "_debug_render_only": True,
                                                 "messages": [{"role": "user", "content": "what", "images": ["iVBORw0KGgo="]}]})
    assert shown["_debug_info"] == {"rendered_template": RENDERED, "image_count": 1}
    assert upstream(engine, "/apply-template")[-1]["messages"][0]["content"][1]["type"] == "image_url"


def test_render_only_passes_runner_refusals_on(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/chat", {"model": "untemplated", "stream": False, "_debug_render_only": True,
                                                                           "messages": [{"role": "user", "content": "hi"}]}))
    assert status == 500 and body["error"] == IMAGE_REFUSAL


def test_raw_generate_completes_the_prompt_as_written(engine):
    reply = http_json(f"{engine.url}/api/generate", {"model": "qwen3:14b", "prompt": RENDERED, "raw": True, "stream": False,
                                                     "options": {"temperature": 0, "num_predict": 40, "seed": 1}})
    assert reply["response"] == WRITTEN and reply["done"] is True and reply["done_reason"] == "stop"
    assert reply["eval_count"] == 24 and reply["prompt_eval_count"] == 167 and "message" not in reply
    sent = upstream(engine, "/completion")[-1]
    assert sent["prompt"] == RENDERED and sent["max_tokens"] == 40 and sent["temperature"] == 0 and sent["seed"] == 1
    assert sent["stream"] is False


def test_raw_generate_hitting_the_limit_says_length(engine):
    reply = http_json(f"{engine.url}/api/generate", {"model": "qwen3:14b", "prompt": RENDERED, "raw": True, "stream": False,
                                                     "options": {"num_predict": 3}})
    assert reply["done_reason"] == "length"


def test_raw_generate_streams_api_chunks(engine):
    import urllib.request
    request = urllib.request.Request(f"{engine.url}/api/generate", method="POST",
                                     data=json.dumps({"model": "qwen3:14b", "prompt": RENDERED, "raw": True,
                                                      "options": {"temperature": 0, "num_predict": 40, "seed": 1}}).encode())
    with urllib.request.urlopen(request, timeout=30) as response:
        chunks = [json.loads(line) for line in response if line.strip()]
    assert "".join(chunk["response"] for chunk in chunks) == WRITTEN
    assert all(chunk["done"] is False for chunk in chunks[:-1])
    assert chunks[-1]["done"] is True and chunks[-1]["done_reason"] == "stop" and chunks[-1]["eval_count"] == 24


def test_the_brains_empty_reply_replay_gets_what_the_model_wrote(engine):
    assert written_reply(engine.url, BRAIN_BODY) == WRITTEN
    assert len(engine.launcher.launched) == 1


def test_raw_images_are_still_refused_clearly(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/generate", {"model": "mimo:9b", "prompt": "x", "raw": True,
                                                                               "images": ["iVBORw0KGgo="], "stream": False}))
    assert status == 501 and "images" in body["error"]
