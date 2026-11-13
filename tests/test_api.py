import json

import pytest

from engine_harness import RunningEngine, error_of, http_json, stream_chat

TIMINGS = {"prompt_n": 40, "prompt_ms": 20.0, "predicted_n": 5, "predicted_ms": 50.0}


def chat_reply(body):
    if body.get("tools"):
        return {"json": {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x1", "type": "function", "function": {"name": "calculate", "arguments": "{\"expression\":\"12*15\"}"}}]}}], "timings": TIMINGS}}
    if body.get("stream"):
        events = [{"choices": [{"delta": {"reasoning_content": "hmm"}}]}, {"choices": [{"delta": {"content": "17:"}}]},
                  {"choices": [{"delta": {"content": "15"}}]}, {"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": TIMINGS}]
        return {"sse": [f"data: {json.dumps(event)}" for event in events] + ["data: [DONE]"]}
    return {"json": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "{\"kind\": \"action\"}"}}], "timings": TIMINGS}}


REPLIES = {"/v1/chat/completions": chat_reply,
           "/v1/embeddings": lambda body: {"json": {"data": [{"index": i, "embedding": [0.1 * i, 0.2]} for i, _ in enumerate(body["input"])],
                                                    "usage": {"prompt_tokens": 7}}}}


@pytest.fixture
def engine(tmp_path):
    running = RunningEngine(tmp_path, REPLIES, default_keep_alive="-1")
    yield running
    running.close()


def test_tags_lists_models_like_ollama(engine):
    models = http_json(f"{engine.url}/api/tags")["models"]
    assert {model["name"] for model in models} >= {"qwen3:14b", "mimo:9b", "nomic-embed-text:latest"}
    assert all(set(model) == {"name", "model", "modified_at", "size", "digest", "details", "capabilities"} for model in models)


def test_show_gives_capabilities_and_details_the_brain_reads(engine):
    shown = http_json(f"{engine.url}/api/show", {"model": "qwen3:14b"})
    assert shown["capabilities"] == ["completion", "tools", "thinking"]
    assert shown["details"]["parameter_size"] and "model_info" in shown


def test_judge_call_shape(engine):
    reply = http_json(f"{engine.url}/api/chat", {"model": "qwen3:14b", "stream": False, "format": "json", "think": False, "keep_alive": -1,
                                                 "options": {"temperature": 0, "num_predict": 120, "num_ctx": 12288},
                                                 "messages": [{"role": "user", "content": "classify"}]})
    assert json.loads(reply["message"]["content"]) == {"kind": "action"}
    assert reply["done"] is True and reply["done_reason"] == "stop" and reply["eval_count"] == 5
    sent = engine.upstream_requests()[-1]["body"]
    assert sent["response_format"] == {"type": "json_object"} and sent["chat_template_kwargs"] == {"enable_thinking": False}


def test_tool_round_returns_ollama_tool_calls(engine):
    reply = http_json(f"{engine.url}/api/chat", {"model": "qwen3:14b", "stream": False, "messages": [{"role": "user", "content": "12x15"}],
                                                 "tools": [{"type": "function", "function": {"name": "calculate", "parameters": {}}}]})
    assert reply["message"]["tool_calls"][0]["function"] == {"index": 0, "name": "calculate", "arguments": {"expression": "12*15"}}


def test_streamed_reply_works_with_the_brains_reader(engine):
    pieces = []
    reply, lines = stream_chat(engine.url, {"model": "qwen3:14b", "think": True, "messages": [{"role": "user", "content": "train"}]}, pieces.append)
    assert pieces == ["17:", "15"] and reply["message"]["content"] == "17:15"
    assert lines[0]["message"]["thinking"] == "hmm" and lines[-1]["done"] is True and lines[-1]["eval_count"] == 5


def test_load_then_ps_then_unload(engine):
    loaded = http_json(f"{engine.url}/api/generate", {"model": "qwen3:14b", "keep_alive": "10m"})
    assert loaded["done"] is True and loaded["done_reason"] == "load"
    running = http_json(f"{engine.url}/api/ps")["models"]
    assert [model["name"] for model in running] == ["qwen3:14b"]
    assert set(running[0]) >= {"name", "model", "size", "digest", "details", "expires_at", "size_vram", "context_length"}
    http_json(f"{engine.url}/api/generate", {"model": "qwen3:14b", "keep_alive": 0})
    assert http_json(f"{engine.url}/api/ps")["models"] == []


def test_embed_and_legacy_embeddings(engine):
    reply = http_json(f"{engine.url}/api/embed", {"model": "nomic-embed-text", "input": ["a", "b"]})
    assert reply["embeddings"] == [[0.0, 0.2], [0.1, 0.2]] and reply["prompt_eval_count"] == 7
    assert http_json(f"{engine.url}/api/embeddings", {"model": "nomic-embed-text", "prompt": "a"})["embedding"] == [0.0, 0.2]


def test_unknown_model_is_404_like_ollama(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/chat", {"model": "nope:1b", "messages": []}))
    assert status == 404 and body == {"error": "model 'nope:1b' not found"}


def test_ollama_engine_only_model_says_why(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/chat", {"model": "qwen2.5vl:7b", "stream": False,
                                                                           "messages": [{"role": "user", "content": "hi"}]}))
    assert status == 501 and "Ollama's own engine format" in body["error"]


def test_chatting_with_an_embedding_model_is_refused(engine):
    status, _ = error_of(lambda: http_json(f"{engine.url}/api/chat", {"model": "nomic-embed-text", "stream": False,
                                                                        "messages": [{"role": "user", "content": "hi"}]}))
    assert status == 400


def test_bad_json_is_a_400(engine):
    import urllib.request
    request = urllib.request.Request(f"{engine.url}/api/chat", data=b"{nope", method="POST")
    status, body = error_of(lambda: urllib.request.urlopen(request))
    assert status == 400 and "invalid JSON" in body["error"]


def test_bad_keep_alive_is_a_400(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/chat", {"model": "qwen3:14b", "keep_alive": "soon", "messages": []}))
    assert status == 400 and "keep_alive" in body["error"]


def test_upstream_errors_come_back_as_ollama_errors(tmp_path):
    running = RunningEngine(tmp_path, {"/v1/chat/completions": lambda body: {"error": "the request exceeds the context size", "status": 400}})
    try:
        status, body = error_of(lambda: http_json(f"{running.url}/api/chat", {"model": "qwen3:14b", "stream": False,
                                                                               "messages": [{"role": "user", "content": "x"}]}))
        assert status == 400 and body == {"error": "the request exceeds the context size"}
    finally:
        running.close()


def test_delete_never_touches_ollamas_store(engine):
    import urllib.request
    request = urllib.request.Request(f"{engine.url}/api/delete", data=b'{"model":"qwen3:14b"}', method="DELETE")
    status, _ = error_of(lambda: urllib.request.urlopen(request))
    assert status == 405


def test_cpu_embeddings_skip_the_trainer_and_the_gpu(tmp_path):
    running = RunningEngine(tmp_path, REPLIES, trainer_url="http://127.0.0.1:9")
    calls = []
    running.engine.trainer.busy = lambda request_id: calls.append("busy")
    running.engine.trainer.done = lambda request_id: calls.append("done")
    try:
        http_json(f"{running.url}/api/embed", {"model": "nomic-embed-text", "input": ["a"], "options": {"num_gpu": 0}})
        assert calls == [] and "--device" in running.launcher.launched[0].arguments
        http_json(f"{running.url}/api/embed", {"model": "nomic-embed-text", "input": ["a"]})
        assert calls == ["busy", "done"]
    finally:
        running.close()


def test_generate_load_reply_matches_ollama(engine):
    loaded = http_json(f"{engine.url}/api/generate", {"model": "qwen3:14b"})
    assert loaded["response"] == "" and "message" not in loaded
    chat_loaded = http_json(f"{engine.url}/api/chat", {"model": "qwen3:14b", "messages": []})
    assert chat_loaded["message"] == {"role": "assistant", "content": ""} and "response" not in chat_loaded


def test_internal_models_stay_out_of_the_model_list_but_still_answer(tmp_path):
    running = RunningEngine(tmp_path, REPLIES, internal_models=["untemplated"])
    try:
        names = [model["name"] for model in http_json(f"{running.url}/api/tags")["models"]]
        assert "untemplated:latest" not in names and "qwen3:14b" in names
        assert http_json(f"{running.url}/api/show", {"model": "untemplated:latest"})["details"]["family"] == "qwen3"
        reply = http_json(f"{running.url}/api/chat", {"model": "untemplated", "stream": False, "messages": [{"role": "user", "content": "hi"}]})
        assert reply["done"] is True
    finally:
        running.close()


def test_ollama_engine_only_model_points_at_a_llama_cpp_build(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/generate", {"model": "qwen2.5vl:7b", "prompt": "hi", "stream": False}))
    assert status == 501 and "hf.co/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M" in body["error"]
