import json
import urllib.request

import pytest

from engine_harness import RunningEngine, error_of, http_json

UPSTREAM_WORD = "ll" + "ama"
CHAT_REPLY = {"json": {"id": "c1", "object": "chat.completion", "model": "qwen3:14b", "system_fingerprint": "b12110-cpu",
                       "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}]}}
CHAT_EVENTS = {"sse": ['data: {"choices":[{"index":0,"delta":{"content":"h"}}],"model":"x","system_fingerprint":"b12110-cpu"}',
                       'data: {"choices":[{"index":0,"delta":{"content":"i"},"finish_reason":"stop"}],"model":"x"}',
                       "data: [DONE]"]}
REPLIES = {
    "/v1/chat/completions": lambda body: CHAT_EVENTS if body.get("stream") else CHAT_REPLY,
    "/v1/completions": {"json": {"object": "text_completion", "choices": [{"text": " world", "index": 0}]}},
    "/v1/embeddings": {"json": {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}]}},
    "/tokenize": {"json": {"tokens": [1, 2, 3]}},
    "/detokenize": {"json": {"content": "abc"}},
    "/apply-template": {"json": {"prompt": "<user>hi"}},
    "/infill": {"json": {"content": "middle"}},
    "/props": {"json": {"default_generation_settings": {"n_ctx": 1024}, "total_slots": 1, "build_info": "b12110-0a1b",
                        "model_path": "/srv/ai/models/blobs/sha256-abc"}},
    "/slots": {"json": [{"id": 0, "is_processing": False}]},
    "/slots/0?action=save": {"json": {"id_slot": 0, "filename": "chat.bin", "n_saved": 12}},
    "/metrics": {"text": "# HELP upstream:prompt_tokens_total Prompt tokens.\n# TYPE upstream:prompt_tokens_total counter\n"
                         "upstream:prompt_tokens_total 42\nrequests{state=\"busy\"} 1\n"
                         f"# HELP upstream:n_decode_total Total number of {UPSTREAM_WORD}_decode() calls\n"
                         "# HELP upstream:brand_new_total Something new\nupstream:brand_new_total 3\n"},
    "/lora-adapters": lambda body: {"json": body if isinstance(body, list) else []},
    "/completion": {"json": {"content": "native", "stop": True}},
    "/v1/messages": {"sse": ['event: message_start\ndata: {"type": "message_start"}', 'event: message_stop\ndata: {"type": "message_stop"}']},
}


@pytest.fixture
def running(tmp_path):
    engine = RunningEngine(tmp_path, REPLIES, vram_budget_mb=100000)
    yield engine
    engine.close()


def runner_requests(running, path):
    return [request["body"] for request in running.upstream_requests() if request["path"] == path]


def read_events(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.headers["Content-Type"] == "text/event-stream"
        return [line.decode().strip()[len("data: "):] for line in response if line.strip()]


def get_text(url):
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


def test_models_are_listed_the_openai_way(running):
    listed = http_json(f"{running.url}/v1/models")
    names = {entry["id"] for entry in listed["data"]}
    assert listed["object"] == "list" and {"qwen3:14b", "nomic-embed-text:latest"} <= names
    assert http_json(f"{running.url}/v1/models/qwen3:14b")["owned_by"] == "arty-engine"


def test_an_unknown_model_answers_an_openai_error(running):
    status, body = error_of(lambda: http_json(f"{running.url}/v1/chat/completions", {"model": "nope", "messages": []}))
    assert status == 404 and body["error"]["type"] == "not_found_error" and "nope" in body["error"]["message"]


def test_chat_completions_carry_the_model_defaults(running):
    answer = http_json(f"{running.url}/v1/chat/completions", {"model": "qwen3:14b", "messages": [{"role": "user", "content": "hi"}]})
    assert answer["choices"][0]["message"]["content"] == "hi"
    sent = runner_requests(running, "/v1/chat/completions")[0]
    assert sent["temperature"] == 0.6 and sent["stop"] == ["<|im_end|>"] and sent["model"] == "qwen3:14b"


def test_the_caller_wins_over_the_model_defaults(running):
    http_json(f"{running.url}/v1/chat/completions", {"model": "qwen3:14b", "temperature": 0.1,
                                                      "messages": [{"role": "user", "content": "hi"}]})
    assert runner_requests(running, "/v1/chat/completions")[0]["temperature"] == 0.1


def test_chat_completions_stream_as_server_sent_events(running):
    events = read_events(f"{running.url}/v1/chat/completions", {"model": "qwen3:14b", "stream": True,
                                                                 "messages": [{"role": "user", "content": "hi"}]})
    assert events[-1] == "[DONE]"
    assert "".join(json.loads(event)["choices"][0]["delta"]["content"] for event in events[:-1]) == "hi"


def test_a_system_prompt_from_the_model_goes_first(running):
    http_json(f"{running.url}/api/create", {"model": "helper", "from": "qwen3:14b", "system": "Be brief.", "stream": False})
    http_json(f"{running.url}/v1/chat/completions", {"model": "helper", "messages": [{"role": "user", "content": "hi"}]})
    sent = runner_requests(running, "/v1/chat/completions")[0]["messages"]
    assert sent == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]


def test_completions_and_embeddings_reach_the_runner(running):
    assert http_json(f"{running.url}/v1/completions", {"model": "qwen3:14b", "prompt": "hello"})["choices"][0]["text"] == " world"
    vectors = http_json(f"{running.url}/v1/embeddings", {"model": "nomic-embed-text", "input": "hello"})
    assert vectors["data"][0]["embedding"] == [0.1, 0.2]
    assert runner_requests(running, "/v1/embeddings")[0]["model"] == "nomic-embed-text:latest"


def test_embeddings_refuse_a_chat_model(running):
    status, body = error_of(lambda: http_json(f"{running.url}/v1/embeddings", {"model": "qwen3:14b", "input": "x"}))
    assert status == 400 and "does not support embeddings" in body["error"]["message"]


def test_rerank_needs_a_reranking_model(running):
    status, body = error_of(lambda: http_json(f"{running.url}/v1/rerank", {"model": "nomic-embed-text", "query": "q",
                                                                            "documents": ["a"]}))
    assert status == 400 and "not a reranking model" in body["error"]["message"]


@pytest.mark.parametrize("path, body, expected", [
    ("/tokenize", {"content": "abc"}, {"tokens": [1, 2, 3]}),
    ("/detokenize", {"tokens": [1, 2, 3]}, {"content": "abc"}),
    ("/infill", {"input_prefix": "a", "input_suffix": "c"}, {"content": "middle"}),
])
def test_runner_tools_are_passed_through(running, path, body, expected):
    assert http_json(f"{running.url}{path}", {"model": "qwen3:14b", **body}) == expected
    assert "model" not in runner_requests(running, path)[0]


def test_apply_template_sees_the_model_messages(running):
    http_json(f"{running.url}/api/create", {"model": "primed", "from": "qwen3:14b", "stream": False,
                                            "modelfile": 'MESSAGE user "Who are you?"\nMESSAGE assistant "Arty."'})
    assert http_json(f"{running.url}/apply-template", {"model": "primed", "messages": [{"role": "user", "content": "hi"}]}) \
        == {"prompt": "<user>hi"}
    assert [message["content"] for message in runner_requests(running, "/apply-template")[0]["messages"]] \
        == ["Who are you?", "Arty.", "hi"]


def test_props_and_slots_use_the_one_loaded_model(running):
    status, body = error_of(lambda: http_json(f"{running.url}/props"))
    assert status == 400 and "?model=" in body["error"]
    http_json(f"{running.url}/tokenize", {"model": "qwen3:14b", "content": "x"})
    assert http_json(f"{running.url}/props")["total_slots"] == 1
    assert http_json(f"{running.url}/slots?model=qwen3:14b")[0]["id"] == 0
    assert len(running.launcher.launched) == 1


def test_a_slot_can_be_saved_and_a_bad_action_is_refused(running):
    saved = http_json(f"{running.url}/slots/0?action=save", {"model": "qwen3:14b", "filename": "chat.bin"})
    assert saved["n_saved"] == 12 and runner_requests(running, "/slots/0?action=save") == [{"filename": "chat.bin"}]
    status, _ = error_of(lambda: http_json(f"{running.url}/slots/0?action=wipe", {"model": "qwen3:14b"}))
    assert status == 400
    assert "--slot-save-path" in running.launcher.launched[0].arguments


def test_lora_adapters_can_be_read_and_set(running):
    assert http_json(f"{running.url}/lora-adapters?model=qwen3:14b") == []
    assert http_json(f"{running.url}/lora-adapters", {"model": "qwen3:14b", "adapters": [{"id": 0, "scale": 0.5}]}) \
        == [{"id": 0, "scale": 0.5}]


def test_metrics_label_each_runner_and_add_the_engine_counts(running):
    http_json(f"{running.url}/tokenize", {"model": "qwen3:14b", "content": "x"})
    text = get_text(f"{running.url}/metrics")
    assert 'arty_runtime_prompt_tokens_total{model="qwen3:14b"} 42' in text
    assert 'arty_runtime_requests{model="qwen3:14b",state="busy"} 1' in text
    assert "arty_engine_loaded_models 1" in text and text.count("# TYPE arty_runtime_prompt_tokens_total counter") == 1
    assert "upstream" not in text


def test_health_answers_ok(running):
    assert http_json(f"{running.url}/health") == {"status": "ok"}


def test_the_native_completion_route_passes_through(running):
    assert http_json(f"{running.url}/completion", {"model": "qwen3:14b", "prompt": "x"})["content"] == "native"
    assert runner_requests(running, "/completion")[0]["model"] == "qwen3:14b"


def test_other_event_streams_pass_through_unchanged(running):
    request = urllib.request.Request(f"{running.url}/v1/messages", method="POST", headers={"Content-Type": "application/json"},
                                     data=json.dumps({"model": "qwen3:14b", "stream": True, "messages": []}).encode())
    with urllib.request.urlopen(request, timeout=30) as response:
        text = response.read().decode()
    assert text == 'event: message_start\ndata: {"type": "message_start"}\n\nevent: message_stop\ndata: {"type": "message_stop"}\n\n'


def test_a_loaded_model_is_reused_and_kept_alive(running):
    http_json(f"{running.url}/api/chat", {"model": "qwen3:14b", "stream": False, "keep_alive": "1m", "options": {"num_ctx": 2048},
                                          "messages": [{"role": "user", "content": "hi"}]})
    http_json(f"{running.url}/v1/chat/completions", {"model": "qwen3:14b", "keep_alive": -1,
                                                      "messages": [{"role": "user", "content": "hi"}]})
    assert len(running.launcher.launched) == 1
    assert running.engine.pool.loaded()[0].expires_at is None


def test_every_engine_call_to_a_runner_carries_that_runner_s_key(running):
    http_json(f"{running.url}/v1/chat/completions", {"model": "qwen3:14b", "messages": [{"role": "user", "content": "hi"}]})
    read_events(f"{running.url}/v1/chat/completions", {"model": "qwen3:14b", "stream": True, "messages": []})
    http_json(f"{running.url}/api/chat", {"model": "qwen3:14b", "stream": False, "messages": [{"role": "user", "content": "hi"}]})
    http_json(f"{running.url}/tokenize", {"model": "qwen3:14b", "content": "x"})
    http_json(f"{running.url}/props")
    get_text(f"{running.url}/metrics")
    process = running.launcher.launched[0]
    assert process.api_key and process.server.refused == []


def test_metric_help_text_is_the_engine_s_own(running):
    http_json(f"{running.url}/tokenize", {"model": "qwen3:14b", "content": "x"})
    text = get_text(f"{running.url}/metrics")
    assert "# HELP arty_runtime_prompt_tokens_total Prompt tokens processed, not counting tokens reused from the cache." in text
    assert "# HELP arty_runtime_n_decode_total Decode calls, not counting speculative or multimodal decoding." in text
    assert "# HELP arty_runtime_brand_new_total Model runner measurement brand_new_total, one series per loaded model." in text
    assert UPSTREAM_WORD not in text and "Something new" not in text


def test_props_leave_out_the_build_and_the_model_s_path(running):
    http_json(f"{running.url}/tokenize", {"model": "qwen3:14b", "content": "x"})
    props = http_json(f"{running.url}/props")
    assert props["total_slots"] == 1 and "build_info" not in props and "model_path" not in props


def test_the_fingerprint_is_the_engine_s_version(running):
    answer = http_json(f"{running.url}/v1/chat/completions", {"model": "qwen3:14b", "messages": [{"role": "user", "content": "hi"}]})
    assert answer["system_fingerprint"] == "arty-engine/0.1.0"
    events = read_events(f"{running.url}/v1/chat/completions", {"model": "qwen3:14b", "stream": True, "messages": []})
    assert json.loads(events[0])["system_fingerprint"] == "arty-engine/0.1.0"


def test_status_names_the_runtime_release_not_its_path(running):
    status = http_json(f"{running.url}/api/status")
    assert status["runtime_release"] == "external" and "runner_binary" not in status


RUNTIME_FAILURE = f"lib{UPSTREAM_WORD}.so: {UPSTREAM_WORD}_decode failed in {UPSTREAM_WORD}-context.cpp"
FAILING_MID_STREAM = {"sse": ['data: {"choices":[{"index":0,"delta":{"content":"h"}}],"model":"x"}',
                              'data: {"error": {"message": "%s", "code": 500}}' % RUNTIME_FAILURE]}


def failing_engine(tmp_path, replies):
    return RunningEngine(tmp_path, replies, vram_budget_mb=100000)


def test_an_error_in_the_middle_of_a_relayed_stream_is_scrubbed(tmp_path):
    engine = failing_engine(tmp_path, {"/v1/chat/completions": FAILING_MID_STREAM})
    try:
        events = read_events(f"{engine.url}/v1/chat/completions", {"model": "qwen3:14b", "stream": True, "messages": []})
    finally:
        engine.close()
    errors = [event for event in events if "error" in event]
    assert errors and UPSTREAM_WORD not in "".join(events) and "runtime_decode failed" in errors[0]


def test_an_error_in_the_middle_of_a_raw_generate_stream_is_scrubbed(tmp_path):
    native_failure = {"sse": ['data: {"content":"h","stop":false}', 'data: {"error": {"message": "%s"}}' % RUNTIME_FAILURE]}
    engine = failing_engine(tmp_path, {"/completion": native_failure})
    try:
        request = urllib.request.Request(f"{engine.url}/api/generate", method="POST", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"model": "qwen3:14b", "prompt": "hi", "raw": True}).encode())
        with urllib.request.urlopen(request, timeout=30) as response:
            lines = [json.loads(line) for line in response if line.strip()]
    finally:
        engine.close()
    assert lines[-1]["error"] and UPSTREAM_WORD not in json.dumps(lines) and "runtime_decode failed" in lines[-1]["error"]


def test_errors_in_passed_through_bodies_and_streams_are_scrubbed_but_answers_are_not(tmp_path):
    engine = failing_engine(tmp_path, {
        "/completion": {"json": {"content": f"a {UPSTREAM_WORD} grazes", "error": {"message": RUNTIME_FAILURE}}},
        "/v1/messages": {"sse": ['event: message_start\ndata: {"type": "message_start"}',
                                 'event: error\ndata: {"type": "error", "error": {"message": "%s"}}' % RUNTIME_FAILURE]},
    })
    try:
        native = http_json(f"{engine.url}/completion", {"model": "qwen3:14b", "prompt": "x"})
        events = read_events(f"{engine.url}/v1/messages", {"model": "qwen3:14b", "messages": [], "stream": True})
    finally:
        engine.close()
    assert native["content"] == f"a {UPSTREAM_WORD} grazes" and UPSTREAM_WORD not in json.dumps(native["error"])
    error_events = [event for event in events if "error" in event]
    assert error_events and UPSTREAM_WORD not in "".join(error_events)


COMPLETE_PASS_THROUGH_STREAMS = {
    "/v1/messages": ['event: message_start\ndata: {"type": "message_start"}', 'event: message_stop\ndata: {"type": "message_stop"}'],
    "/completion": ['data: {"content": "h", "stop": false}', 'data: {"content": "", "stop": true}'],
    "/v1/responses": ['event: response.created\ndata: {"type": "response.created"}',
                      'event: response.completed\ndata: {"type": "response.completed"}'],
}


def raw_event_lines(url, body):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as response:
        assert response.headers["Content-Type"] == "text/event-stream"
        return [line.decode().strip() for line in response if line.strip()]


PASS_THROUGH_REQUEST = {"model": "qwen3:14b", "stream": True, "prompt": "x", "messages": [{"role": "user", "content": "x"}],
                        "input": "x"}


@pytest.mark.parametrize("path", sorted(COMPLETE_PASS_THROUGH_STREAMS))
def test_a_streamed_pass_through_whose_runner_closed_before_answering_is_asked_again(tmp_path, path):
    engine = failing_engine(tmp_path, {})
    launched = engine.launcher.launched

    def closes_after_the_headers_on_the_first_runner(body):
        if len(launched) == 1:
            launched[0].returncode = -6
            return {"cut_sse": []}
        return {"sse": COMPLETE_PASS_THROUGH_STREAMS[path]}

    engine.launcher.replies[path] = closes_after_the_headers_on_the_first_runner
    try:
        lines = raw_event_lines(f"{engine.url}{path}", PASS_THROUGH_REQUEST)
    finally:
        engine.close()
    assert lines == [line for event in COMPLETE_PASS_THROUGH_STREAMS[path] for line in event.split("\n")] and len(launched) == 2


@pytest.mark.parametrize("path", sorted(COMPLETE_PASS_THROUGH_STREAMS))
def test_a_pass_through_stream_that_stops_midway_ends_with_an_error_event(tmp_path, path):
    engine = failing_engine(tmp_path, {path: {"cut_sse": COMPLETE_PASS_THROUGH_STREAMS[path][:1]}})
    try:
        lines = raw_event_lines(f"{engine.url}{path}", PASS_THROUGH_REQUEST)
    finally:
        engine.close()
    first_event = COMPLETE_PASS_THROUGH_STREAMS[path][0].split("\n")
    assert lines[:len(first_event)] == first_event and lines[-2] == "event: error"
    assert "stopped before its end" in json.loads(lines[-1][len("data: "):])["error"]["message"]


@pytest.mark.parametrize("path", sorted(COMPLETE_PASS_THROUGH_STREAMS))
def test_a_complete_pass_through_stream_gets_no_error_event(tmp_path, path):
    engine = failing_engine(tmp_path, {path: {"cut_sse": COMPLETE_PASS_THROUGH_STREAMS[path]}})
    try:
        lines = raw_event_lines(f"{engine.url}{path}", PASS_THROUGH_REQUEST)
    finally:
        engine.close()
    assert "event: error" not in lines


BIG_TEXT = "x" * (100 * 1024)
BIG_FINAL_EVENTS = {
    "/completion": ['data: {"content": "h", "stop": false}', 'data: {"content": "", "stop": true, "prompt": "%s"}' % BIG_TEXT],
    "/v1/responses": ['event: response.created\ndata: {"type": "response.created"}',
                      'event: response.completed\ndata: {"type": "response.completed", "response": {"output_text": "%s"}}' % BIG_TEXT],
}


@pytest.mark.parametrize("path", sorted(BIG_FINAL_EVENTS))
def test_a_final_event_longer_than_one_read_still_ends_the_stream(tmp_path, path):
    engine = failing_engine(tmp_path, {path: {"cut_sse": BIG_FINAL_EVENTS[path]}})
    try:
        lines = raw_event_lines(f"{engine.url}{path}", PASS_THROUGH_REQUEST)
    finally:
        engine.close()
    assert "event: error" not in lines and BIG_TEXT in lines[-1]


@pytest.mark.parametrize("path", sorted(BIG_FINAL_EVENTS))
def test_a_stream_cut_after_a_big_event_that_is_not_its_end_gets_an_error_event(tmp_path, path):
    big_middle = 'data: {"content": "%s", "stop": false}' % BIG_TEXT
    engine = failing_engine(tmp_path, {path: {"cut_sse": [BIG_FINAL_EVENTS[path][0], big_middle]}})
    try:
        lines = raw_event_lines(f"{engine.url}{path}", PASS_THROUGH_REQUEST)
    finally:
        engine.close()
    assert lines[-2] == "event: error" and BIG_TEXT in lines[-3]


def test_a_final_event_past_the_checked_size_is_not_called_a_cut(tmp_path, monkeypatch):
    monkeypatch.setattr("arty_engine.compat.LONGEST_CHECKED_LINE_BYTES", 1024)
    engine = failing_engine(tmp_path, {"/completion": {"cut_sse": BIG_FINAL_EVENTS["/completion"]}})
    try:
        lines = raw_event_lines(f"{engine.url}/completion", PASS_THROUGH_REQUEST)
    finally:
        engine.close()
    assert "event: error" not in lines and BIG_TEXT in lines[-1]


def test_the_runtime_release_is_the_engine_s_folder_name_or_external():
    from arty_engine.server import runtime_release_name
    upstream = "ll" + "ama"
    assert runtime_release_name("/srv/runtime/runtime-b12110-linux-cuda-12.8-x64/arty-runtime-server") == \
        "runtime-b12110-linux-cuda-12.8-x64"
    assert runtime_release_name(f"/srv/{upstream}-b12110-bin-ubuntu-x64/{upstream}-server") == "external"
    assert runtime_release_name(f"/srv/{upstream}-b12110/server") == "external"


def test_reading_props_slots_and_adapters_keeps_the_model_s_keep_alive(running):
    http_json(f"{running.url}/v1/chat/completions", {"model": "qwen3:14b", "keep_alive": -1,
                                                      "messages": [{"role": "user", "content": "hi"}]})
    for path in ("/props", "/slots", "/lora-adapters"):
        http_json(f"{running.url}{path}")
    runner = running.engine.pool.loaded()[0]
    assert runner.expires_at is None and runner.in_flight == 0


@pytest.mark.parametrize("name", ["../escape.bin", "/etc/passwd", "a/b.bin", ".hidden", "x" * 101, "", 7])
def test_slot_file_names_must_be_plain(running, name):
    for action in ("save", "restore"):
        status, body = error_of(lambda: http_json(f"{running.url}/slots/0?action={action}", {"model": "qwen3:14b", "filename": name}))
        assert status == 400 and "plain file name" in body["error"]
    assert runner_requests(running, "/slots/0?action=save") == []


def test_saved_slot_files_are_capped(running, monkeypatch):
    monkeypatch.setattr("arty_engine.compat.MOST_SLOT_FILES", 2)
    http_json(f"{running.url}/tokenize", {"model": "qwen3:14b", "content": "x"})
    folder = running.engine.pool.loaded()[0].plan.slot_dir
    for name in ("one.bin", "two.bin"):
        (folder / name).write_bytes(b"x")
    assert http_json(f"{running.url}/slots/0?action=save", {"model": "qwen3:14b", "filename": "one.bin"})["n_saved"] == 12
    status, body = error_of(lambda: http_json(f"{running.url}/slots/0?action=save", {"model": "qwen3:14b", "filename": "three.bin"}))
    assert status == 409 and "at most 2" in body["error"] and "state_dir" in body["error"] and "delete some" not in body["error"]
