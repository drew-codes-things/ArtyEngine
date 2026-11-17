import json
import urllib.request

import pytest

from engine_harness import RunningEngine, error_of, http_json, stream_chat

TIMINGS = {"prompt_n": 40, "prompt_ms": 20.0, "predicted_n": 5, "predicted_ms": 50.0}


LEAK_PROMPT = "how many orange stars?"
LEAK_PIECES = ["Yes, 14 ", "orange stars.</th", "ink>14"]
COUNT_PROMPT = "the numbers from 1 to 5 separated by spaces, nothing else"
COUNT_PIECES = ["1", " 2", " 3", " 4", " 5"]
CODE_PROMPT = "print a Python line containing the literal text <think>a</think>"
UNLISTED_OPENER_PROMPT = "what is the invoice total?"
UNLISTED_OPENER_REASONING = ("Got it, let's look at the picture. The invoice has two rows, a line for the parts and a line "
                             "for the labour, and a total at the bottom right that reads thirty six pounds and twenty pence, "
                             "so the answer is that figure written as a number.")
INVOICE_ANSWER = "The invoice total is 36.20 pounds."
BLANK_LINES_PROMPT = "how many rows?"
BLANK_LINES_PIECES = ["Okay, count the rows.", "</think>", "\n\nTwo rows."]
CUT_PROMPT = "count every star"
LATE_TAG_PROMPT = "how many rows, twice?"
CUT_REASONING = "<think>\nOkay, the user wants every star counted, so I start at the top left and go"
CODE_PIECES = ["Here it is:\n```python\nprint(\"<thi", "nk>a</th", "ink>\")\n```"]
TAGLESS_CUT_PROMPT = "what is on the invoice?"
TAG_ANSWER_PROMPT = "what does the think tag do?"
TAG_ANSWER = "<think> is the tag some models put in front of their reasoning. It is not HTML."
REASONED_CODE_PROMPT = "write a function that strips reasoning blocks"
REASONED_CODE_REASONING = "The user wants a regex that removes the blocks."
REASONED_CODE_PIECES = ["return re.sub(r\"<thi", "nk>.*?</th", "ink>\", \"\", text, flags=re.DOTALL)"]
CUT_MID_TAG_PROMPT = "what is the total, briefly?"
CUT_MID_TAG_REPLY = "Okay.\n<think>\n" + UNLISTED_OPENER_REASONING


def leaked_reply(body):
    if body.get("stream"):
        events = [{"choices": [{"delta": {"content": piece}}]} for piece in LEAK_PIECES]
        events.append({"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": TIMINGS})
        return {"sse": [f"data: {json.dumps(event)}" for event in events] + ["data: [DONE]"]}
    return {"json": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "".join(LEAK_PIECES)}}],
                     "timings": TIMINGS}}


def pieces_reply(body, pieces, finish="stop"):
    if body.get("stream"):
        events = [{"choices": [{"delta": {"content": piece}}]} for piece in pieces]
        events.append({"choices": [{"delta": {}, "finish_reason": finish}], "timings": TIMINGS})
        return {"sse": [f"data: {json.dumps(event)}" for event in events] + ["data: [DONE]"]}
    return {"json": {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": "".join(pieces)}}],
                     "timings": TIMINGS}}


def reasoned_reply(body, reasoning, pieces):
    if body.get("stream"):
        events = [{"choices": [{"delta": {"reasoning_content": reasoning}}]}]
        events += [{"choices": [{"delta": {"content": piece}}]} for piece in pieces]
        events.append({"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": TIMINGS})
        return {"sse": [f"data: {json.dumps(event)}" for event in events] + ["data: [DONE]"]}
    return {"json": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "".join(pieces),
                                                                       "reasoning_content": reasoning}}], "timings": TIMINGS}}


def counted_reply():
    events = [{"choices": [{"delta": {"content": piece}}]} for piece in COUNT_PIECES]
    events.append({"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": TIMINGS})
    return {"sse": [f"data: {json.dumps(event)}" for event in events] + ["data: [DONE]"]}


def chat_reply(body):
    if any(message.get("content") == LEAK_PROMPT for message in body.get("messages", [])):
        return leaked_reply(body)
    if any(message.get("content") == COUNT_PROMPT for message in body.get("messages", [])):
        return counted_reply()
    if any(message.get("content") == CODE_PROMPT for message in body.get("messages", [])):
        return pieces_reply(body, CODE_PIECES)
    if any(message.get("content") == UNLISTED_OPENER_PROMPT for message in body.get("messages", [])):
        words = [word + " " for word in UNLISTED_OPENER_REASONING.split(" ")]
        return pieces_reply(body, words + ["\n</think>", "\n\n" + INVOICE_ANSWER])
    if any(message.get("content") == BLANK_LINES_PROMPT for message in body.get("messages", [])):
        return pieces_reply(body, BLANK_LINES_PIECES)
    if any(message.get("content") == LATE_TAG_PROMPT for message in body.get("messages", [])):
        return pieces_reply(body, ["a</think>Two", " rows.</think>"])
    if any(message.get("content") == TAGLESS_CUT_PROMPT for message in body.get("messages", [])):
        return pieces_reply(body, [word + " " for word in UNLISTED_OPENER_REASONING.split(" ")], "length")
    if any(message.get("content") == TAG_ANSWER_PROMPT for message in body.get("messages", [])):
        return pieces_reply(body, [TAG_ANSWER[index:index + 6] for index in range(0, len(TAG_ANSWER), 6)])
    if any(message.get("content") == REASONED_CODE_PROMPT for message in body.get("messages", [])):
        return reasoned_reply(body, REASONED_CODE_REASONING, REASONED_CODE_PIECES)
    if any(message.get("content") == CUT_MID_TAG_PROMPT for message in body.get("messages", [])):
        return pieces_reply(body, [CUT_MID_TAG_REPLY[index:index + 9] for index in range(0, len(CUT_MID_TAG_REPLY), 9)], "length")
    if any(message.get("content") == CUT_PROMPT for message in body.get("messages", [])):
        return pieces_reply(body, [CUT_REASONING[index:index + 7] for index in range(0, len(CUT_REASONING), 7)], "length")
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


def test_tags_lists_every_model(engine):
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


def test_tool_round_returns_tool_calls_with_object_arguments(engine):
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


def test_unknown_model_is_404(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/chat", {"model": "nope:1b", "messages": []}))
    assert status == 404 and body == {"error": "model 'nope:1b' not found"}


def test_an_unsupported_model_says_why(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/chat", {"model": "qwen2.5vl:7b", "stream": False,
                                                                           "messages": [{"role": "user", "content": "hi"}]}))
    assert status == 501 and "not supported by ArtyEngine" in body["error"]


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


def chat_answer(engine, **fields):
    body = {"model": "qwen3:14b", "stream": False, "messages": [{"role": "user", "content": "x"}], **fields}
    return error_of(lambda: http_json(f"{engine.url}/api/chat", body))


@pytest.mark.parametrize("messages", [1, 1.5, True, "hi"])
def test_messages_that_are_not_a_list_are_a_400(engine, messages):
    status, body = chat_answer(engine, messages=messages)
    assert status == 400 and "messages" in body["error"]


@pytest.mark.parametrize("calls", ["x", [1], [{"function": "x"}], [{"function": {"name": 5}}]])
def test_tool_calls_of_the_wrong_shape_are_a_400(engine, calls):
    status, body = chat_answer(engine, messages=[{"role": "assistant", "content": "", "tool_calls": calls}])
    assert status == 400 and "tool_calls" in body["error"]


@pytest.mark.parametrize("option", ["num_predict", "num_ctx"])
@pytest.mark.parametrize("value", ["abc", "", [], {}, [1], float("nan"), float("inf"), 1.5])
def test_count_options_that_are_not_whole_numbers_are_a_400(engine, option, value):
    status, body = chat_answer(engine, options={option: value})
    assert status == 400 and option in body["error"]


@pytest.mark.parametrize("value", [-1, -(2 ** 70), 2 ** 70])
def test_a_num_ctx_out_of_range_is_a_400(engine, value):
    status, body = chat_answer(engine, options={"num_ctx": value})
    assert status == 400 and "num_ctx" in body["error"]


@pytest.mark.parametrize("body", [b"[" * 200_000, b'{"a":' * 200_000])
def test_json_nested_too_deep_is_a_400(engine, body):
    import urllib.request
    request = urllib.request.Request(f"{engine.url}/api/chat", data=body, method="POST")
    status, answer = error_of(lambda: urllib.request.urlopen(request))
    assert status == 400 and "nested too deeply" in answer["error"]


@pytest.mark.parametrize("images", [5, "abc", {"a": 1}])
def test_images_that_are_not_a_list_are_a_400(engine, images):
    message = {"role": "user", "content": "x", "images": images}
    status, body = chat_answer(engine, model="mimo:9b", messages=[message])
    assert status == 400 and "images" in body["error"]
    status, body = error_of(lambda: http_json(f"{engine.url}/api/generate", {"model": "mimo:9b", "stream": False, "prompt": "x", "images": images}))
    assert status == 400 and "images" in body["error"]
    assert engine.upstream_requests() == []


@pytest.mark.parametrize("option, value", [
    ("temperature", "hot"), ("temperature", float("nan")), ("top_p", [0.9]), ("top_k", {}), ("seed", "x"), ("seed", 1.5),
    ("repeat_penalty", True), ("stop", 5), ("stop", ["a", 1]), ("num_predict", 2 ** 40), ("min_p", float("inf")),
])
def test_sampling_options_of_the_wrong_type_are_a_400(engine, option, value):
    status, body = chat_answer(engine, options={option: value})
    assert status == 400 and option in body["error"]


def test_sampling_options_of_the_right_type_reach_the_runner(engine):
    options = {"temperature": 0, "top_p": 0.9, "top_k": 20, "seed": 7, "stop": "END", "num_predict": 12, "repeat_penalty": 1.1}
    http_json(f"{engine.url}/api/chat", {"model": "qwen3:14b", "stream": False, "messages": [{"role": "user", "content": "x"}], "options": options})
    sent = engine.upstream_requests()[-1]["body"]
    assert (sent["temperature"], sent["top_p"], sent["top_k"], sent["seed"], sent["stop"], sent["max_tokens"]) == (0, 0.9, 20, 7, ["END"], 12)


@pytest.mark.parametrize("value", ["abc", [], -1, 1.5, float("nan")])
def test_embed_dimensions_that_are_not_a_positive_whole_number_are_a_400(engine, value):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/embed", {"model": "nomic-embed-text", "input": "x", "dimensions": value}))
    assert status == 400 and "dimensions" in body["error"]


def test_a_nan_keep_alive_is_a_400(engine):
    status, body = chat_answer(engine, keep_alive=float("nan"))
    assert status == 400 and "keep_alive" in body["error"]


@pytest.mark.parametrize("value", [2 ** 70, 1e308, float("inf"), "1e308"])
def test_a_keep_alive_too_long_to_date_keeps_the_model_loaded(engine, value):
    http_json(f"{engine.url}/api/chat", {"model": "qwen3:14b", "keep_alive": value})
    loaded = http_json(f"{engine.url}/api/ps")["models"]
    assert [model["name"] for model in loaded] == ["qwen3:14b"]


def test_a_model_name_too_long_for_a_file_name_is_a_client_error(engine):
    import urllib.request
    name = "x" * 100_000
    status, body = error_of(lambda: http_json(f"{engine.url}/api/show", {"model": name}))
    assert status == 404 and "not found" in body["error"]
    request = urllib.request.Request(f"{engine.url}/api/delete", data=json.dumps({"model": name}).encode(), method="DELETE")
    status, body = error_of(lambda: urllib.request.urlopen(request))
    assert status == 400 and "invalid model name" in body["error"]


def test_a_model_name_with_a_trailing_newline_is_not_found(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/show", {"model": "qwen3:14b\n"}))
    assert status == 404


def test_runner_errors_come_back_as_api_errors(tmp_path):
    running = RunningEngine(tmp_path, {"/v1/chat/completions": lambda body: {"error": "the request exceeds the context size", "status": 400}})
    try:
        status, body = error_of(lambda: http_json(f"{running.url}/api/chat", {"model": "qwen3:14b", "stream": False,
                                                                               "messages": [{"role": "user", "content": "x"}]}))
        assert status == 400 and body == {"error": "the request exceeds the context size"}
    finally:
        running.close()


def test_delete_never_touches_a_store_read_in_place(engine):
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


def test_generate_load_reply_has_no_message(engine):
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


def test_an_unsupported_model_points_at_a_build_with_a_projector_file(engine):
    status, body = error_of(lambda: http_json(f"{engine.url}/api/generate", {"model": "qwen2.5vl:7b", "prompt": "hi", "stream": False}))
    assert status == 501 and "projector file from Hugging Face" in body["error"]


def test_a_damaged_manifest_never_breaks_the_model_list_or_delete(tmp_path):
    from arty_engine.store import MANIFEST_MEDIA_TYPE
    from fake_store import QWEN3, add_model
    path = add_model(tmp_path / "own", "broken", "1", QWEN3, {"blk.0.w": (4, 4)})
    path.write_text(json.dumps({"mediaType": MANIFEST_MEDIA_TYPE, "layers": [{"mediaType": ["x"], "digest": 5}, 5], "manifests": [5]}))
    running = RunningEngine(tmp_path, REPLIES)
    try:
        names = [model["name"] for model in http_json(f"{running.url}/api/tags")["models"]]
        assert "qwen3:14b" in names and "broken:1" not in names
        request = urllib.request.Request(f"{running.url}/api/delete", data=b'{"model":"broken:1"}', method="DELETE")
        with urllib.request.urlopen(request, timeout=30) as response:
            assert response.status == 200
        assert not path.exists()
    finally:
        running.close()


@pytest.mark.parametrize("text", ["", "{", "\u0000\u0001binary", "[" * 200000])
def test_a_manifest_that_is_not_json_is_a_clear_error_never_an_engine_error(tmp_path, text):
    from fake_store import QWEN3, add_model
    path = add_model(tmp_path / "own", "broken", "1", QWEN3, {"blk.0.w": (4, 4)})
    path.write_text(text)
    running = RunningEngine(tmp_path, REPLIES)
    try:
        for call in (lambda: http_json(f"{running.url}/api/show", {"model": "broken:1"}),
                     lambda: http_json(f"{running.url}/api/chat", {"model": "broken:1", "stream": False, "messages": []})):
            status, body = error_of(call)
            assert status == 500 and body["error"].startswith("couldn't read broken:1")
    finally:
        running.close()


@pytest.mark.parametrize("params", ["[1, 2]", "5", "\"text\""])
def test_params_that_are_not_an_object_are_ignored(tmp_path, params):
    from fake_store import QWEN3, add_model
    add_model(tmp_path / "own", "odd", "1", QWEN3, {"blk.0.w": (4, 4)}, params={})
    store_path = tmp_path / "own"
    running = RunningEngine(tmp_path, REPLIES)
    try:
        model = running.engine.store.get("odd:1")
        params_layer = next(layer for layer in model.manifest["layers"] if layer["mediaType"].endswith("params"))
        (store_path / "blobs" / params_layer["digest"].replace(":", "-")).write_text(params)
        assert http_json(f"{running.url}/api/show", {"model": "odd:1"})["parameters"] == ""
    finally:
        running.close()


def streamed_lines(engine, path, body):
    request = urllib.request.Request(f"{engine.url}{path}", data=json.dumps({**body, "stream": True}).encode(), method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        return [json.loads(line) for line in response if line.strip()]


@pytest.mark.parametrize("think, thinking", [(None, "Yes, 14 orange stars."), (False, "")])
def test_leaked_reasoning_never_reaches_chat_or_generate_as_content(engine, think, thinking):
    extra = {} if think is None else {"think": think}
    chat = {"model": "mimo:9b", "messages": [{"role": "user", "content": LEAK_PROMPT}], **extra}
    generate = {"model": "mimo:9b", "prompt": LEAK_PROMPT, **extra}
    reply = http_json(f"{engine.url}/api/chat", {**chat, "stream": False})
    assert reply["message"]["content"] == "14" and reply["message"].get("thinking", "") == thinking
    assert http_json(f"{engine.url}/api/generate", {**generate, "stream": False})["response"] == "14"
    lines = streamed_lines(engine, "/api/chat", chat)
    assert "".join(line["message"].get("content", "") for line in lines) == "14"
    assert "".join(line["message"].get("thinking", "") for line in lines) == thinking
    lines = streamed_lines(engine, "/api/generate", generate)
    assert "".join(line.get("response", "") for line in lines) == "14"


@pytest.mark.parametrize("path, field", [("/api/chat", None), ("/api/generate", "response")])
def test_a_template_that_honours_the_thinking_switch_streams_think_false_replies_at_once(engine, path, field):
    body = ({"messages": [{"role": "user", "content": COUNT_PROMPT}]} if field is None else {"prompt": COUNT_PROMPT})
    lines = streamed_lines(engine, path, {"model": "qwen3:14b", "think": False, **body})
    pieces = [(line.get(field) if field else line["message"].get("content")) for line in lines]
    assert [piece for piece in pieces if piece] == COUNT_PIECES
    assert not lines[0]["done"] and lines[-1]["done"]


@pytest.mark.parametrize("think", [None, False])
@pytest.mark.parametrize("stream", [False, True])
def test_think_tags_in_the_answer_of_a_model_that_cannot_leak_arrive_intact(engine, think, stream):
    extra = {} if think is None else {"think": think}
    chat = {"model": "qwen3:14b", "messages": [{"role": "user", "content": CODE_PROMPT}], **extra}
    generate = {"model": "qwen3:14b", "prompt": CODE_PROMPT, **extra}
    if stream:
        chat_lines = streamed_lines(engine, "/api/chat", chat)
        generate_lines = streamed_lines(engine, "/api/generate", generate)
        chat_content = "".join(line["message"].get("content", "") for line in chat_lines)
        chat_thinking = "".join(line["message"].get("thinking", "") for line in chat_lines)
        generate_content = "".join(line.get("response", "") for line in generate_lines)
    else:
        reply = http_json(f"{engine.url}/api/chat", {**chat, "stream": False})
        chat_content, chat_thinking = reply["message"]["content"], reply["message"].get("thinking", "")
        generate_content = http_json(f"{engine.url}/api/generate", {**generate, "stream": False})["response"]
    assert chat_content == "".join(CODE_PIECES) and chat_thinking == ""
    assert generate_content == "".join(CODE_PIECES)


def leaker_answers(engine, prompt, think, stream):
    extra = {} if think is None else {"think": think}
    chat = {"model": "mimo:9b", "messages": [{"role": "user", "content": prompt}], **extra}
    generate = {"model": "mimo:9b", "prompt": prompt, **extra}
    if not stream:
        reply = http_json(f"{engine.url}/api/chat", {**chat, "stream": False})
        response = http_json(f"{engine.url}/api/generate", {**generate, "stream": False})["response"]
        return reply["message"]["content"], reply["message"].get("thinking", ""), response
    lines = streamed_lines(engine, "/api/chat", chat)
    response = "".join(line.get("response", "") for line in streamed_lines(engine, "/api/generate", generate))
    return ("".join(line["message"].get("content", "") for line in lines),
            "".join(line["message"].get("thinking", "") for line in lines), response)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("think", [None, False])
def test_a_leak_that_opens_with_any_words_never_reaches_the_answer(engine, think, stream):
    content, thinking, response = leaker_answers(engine, UNLISTED_OPENER_PROMPT, think, stream)
    assert content == INVOICE_ANSWER and response == INVOICE_ANSWER
    assert thinking == ("" if think is False else UNLISTED_OPENER_REASONING)


@pytest.mark.parametrize("stream", [False, True])
def test_the_answer_after_a_leak_starts_without_the_blank_lines(engine, stream):
    content, _, response = leaker_answers(engine, BLANK_LINES_PROMPT, False, stream)
    assert content == "Two rows." and response == "Two rows."


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("think", [None, False])
def test_a_reply_cut_off_inside_its_reasoning_is_thinking_not_the_answer(engine, think, stream):
    content, thinking, response = leaker_answers(engine, CUT_PROMPT, think, stream)
    assert content == "" and response == ""
    assert thinking == ("" if think is False else CUT_REASONING.replace("<think>", "").strip())


def test_a_think_tag_after_a_leaker_s_answer_started_is_logged(engine, caplog):
    caplog.set_level("WARNING")
    leaker_answers(engine, BLANK_LINES_PROMPT, False, True)
    assert "think tag" not in caplog.text
    content, _, _ = leaker_answers(engine, LATE_TAG_PROMPT, False, True)
    assert content == "Two rows."
    assert "mimo:9b" in caplog.text and "think tag" in caplog.text


@pytest.mark.parametrize("stream", [False, True])
def test_a_tagless_reply_cut_by_the_length_limit_is_delivered_and_logged(engine, stream, caplog):
    caplog.set_level("WARNING")
    content, thinking, response = leaker_answers(engine, TAGLESS_CUT_PROMPT, False, stream)
    expected = "".join(word + " " for word in UNLISTED_OPENER_REASONING.split(" "))
    assert content == expected and response == expected and thinking == ""
    assert "mimo:9b stopped at the length limit before any think tag" in caplog.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("think", [None, False])
def test_a_finished_answer_that_opens_with_the_think_tag_arrives_whole(engine, think, stream, caplog):
    caplog.set_level("WARNING")
    content, thinking, response = leaker_answers(engine, TAG_ANSWER_PROMPT, think, stream)
    assert content == TAG_ANSWER and response == TAG_ANSWER and thinking == ""
    assert "length limit" not in caplog.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("think", [None, False])
def test_a_template_only_leaker_s_answer_is_left_alone_after_the_runner_s_reasoning(engine, think, stream, caplog):
    caplog.set_level("WARNING")
    content, thinking, response = leaker_answers(engine, REASONED_CODE_PROMPT, think, stream)
    assert content == "".join(REASONED_CODE_PIECES) and response == "".join(REASONED_CODE_PIECES)
    assert thinking == REASONED_CODE_REASONING
    assert "think tag" not in caplog.text


@pytest.mark.parametrize("think", [None, False])
def test_a_named_leaker_s_reply_after_the_runner_s_reasoning_is_the_same_streamed_and_not(tmp_path, think, caplog):
    caplog.set_level("WARNING")
    running = RunningEngine(tmp_path, REPLIES, default_keep_alive="-1", thinking_leakers=["mimo:9b"])
    try:
        plain = leaker_answers(running, REASONED_CODE_PROMPT, think, False)
        streamed = leaker_answers(running, REASONED_CODE_PROMPT, think, True)
    finally:
        running.close()
    assert streamed == plain
    assert plain[0] == 'return re.sub(r"", "", text, flags=re.DOTALL)'
    assert "mimo:9b wrote 2 think tag(s)" in caplog.text


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("think", [None, False])
def test_a_reply_cut_inside_reasoning_that_follows_its_first_words_keeps_only_those_words(engine, think, stream, caplog):
    caplog.set_level("WARNING")
    content, thinking, response = leaker_answers(engine, CUT_MID_TAG_PROMPT, think, stream)
    assert content == "Okay.\n" and response == "Okay.\n"
    assert thinking == ("" if think is False else UNLISTED_OPENER_REASONING)
    assert "mimo:9b wrote 1 think tag(s) after its answer had started" in caplog.text


def test_every_reply_with_a_caught_leak_is_logged_and_counted_at_the_normal_level(engine, caplog):
    caplog.set_level("INFO")
    leaker_answers(engine, TAG_ANSWER_PROMPT, False, False)
    leaker_answers(engine, TAG_ANSWER_PROMPT, False, True)
    assert "leak caught" not in caplog.text
    leaker_answers(engine, BLANK_LINES_PROMPT, False, False)
    leaker_answers(engine, BLANK_LINES_PROMPT, False, True)
    caught = [record for record in caplog.records if record.getMessage().startswith("leak caught: mimo:9b")]
    assert [record.levelname for record in caught] == ["INFO"] * 4
    assert [record.getMessage().split("(")[-1] for record in caught] == [
        f"{count} replies of mimo:9b with a caught leak since the engine started)" for count in range(1, 5)]
