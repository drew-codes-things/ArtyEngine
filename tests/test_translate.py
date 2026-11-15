import json

import pytest

from arty_engine import translate
from arty_engine.runner import FOREVER, parse_keep_alive
from arty_engine.store import ModelStore
from fake_store import build_store


@pytest.fixture
def store(tmp_path):
    return ModelStore([build_store(tmp_path / "other")])


@pytest.mark.parametrize("value, expected", [
    (-1, FOREVER), ("-1", FOREVER), ("-1m", FOREVER), (0, 0.0), ("0", 0.0), ("10m", 600.0), ("1h30m", 5400.0),
    ("45s", 45.0), ("90", 90.0), (300, 300.0), ("500ms", 0.5),
])
def test_keep_alive_durations(value, expected):
    assert parse_keep_alive(value, "5m") == expected


def test_keep_alive_falls_back_to_the_default():
    assert parse_keep_alive(None, "-1") is FOREVER
    assert parse_keep_alive("", "10m") == 600.0


@pytest.mark.parametrize("value", ["soon", "10 minutes", True, "5x"])
def test_keep_alive_rejects_nonsense(value):
    with pytest.raises(ValueError):
        parse_keep_alive(value, "5m")


def test_brain_judge_request_becomes_json_mode_without_thinking(store):
    request = {"model": "qwen3:14b", "stream": False, "format": "json", "think": False, "keep_alive": -1,
               "options": {"temperature": 0, "num_predict": 120, "num_ctx": 12288},
               "messages": [{"role": "user", "content": "classify"}]}
    body = translate.chat_body(request, store.get("qwen3:14b"), stream=False)
    assert body["response_format"] == {"type": "json_object"}
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0 and body["max_tokens"] == 120
    assert body["stop"] == ["<|im_end|>"]
    assert "num_ctx" not in body and "keep_alive" not in body and "stream_options" not in body


def test_unlimited_predict_sends_no_max_tokens(store):
    for predict in (-1, -2):
        body = translate.chat_body({"messages": [], "options": {"num_predict": predict}}, store.get("qwen3:14b"), stream=True)
        assert "max_tokens" not in body
        assert body["stream_options"] == {"include_usage": True}


def test_schema_format_becomes_json_schema(store):
    schema = {"type": "object", "properties": {"kind": {"type": "string"}}}
    body = translate.chat_body({"messages": [], "format": schema}, store.get("qwen3:14b"), stream=False)
    assert body["response_format"] == {"type": "json_schema", "json_schema": {"name": "response", "schema": schema}}


def test_tool_history_gets_matching_call_ids():
    messages = translate.convert_messages([
        {"role": "user", "content": "12 x 15 and the weather"},
        {"role": "assistant", "content": "", "thinking": "need tools", "tool_calls": [
            {"function": {"name": "calculate", "arguments": {"expression": "12*15"}}},
            {"function": {"name": "weather", "arguments": {}}}]},
        {"role": "tool", "content": "sunny", "tool_name": "weather"},
        {"role": "tool", "content": "180", "tool_name": "calculate"},
    ])
    calls = messages[1]["tool_calls"]
    assert [call["id"] for call in calls] == ["call_0", "call_1"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"expression": "12*15"}
    assert messages[1]["reasoning_content"] == "need tools"
    assert messages[2]["tool_call_id"] == "call_1" and messages[3]["tool_call_id"] == "call_0"


def test_tool_results_without_names_match_in_order():
    messages = translate.convert_messages([
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "a", "arguments": {}}}, {"function": {"name": "b", "arguments": {}}}]},
        {"role": "tool", "content": "1"}, {"role": "tool", "content": "2"}])
    assert [message.get("tool_call_id") for message in messages[1:]] == ["call_0", "call_1"]


def test_images_become_data_urls_in_content_parts():
    png = "iVBORw0KGgoAAAANSUhEUg"
    messages = translate.convert_messages([{"role": "user", "content": "what is this?", "images": [png, "/9j/4AAQ"]}])
    parts = messages[0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this?"}
    assert parts[1]["image_url"]["url"] == f"data:image/png;base64,{png}"
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_images_need_a_vision_model(store):
    with pytest.raises(translate.RequestError):
        translate.chat_body({"messages": [{"role": "user", "content": "x", "images": ["iVBOR"]}]}, store.get("qwen3:14b"), stream=False)


def test_model_system_prompt_is_used_when_the_request_has_none():
    assert translate.convert_messages([{"role": "user", "content": "hi"}], "Be brief.")[0] == {"role": "system", "content": "Be brief."}
    assert translate.convert_messages([{"role": "system", "content": "Mine"}], "Be brief.") == [{"role": "system", "content": "Mine"}]


@pytest.mark.parametrize("bad", [{"messages": "hi"}, {"messages": [{"role": "robot", "content": "x"}]},
                                 {"messages": [{"role": "user", "content": 5}]}, {"messages": [], "think": "maybe"},
                                 {"messages": [], "format": "yaml"}, {"messages": [], "options": {"num_predict": -7}}])
def test_bad_requests_are_refused(store, bad):
    with pytest.raises(translate.RequestError):
        translate.chat_body(bad, store.get("qwen3:14b"), stream=False)


def test_tools_on_a_model_without_tool_support_are_refused(store):
    with pytest.raises(translate.RequestError):
        translate.chat_body({"messages": [], "tools": [{"type": "function"}]}, store.get("nomic-embed-text"), stream=False)


def test_non_streamed_reply_has_the_api_shape():
    completion = {"choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": "", "reasoning_content": "hmm",
                  "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "calculate", "arguments": "{\"expression\": \"12*15\"}"}}]}}],
                  "timings": {"prompt_n": 30, "prompt_ms": 15.5, "predicted_n": 12, "predicted_ms": 240.0}}
    reply = translate.chat_reply("qwen3:14b", completion, 2.5, 0.0)
    assert reply["done"] is True and reply["done_reason"] == "stop"
    assert reply["message"]["thinking"] == "hmm"
    assert reply["message"]["tool_calls"] == [{"id": "abc", "function": {"index": 0, "name": "calculate", "arguments": {"expression": "12*15"}}}]
    assert reply["prompt_eval_count"] == 30 and reply["eval_count"] == 12 and reply["prompt_eval_cached_count"] == 0
    assert reply["eval_duration"] == 240_000_000 and reply["load_duration"] == 2_500_000_000


def test_length_finish_is_reported_as_length():
    reply = translate.chat_reply("m", {"choices": [{"finish_reason": "length", "message": {"content": "cut"}}]}, 0, 0)
    assert reply["done_reason"] == "length"


def test_stream_assembles_tool_call_pieces_into_whole_calls():
    stream = translate.StreamTranslator("qwen3:14b", 0.0, 0.0)
    events = [
        {"choices": [{"delta": {"reasoning_content": "think"}}]},
        {"choices": [{"delta": {"content": "Let me "}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "calc", "arguments": "{\"expr"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "ession\": \"2+2\"}"}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 9}, "timings": {"predicted_n": 9, "predicted_ms": 90}},
    ]
    chunks = [chunk for event in events for chunk in stream.feed(event)] + stream.finish_chunks()
    assert chunks[0]["message"] == {"role": "assistant", "content": "", "thinking": "think"}
    assert chunks[1]["message"]["content"] == "Let me "
    assert chunks[2]["message"]["tool_calls"] == [{"id": "c1", "function": {"index": 0, "name": "calc", "arguments": {"expression": "2+2"}}}]
    final = chunks[-1]
    assert final["done"] is True and final["done_reason"] == "stop" and final["eval_count"] == 9
    assert all(chunk["done"] is False for chunk in chunks[:-1])


def test_unparseable_arguments_are_passed_through_as_text():
    assert translate.parse_arguments("not json") == "not json"
    assert translate.parse_arguments("") == {}


def test_generate_turns_prompt_into_messages(store):
    model = store.get("qwen2.5vl:7b")
    assert translate.generate_messages({"prompt": "hi"}, model) == [{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "hi"}]
    assert translate.generate_reply_from_chat({"message": {"content": "yo", "thinking": "t"}, "done": True}) == {"response": "yo", "thinking": "t", "done": True}


def test_embed_inputs_accept_text_or_lists():
    assert translate.embed_inputs({"input": "a"}) == ["a"]
    assert translate.embed_inputs({"input": ["a", "b"]}) == ["a", "b"]
    with pytest.raises(translate.RequestError):
        translate.embed_inputs({"input": [1]})


def test_cached_prompt_tokens_are_counted_in_the_prompt():
    reply = translate.chat_reply("m", {"choices": [{"message": {"content": "x"}}], "timings": {"prompt_n": 12, "cache_n": 150}}, 0, 0)
    assert reply["prompt_eval_count"] == 162 and reply["prompt_eval_cached_count"] == 150


LEAKED = "Yes, 14 orange stars.</think>14"


@pytest.mark.parametrize("keep_thinking, thinking", [(True, "Yes, 14 orange stars."), (False, None)])
def test_reasoning_leaked_into_a_reply_is_moved_or_dropped(keep_thinking, thinking):
    completion = {"choices": [{"finish_reason": "stop", "message": {"content": LEAKED}}]}
    reply = translate.chat_reply("mimo-v2.6:9b", completion, 0, 0, keep_thinking=keep_thinking)
    assert reply["message"]["content"] == "14"
    assert reply["message"].get("thinking") == thinking


def stream_of(pieces, hold_content=True, keep_thinking=True, reasoning_first=False):
    stream = translate.StreamTranslator("mimo-v2.6:9b", 0.0, 0.0, hold_content=hold_content, keep_thinking=keep_thinking)
    chunks = []
    if reasoning_first:
        chunks.append(stream.feed({"choices": [{"delta": {"reasoning_content": "plan"}}]}))
    for piece in pieces:
        chunks.append(stream.feed({"choices": [{"delta": {"content": piece}}]}))
    chunks.append(stream.finish_chunks())
    return chunks


def contents(chunks) -> list[str]:
    return [chunk["message"]["content"] for batch in chunks for chunk in batch if chunk["message"].get("content")]


def test_a_streamed_leak_never_reaches_the_client_as_content():
    chunks = stream_of(["Yes, 14 ", "orange stars.</th", "ink>14"])
    assert contents(chunks) == ["14"]
    assert chunks[0] == [] and chunks[1] == []
    assert [chunk["message"]["thinking"] for batch in chunks for chunk in batch if chunk["message"].get("thinking")] == ["Yes, 14 orange stars."]


def test_a_streamed_leak_is_dropped_when_thinking_was_turned_off():
    chunks = stream_of(["Yes, 14 orange stars.</think>", "14"], keep_thinking=False)
    assert contents(chunks) == ["14"]
    assert not any(chunk["message"].get("thinking") for batch in chunks for chunk in batch)


def test_held_text_without_a_tag_comes_out_before_the_end():
    chunks = stream_of(["Fourteen", " stars."])
    assert contents(chunks) == ["Fourteen stars."]
    assert chunks[-1][-1]["done"] is True


def test_reasoning_from_the_runner_is_a_safe_point_and_content_flows_at_once():
    chunks = stream_of(["Four", "teen"], reasoning_first=True)
    assert contents(chunks) == ["Four", "teen"]


def test_a_model_without_thinking_streams_at_once():
    chunks = stream_of(["Four", "teen"], hold_content=False)
    assert chunks[0][0]["message"]["content"] == "Four"
