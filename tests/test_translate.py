import json
import re

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
    reply = translate.chat_reply("mimo-v2.6:9b", completion, 0, 0, keep_thinking=keep_thinking,
                                 splitter=translate.LeakSplitter())
    assert reply["message"]["content"] == "14"
    assert reply["message"].get("thinking") == thinking


def stream_of(pieces, can_leak=True, keep_thinking=True, reasoning_first=False):
    splitter = translate.LeakSplitter() if can_leak else None
    stream = translate.StreamTranslator("mimo-v2.6:9b", 0.0, 0.0, keep_thinking=keep_thinking, splitter=splitter)
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
    chunks = stream_of(["Four", "teen"], can_leak=False)
    assert chunks[0][0]["message"]["content"] == "Four"


REASONING_LEAK = "Okay, the user wants the count of orange stars, so I should look at the picture again and count. " * 3
PLAIN_ANSWER = "There are fourteen orange stars in the picture, set out in two rows of seven across the top of the page. " * 3


def test_a_leaker_s_reply_without_a_tag_is_held_to_the_end_however_it_opens():
    pieces = [PLAIN_ANSWER[index:index + 10] for index in range(0, len(PLAIN_ANSWER), 10)]
    chunks = stream_of(pieces)
    assert not any(chunk["message"].get("content") for batch in chunks[:-1] for chunk in batch)
    assert "".join(contents(chunks)) == PLAIN_ANSWER


def test_text_that_reads_like_reasoning_stays_held_until_its_tag():
    pieces = [REASONING_LEAK[index:index + 10] for index in range(0, len(REASONING_LEAK), 10)] + ["</think>", "14"]
    chunks = stream_of(pieces)
    assert contents(chunks) == ["14"]


def test_a_tag_that_turns_up_after_the_answer_was_released_is_dropped_from_the_rest():
    chunks = stream_of(["count</think>Fourteen", " stars.</th", "ink> Fourteen."])
    assert "".join(contents(chunks)) == "Fourteen stars. Fourteen."


def test_an_opening_tag_after_the_answer_started_that_never_closes_stays_in_the_answer():
    chunks = stream_of(["count</think>Fourteen <think>", "stars."])
    assert "".join(contents(chunks)) == "Fourteen <think>stars."


def test_a_lone_angle_bracket_at_the_end_still_comes_out():
    assert "".join(contents(stream_of(["count</think>5 <"]))) == "5 <"


def test_a_model_that_cannot_leak_streams_think_tags_as_it_wrote_them():
    pieces = ["print(\"<thi", "nk>a</th", "ink>\")"]
    assert "".join(contents(stream_of(pieces, can_leak=False))) == "".join(pieces)
    completion = {"choices": [{"finish_reason": "stop", "message": {"content": "".join(pieces)}}]}
    assert translate.chat_reply("qwen3:14b", completion, 0, 0, keep_thinking=False)["message"]["content"] == "".join(pieces)


def leaker_reply(text, keep_thinking, finish="stop", reasoning="", piece_size=None, named_leaker=False):
    stream = translate.StreamTranslator("mimo-v2.6:9b", 0.0, 0.0, keep_thinking=keep_thinking,
                                        splitter=translate.LeakSplitter(named_leaker=named_leaker))
    chunks = stream.feed({"choices": [{"delta": {"reasoning_content": reasoning}}]}) if reasoning else []
    pieces = re.findall(r"\s*\S+\s*|\s+", text) if piece_size is None else [text[index:index + piece_size]
                                                                           for index in range(0, len(text), piece_size)]
    for piece in pieces:
        chunks += stream.feed({"choices": [{"delta": {"content": piece}}]})
    chunks += stream.feed({"choices": [{"delta": {}, "finish_reason": finish}]}) + stream.finish_chunks()
    streamed = ("".join(chunk["message"].get("content", "") for chunk in chunks),
                "".join(chunk["message"].get("thinking", "") for chunk in chunks))
    message = translate.chat_reply("mimo-v2.6:9b", {"choices": [{"finish_reason": finish, "message": {
        "content": text, **({"reasoning_content": reasoning} if reasoning else {})}}]}, 0, 0,
        keep_thinking=keep_thinking, splitter=translate.LeakSplitter(named_leaker=named_leaker))["message"]
    return streamed, (message["content"], message.get("thinking", ""))


TAG_ANSWER = "<think> is the tag some models put in front of their reasoning. It is not HTML."
CODE_WITH_TAGS = 'return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)'
LEAKER_SHAPES = {
    "a leak, its tag, then the answer": ("Count the rows first.\n</think>\n\nTwo rows.", "stop", "Two rows."),
    "a plain answer": ("Two rows.", "stop", "Two rows."),
    "an answer that starts with an indented line": ("    for row in rows:\n        print(row)\n", "stop",
                                                    "    for row in rows:\n        print(row)\n"),
    "an answer that starts with blank lines": ("\n\nTwo rows.", "stop", "\n\nTwo rows."),
    "an answer that names the opening tag": ("The tag <think> opens a reasoning block in some models.", "stop",
                                             "The tag <think> opens a reasoning block in some models."),
    "a finished answer that opens with the tag": (TAG_ANSWER, "stop", TAG_ANSWER),
    "two reasoning blocks": ("plan one</think>Blue. <think>or red, the user may want red</think>Red.", "stop", "Blue. Red."),
    "a tagless leak cut by the length limit": ("Count the rows first, the top one has a cable and", "length",
                                               "Count the rows first, the top one has a cable and"),
    "a leak cut by the length limit after its opening tag": ("<think>\nCount the rows first, the top one", "length", ""),
    "a late block cut by the length limit": ("plan</think>Two rows. <think>or three, the", "length", "Two rows. "),
    "a leak cut by the length limit after a first line": ("Okay.\n<think>\nCount the rows first, the top one", "length",
                                                          "Okay.\n"),
}


@pytest.mark.parametrize("named_leaker", [False, True])
@pytest.mark.parametrize("piece_size", [None, 1, 5])
@pytest.mark.parametrize("keep_thinking", [False, True])
@pytest.mark.parametrize("label", list(LEAKER_SHAPES))
def test_a_leaker_s_reply_is_the_same_streamed_and_not_streamed(label, keep_thinking, piece_size, named_leaker):
    text, finish, answer = LEAKER_SHAPES[label]
    streamed, plain = leaker_reply(text, keep_thinking, finish, piece_size=piece_size, named_leaker=named_leaker)
    assert streamed == plain
    assert plain[0] == answer


def test_a_second_reasoning_block_goes_to_thinking_streamed_and_not():
    streamed, plain = leaker_reply(LEAKER_SHAPES["two reasoning blocks"][0], True)
    assert streamed == plain == ("Blue. Red.", "plan one\nor red, the user may want red")


@pytest.mark.parametrize("keep_thinking", [False, True])
def test_a_finished_answer_that_opens_with_the_tag_is_not_emptied(keep_thinking):
    streamed, plain = leaker_reply(TAG_ANSWER, keep_thinking)
    assert streamed == plain == (TAG_ANSWER, "")


def test_a_reply_cut_after_its_opening_tag_is_thinking():
    streamed, plain = leaker_reply("<think>\nCount the rows first", True, "length")
    assert streamed == plain == ("", "Count the rows first")


@pytest.mark.parametrize("named_leaker", [False, True])
@pytest.mark.parametrize("piece_size", [None, 1])
def test_a_reply_cut_after_an_opening_tag_that_follows_its_first_words_keeps_only_those_words(named_leaker, piece_size):
    text = "Okay.\n<think>\nGot it, let's look at the picture. The total is in the corner, with the currency."
    streamed, plain = leaker_reply(text, True, "length", piece_size=piece_size, named_leaker=named_leaker)
    assert streamed == plain == ("Okay.\n", "Got it, let's look at the picture. The total is in the corner, with the currency.")


def splitter_after(text, finish):
    splitter = translate.LeakSplitter()
    splitter.feed(text)
    splitter.finish(finish)
    return splitter


def test_a_cut_after_an_opening_tag_that_follows_the_first_words_is_counted_and_flagged():
    splitter = splitter_after("Okay.\n<think>\nGot it, the total", "length")
    assert (splitter.leaks_caught, splitter.late_tags, splitter.cut_without_tag) == (1, 1, False)


@pytest.mark.parametrize("text, finish, caught", [
    ("Count the rows first.\n</think>\n\nTwo rows.", "stop", 1),
    ("plan one</think>Blue. <think>or red</think>Red.", "stop", 2),
    ("<think>\nCount the rows first", "length", 1),
    ("plan</think>Two rows. <think>or three, the", "length", 2),
    ("Two rows.", "stop", 0),
    ("The tag <think> opens a reasoning block.", "stop", 0),
    ("Count the rows first, the top one", "length", 0),
])
def test_every_leak_the_splitter_takes_out_of_the_answer_is_counted(text, finish, caught):
    assert splitter_after(text, finish).leaks_caught == caught


@pytest.mark.parametrize("piece_size", [None, 1])
def test_a_template_only_leaker_s_answer_is_left_alone_once_the_runner_gave_its_reasoning(piece_size):
    streamed, plain = leaker_reply(CODE_WITH_TAGS, False, reasoning="The user wants a regex.", piece_size=piece_size)
    assert streamed == plain == (CODE_WITH_TAGS, "The user wants a regex.")


def test_tagged_blocks_taken_out_after_the_runner_s_reasoning_are_not_counted_as_leaks():
    splitter = translate.LeakSplitter(named_leaker=True)
    splitter.runner_reasoning()
    events = splitter.feed('return re.sub(r"<think>.*?</think>", "", text)')
    splitter.finish("stop")
    assert translate.joined(events, translate.CONTENT) == 'return re.sub(r"", "", text)'
    assert (splitter.leaks_caught, splitter.late_tags) == (0, 2)


def test_each_caught_leak_records_its_rule_and_the_length_moved():
    assert splitter_after("plan one</think>Blue. <think>or red</think>Red.", "stop").leaks == [
        (translate.BEFORE_FIRST_CLOSE, 8), (translate.TAGGED_BLOCK, 6)]
    assert splitter_after("Okay.\n<think>\nGot it", "length").leaks == [(translate.CUT_AFTER_OPEN, len("\nGot it"))]
