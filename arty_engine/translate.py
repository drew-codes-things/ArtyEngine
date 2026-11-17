import json
import math
import time
from datetime import datetime, timezone

OPTION_FIELDS = {
    "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p", "typical_p": "typical_p",
    "repeat_penalty": "repeat_penalty", "repeat_last_n": "repeat_last_n", "presence_penalty": "presence_penalty",
    "frequency_penalty": "frequency_penalty", "seed": "seed", "stop": "stop", "mirostat": "mirostat",
    "mirostat_tau": "mirostat_tau", "mirostat_eta": "mirostat_eta",
}
LOAD_OPTIONS = {"num_ctx", "num_gpu", "num_thread", "num_batch", "main_gpu", "use_mmap", "use_mlock", "numa", "low_vram", "num_keep"}
FILL_CONTEXT = -2
NO_LIMIT = -1
THINK_LEVELS = {"low", "medium", "high"}
FINISH_REASONS = {"stop": "stop", "length": "length", "tool_calls": "stop", "content_filter": "stop"}
NANOSECONDS_PER_MS = 1_000_000
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
THINK_TAGS = (THINK_OPEN, THINK_CLOSE)
THINKING_SWITCH = "enable_thinking"
LARGEST_TOKEN_COUNT = 2 ** 31 - 1
INT64_LIMITS = (-(2 ** 63), 2 ** 63 - 1)
WHOLE_NUMBER_FIELDS = {"top_k", "seed", "repeat_last_n", "mirostat"}


class RequestError(ValueError):
    pass


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def whole_number(value, label: str) -> int:
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    raise RequestError(f"{label} must be a whole number")


def bounded_whole_number(value, label: str, lowest: int, highest: int) -> int:
    number = whole_number(value, label)
    if not lowest <= number <= highest:
        raise RequestError(f"{label} must be between {lowest} and {highest}")
    return number


def context_size(value) -> int:
    return bounded_whole_number(value, "num_ctx", 0, LARGEST_TOKEN_COUNT)


def finite_number(value, label: str):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestError(f"{label} must be a number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise RequestError(f"{label} must be a finite number")
    return value


def stop_list(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise RequestError("stop must be a string or a list of strings")


def sampling_value(name: str, value):
    if name == "stop":
        return stop_list(value)
    if name in WHOLE_NUMBER_FIELDS:
        return bounded_whole_number(value, name, *INT64_LIMITS)
    return finite_number(value, name)


def merged_options(model_params: dict, request_options: dict | None) -> dict:
    if request_options is not None and not isinstance(request_options, dict):
        raise RequestError("options must be an object")
    merged = {**model_params, **(request_options or {})}
    if merged.get("num_ctx") is not None:
        merged["num_ctx"] = context_size(merged["num_ctx"])
    return merged


def sampling_fields(options: dict) -> dict:
    body = {}
    for option_name, runner_name in OPTION_FIELDS.items():
        if option_name in options and options[option_name] is not None:
            body[runner_name] = sampling_value(option_name, options[option_name])
    predict = options.get("num_predict")
    if predict is not None:
        predict = bounded_whole_number(predict, "num_predict", FILL_CONTEXT, LARGEST_TOKEN_COUNT)
    if predict is not None and predict not in (NO_LIMIT, FILL_CONTEXT):
        body["max_tokens"] = predict
    return body


def _image_part(image: str) -> dict:
    if not isinstance(image, str) or not image:
        raise RequestError("images must be base64 strings")
    url = image if image.startswith("data:") else f"data:{sniff_image_type(image)};base64,{image}"
    return {"type": "image_url", "image_url": {"url": url}}


def sniff_image_type(encoded: str) -> str:
    prefix = encoded[:16]
    if prefix.startswith("iVBOR"):
        return "image/png"
    if prefix.startswith("/9j/"):
        return "image/jpeg"
    if prefix.startswith("R0lGOD"):
        return "image/gif"
    if prefix.startswith("UklGR"):
        return "image/webp"
    return "image/png"


def _arguments_text(arguments) -> str:
    if isinstance(arguments, str):
        return arguments
    return json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)


def convert_messages(messages, default_system: str = "") -> list[dict]:
    if not isinstance(messages, list):
        raise RequestError("messages must be a list")
    converted = []
    pending_ids: list[tuple[str, str]] = []
    call_number = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise RequestError(f"message {index} must be an object")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise RequestError(f"message {index} has an unknown role {role!r}")
        content = message.get("content") or ""
        if not isinstance(content, str):
            raise RequestError(f"message {index} content must be a string")
        item: dict = {"role": role}
        images = message.get("images") or []
        if not isinstance(images, list):
            raise RequestError(f"message {index} images must be a list of base64 strings")
        if images and role == "user":
            item["content"] = ([{"type": "text", "text": content}] if content else []) + [_image_part(image) for image in images]
        else:
            item["content"] = content
        if role == "assistant":
            if message.get("thinking"):
                item["reasoning_content"] = message["thinking"]
            calls = []
            for call in checked_tool_calls(message.get("tool_calls"), index):
                function = call.get("function") or {}
                call_id = call.get("id") or f"call_{call_number}"
                call_number += 1
                calls.append({"id": call_id, "type": "function",
                              "function": {"name": function.get("name", ""), "arguments": _arguments_text(function.get("arguments"))}})
                pending_ids.append((call_id, function.get("name", "")))
            if calls:
                item["tool_calls"] = calls
        if role == "tool":
            item["tool_call_id"] = message.get("tool_call_id") or _match_call(pending_ids, message.get("tool_name"))
            if message.get("tool_name"):
                item["name"] = message["tool_name"]
        converted.append(item)
    if default_system and not any(item["role"] == "system" for item in converted):
        converted.insert(0, {"role": "system", "content": default_system})
    return converted


def is_tool_call(call) -> bool:
    if not isinstance(call, dict):
        return False
    function = call.get("function") or {}
    return isinstance(function, dict) and isinstance(function.get("name", ""), str)


def checked_tool_calls(calls, index: int) -> list:
    if not calls:
        return []
    if not isinstance(calls, list) or not all(is_tool_call(call) for call in calls):
        raise RequestError(f"message {index} tool_calls must be a list of objects, each with a function object and a string name")
    return calls


def _match_call(pending: list[tuple[str, str]], name: str | None) -> str:
    for position, (call_id, call_name) in enumerate(pending):
        if not name or call_name == name:
            pending.pop(position)
            return call_id
    return f"call_unmatched_{len(pending)}"


def response_format(format_value) -> dict | None:
    if format_value in (None, ""):
        return None
    if format_value == "json":
        return {"type": "json_object"}
    if isinstance(format_value, dict):
        return {"type": "json_schema", "json_schema": {"name": "response", "schema": format_value}}
    raise RequestError('format must be "json" or a JSON schema object')


def thinking_fields(think, thinking_capable: bool) -> dict:
    if think is None or not thinking_capable:
        return {}
    if isinstance(think, str):
        if think not in THINK_LEVELS:
            raise RequestError('think must be true, false, "low", "medium" or "high"')
        return {"reasoning_effort": think, "chat_template_kwargs": {THINKING_SWITCH: True}}
    if not isinstance(think, bool):
        raise RequestError("think must be true or false")
    return {"chat_template_kwargs": {THINKING_SWITCH: think}}


def chat_body(request: dict, model, stream: bool) -> dict:
    options = merged_options(model.params, request.get("options"))
    capabilities = model.capabilities
    tools = request.get("tools")
    if tools is not None and not isinstance(tools, list):
        raise RequestError("tools must be a list")
    if tools and "tools" not in capabilities:
        raise RequestError(f"{model.name} does not support tools")
    messages = request.get("messages") or []
    if not isinstance(messages, list):
        raise RequestError("messages must be a list")
    if any(message.get("images") for message in messages if isinstance(message, dict)) and "vision" not in capabilities:
        raise RequestError(f"{model.name} does not support images")
    if request.get("think") and "thinking" not in capabilities:
        raise RequestError(f'"{model.name}" does not support thinking')
    body = {"model": model.name, "messages": convert_messages(messages, model.system), "stream": stream,
            **sampling_fields(options), **thinking_fields(request.get("think"), "thinking" in capabilities)}
    if tools:
        body["tools"] = tools
    shaped = response_format(request.get("format"))
    if shaped:
        body["response_format"] = shaped
    if stream:
        body["stream_options"] = {"include_usage": True}
    return body


def parse_arguments(text):
    if isinstance(text, dict):
        return text
    if not text:
        return {}
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return value


def reply_tool_calls(calls) -> list[dict]:
    converted = []
    for index, call in enumerate(calls or []):
        function = call.get("function") or {}
        converted.append({"id": call.get("id") or f"call_{index}",
                          "function": {"index": index, "name": function.get("name", ""), "arguments": parse_arguments(function.get("arguments"))}})
    return converted


def timing_fields(timings: dict | None, usage: dict | None, load_seconds: float, total_seconds: float) -> dict:
    timings = timings or {}
    usage = usage or {}
    return {
        "total_duration": int(total_seconds * 1e9),
        "load_duration": int(load_seconds * 1e9),
        "prompt_eval_count": int(timings.get("prompt_n", usage.get("prompt_tokens", 0)) or 0) + int(timings.get("cache_n", 0) or 0),
        "prompt_eval_cached_count": int(timings.get("cache_n", 0) or 0),
        "prompt_eval_duration": int(float(timings.get("prompt_ms", 0) or 0) * NANOSECONDS_PER_MS),
        "eval_count": int(timings.get("predicted_n", usage.get("completion_tokens", 0)) or 0),
        "eval_duration": int(float(timings.get("predicted_ms", 0) or 0) * NANOSECONDS_PER_MS),
    }


def finish_reason(reason: str | None) -> str:
    return FINISH_REASONS.get(reason or "stop", "stop")


HOLDING = "holding"
ANSWER = "answer"
LATE_BLOCK = "late block"
AS_WRITTEN = "as written"
CONTENT = "content"
THINKING = "thinking"
CUT_FINISH = "length"


def without_think_tags(text: str) -> str:
    for tag in THINK_TAGS:
        text = text.replace(tag, "")
    return text


BEFORE_FIRST_CLOSE = "text before the first closing tag"
TAGGED_BLOCK = "a tagged block in the answer"
CUT_AFTER_OPEN = "text after an opening tag in a reply cut by the length limit"


class LeakSplitter:
    def __init__(self, named_leaker: bool = False):
        self.named_leaker = named_leaker
        self.phase = HOLDING
        self.held = ""
        self.tail = ""
        self.trim_answer_start = False
        self.thinking_given = False
        self.runner_reasoned = False
        self.late_tags = 0
        self.leaks: list[tuple[str, int]] = []
        self.cut_without_tag = False

    def runner_reasoning(self) -> list[tuple[str, str]]:
        self.runner_reasoned = True
        if self.phase != HOLDING:
            self.thinking_given = True
            return []
        held, self.held = self.held, ""
        if not self.named_leaker:
            self.phase = AS_WRITTEN
            self.thinking_given = True
            return [(CONTENT, held)] if held else []
        self.phase = ANSWER
        out = self._answer(held)
        self.thinking_given = True
        return out

    def feed(self, text: str) -> list[tuple[str, str]]:
        if self.phase == AS_WRITTEN:
            return [(CONTENT, text)] if text else []
        if self.phase != HOLDING:
            return self._answer(text)
        self.held += text
        if THINK_CLOSE not in self.held:
            return []
        leaked, _, rest = self.held.partition(THINK_CLOSE)
        self.held, self.phase, self.trim_answer_start = "", ANSWER, True
        self._caught(BEFORE_FIRST_CLOSE, leaked)
        return self._thinking(leaked) + self._answer(rest)

    def finish(self, finish_reason: str | None) -> list[tuple[str, str]]:
        was_cut = finish_reason == CUT_FINISH
        phase, held, tail = self.phase, self.held, self.tail
        self.phase, self.held, self.tail = AS_WRITTEN, "", ""
        if phase == HOLDING:
            if was_cut and THINK_OPEN in held:
                return self._cut_after_opening_tag(held)
            self.cut_without_tag = was_cut
            return [(CONTENT, held)] if held else []
        if phase == LATE_BLOCK:
            if was_cut:
                self.late_tags += 1
                self._caught(CUT_AFTER_OPEN, held + tail)
                return self._thinking(held + tail)
            return self._content(THINK_OPEN + held + tail)
        return self._content(tail)

    @property
    def leaks_caught(self) -> int:
        return len(self.leaks)

    def leaks_by_rule(self) -> dict[str, tuple[int, int]]:
        grouped: dict[str, tuple[int, int]] = {}
        for rule, length in self.leaks:
            moves, characters = grouped.get(rule, (0, 0))
            grouped[rule] = (moves + 1, characters + length)
        return grouped

    def _caught(self, rule: str, moved: str) -> None:
        reasoning = without_think_tags(moved).strip()
        if reasoning and not self.runner_reasoned:
            self.leaks.append((rule, len(reasoning)))

    def _cut_after_opening_tag(self, held: str) -> list[tuple[str, str]]:
        preamble, _, reasoning = held.partition(THINK_OPEN)
        self._caught(CUT_AFTER_OPEN, reasoning)
        if not preamble.strip():
            return self._thinking(reasoning)
        self.late_tags += 1
        return [(CONTENT, preamble)] + self._thinking(reasoning)

    def _answer(self, text: str) -> list[tuple[str, str]]:
        text, self.tail = self.tail + text, ""
        out = []
        while text:
            if self.phase == LATE_BLOCK:
                text = self._late_block(text, out)
                continue
            opening, closing = text.find(THINK_OPEN), text.find(THINK_CLOSE)
            found = [position for position in (opening, closing) if position >= 0]
            if not found:
                kept = partial_tag_length(text)
                out += self._content(text[:len(text) - kept])
                self.tail = text[len(text) - kept:]
                return out
            first = min(found)
            out += self._content(text[:first])
            if first == opening:
                self.phase = LATE_BLOCK
                text = text[first + len(THINK_OPEN):]
            else:
                self.late_tags += 1
                text = text[first + len(THINK_CLOSE):]
        return out

    def _late_block(self, text: str, out: list) -> str:
        end = text.find(THINK_CLOSE)
        if end < 0:
            kept = partial_tag_length(text)
            self.held += text[:len(text) - kept]
            self.tail = text[len(text) - kept:]
            return ""
        block, self.held = self.held + text[:end], ""
        self.late_tags += 2
        self._caught(TAGGED_BLOCK, block)
        self.phase = ANSWER
        out += self._thinking(block)
        return text[end + len(THINK_CLOSE):]

    def _content(self, text: str) -> list[tuple[str, str]]:
        if self.trim_answer_start:
            text = text.lstrip()
            self.trim_answer_start = not text
        return [(CONTENT, text)] if text else []

    def _thinking(self, text: str) -> list[tuple[str, str]]:
        text = without_think_tags(text).strip()
        if not text:
            return []
        separator = "\n" if self.thinking_given else ""
        self.thinking_given = True
        return [(THINKING, separator + text)]


def joined(events: list[tuple[str, str]], kind: str) -> str:
    return "".join(text for event_kind, text in events if event_kind == kind)


def chat_reply(model_name: str, completion: dict, load_seconds: float, started: float, keep_thinking: bool = True,
               splitter: LeakSplitter | None = None) -> dict:
    choice = (completion.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    leaked = ""
    if splitter:
        events = (splitter.runner_reasoning() if reasoning else []) + splitter.feed(content)
        events += splitter.finish(choice.get("finish_reason"))
        content, leaked = joined(events, CONTENT), joined(events, THINKING)
    reply_message = {"role": "assistant", "content": content}
    thinking = reasoning + (leaked if keep_thinking else "")
    if thinking:
        reply_message["thinking"] = thinking
    calls = reply_tool_calls(message.get("tool_calls"))
    if calls:
        reply_message["tool_calls"] = calls
    return {"model": model_name, "created_at": now_iso(), "message": reply_message,
            "done_reason": finish_reason(choice.get("finish_reason")), "done": True,
            **timing_fields(completion.get("timings"), completion.get("usage"), load_seconds, time.monotonic() - started)}


def partial_tag_length(text: str) -> int:
    longest = max(len(tag) for tag in THINK_TAGS) - 1
    for length in range(min(longest, len(text)), 0, -1):
        if any(tag.startswith(text[-length:]) for tag in THINK_TAGS):
            return length
    return 0


class StreamTranslator:
    def __init__(self, model_name: str, load_seconds: float, started: float, keep_thinking: bool = True,
                 splitter: LeakSplitter | None = None):
        self.model_name = model_name
        self.load_seconds = load_seconds
        self.started = started
        self.splitter = splitter
        self.keep_thinking = keep_thinking
        self.calls: dict[int, dict] = {}
        self.finish = None
        self.timings = None
        self.usage = None

    @property
    def late_tags(self) -> int:
        return self.splitter.late_tags if self.splitter else 0

    def _chunk(self, message: dict) -> dict:
        return {"model": self.model_name, "created_at": now_iso(), "message": {"role": "assistant", **message}, "done": False}

    def _chunks(self, events: list[tuple[str, str]]) -> list[dict]:
        out = []
        for kind, text in events:
            if kind == CONTENT:
                out.append(self._chunk({"content": text}))
            elif self.keep_thinking:
                out.append(self._chunk({"content": "", "thinking": text}))
        return out

    def feed(self, event: dict) -> list[dict]:
        if event.get("timings"):
            self.timings = event["timings"]
        if event.get("usage"):
            self.usage = event["usage"]
        out = []
        for choice in event.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("reasoning_content"):
                out += self._chunks(self.splitter.runner_reasoning()) if self.splitter else []
                out.append(self._chunk({"content": "", "thinking": delta["reasoning_content"]}))
            if delta.get("content"):
                text = delta["content"]
                out += self._chunks(self.splitter.feed(text)) if self.splitter else [self._chunk({"content": text})]
            for piece in delta.get("tool_calls") or []:
                self._add_call_piece(piece)
            if choice.get("finish_reason"):
                self.finish = choice["finish_reason"]
        return out

    def _add_call_piece(self, piece: dict) -> None:
        slot = self.calls.setdefault(int(piece.get("index", len(self.calls))), {"id": "", "function": {"name": "", "arguments": ""}})
        if piece.get("id"):
            slot["id"] = piece["id"]
        function = piece.get("function") or {}
        if function.get("name"):
            slot["function"]["name"] += function["name"]
        if function.get("arguments"):
            slot["function"]["arguments"] += function["arguments"]

    def finish_chunks(self) -> list[dict]:
        out = self._chunks(self.splitter.finish(self.finish)) if self.splitter else []
        if self.calls:
            ordered = [self.calls[index] for index in sorted(self.calls)]
            out.append(self._chunk({"content": "", "tool_calls": reply_tool_calls(ordered)}))
        out.append({"model": self.model_name, "created_at": now_iso(), "message": {"role": "assistant", "content": ""},
                    "done_reason": finish_reason(self.finish), "done": True,
                    **timing_fields(self.timings, self.usage, self.load_seconds, time.monotonic() - self.started)})
        return out


def generate_messages(request: dict, model) -> list[dict]:
    prompt = request.get("prompt") or ""
    if not isinstance(prompt, str):
        raise RequestError("prompt must be a string")
    system = request.get("system") or model.system
    messages = [{"role": "system", "content": system}] if system else []
    messages.append({"role": "user", "content": prompt, **({"images": request["images"]} if request.get("images") else {})})
    return messages


def generate_reply_from_chat(chat: dict) -> dict:
    message = chat.pop("message", {})
    chat["response"] = message.get("content", "")
    if message.get("thinking"):
        chat["thinking"] = message["thinking"]
    return chat


def render_body(request: dict, model) -> dict:
    body = chat_body(request, model, stream=False)
    body.pop("stream", None)
    return body


def image_count(messages) -> int:
    return sum(len(message.get("images") or []) for message in messages or [] if isinstance(message, dict))


def raw_completion_body(request: dict, model, stream: bool) -> dict:
    prompt = request.get("prompt") or ""
    if not isinstance(prompt, str):
        raise RequestError("prompt must be a string")
    return {"prompt": prompt, "stream": stream, **sampling_fields(merged_options(model.params, request.get("options")))}


def raw_reply(model_name: str, completion: dict, load_seconds: float, started: float) -> dict:
    usage = {"prompt_tokens": completion.get("tokens_evaluated", 0), "completion_tokens": completion.get("tokens_predicted", 0)}
    return {"model": model_name, "created_at": now_iso(), "response": completion.get("content") or "", "done": True,
            "done_reason": "length" if completion.get("stop_type") == "limit" else "stop",
            **timing_fields(completion.get("timings"), usage, load_seconds, time.monotonic() - started)}


def raw_stream_chunks(model_name: str, event: dict, load_seconds: float, started: float) -> list[dict]:
    chunks = []
    if event.get("content"):
        chunks.append({"model": model_name, "created_at": now_iso(), "response": event["content"], "done": False})
    if event.get("stop"):
        chunks.append(raw_reply(model_name, {**event, "content": ""}, load_seconds, started))
    return chunks


def embed_dimensions(request: dict) -> int | None:
    value = request.get("dimensions")
    if value is None or value == 0:
        return None
    size = whole_number(value, "dimensions")
    if size < 1:
        raise RequestError("dimensions must be a positive whole number")
    return size


def embed_inputs(request: dict) -> list[str]:
    value = request.get("input", request.get("prompt"))
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise RequestError("input must be a string or a list of strings")
