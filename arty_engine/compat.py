import itertools
import json
import re
import time
import urllib.error
import urllib.request
from http import HTTPStatus
from urllib.parse import parse_qs, unquote, urlencode, urlsplit

from . import remote, translate
from .runner import is_reranker

OPENAI_PREFIX = "/v1/"
RUNNER_TIMEOUT_SECONDS = 3600
SLOT_ACTIONS = ("save", "restore", "erase")
SAMPLE_LINE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?(\s.*)$")
DESCRIBED_LINE = re.compile(r"^# (HELP|TYPE) (\S+)(.*)$")
RUNTIME_METRIC_PREFIX = "arty_runtime_"
ERROR_TYPES = {400: "invalid_request_error", 401: "authentication_error", 403: "permission_error", 404: "not_found_error",
               429: "rate_limit_error", 501: "not_supported_error", 503: "unavailable_error"}
OWNER = "arty-engine"
EVENT_STREAM = "text/event-stream"
PASS_THROUGH_CHUNK_BYTES = 65536


class RelayError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def openai_error(status: int, message: str) -> dict:
    return {"error": {"message": message, "type": ERROR_TYPES.get(status, "server_error"), "code": status}}


def runner_message(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read() or b"{}")
    except ValueError:
        return f"model runner error {error.code}"
    detail = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail or payload)


def runner_call(url: str, body=None, method: str | None = None):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method or ("POST" if data is not None else "GET"),
                                     headers={"Content-Type": "application/json"})
    try:
        return urllib.request.urlopen(request, timeout=RUNNER_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as error:
        raise RelayError(error.code if error.code < 500 or error.code == HTTPStatus.NOT_IMPLEMENTED else 500,
                         runner_message(error)) from error
    except (urllib.error.URLError, OSError) as error:
        raise RelayError(500, f"the model runner stopped answering: {error}") from error


def runner_json(url: str, body=None, method: str | None = None):
    with runner_call(url, body, method) as response:
        raw = response.read()
    try:
        return json.loads(raw) if raw else {}
    except ValueError as error:
        raise RelayError(500, "the model runner answered with something other than JSON") from error


def runner_text(url: str) -> str:
    with runner_call(url) as response:
        return response.read().decode(errors="replace")


def model_entry(model) -> dict:
    modified = getattr(model, "modified_at", None)
    created = int(modified.timestamp()) if modified else int(time.time())
    return {"id": model.name, "object": "model", "created": created, "owned_by": OWNER}


def with_model_defaults(request: dict, model, prompt_field: str | None = None) -> dict:
    body = {key: value for key, value in request.items() if key not in ("keep_alive", "options")}
    body["model"] = model.name
    for option_name, runner_name in translate.OPTION_FIELDS.items():
        if runner_name not in body and model.params.get(option_name) is not None:
            body[runner_name] = translate.sampling_value(option_name, model.params[option_name])
    predict = model.params.get("num_predict")
    if prompt_field is None and "max_tokens" not in body and "max_completion_tokens" not in body and isinstance(predict, int) \
            and predict > 0:
        body["max_tokens"] = predict
    return body


def chat_messages(messages, model) -> list:
    if not isinstance(messages, list) or not all(isinstance(message, dict) for message in messages):
        raise translate.RequestError("messages must be a list of objects")
    combined = translate.with_recipe_messages(messages, model.messages)
    if model.system and not any(message.get("role") == "system" for message in combined):
        combined = [{"role": "system", "content": model.system}] + combined
    return combined


def runtime_metric_name(name: str) -> str:
    return RUNTIME_METRIC_PREFIX + name.rsplit(":", 1)[-1]


def labelled_samples(text: str, model_name: str, seen: set) -> list[str]:
    label = 'model="' + model_name.replace("\\", "\\\\").replace('"', '\\"') + '"'
    lines = []
    for line in text.splitlines():
        described = DESCRIBED_LINE.match(line)
        if described:
            kind, name, rest = described.groups()
            renamed = f"# {kind} {runtime_metric_name(name)}{rest}"
            if renamed not in seen:
                seen.add(renamed)
                lines.append(renamed)
            continue
        matched = SAMPLE_LINE.match(line)
        if not matched:
            continue
        name, labels, rest = matched.groups()
        inner = labels[1:-1] if labels else ""
        lines.append(f"{runtime_metric_name(name)}{{{label}{',' + inner if inner else ''}}}{rest}")
    return lines


def engine_samples(loaded: int, workers: list[dict]) -> list[str]:
    lines = ["# HELP arty_engine_loaded_models Models loaded right now.", "# TYPE arty_engine_loaded_models gauge",
             f"arty_engine_loaded_models {loaded}"]
    if workers:
        lines += ["# HELP arty_engine_rpc_worker_up 1 when the rpc worker answered its last check.",
                  "# TYPE arty_engine_rpc_worker_up gauge"]
        for worker in workers:
            label = worker["label"].replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'arty_engine_rpc_worker_up{{worker="{label}",address="{worker["address"]}"}} {1 if worker["up"] else 0}')
    return lines


class CompatRoutes:
    engine = None

    def _query(self) -> dict:
        return {key: values[-1] for key, values in parse_qs(urlsplit(self.path).query).items()}

    def _wants_openai_errors(self) -> bool:
        return urlsplit(self.path).path.startswith(OPENAI_PREFIX)

    def _send_text(self, status: int, text: str, content_type: str) -> None:
        data = text.encode()
        self.send_response(status)
        self.send_header("Connection", "close")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _start_events(self) -> None:
        self.send_response(200)
        self.send_header("Connection", "close")
        self.send_header("Content-Type", EVENT_STREAM)
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self._streaming = True
        self._events = True

    def _event(self, payload) -> None:
        text = payload if isinstance(payload, str) else json.dumps(payload)
        data = f"data: {text}\n\n".encode()
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _relay_events(self, events, rename: str = "") -> None:
        first = next(events, None)
        self._start_events()
        for event in itertools.chain([first] if first is not None else [], events):
            self._event({**event, "model": rename} if rename and "model" in event else event)
        self._event("[DONE]")
        self._end_stream()

    def _runner_events(self, url: str, body: dict):
        with runner_call(url, body) as response:
            for event in remote.server_sent_events(response):
                if event.get("error"):
                    detail = event["error"]
                    raise RelayError(500, str(detail.get("message") if isinstance(detail, dict) else detail))
                yield event

    def _openai_model(self, request: dict):
        model = self.engine.model(request.get("model"))
        return model, self.engine.keep_alive(request)

    def _with_local(self, model, request: dict, keep_alive, work):
        if not model.runnable:
            raise RelayError(501, model.unrunnable_reason)
        options = translate.merged_options(model.params, request.get("options"))
        return self.engine.with_runner(model, options, keep_alive, work, prefer_loaded=request.get("options") is None)

    def _relay(self, model, request: dict, keep_alive, path: str, body: dict) -> None:
        def run(runner, load_seconds):
            url = f"{runner.base_url}{path}"
            if body.get("stream"):
                self._relay_events(self._runner_events(url, body))
                return
            self._send_json(200, runner_json(url, body))
        self._with_local(model, request, keep_alive, run)

    def v1_models(self):
        models = self.engine.listed_models() + self.engine.remotes.models()
        self._send_json(200, {"object": "list", "data": [model_entry(model) for model in models]})

    def v1_model(self):
        name = unquote(urlsplit(self.path).path[len(f"{OPENAI_PREFIX}models/"):])
        self._send_json(200, model_entry(self.engine.model(name)))

    def v1_chat(self):
        request = self._body()
        model, keep_alive = self._openai_model(request)
        if remote.is_remote(model):
            chat = self.engine.remotes.chat(model)
            if request.get("stream"):
                self._relay_events(chat.stream(request), rename=model.name)
            else:
                self._send_json(200, {**chat.post(request), "model": model.name})
            return
        if model.is_embedding:
            raise RelayError(400, f'"{model.name}" does not support chat')
        body = with_model_defaults(request, model)
        body["messages"] = chat_messages(request.get("messages"), model)
        self._relay(model, request, keep_alive, "/v1/chat/completions", body)

    def v1_completions(self):
        request = self._body()
        model, keep_alive = self._openai_model(request)
        self._refuse_remote(model)
        if model.is_embedding:
            raise RelayError(400, f'"{model.name}" does not support completions')
        self._relay(model, request, keep_alive, "/v1/completions", with_model_defaults(request, model, "prompt"))

    def v1_embeddings(self):
        request = self._body()
        model, keep_alive = self._openai_model(request)
        self._refuse_remote(model)
        if not model.is_embedding:
            raise RelayError(400, f'"{model.name}" does not support embeddings')
        body = {key: value for key, value in request.items() if key not in ("keep_alive", "options")}
        self._relay(model, request, keep_alive, "/v1/embeddings", {**body, "model": model.name})

    def rerank(self):
        request = self._body()
        model, keep_alive = self._openai_model(request)
        self._refuse_remote(model)
        if not is_reranker(model):
            raise RelayError(400, f'"{model.name}" is not a reranking model')
        body = {key: value for key, value in request.items() if key not in ("keep_alive", "options")}
        self._relay(model, request, keep_alive, "/v1/rerank", {**body, "model": model.name})

    def _refuse_remote(self, model) -> None:
        if remote.is_remote(model):
            raise RelayError(501, f"{model.name}: a remote model only answers chats")

    def _runner_post(self, path: str) -> None:
        request = self._body()
        model, keep_alive = self._openai_model(request)
        self._refuse_remote(model)
        body = {key: value for key, value in request.items() if key not in ("model", "keep_alive", "options")}
        self._relay(model, request, keep_alive, path, body)

    def _pass_through(self, path: str) -> None:
        request = self._body()
        model, keep_alive = self._openai_model(request)
        self._refuse_remote(model)
        body = {**{key: value for key, value in request.items() if key not in ("keep_alive", "options")}, "model": model.name}

        def run(runner, load_seconds):
            with runner_call(f"{runner.base_url}{path}", body) as response:
                content_type = response.headers.get("Content-Type") or "application/json"
                if not content_type.startswith(EVENT_STREAM):
                    self._send_text(200, response.read().decode(errors="replace"), content_type)
                    return
                self._start_events()
                while chunk := response.read1(PASS_THROUGH_CHUNK_BYTES):
                    self.wfile.write(f"{len(chunk):x}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                self._end_stream()
        self._with_local(model, request, keep_alive, run)

    def native_completion(self):
        self._pass_through("/completion")

    def v1_responses(self):
        self._pass_through("/v1/responses")

    def v1_messages(self):
        self._pass_through("/v1/messages")

    def v1_count_tokens(self):
        self._pass_through("/v1/messages/count_tokens")

    def tokenize(self):
        self._runner_post("/tokenize")

    def detokenize(self):
        self._runner_post("/detokenize")

    def apply_template(self):
        request = self._body()
        model, keep_alive = self._openai_model(request)
        self._refuse_remote(model)
        body = {"messages": chat_messages(request.get("messages") or [], model)}
        if request.get("tools"):
            body["tools"] = request["tools"]
        self._relay(model, request, keep_alive, "/apply-template", body)

    def infill(self):
        self._runner_post("/infill")

    def _named_model(self, request: dict | None = None):
        name = (request or {}).get("model") or self._query().get("model")
        if name:
            return self.engine.model(name)
        loaded = self.engine.pool.loaded()
        if len(loaded) == 1:
            return loaded[0].plan.model
        raise RelayError(400, "name the model with ?model=<name> (it is needed when not exactly one model is loaded)")

    def _runner_get(self, path: str) -> None:
        model = self._named_model()
        self._refuse_remote(model)
        keep_alive = self.engine.keep_alive({})
        result = {}
        self._with_local(model, {}, keep_alive, lambda runner, load_seconds: result.update(
            payload=runner_json(f"{runner.base_url}{path}")))
        self._send_json(200, result["payload"])

    def props(self):
        self._runner_get("/props")

    def slots(self):
        self._runner_get("/slots")

    def slot_action(self):
        slot = urlsplit(self.path).path.rsplit("/", 1)[1]
        action = self._query().get("action", "")
        if not slot.isdigit() or action not in SLOT_ACTIONS:
            raise RelayError(400, f"use /slots/<number>?action=<{'|'.join(SLOT_ACTIONS)}>")
        request = self._body()
        model = self._named_model(request)
        self._refuse_remote(model)
        body = {key: value for key, value in request.items() if key != "model"}
        result = {}
        self._with_local(model, {}, self.engine.keep_alive({}), lambda runner, load_seconds: result.update(
            payload=runner_json(f"{runner.base_url}/slots/{slot}?{urlencode({'action': action})}", body)))
        self._send_json(200, result["payload"])

    def lora_adapters(self):
        if self.command == "GET":
            self._runner_get("/lora-adapters")
            return
        request = self._body()
        model = self._named_model(request)
        self._refuse_remote(model)
        adapters = request.get("adapters")
        if not isinstance(adapters, list):
            raise RelayError(400, "adapters must be a list of {id, scale} objects")
        result = {}
        self._with_local(model, {}, self.engine.keep_alive({}), lambda runner, load_seconds: result.update(
            payload=runner_json(f"{runner.base_url}/lora-adapters", adapters)))
        self._send_json(200, result["payload"])

    def metrics(self):
        seen: set = set()
        lines = engine_samples(len(self.engine.pool.loaded()), self.engine.pool.workers_status())
        for runner in self.engine.pool.loaded():
            try:
                lines += labelled_samples(runner_text(f"{runner.base_url}/metrics"), runner.name, seen)
            except RelayError:
                continue
        self._send_text(200, "\n".join(lines) + "\n", "text/plain; version=0.0.4")

    def health(self):
        self._send_json(200, {"status": "ok"})
