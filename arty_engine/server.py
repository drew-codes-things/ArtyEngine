import argparse
import itertools
import json
import logging
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from . import gpu, translate
from .config import Config, load_config
from .gguf import GgufError
from .pull import Puller, PullError
from .runner import LoadError, Runner, RunnerPool, parse_keep_alive
from .store import ModelNotFound, ModelStore, NotOwned, iso_time, parse_name

log = logging.getLogger("arty-engine")

VERSION = "0.1.0"
UPSTREAM_TIMEOUT_SECONDS = 3600
TRAINER_BUSY_TIMEOUT_SECONDS = 180
TRAINER_DONE_TIMEOUT_SECONDS = 5
MAX_BODY_BYTES = 256 * 1024 * 1024
FAR_FUTURE = "2318-08-18T00:00:00Z"
LOCAL_HOSTNAMES = ("localhost", "127.0.0.1", "::1")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class TrainerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post(self, path: str, request_id: str, timeout: int) -> dict:
        request = urllib.request.Request(f"{self.base_url}{path}", data=json.dumps({"id": request_id}).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")

    def busy(self, request_id: str) -> None:
        if not self.base_url:
            return
        try:
            answer = self._post("/busy", request_id, TRAINER_BUSY_TIMEOUT_SECONDS)
        except (urllib.error.URLError, OSError, ValueError) as error:
            log.warning("trainer controller unreachable, going ahead anyway: %s", error)
            return
        if answer.get("waited_seconds", 0) > 1:
            log.info("waited %.1f s for training to hand the GPU back", answer["waited_seconds"])

    def done(self, request_id: str) -> None:
        if not self.base_url:
            return
        try:
            self._post("/done", request_id, TRAINER_DONE_TIMEOUT_SECONDS)
        except (urllib.error.URLError, OSError, ValueError) as error:
            log.warning("couldn't tell the trainer controller a request finished: %s", error)


class Engine:
    def __init__(self, config: Config, pool: RunnerPool | None = None, store: ModelStore | None = None):
        self.config = config
        self.store = store or ModelStore(config.roots)
        self.pool = pool or RunnerPool(config, self.store)
        self.trainer = TrainerClient(config.trainer_url)
        self.puller = Puller(config.roots[0], config.pull_rate_bytes)
        self.pull_lock = threading.Lock()
        self.internal_names = {parse_name(name).display for name in config.internal_models}

    def listed_models(self) -> list:
        return [model for model in self.store.models() if model.name not in self.internal_names]

    def model(self, name) -> object:
        if not isinstance(name, str) or not name:
            raise ApiError(400, "model is required")
        try:
            model = self.store.get(name)
        except ModelNotFound as error:
            raise ApiError(404, f"model '{name}' not found") from error
        except (GgufError, OSError) as error:
            raise ApiError(500, f"couldn't read {name}: {error}") from error
        return model

    def runnable_model(self, name):
        model = self.model(name)
        if not model.runnable:
            raise ApiError(501, model.unrunnable_reason)
        return model

    def keep_alive(self, request: dict) -> float | None:
        try:
            return parse_keep_alive(request.get("keep_alive"), self.config.default_keep_alive)
        except ValueError as error:
            raise ApiError(400, str(error)) from error

    def with_runner(self, model, options: dict, keep_alive, work):
        cpu_only = options.get("num_gpu") == 0
        request_id = uuid.uuid4().hex
        if not cpu_only:
            self.trainer.busy(request_id)
        try:
            try:
                runner, load_seconds = self.pool.acquire(model, options.get("num_ctx"), cpu_only)
            except LoadError as error:
                raise ApiError(500, str(error)) from error
            try:
                return work(runner, load_seconds)
            finally:
                self.pool.release(runner, keep_alive)
        finally:
            if not cpu_only:
                self.trainer.done(request_id)


def post_json(url: str, body: dict, timeout: int = UPSTREAM_TIMEOUT_SECONDS) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise ApiError(error.code if error.code < 500 else 500, upstream_message(error)) from error
    except (urllib.error.URLError, OSError) as error:
        raise ApiError(500, f"the model runner stopped answering: {error}") from error


def upstream_message(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read() or b"{}")
    except ValueError:
        return f"model runner error {error.code}"
    detail = payload.get("error")
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail or payload)


def stream_events(url: str, body: dict):
    request = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    try:
        response = urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as error:
        raise ApiError(error.code if error.code < 500 else 500, upstream_message(error)) from error
    except (urllib.error.URLError, OSError) as error:
        raise ApiError(500, f"the model runner stopped answering: {error}") from error
    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                return
            event = json.loads(data)
            if event.get("error"):
                detail = event["error"]
                raise ApiError(500, str(detail.get("message") if isinstance(detail, dict) else detail))
            yield event


def ps_entry(runner: Runner) -> dict:
    model = runner.plan.model
    measured = runner.vram_mb()
    size = runner.estimate_mb * gpu.MIB
    expires = (datetime.fromtimestamp(runner.expires_wall, tz=timezone.utc) if runner.expires_wall else None)
    return {"name": model.name, "model": model.name, "size": max(size, measured * gpu.MIB), "digest": model.digest, "details": model.details,
            "expires_at": iso_time(expires) if expires else FAR_FUTURE, "size_vram": measured * gpu.MIB if measured else size,
            "context_length": runner.plan.ctx, "speculative": runner.plan.speculative.draft or runner.plan.speculative.spec_type or ""}


def hostname_of(authority: str) -> str:
    try:
        return (urlsplit(f"//{authority}").hostname or "").lower()
    except ValueError:
        return ""


def origin_hostname(origin: str) -> str:
    try:
        return (urlsplit(origin).hostname or "").lower()
    except ValueError:
        return ""


def listens_on_loopback(listen: str) -> bool:
    hostname = hostname_of(listen)
    return hostname in LOCAL_HOSTNAMES or hostname.startswith("127.")


def lowered(names: list) -> set[str]:
    return {str(name).lower() for name in names}


def refusal_for(config: Config, host: str | None, origin: str | None) -> str:
    if origin is not None:
        hostname = origin_hostname(origin)
        if hostname not in LOCAL_HOSTNAMES and hostname not in lowered(config.allowed_origins):
            return f"requests from the web page at {origin} are not allowed (allowed_origins)"
    if host is not None and listens_on_loopback(config.listen):
        hostname = hostname_of(host)
        if hostname not in LOCAL_HOSTNAMES and hostname not in lowered(config.allowed_hosts):
            return f"requests for host {host} are not allowed (allowed_hosts)"
    return ""


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = f"arty-engine/{VERSION}"

        def log_message(self, format, *args):
            log.debug("%s", format % args)

        def _send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Connection", "close")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _start_stream(self) -> None:
            self.send_response(200)
            self.send_header("Connection", "close")
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._streaming = True

        def _stream_line(self, payload: dict) -> None:
            data = json.dumps(payload).encode() + b"\n"
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        def _end_stream(self) -> None:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

        def _body(self) -> dict:
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError as error:
                raise ApiError(400, "Content-Length must be a number") from error
            if length < 0:
                raise ApiError(400, "Content-Length can't be negative")
            if length > MAX_BODY_BYTES:
                raise ApiError(413, "request body too large")
            raw = self.rfile.read(length) if length else b""
            if not raw:
                return {}
            try:
                body = json.loads(raw)
            except RecursionError as error:
                raise ApiError(400, "invalid JSON: nested too deeply") from error
            except ValueError as error:
                raise ApiError(400, f"invalid JSON: {error}") from error
            if not isinstance(body, dict):
                raise ApiError(400, "the request body must be a JSON object")
            return body

        def _dispatch(self, routes: dict) -> None:
            self._streaming = False
            self.close_connection = True
            path = urlsplit(self.path).path.rstrip("/") or "/"
            handler = routes.get(path)
            try:
                refusal = refusal_for(engine.config, self.headers.get("Host"), self.headers.get("Origin"))
                if refusal:
                    log.warning("refused %s %s: %s", self.command, path, refusal)
                    raise ApiError(403, refusal)
                if handler is None:
                    raise ApiError(404, f"no route {self.command} {path}")
                handler()
            except ApiError as error:
                self._fail(error.status, error.message)
            except translate.RequestError as error:
                self._fail(400, str(error))
            except (BrokenPipeError, ConnectionResetError):
                log.info("client went away during %s %s", self.command, path)
            except Exception as error:
                log.exception("%s %s failed", self.command, path)
                self._fail(500, f"engine error: {error}")

        def _fail(self, status: int, message: str) -> None:
            if self._streaming:
                try:
                    self._stream_line({"error": message})
                    self._end_stream()
                except OSError:
                    pass
                return
            try:
                self._send_json(status, {"error": message})
            except (BrokenPipeError, ConnectionResetError):
                log.info("client went away before the %d answer to %s %s", status, self.command, self.path)

        def do_GET(self):
            self._dispatch({"/": self.root, "/api/tags": self.tags, "/api/ps": self.ps, "/api/version": self.version})

        def do_HEAD(self):
            self.close_connection = True
            self.send_response(200)
            self.send_header("Connection", "close")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_POST(self):
            self._dispatch({"/api/chat": self.chat, "/api/generate": self.generate, "/api/show": self.show, "/api/embed": self.embed,
                            "/api/embeddings": self.embeddings, "/api/pull": self.pull, "/api/tags": self.tags, "/api/ps": self.ps,
                            "/api/create": self.not_in_engine, "/api/copy": self.not_in_engine, "/api/push": self.not_in_engine})

        def do_DELETE(self):
            self._dispatch({"/api/delete": self.delete})

        def root(self):
            data = b"Arty engine is running"
            self.send_response(200)
            self.send_header("Connection", "close")
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def version(self):
            self._send_json(200, {"version": VERSION})

        def tags(self):
            self._send_json(200, {"models": [model.describe() for model in engine.listed_models()]})

        def ps(self):
            self._send_json(200, {"models": [ps_entry(runner) for runner in engine.pool.loaded()]})

        def show(self):
            request = self._body()
            model = engine.model(request.get("model") or request.get("name"))
            params = "\n".join(f"{key:<30} {json.dumps(value) if not isinstance(value, str) else value}"
                               for key, value in model.params.items())
            self._send_json(200, {"license": model.license, "modelfile": f"FROM {model.model_path}\n", "parameters": params,
                                  "template": model.stored_template or model.gguf.chat_template, "system": model.system,
                                  "details": model.details, "model_info": engine.store.gguf_metadata(model),
                                  "capabilities": model.capabilities, "modified_at": iso_time(model.modified_at),
                                  "runnable": model.runnable})

        def delete(self):
            request = self._body()
            name = request.get("model") or request.get("name")
            if not isinstance(name, str) or not name:
                raise ApiError(400, "model is required")
            try:
                display = parse_name(name).display
            except ModelNotFound as error:
                raise ApiError(400, str(error)) from error
            if display in engine.internal_names:
                raise ApiError(409, f"{display} is kept for the engine's own use (internal_models); take it out of the config first")
            if not engine.pull_lock.acquire(blocking=False):
                raise ApiError(409, "a pull is running; delete when it is done")
            try:
                engine.store.owned_manifest(name)
                engine.pool.unload(display)
                removed = engine.store.remove_owned(name)
            except ModelNotFound as error:
                raise ApiError(404, f"model '{name}' not found") from error
            except NotOwned as error:
                raise ApiError(405, str(error)) from error
            finally:
                engine.pull_lock.release()
            log.info("deleted %s and %d blobs only it used", display, len(removed))
            self.send_response(200)
            self.send_header("Connection", "close")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def not_in_engine(self):
            raise ApiError(501, f"the engine doesn't do {urlsplit(self.path).path}: it pulls models from Hugging Face and "
                                "deletes its own, and does not create, copy or push models")

        def _load_or_unload(self, request: dict, model, keep_alive, generate: bool) -> bool:
            if request.get("messages") or request.get("prompt"):
                return False
            reply = {"response": ""} if generate else {"message": {"role": "assistant", "content": ""}}
            if keep_alive == 0:
                engine.pool.unload(model.name)
                self._send_json(200, {"model": model.name, "created_at": translate.now_iso(), **reply, "done_reason": "unload", "done": True})
                return True
            started = time.monotonic()
            engine.with_runner(model, translate.merged_options(model.params, request.get("options")), keep_alive, lambda runner, load_seconds: None)
            self._send_json(200, {"model": model.name, "created_at": translate.now_iso(), **reply, "done_reason": "load", "done": True,
                                  "load_duration": int((time.monotonic() - started) * 1e9)})
            return True

        def chat(self):
            request = self._body()
            model = engine.model(request.get("model"))
            keep_alive = engine.keep_alive(request)
            if request.get("_debug_render_only"):
                self._render_only(request, model, keep_alive)
                return
            if not request.get("messages") and self._load_or_unload_checked(request, model, keep_alive, generate=False):
                return
            self._run_chat(request, model, keep_alive, generate=False)

        def _render_only(self, request: dict, model, keep_alive):
            self._check_chat_model(model)
            body = translate.render_body(request, model)
            options = translate.merged_options(model.params, request.get("options"))
            rendered = engine.with_runner(model, options, keep_alive,
                                          lambda runner, load_seconds: post_json(f"{runner.base_url}/apply-template", body))
            info = {"rendered_template": rendered.get("prompt", "")}
            images = translate.image_count(request.get("messages"))
            if images:
                info["image_count"] = images
            self._send_json(200, {"model": model.name, "created_at": translate.now_iso(), "message": {"role": "assistant", "content": ""},
                                  "done": True, "_debug_info": info})

        @staticmethod
        def _check_chat_model(model) -> None:
            if not model.runnable:
                raise ApiError(501, model.unrunnable_reason)
            if model.is_embedding:
                raise ApiError(400, f'"{model.name}" does not support chat')

        def _load_or_unload_checked(self, request, model, keep_alive, generate: bool) -> bool:
            if not model.runnable:
                raise ApiError(501, model.unrunnable_reason)
            return self._load_or_unload(request, model, keep_alive, generate)

        def generate(self):
            request = self._body()
            model = engine.model(request.get("model"))
            keep_alive = engine.keep_alive(request)
            if not request.get("prompt") and self._load_or_unload_checked(request, model, keep_alive, generate=True):
                return
            if request.get("suffix"):
                raise ApiError(501, "suffix (fill in the middle) isn't supported by the engine yet")
            if request.get("raw"):
                self._run_raw(request, model, keep_alive)
                return
            chat_request = {**request, "messages": translate.generate_messages(request, model)}
            self._run_chat(chat_request, model, keep_alive, generate=True)

        def _run_raw(self, request: dict, model, keep_alive):
            self._check_chat_model(model)
            if request.get("images") or request.get("format"):
                raise ApiError(501, "raw prompts with images or a format aren't supported by the engine yet")
            stream = request.get("stream", True) is not False
            body = translate.raw_completion_body(request, model, stream)
            options = translate.merged_options(model.params, request.get("options"))
            started = time.monotonic()

            def run(runner: Runner, load_seconds: float):
                url = f"{runner.base_url}/completion"
                if not stream:
                    self._send_json(200, translate.raw_reply(model.name, post_json(url, body), load_seconds, started))
                    return
                events = stream_events(url, body)
                first = next(events, None)
                self._start_stream()
                finished = False
                for event in itertools.chain([first] if first else [], events):
                    for chunk in translate.raw_stream_chunks(model.name, event, load_seconds, started):
                        finished = finished or chunk["done"]
                        self._stream_line(chunk)
                if not finished:
                    self._stream_line(translate.raw_reply(model.name, {}, load_seconds, started))
                self._end_stream()

            engine.with_runner(model, options, keep_alive, run)

        def _run_chat(self, request: dict, model, keep_alive, generate: bool):
            self._check_chat_model(model)
            stream = request.get("stream", True) is not False
            body = translate.chat_body(request, model, stream)
            options = translate.merged_options(model.params, request.get("options"))
            started = time.monotonic()
            shape = translate.generate_reply_from_chat if generate else (lambda reply: reply)

            def run(runner: Runner, load_seconds: float):
                url = f"{runner.base_url}/v1/chat/completions"
                if not stream:
                    completion = post_json(url, body)
                    self._send_json(200, shape(translate.chat_reply(model.name, completion, load_seconds, started)))
                    return
                translator = translate.StreamTranslator(model.name, load_seconds, started)
                events = stream_events(url, body)
                first = next(events, None)
                self._start_stream()
                for event in ([first] if first else []):
                    for chunk in translator.feed(event):
                        self._stream_line(shape(chunk))
                for event in events:
                    for chunk in translator.feed(event):
                        self._stream_line(shape(chunk))
                for chunk in translator.finish_chunks():
                    self._stream_line(shape(chunk))
                self._end_stream()

            engine.with_runner(model, options, keep_alive, run)

        def _embed_vectors(self, request: dict, inputs: list[str]):
            model = engine.runnable_model(request.get("model"))
            if not model.is_embedding:
                raise ApiError(400, f'"{model.name}" does not support embeddings')
            keep_alive = engine.keep_alive(request)
            result = {}

            def run(runner: Runner, load_seconds: float):
                result["reply"] = post_json(f"{runner.base_url}/v1/embeddings", {"input": inputs, "model": model.name})
                result["load"] = load_seconds

            started = time.monotonic()
            engine.with_runner(model, translate.merged_options(model.params, request.get("options")), keep_alive, run)
            data = sorted(result["reply"].get("data") or [], key=lambda item: item.get("index", 0))
            return [item["embedding"] for item in data], result["reply"].get("usage") or {}, time.monotonic() - started, model, result["load"]

        def embed(self):
            request = self._body()
            inputs = translate.embed_inputs(request)
            vectors, usage, seconds, model, load_seconds = self._embed_vectors(request, inputs)
            dimensions = translate.embed_dimensions(request)
            if dimensions:
                vectors = [vector[:dimensions] for vector in vectors]
            self._send_json(200, {"model": model.name, "embeddings": vectors, "total_duration": int(seconds * 1e9),
                                  "load_duration": int(load_seconds * 1e9), "prompt_eval_count": int(usage.get("prompt_tokens", 0))})

        def embeddings(self):
            request = self._body()
            prompt = request.get("prompt", "")
            if not isinstance(prompt, str):
                raise ApiError(400, "prompt must be a string")
            vectors, _, _, _, _ = self._embed_vectors(request, [prompt])
            self._send_json(200, {"embedding": vectors[0] if vectors else []})

        def pull(self):
            request = self._body()
            name = request.get("model") or request.get("name")
            if not isinstance(name, str) or not name:
                raise ApiError(400, "model is required")
            stream = request.get("stream", True) is not False
            if not engine.pull_lock.acquire(blocking=False):
                raise ApiError(409, "another pull is running; the engine pulls one model at a time to spare the link")
            try:
                if stream:
                    self._start_stream()
                    engine.puller.pull(name, self._stream_line)
                    self._end_stream()
                else:
                    engine.puller.pull(name, lambda update: None)
                    self._send_json(200, {"status": "success"})
            except PullError as error:
                raise ApiError(error.status, str(error)) from error
            except ModelNotFound as error:
                raise ApiError(400, str(error)) from error
            finally:
                engine.pull_lock.release()

    return Handler


def serve(config: Config) -> None:
    engine = Engine(config)
    engine.pool.start_reaper()
    host, port = config.listen.rsplit(":", 1)
    server = ThreadingHTTPServer((host, int(port)), make_handler(engine))
    server.daemon_threads = True
    signal.signal(signal.SIGTERM, exit_on_terminate)
    log.info("arty-engine %s on %s, models from %s, model runner %s, VRAM budget %d MiB", VERSION, config.listen,
             ", ".join(str(root) for root in config.roots), config.runner_binary, config.vram_budget_mb)
    try:
        server.serve_forever()
    finally:
        log.info("arty-engine stopping; unloading %d models", len(engine.pool.loaded()))
        engine.pool.shutdown()


def exit_on_terminate(*_) -> None:
    raise SystemExit(0)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="ArtyEngine, Arty's own model engine")
    parser.add_argument("--config", default=None, help="JSON settings file (default: $ARTY_ENGINE_CONFIG or built-in defaults)")
    parser.add_argument("--listen", default=None, help="host:port to listen on")
    parser.add_argument("--verbose", action="store_true")
    arguments = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if arguments.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s",
                        stream=sys.stdout)
    config = load_config(arguments.config)
    if arguments.listen:
        config.listen = arguments.listen
    serve(config)
