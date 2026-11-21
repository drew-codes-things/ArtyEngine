import json
import logging
import os
import re
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.parse import urlsplit

log = logging.getLogger("arty-engine")

REMOTE_PREFIX = "remote/"
REMOTE_NAME = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
SLUG_UNSAFE = re.compile(r"[^a-z0-9._-]+")
OPTIONAL_CAPABILITIES = ("tools", "vision", "thinking")
DEFAULT_CAPABILITIES = ("tools",)
REMOTE_SETTINGS = ("url", "key_env", "key_file", "ca_file", "models", "capabilities", "timeout_seconds")
KEY_IN_CONFIG = ("key", "api_key", "token", "secret")
DEFAULT_TIMEOUT_SECONDS = 300
LONGEST_TIMEOUT_SECONDS = 3600
LIST_TIMEOUT_SECONDS = 10
LIST_CACHE_SECONDS = 60
LIST_FAILURE_CACHE_SECONDS = 15
STANDARD_SAMPLING = ("temperature", "top_p", "presence_penalty", "frequency_penalty", "seed", "stop", "max_tokens")
OPENAI_CHAT_FIELDS = ("messages", "stream", "stream_options", "tools", "response_format", *STANDARD_SAMPLING)
LOCAL_RUNNER_ONLY = ("chat_template_kwargs", "reasoning_effort")
OTHERS_CAN_READ = 0o077
MESSAGE_LIMIT = 300
KEY_HIDDEN = "[key hidden]"
GATEWAY_FAILED = 502
GATEWAY_TIMEOUT = 504
TOO_MANY_REQUESTS = 429
NOT_FOUND = 404
SETUP_FAILED = 500
REMOTE_FAMILY = "remote"
WEB_SCHEMES = ("http", "https")


class NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def keyed_opener(tls: ssl.SSLContext | None):
    return urllib.request.build_opener(NoRedirects, urllib.request.HTTPSHandler(context=tls)).open


class RemoteError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def is_remote_name(name) -> bool:
    return isinstance(name, str) and name.startswith(REMOTE_PREFIX)


def is_remote(model) -> bool:
    return isinstance(model, RemoteModel)


def slug_of(upstream_id: str) -> str:
    return SLUG_UNSAFE.sub("-", upstream_id.lower()).strip("-.")


def check_remote_models(remotes) -> None:
    if not isinstance(remotes, dict):
        raise ValueError("remote_models must be an object of remote name to settings")
    for name, settings in remotes.items():
        check_remote(name, settings)


def check_remote(name, settings) -> None:
    if not isinstance(name, str) or not REMOTE_NAME.fullmatch(name):
        raise ValueError(f"remote_models: {name!r} must be lower case letters, digits, - or _ (at most 64)")
    if not isinstance(settings, dict):
        raise ValueError(f"remote_models.{name} must be an object with url and key_file or key_env")
    keys_written_in = sorted(set(settings) & set(KEY_IN_CONFIG))
    if keys_written_in:
        raise ValueError(f"remote_models.{name}: never put the key in the config ({', '.join(keys_written_in)}); "
                         "name a file with key_file or a variable with key_env")
    unknown = sorted(set(settings) - set(REMOTE_SETTINGS))
    if unknown:
        raise ValueError(f"remote_models.{name}: unknown settings {', '.join(unknown)}")
    check_address(name, settings.get("url"))
    check_key_source(name, settings)
    models = settings.get("models", [])
    if not isinstance(models, list) or not all(isinstance(model, str) and slug_of(model) for model in models):
        raise ValueError(f"remote_models.{name}.models must be a list of the server's model names")
    capabilities = settings.get("capabilities", list(DEFAULT_CAPABILITIES))
    if not isinstance(capabilities, list) or not set(capabilities) <= set(OPTIONAL_CAPABILITIES):
        raise ValueError(f"remote_models.{name}.capabilities must be a list of {', '.join(OPTIONAL_CAPABILITIES)}")
    timeout = settings.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= LONGEST_TIMEOUT_SECONDS:
        raise ValueError(f"remote_models.{name}.timeout_seconds must be a number of seconds from 1 to {LONGEST_TIMEOUT_SECONDS}")
    ca_file = settings.get("ca_file", "")
    if not isinstance(ca_file, str):
        raise ValueError(f"remote_models.{name}.ca_file must be the path of a PEM certificate file")


def check_address(name: str, url) -> None:
    if not isinstance(url, str) or not url:
        raise ValueError(f"remote_models.{name}.url is required: the server's OpenAI-compatible address, ending in /v1")
    try:
        parts = urlsplit(url)
    except ValueError as error:
        raise ValueError(f"remote_models.{name}.url is not an address: {error}") from error
    if parts.scheme not in WEB_SCHEMES or not parts.hostname:
        raise ValueError(f"remote_models.{name}.url must start with http:// or https:// and name a host")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError(f"remote_models.{name}.url must be a plain address with no user, password, query or fragment")


def check_key_source(name: str, settings: dict) -> None:
    key_env, key_file = settings.get("key_env", ""), settings.get("key_file", "")
    if not isinstance(key_env, str) or not isinstance(key_file, str):
        raise ValueError(f"remote_models.{name}: key_env and key_file must be text")
    if key_env and key_file:
        raise ValueError(f"remote_models.{name}: use key_env or key_file, not both")


@dataclass
class RemoteModel:
    name: str
    remote: str
    upstream: str
    capabilities: list
    params: dict = field(default_factory=dict)
    system: str = ""
    runnable: bool = True
    unrunnable_reason: str = ""
    is_embedding: bool = False

    @property
    def details(self) -> dict:
        return {"format": REMOTE_FAMILY, "family": REMOTE_FAMILY, "families": [REMOTE_FAMILY], "parameter_size": "",
                "quantization_level": "", "remote": self.remote}

    def describe(self) -> dict:
        return {"name": self.name, "model": self.name, "modified_at": now_iso(), "size": 0, "digest": "",
                "details": self.details, "capabilities": self.capabilities, "remote": True}

    def show(self) -> dict:
        return {"license": "", "modelfile": "", "parameters": "", "template": "", "system": "", "details": self.details,
                "model_info": {}, "capabilities": self.capabilities, "modified_at": now_iso(), "runnable": True,
                "remote": {"name": self.remote, "model": self.upstream}}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def server_sent_events(lines):
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[len("data:"):].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def upstream_detail(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read() or b"{}")
    except (ValueError, OSError):
        return ""
    detail = payload.get("error") if isinstance(payload, dict) else payload
    if isinstance(detail, dict):
        detail = detail.get("message") or detail
    return str(detail or "")[:MESSAGE_LIMIT]


class Remote:
    def __init__(self, name: str, settings: dict, opener=None):
        self.name = name
        self.url = settings["url"].rstrip("/")
        self.key_env = settings.get("key_env", "")
        self.key_file = settings.get("key_file", "")
        self.models_set = list(settings.get("models", []))
        self.capabilities = ["completion", *settings.get("capabilities", DEFAULT_CAPABILITIES)]
        self.timeout = settings.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        tls = ssl.create_default_context(cafile=settings["ca_file"]) if settings.get("ca_file") else None
        self.opener = opener or keyed_opener(tls)
        self.list_lock = threading.Lock()
        self.listed: dict[str, str] = {}
        self.list_failure: RemoteError | None = None
        self.listed_at = 0.0

    @property
    def label(self) -> str:
        return f"remote server '{self.name}'"

    @property
    def key_source(self) -> str:
        if self.key_file:
            return f"the key in {self.key_file}"
        return f"the key in ${self.key_env}" if self.key_env else "the server (no key is set for it)"

    def key(self) -> str:
        if self.key_file:
            return self.key_from_file()
        if not self.key_env:
            return ""
        key = os.environ.get(self.key_env, "").strip()
        if not key:
            raise RemoteError(SETUP_FAILED, f"{self.label}: ${self.key_env} is not set in the engine's environment")
        return key

    def key_from_file(self) -> str:
        try:
            mode = os.stat(self.key_file).st_mode
            if mode & OTHERS_CAN_READ:
                raise RemoteError(SETUP_FAILED, f"{self.label}: other users can read {self.key_file}; "
                                                f"run chmod 600 {self.key_file} and try again")
            with open(self.key_file) as handle:
                key = handle.read().strip()
        except OSError as error:
            raise RemoteError(SETUP_FAILED, f"{self.label}: couldn't read its key file {self.key_file}: "
                                            f"{error.strerror or error}") from None
        if not key:
            raise RemoteError(SETUP_FAILED, f"{self.label}: its key file {self.key_file} is empty")
        return key

    def hidden(self, text: str, key: str) -> str:
        return text.replace(key, KEY_HIDDEN) if key else text

    def open(self, path: str, body: dict | None, timeout: float):
        key = self.key()
        headers = {"Accept": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        request = urllib.request.Request(f"{self.url}{path}", data=data, headers=headers,
                                         method="POST" if data is not None else "GET")
        try:
            return self.opener(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            raise self.refusal(error.code, upstream_detail(error), error.headers, key) from None
        except (TimeoutError, socket.timeout):
            raise RemoteError(GATEWAY_TIMEOUT, f"{self.label} did not answer within {timeout:g} s") from None
        except (urllib.error.URLError, OSError) as error:
            reason = getattr(error, "reason", error)
            if isinstance(reason, (TimeoutError, socket.timeout)):
                raise RemoteError(GATEWAY_TIMEOUT, f"{self.label} did not answer within {timeout:g} s") from None
            raise RemoteError(GATEWAY_FAILED, self.hidden(f"{self.label} isn't reachable at {self.url}: {reason}", key)) from None

    def refusal(self, status: int, detail: str, headers, key: str) -> RemoteError:
        detail = self.hidden(detail, key)
        said = f": {detail}" if detail else ""
        if status == 401:
            return RemoteError(GATEWAY_FAILED, f"{self.label} refused the key (HTTP 401): check {self.key_source}{said}")
        if status == 403:
            return RemoteError(GATEWAY_FAILED, f"{self.label} refused access (HTTP 403): the key may not be allowed to use "
                                               f"this model{said}")
        if status == TOO_MANY_REQUESTS:
            wait = (headers.get("Retry-After") if headers else None) or ""
            later = f"try again in {wait} s" if wait.isdigit() else "try again later"
            return RemoteError(TOO_MANY_REQUESTS, f"{self.label} is limiting requests (HTTP 429), {later}{said}")
        if 300 <= status < 400:
            return RemoteError(GATEWAY_FAILED, f"{self.label} answered with a redirect (HTTP {status}), which is not followed so "
                                               "the key stays with that server; set url to the address it moved to")
        if status == NOT_FOUND:
            return RemoteError(NOT_FOUND, f"{self.label} has no such model or address (HTTP 404){said}")
        if status < 500:
            return RemoteError(GATEWAY_FAILED, f"{self.label} refused the request (HTTP {status}){said}")
        return RemoteError(GATEWAY_FAILED, f"{self.label} failed (HTTP {status}){said}")

    def json_from(self, path: str, body: dict | None, timeout: float) -> dict:
        with self.open(path, body, timeout) as response:
            raw = self.read_within_time(lambda: response.read())
        try:
            answer = json.loads(raw or b"{}")
        except ValueError:
            raise RemoteError(GATEWAY_FAILED, f"{self.label} sent an answer that isn't JSON") from None
        if not isinstance(answer, dict):
            raise RemoteError(GATEWAY_FAILED, f"{self.label} sent an answer that isn't a JSON object")
        return answer

    def read_within_time(self, read):
        try:
            return read()
        except (TimeoutError, socket.timeout):
            raise RemoteError(GATEWAY_TIMEOUT, f"{self.label} stopped sending for {self.timeout:g} s") from None
        except OSError as error:
            raise RemoteError(GATEWAY_FAILED, f"{self.label} broke off its answer: {error}") from None

    def upstream_ids(self) -> list[str]:
        if self.models_set:
            return self.models_set
        with self.list_lock:
            age = time.monotonic() - self.listed_at
            if self.listed_at and self.list_failure is None and age < LIST_CACHE_SECONDS:
                return list(self.listed.values())
            if self.listed_at and self.list_failure is not None and age < LIST_FAILURE_CACHE_SECONDS:
                raise self.list_failure
            try:
                answer = self.json_from("/models", None, LIST_TIMEOUT_SECONDS)
            except RemoteError as error:
                self.list_failure, self.listed_at = error, time.monotonic()
                raise
            data = answer.get("data")
            ids = [item.get("id") for item in data if isinstance(item, dict)] if isinstance(data, list) else []
            self.listed = {slug_of(name): name for name in ids if isinstance(name, str) and slug_of(name)}
            self.list_failure, self.listed_at = None, time.monotonic()
            return list(self.listed.values())

    def models(self) -> list[RemoteModel]:
        by_slug: dict[str, RemoteModel] = {}
        for upstream in self.upstream_ids():
            slug = slug_of(upstream)
            if slug in by_slug:
                log.warning("%s lists %s and %s under the same name %s; using the first", self.label,
                            by_slug[slug].upstream, upstream, slug)
                continue
            by_slug[slug] = RemoteModel(name=f"{REMOTE_PREFIX}{self.name}/{slug}", remote=self.name, upstream=upstream,
                                        capabilities=list(self.capabilities))
        return list(by_slug.values())

    def model(self, slug: str) -> RemoteModel:
        for model in self.models():
            if model.name == f"{REMOTE_PREFIX}{self.name}/{slug}":
                return model
        raise RemoteError(NOT_FOUND, f"model '{REMOTE_PREFIX}{self.name}/{slug}' not found: {self.label} doesn't list it")

    def chat_body(self, model: RemoteModel, body: dict) -> dict:
        sent = set(OPENAI_CHAT_FIELDS) | (set(LOCAL_RUNNER_ONLY) if "thinking" in model.capabilities else set())
        return {**{key: value for key, value in body.items() if key in sent}, "model": model.upstream}

    def post_chat(self, model: RemoteModel, body: dict) -> dict:
        return self.json_from("/chat/completions", self.chat_body(model, body), self.timeout)

    def stream_chat(self, model: RemoteModel, body: dict):
        response = self.open("/chat/completions", self.chat_body(model, body), self.timeout)
        with response:
            events = server_sent_events(response)
            while True:
                try:
                    event = self.read_within_time(lambda: next(events, None))
                except ValueError:
                    raise RemoteError(GATEWAY_FAILED, f"{self.label} sent a stream event that isn't JSON") from None
                if event is None:
                    return
                if event.get("error"):
                    detail = event["error"]
                    message = detail.get("message") if isinstance(detail, dict) else detail
                    raise RemoteError(GATEWAY_FAILED, f"{self.label} failed during the answer: {str(message)[:MESSAGE_LIMIT]}")
                yield event


@dataclass(frozen=True)
class RemoteChat:
    remote: Remote
    model: RemoteModel

    def post(self, body: dict) -> dict:
        return self.remote.post_chat(self.model, body)

    def stream(self, body: dict):
        return self.remote.stream_chat(self.model, body)


class Remotes:
    def __init__(self, settings: dict, opener=None):
        self.by_name = {name: Remote(name, values, opener) for name, values in settings.items()}
        for remote in self.by_name.values():
            if remote.url.startswith("http://") and (remote.key_file or remote.key_env):
                log.warning("%s is reached over plain http, so its key crosses the network unencrypted", remote.label)

    def models(self) -> list[RemoteModel]:
        listed = []
        for remote in self.by_name.values():
            try:
                listed += remote.models()
            except RemoteError as error:
                log.warning("left %s's models out of the model list: %s", remote.label, error.message)
        return listed

    def model(self, name: str) -> RemoteModel:
        remote_name, _, slug = name[len(REMOTE_PREFIX):].partition("/")
        remote = self.by_name.get(remote_name)
        if remote is None or not slug:
            raise RemoteError(NOT_FOUND, f"model '{name}' not found: no remote server named '{remote_name}' in remote_models")
        return remote.model(slug.removesuffix(":latest"))

    def chat(self, model: RemoteModel) -> RemoteChat:
        return RemoteChat(self.by_name[model.remote], model)
