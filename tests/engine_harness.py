import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from arty_engine.config import Config
from arty_engine.gpu import GpuMemory
from arty_engine.runner import RunnerPool
from arty_engine.server import Engine, make_handler
from arty_engine.store import ModelStore
from fake_llama import FakeLauncher
from fake_store import build_store


class RunningEngine:
    def __init__(self, tmp_path, replies: dict, **settings):
        build_store(tmp_path / "ollama")
        self.config = Config(store=str(tmp_path / "own"), model_roots=[str(tmp_path / "ollama")], state_dir=str(tmp_path / "state"),
                             llama_server="/opt/llama/llama-server", load_overhead_mb=100, **settings)
        store = ModelStore(self.config.roots)
        self.launcher = FakeLauncher(replies)
        pool = RunnerPool(self.config, store, launcher=self.launcher, memory=lambda: GpuMemory(total_mb=12288, used_mb=0))
        self.engine = Engine(self.config, pool=pool, store=store)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.engine))
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"

    def upstream_requests(self) -> list[dict]:
        return [request for process in self.launcher.launched if process.server for request in process.server.requests]

    def close(self):
        self.http.shutdown()
        self.engine.pool.shutdown()


def http_json(url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        return json.loads(raw) if raw else None


def stream_chat(url, body, on_text, timeout=30):
    request = urllib.request.Request(f"{url}/api/chat", data=json.dumps({**body, "stream": True}).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    content, calls, last, lines = [], [], {}, []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for line in response:
            if not line.strip():
                continue
            last = json.loads(line)
            lines.append(last)
            if last.get("error"):
                raise RuntimeError(f"model error: {last['error']}")
            piece = last.get("message") or {}
            calls += piece.get("tool_calls") or []
            if piece.get("content"):
                content.append(piece["content"])
                on_text(piece["content"])
    return {**last, "message": {"role": "assistant", "content": "".join(content), "tool_calls": calls}}, lines


def error_of(call):
    try:
        call()
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read())
    raise AssertionError("expected an HTTP error")
