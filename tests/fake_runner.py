import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeRunnerServer:
    def __init__(self, port: int, replies: dict, loading: bool = False, api_key: str = ""):
        self.replies = replies
        self.loading = loading
        self.api_key = api_key
        self.requests: list[dict] = []
        self.refused: list[str] = []
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, payload, content_type="application/json"):
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _send_cut_stream(self, events: list[str]):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write("".join(f"{event}\n\n" for event in events).encode())
                self.wfile.flush()
                self.close_connection = True

            def _refused(self) -> bool:
                if not server.api_key or self.path == "/health" or self.headers.get("Authorization") == f"Bearer {server.api_key}":
                    return False
                server.refused.append(self.path)
                self._send(401, {"error": {"message": "Invalid API Key", "type": "authentication_error"}})
                return True

            def do_GET(self):
                if self._refused():
                    return
                if server.loading:
                    self._send(503, {"error": {"message": "Loading model"}})
                    return
                reply = server.replies.get(self.path) if self.path != "/health" else None
                if callable(reply):
                    reply = reply({})
                if reply is None:
                    self._send(200, {"status": "ok"})
                elif "text" in reply:
                    self._send(200, reply["text"].encode(), reply.get("content_type", "text/plain"))
                else:
                    self._send(200, reply["json"])

            def do_POST(self):
                if self._refused():
                    return
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                server.requests.append({"path": self.path, "body": body})
                reply = server.replies.get(self.path)
                if callable(reply):
                    reply = reply(body)
                if reply is None:
                    self._send(404, {"error": {"message": "no route"}})
                elif "sse" in reply:
                    self._send(200, ("\n\n".join(reply["sse"]) + "\n\n").encode(), "text/event-stream")
                elif "cut_sse" in reply:
                    self._send_cut_stream(reply["cut_sse"])
                elif "error" in reply:
                    self._send(reply.get("status", 400), {"error": {"message": reply["error"]}})
                else:
                    self._send(200, reply["json"])

        self.http = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.http.shutdown()
        self.http.server_close()


KEY_FILE_MODES: list[int] = []


def read_api_key(arguments: list[str]) -> str:
    if "--api-key-file" not in arguments:
        return ""
    path = arguments[arguments.index("--api-key-file") + 1]
    KEY_FILE_MODES.append(os.stat(path).st_mode & 0o777)
    with open(path) as key_file:
        return key_file.read().strip()


class FakeProcess:
    next_pid = 50000

    def __init__(self, arguments: list[str], replies: dict, fail: bool = False, loading: bool = False):
        self.arguments = arguments
        FakeProcess.next_pid += 1
        self.pid = FakeProcess.next_pid
        self.returncode = 1 if fail else None
        self.api_key = read_api_key(arguments)
        self.server = None if fail else FakeRunnerServer(int(arguments[arguments.index("--port") + 1]), replies, loading,
                                                         self.api_key)

    def poll(self):
        return self.returncode

    def terminate(self):
        if self.returncode is None:
            self.returncode = -15
            self.server.close()

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode


class FakeLauncher:
    def __init__(self, replies: dict | None = None, fail: bool = False, loading_when=None):
        self.replies = replies or {}
        self.fail = fail
        self.loading_when = loading_when or (lambda arguments: False)
        self.launched: list[FakeProcess] = []
        self.environments: list[dict] = []

    def __call__(self, arguments, **kwargs):
        process = FakeProcess(arguments, self.replies, self.fail, self.loading_when(arguments))
        self.launched.append(process)
        self.environments.append(kwargs.get("env") or {})
        return process
