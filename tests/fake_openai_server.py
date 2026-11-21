import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GOOD_KEY = "sf_test_0123456789abcdef"
LISTED_MODELS = ["Qwen3-14B-Q4_K_M", "friend/Big-Model:Q8"]
LIMITED_MODEL = "limited"
FORBIDDEN_MODEL = "forbidden"
SLOW_MODEL = "slow"
BROKEN_MODEL = "broken"
ECHO_KEY_MODEL = "echo-key"
REDIRECT_MODEL = "moved"
SLOW_SECONDS = 3


class FakeOpenAiServer:
    def __init__(self, models=None):
        self.models = list(models if models is not None else LISTED_MODELS)
        self.requests: list[dict] = []
        self.listings = 0
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send(self, status, payload, content_type="application/json", headers=None):
                data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Connection", "close")
                for name, value in (headers or {}).items():
                    self.send_header(name, value)
                self.end_headers()
                self.wfile.write(data)

            def _key_is_good(self) -> bool:
                if self.headers.get("Authorization") == f"Bearer {GOOD_KEY}":
                    return True
                self._send(401, {"error": {"message": "invalid app key", "type": "invalid_request_error"}})
                return False

            def do_GET(self):
                server.listings += 1
                if not self._key_is_good():
                    return
                if self.path != "/v1/models":
                    self._send(404, {"error": {"message": "no route"}})
                    return
                self._send(200, {"object": "list", "data": [{"id": name, "object": "model"} for name in server.models]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                server.requests.append({"path": self.path, "body": body, "authorization": self.headers.get("Authorization")})
                if not self._key_is_good():
                    return
                if self.path != "/v1/chat/completions":
                    self._send(404, {"error": {"message": "no route"}})
                    return
                model = body.get("model")
                if model == LIMITED_MODEL:
                    self._send(429, {"error": {"message": "slow down"}}, headers={"Retry-After": "7"})
                elif model == FORBIDDEN_MODEL:
                    self._send(403, {"error": {"message": "this key may not use that model"}})
                elif model == BROKEN_MODEL:
                    self._send(500, {"error": {"message": "upstream crashed"}})
                elif model == ECHO_KEY_MODEL:
                    self._send(400, {"error": {"message": f"bad request from {self.headers.get('Authorization')}"}})
                elif model == REDIRECT_MODEL:
                    self._send(302, {}, headers={"Location": f"{server.url}/elsewhere/chat/completions"})
                elif model == SLOW_MODEL:
                    time.sleep(SLOW_SECONDS)
                    self._send(200, completion("late"))
                elif body.get("stream"):
                    self._send(200, streamed(body), "text/event-stream")
                else:
                    self._send(200, completion(reply_text(body), with_tools=bool(body.get("tools"))))

        self.http = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}/v1"
        threading.Thread(target=self.http.serve_forever, daemon=True).start()

    def close(self):
        self.http.shutdown()
        self.http.server_close()


def reply_text(body: dict) -> str:
    last = (body.get("messages") or [{}])[-1]
    return f"remote says: {last.get('content', '')}"


def completion(text: str, with_tools: bool = False) -> dict:
    message = {"role": "assistant", "content": text, "reasoning_content": "thought first"}
    if with_tools:
        message["tool_calls"] = [{"id": "call_a", "type": "function",
                                  "function": {"name": "get_time", "arguments": "{\"zone\": \"UTC\"}"}}]
    return {"id": "chatcmpl-1", "object": "chat.completion", "choices": [{"index": 0, "message": message,
                                                                          "finish_reason": "tool_calls" if with_tools else "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3}}


def streamed(body: dict) -> bytes:
    events = [{"choices": [{"delta": {"reasoning_content": "thinking it over"}}]},
              {"choices": [{"delta": {"content": "remote "}}]},
              {"choices": [{"delta": {"content": "stream"}}]}]
    if body.get("tools"):
        events.append({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call_b", "function": {"name": "get_time",
                                                                                                     "arguments": "{\"zone\""}}]}}]})
        events.append({"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": \"UTC\"}"}}]}}]})
    events.append({"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 2}})
    return ("\n\n".join(f"data: {json.dumps(event)}" for event in events) + "\n\ndata: [DONE]\n\n").encode()
