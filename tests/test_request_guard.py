import http.client
import json
import socket

import pytest

from arty_engine.server import ApiError, make_handler
from engine_harness import RunningEngine

REPLIES = {"/v1/chat/completions": lambda body: {"json": {"choices": [{"finish_reason": "stop", "message": {"content": "hi"}}]}}}
LOAD = json.dumps({"model": "qwen3:14b", "keep_alive": -1})
FOREIGN = "attacker.example"
SOCKET_TIMEOUT_SECONDS = 5


def raw_request(running, method, path, body="", headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", running.http.server_address[1], timeout=SOCKET_TIMEOUT_SECONDS)
    headers = headers or {}
    connection.putrequest(method, path, skip_host=True)
    for key, value in headers.items():
        connection.putheader(key, value)
    data = body.encode()
    connection.putheader("Content-Length", str(len(data)))
    connection.endheaders()
    connection.send(data)
    response = connection.getresponse()
    return response.status, response.read()


@pytest.fixture
def running(tmp_path):
    engine = RunningEngine(tmp_path, REPLIES)
    yield engine
    engine.close()


@pytest.fixture
def opened(tmp_path):
    engine = RunningEngine(tmp_path, REPLIES, allowed_hosts=["Arty.lan"], allowed_origins=["chat.example.org"])
    yield engine
    engine.close()


def loaded(engine) -> list:
    return [runner.name for runner in engine.engine.pool.loaded()]


def test_a_foreign_origin_cannot_load_a_model(running):
    status, body = raw_request(running, "POST", "/api/generate", LOAD,
                               {"Host": "127.0.0.1:11438", "Origin": f"http://{FOREIGN}", "Content-Type": "text/plain"})
    assert status == 403 and FOREIGN in json.loads(body)["error"]
    assert loaded(running) == []


def test_a_foreign_host_cannot_read_the_models(running, caplog):
    status, _ = raw_request(running, "GET", "/api/tags", "", {"Host": f"{FOREIGN}:11438"})
    assert status == 403
    assert any("refused GET /api/tags" in record.getMessage() and FOREIGN in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("host", ["127.0.0.1:11438", "localhost:11438", "[::1]:11438", "localhost"])
def test_local_hosts_and_origins_are_served(running, host):
    status, _ = raw_request(running, "GET", "/api/tags", "", {"Host": host, "Origin": "http://localhost:3000"})
    assert status == 200


def test_a_request_without_host_or_origin_is_served(running):
    status, _ = raw_request(running, "GET", "/api/version", "", {})
    assert status == 200


def test_listed_hosts_and_origins_are_served(opened):
    status, _ = raw_request(opened, "POST", "/api/generate", LOAD,
                            {"Host": "arty.lan:11438", "Origin": "https://chat.example.org"})
    assert status == 200
    assert raw_request(opened, "GET", "/api/tags", "", {"Host": "arty.lan", "Origin": f"http://{FOREIGN}"})[0] == 403
    assert raw_request(opened, "GET", "/api/tags", "", {"Host": f"{FOREIGN}"})[0] == 403


def test_the_host_is_not_checked_when_the_engine_listens_beyond_loopback(tmp_path):
    engine = RunningEngine(tmp_path, REPLIES, listen="0.0.0.0:11438")
    try:
        assert raw_request(engine, "GET", "/api/tags", "", {"Host": "arty.lan:11438"})[0] == 200
        assert raw_request(engine, "GET", "/api/tags", "", {"Host": "arty.lan", "Origin": f"http://{FOREIGN}"})[0] == 403
    finally:
        engine.close()


def answer_to(running, length: str) -> bytes:
    probe = socket.create_connection(("127.0.0.1", running.http.server_address[1]), timeout=SOCKET_TIMEOUT_SECONDS)
    try:
        probe.sendall(f"POST /api/show HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {length}\r\n\r\n".encode())
        return probe.recv(200)
    finally:
        probe.close()


@pytest.mark.parametrize("length", ["-1", "lots"])
def test_a_bad_content_length_is_answered_with_400(running, length):
    assert answer_to(running, length).startswith(b"HTTP/1.1 400")


class ClosedConnection:
    def write(self, data):
        raise BrokenPipeError("client went away")

    def flush(self):
        raise BrokenPipeError("client went away")


def test_an_error_for_a_client_that_left_raises_nothing(running):
    handler_class = make_handler(running.engine)
    handler = handler_class.__new__(handler_class)
    handler.wfile = ClosedConnection()
    handler.request_version = "HTTP/1.1"
    handler.requestline = "POST /api/show HTTP/1.1"
    handler.command = "POST"
    handler.path = "/api/show"
    handler.client_address = ("127.0.0.1", 0)
    handler.headers = http.client.HTTPMessage()

    def refuse():
        raise ApiError(404, "no such model")

    handler._dispatch({"/api/show": refuse})
