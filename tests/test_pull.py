import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from arty_engine import pull
from arty_engine.store import ModelStore
from gguf_writer import write_gguf


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.slept = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_holds_the_average_to_the_cap():
    clock = FakeClock()
    limiter = pull.RateLimiter(4_000_000, clock=clock.time, sleep=clock.sleep)
    for _ in range(40):
        limiter.take(1_000_000)
    assert clock.now == pytest.approx(10.0)


def test_rate_limiter_off_when_zero():
    clock = FakeClock()
    limiter = pull.RateLimiter(0, clock=clock.time, sleep=clock.sleep)
    limiter.take(10**9)
    assert clock.slept == []


SIBLINGS = [{"rfilename": "Model-Q8_0.gguf"}, {"rfilename": "Model-Q4_K_M.gguf"}, {"rfilename": "mmproj-Model-f16.gguf"},
            {"rfilename": "Model-Q2_K-00001-of-00002.gguf"}, {"rfilename": "README.md"}]


def test_choose_files_picks_the_quant_and_projector():
    chosen, projector = pull.choose_hf_files(SIBLINGS, "Q8_0")
    assert chosen["rfilename"] == "Model-Q8_0.gguf" and projector["rfilename"] == "mmproj-Model-f16.gguf"
    assert pull.choose_hf_files(SIBLINGS, "")[0]["rfilename"] == "Model-Q4_K_M.gguf"


def test_choose_files_explains_what_is_there():
    with pytest.raises(pull.PullError, match="Model-Q8_0.gguf"):
        pull.choose_hf_files(SIBLINGS, "Q5_K_M")
    with pytest.raises(pull.PullError, match="split"):
        pull.choose_hf_files(SIBLINGS, "Q2_K")


@pytest.fixture
def fake_hf(tmp_path, monkeypatch):
    model = write_gguf(tmp_path / "Tiny-Q8_0.gguf", {"general.architecture": "qwen3", "tokenizer.chat_template": "{{ tools }}"}, {"w": (4, 4)})
    data = model.read_bytes()
    state = {"data": data, "sha": hashlib.sha256(data).hexdigest(), "ranges": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path.startswith("/api/"):
                payload = json.dumps({"siblings": [{"rfilename": "Tiny-Q8_0.gguf", "lfs": {"sha256": state["sha"], "size": len(state["data"])}}]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            start = int(self.headers.get("Range", "bytes=0-")[6:].rstrip("-") or 0)
            state["ranges"].append(start)
            body = state["data"][start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(pull, "HF_API", base + "/api/models/{repo}?blobs=true")
    monkeypatch.setattr(pull, "HF_FILE", base + "/{repo}/resolve/main/{file}")
    yield state
    server.shutdown()


def test_pull_from_hugging_face_lands_in_the_engine_store(tmp_path, fake_hf):
    updates = []
    pull.Puller(tmp_path / "own", 0).pull("hf.co/Someone/Tiny-GGUF:Q8_0", updates.append)
    assert updates[0] == {"status": "pulling manifest"} and updates[-1] == {"status": "success"}
    model = ModelStore([tmp_path / "own"]).get("hf.co/Someone/Tiny-GGUF:Q8_0")
    assert model.model_path.read_bytes() == fake_hf["data"] and model.capabilities == ["completion", "tools"]


def test_pull_resumes_a_partial_download(tmp_path, fake_hf):
    blob = tmp_path / "own" / "blobs" / f"sha256-{fake_hf['sha']}"
    blob.parent.mkdir(parents=True)
    blob.with_name(blob.name + "-partial").write_bytes(fake_hf["data"][:10])
    pull.Puller(tmp_path / "own", 0).pull("hf.co/Someone/Tiny-GGUF:Q8_0", lambda update: None)
    assert fake_hf["ranges"] == [10] and blob.read_bytes() == fake_hf["data"]


def test_corrupt_download_is_thrown_away(tmp_path, fake_hf):
    fake_hf["data"] = b"GGUF" + bytes(50)
    with pytest.raises(pull.PullError, match="sha256"):
        pull.Puller(tmp_path / "own", 0).pull("hf.co/Someone/Tiny-GGUF:Q8_0", lambda update: None)
    assert not list((tmp_path / "own" / "blobs").iterdir())


def test_unknown_hosts_are_refused(tmp_path):
    with pytest.raises(pull.PullError):
        pull.Puller(tmp_path, 0).pull("example.com/a/b:c", lambda update: None)


def serve(handler_class):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_class)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_the_hugging_face_token_does_not_follow_a_redirect(monkeypatch):
    seen = {}

    class Cdn(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen["cdn"] = self.headers.get("Authorization")
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    cdn = serve(Cdn)

    class Hub(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            seen["hub"] = self.headers.get("Authorization")
            self.send_response(302)
            self.send_header("Location", f"http://localhost:{cdn.server_address[1]}/blob")
            self.send_header("Content-Length", "0")
            self.end_headers()

    hub = serve(Hub)
    monkeypatch.setenv("HF_TOKEN", "test-token-value")
    monkeypatch.setattr(pull, "TOKEN_HOSTS", ("127.0.0.1",))
    try:
        with pull._open(f"http://127.0.0.1:{hub.server_address[1]}/file") as response:
            assert response.read() == b"ok"
    finally:
        hub.shutdown()
        cdn.shutdown()
    assert seen == {"hub": "Bearer test-token-value", "cdn": None}


@pytest.mark.parametrize("digest", ["../../victim.txt", "sha256:../../victim", "sha256:" + "A" * 64, "sha256:" + "0" * 63])
def test_a_registry_digest_that_is_not_a_sha256_is_refused(tmp_path, monkeypatch, digest):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    manifest = {"schemaVersion": 2, "config": {}, "layers": [{"mediaType": "application/vnd.ollama.image.template", "digest": digest, "size": 8}]}

    class Registry(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            data = json.dumps(manifest).encode() if "/manifests/" in self.path else b"precious"
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    registry = serve(Registry)
    monkeypatch.setattr(pull, "OLLAMA_REGISTRY", f"http://127.0.0.1:{registry.server_address[1]}" + "/v2/{namespace}/{model}/{kind}/{reference}")
    try:
        with pytest.raises(pull.PullError, match="digest"):
            pull.Puller(tmp_path / "own" / "store", 0).pull("evil:1", lambda update: None)
    finally:
        registry.shutdown()
    assert victim.read_text() == "precious"
    assert not (tmp_path / "own" / "store" / "manifests-v2").exists()
