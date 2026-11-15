import hashlib
import json
import socketserver
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from arty_engine import pull
from arty_engine.store import MANIFEST_MEDIA_TYPE, MODEL_LAYER, ModelStore, parse_name
from engine_harness import RunningEngine, error_of, http_json
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
    state = {"data": data, "sha": hashlib.sha256(data).hexdigest(), "ranges": [], "paths": [], "listed_size": None,
             "refuse_ranges": 0, "revision": "a" * 40, "more_files": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["paths"].append(self.path)
            if self.path.startswith("/api/"):
                listed = len(state["data"]) if state["listed_size"] is None else state["listed_size"]
                payload = json.dumps({"sha": state["revision"], "siblings": [
                    {"rfilename": "Tiny-Q8_0.gguf", "lfs": {"sha256": state["sha"], "size": listed}}, *state["more_files"]]}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            start = int(self.headers.get("Range", "bytes=0-")[6:].rstrip("-") or 0)
            state["ranges"].append(start)
            if start and state["refuse_ranges"]:
                state["refuse_ranges"] -= 1
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = state["data"][start:]
            self.send_response(206 if start else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setattr(pull, "HF_API", base + "/api/models/{repo}?blobs=true")
    monkeypatch.setattr(pull, "HF_FILE", base + "/{repo}/resolve/{revision}/{file}")
    yield state
    server.shutdown()


def test_pull_from_hugging_face_lands_in_the_engine_store(tmp_path, fake_hf):
    updates = []
    pull.Puller(tmp_path / "own", 0).pull("hf.co/Someone/Tiny-GGUF:Q8_0", updates.append)
    assert updates[0] == {"status": "pulling manifest"} and updates[-1] == {"status": "success"}
    model = ModelStore([tmp_path / "own"]).get("hf.co/Someone/Tiny-GGUF:Q8_0")
    assert model.model_path.read_bytes() == fake_hf["data"] and model.capabilities == ["completion", "tools"]
    manifest = json.loads((tmp_path / "own" / "catalog" / "hf.co" / "Someone" / "Tiny-GGUF" / "Q8_0.json").read_text())
    assert manifest["mediaType"] == MANIFEST_MEDIA_TYPE and [layer["mediaType"] for layer in manifest["layers"]] == [MODEL_LAYER]


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


@pytest.mark.parametrize("sha", ["../../victim.txt", "A" * 64, "0" * 63])
def test_a_hugging_face_sha256_that_is_not_one_is_refused(tmp_path, monkeypatch, sha):
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    info = {"siblings": [{"rfilename": "M-Q8_0.gguf", "lfs": {"sha256": sha, "size": 8}}]}
    hub = serve(hub_answering(200, info))
    point_at(monkeypatch, f"http://127.0.0.1:{hub.server_address[1]}")
    try:
        with pytest.raises(pull.PullError, match="digest"):
            pull.Puller(tmp_path / "own" / "store", 0).pull("hf.co/Someone/M-GGUF:Q8_0", lambda update: None)
    finally:
        hub.shutdown()
    assert victim.read_text() == "precious"
    assert not (tmp_path / "own" / "store" / "catalog").exists()


def hub_answering(status: int, payload) -> type:
    class Hub(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Hub


def point_at(monkeypatch, base: str) -> None:
    monkeypatch.setattr(pull, "HF_API", base + "/api/models/{repo}?blobs=true")
    monkeypatch.setattr(pull, "HF_FILE", base + "/{repo}/resolve/{revision}/{file}")


@pytest.fixture
def engine(tmp_path):
    running = RunningEngine(tmp_path, {})
    yield running
    running.close()


def pull_answer(engine, model: str):
    return error_of(lambda: http_json(f"{engine.url}/api/pull", {"model": model, "stream": False}))


def pull_from_hub(engine, monkeypatch, status: int, payload, model: str = "hf.co/Someone/M-GGUF:Q8_0"):
    hub = serve(hub_answering(status, payload))
    point_at(monkeypatch, f"http://127.0.0.1:{hub.server_address[1]}")
    try:
        return pull_answer(engine, model)
    finally:
        hub.shutdown()


BAD_DIGEST_INFO = {"siblings": [{"rfilename": "M-Q8_0.gguf", "lfs": {"sha256": "../../x", "size": 1}}]}


def test_a_bad_blob_digest_from_hugging_face_is_a_502_not_a_500(engine, monkeypatch, caplog):
    status, body = pull_from_hub(engine, monkeypatch, 200, BAD_DIGEST_INFO)
    assert status == 502
    assert "Hugging Face" in body["error"] and "digest" in body["error"] and "Traceback" not in body["error"]
    assert not [record for record in caplog.records if record.exc_info]


def test_a_bad_blob_digest_while_streaming_ends_the_stream_with_the_error(engine, monkeypatch):
    hub = serve(hub_answering(200, BAD_DIGEST_INFO))
    point_at(monkeypatch, f"http://127.0.0.1:{hub.server_address[1]}")
    import urllib.request
    request = urllib.request.Request(f"{engine.url}/api/pull", data=json.dumps({"model": "hf.co/Someone/M-GGUF:Q8_0"}).encode(),
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            lines = [json.loads(line) for line in response if line.strip()]
    finally:
        hub.shutdown()
    assert "digest" in lines[-1]["error"]


def test_a_repo_hugging_face_does_not_have_is_a_404(engine, monkeypatch):
    status, body = pull_from_hub(engine, monkeypatch, 404, {"error": "Repository not found"}, "hf.co/Someone/Nothere-GGUF:Q8_0")
    assert status == 404 and "Nothere" in body["error"]


@pytest.mark.parametrize("hub_status", [500, 503, 429])
def test_other_hugging_face_failures_are_a_502(engine, monkeypatch, hub_status):
    status, body = pull_from_hub(engine, monkeypatch, hub_status, {"error": "x"})
    assert status == 502 and str(hub_status) in body["error"]


def test_an_unreachable_hugging_face_is_a_502(engine, monkeypatch):
    point_at(monkeypatch, "http://127.0.0.1:9")
    status, body = pull_answer(engine, "hf.co/Someone/M-GGUF:Q8_0")
    assert status == 502 and "couldn't read" in body["error"]


def test_an_unknown_host_is_a_400(engine):
    status, body = pull_answer(engine, "example.invalid/a/b:c")
    assert status == 400 and "example.invalid" in body["error"]


@pytest.mark.parametrize("name", ["registry.example/library/qwen3:14b", "example.com/a/b:c"])
def test_names_from_other_hosts_are_refused_with_the_form_to_use(engine, name):
    status, body = pull_answer(engine, name)
    assert status == 400
    assert "Hugging Face only" in body["error"] and "hf.co/<owner>/<repo>:<quant>" in body["error"] and name in body["error"]
    assert not (engine.config.roots[0] / "catalog").exists()


@pytest.mark.parametrize("name", ["nothere:1", "someone/custom:1"])
def test_an_unknown_short_name_is_refused_with_the_form_to_use(engine, name):
    status, body = pull_answer(engine, name)
    assert status == 400
    assert "short name" in body["error"] and "hf.co/<owner>/<repo>:<quant>" in body["error"] and name in body["error"]
    assert not (engine.config.roots[0] / "catalog").exists()


def test_a_short_name_is_pulled_from_its_hugging_face_repo_and_keeps_its_name(tmp_path, fake_hf):
    updates = []
    puller = pull.Puller(tmp_path / "own", 0, short_names={"tiny:1": pull.ShortName("hf.co/Someone/Tiny-GGUF:Q8_0")})
    puller.pull("tiny:1", updates.append)
    assert updates[-1] == {"status": "success"}
    store = ModelStore([tmp_path / "own"])
    assert store.names() == ["tiny:1"]
    assert store.get("tiny:1").model_path.read_bytes() == fake_hf["data"]
    assert store.get("tiny:1").manifest["layers"][0]["from"] == "Tiny-Q8_0.gguf"


def test_an_installed_short_name_is_not_downloaded_again(tmp_path, fake_hf):
    puller = pull.Puller(tmp_path / "own", 0, short_names={"tiny:1": pull.ShortName("hf.co/Someone/Tiny-GGUF:Q8_0")})
    puller.pull("tiny:1", lambda update: None)
    manifest = (tmp_path / "own" / "catalog" / "local" / "library" / "tiny" / "1.json")
    before = manifest.read_bytes()
    fake_hf["ranges"].clear()
    updates = []
    puller.pull("tiny:1", updates.append)
    assert fake_hf["ranges"] == [] and manifest.read_bytes() == before and updates[-1] == {"status": "success"}


def test_the_shipped_short_names_point_at_hugging_face_and_cover_what_arty_pulls():
    names = pull.load_short_names()
    assert {"qwen3:14b", "mimo-v2.6:9b", "nomic-embed-text:latest"} <= set(names)
    for short, entry in names.items():
        assert parse_name(short).host == "local" and parse_name(entry.source).host in pull.HF_HOSTS
        assert parse_name(entry.source).tag != "latest"
        assert len(entry.model_sha256) == 64 and int(entry.model_sha256, 16) >= 0
    assert all(len(names[name].projector_sha256 or "") == 64 for name in ("mimo-v2.6:9b", "qwen2.5vl:7b"))


def test_a_short_name_list_pointing_elsewhere_is_refused(tmp_path):
    listed = tmp_path / "short_names.json"
    listed.write_text(json.dumps({"tiny:1": {"source": "registry.example/library/tiny:1", "model_sha256": "0" * 64}}))
    with pytest.raises(ValueError, match="Hugging Face"):
        pull.load_short_names(listed)


@pytest.mark.parametrize("info", [
    [], {"siblings": "x"}, {"siblings": [1]}, {"siblings": [{"rfilename": 5}]},
    {"siblings": [{"rfilename": "M-Q8_0.gguf", "lfs": "x"}]},
    {"siblings": [{"rfilename": "M-Q8_0.gguf", "lfs": {"sha256": "0" * 64, "size": "big"}}]},
    {"siblings": [{"rfilename": "M-Q8_0.gguf", "lfs": {"sha256": "zz", "size": 1}}]},
])
def test_hugging_face_file_lists_the_engine_cannot_read_are_a_502(engine, monkeypatch, info):
    status, body = pull_from_hub(engine, monkeypatch, 200, info, "hf.co/Someone/M-GGUF:Q8_0")
    assert status == 502 and "Hugging Face" in body["error"]


def test_a_quant_the_repo_does_not_have_is_a_404_and_a_split_one_a_501(engine, monkeypatch):
    info = {"siblings": [{"rfilename": "M-Q8_0.gguf"}, {"rfilename": "M-Q2_K-00001-of-00002.gguf"}]}
    status, body = pull_from_hub(engine, monkeypatch, 200, info, "hf.co/Someone/M-GGUF:Q5_K_M")
    assert status == 404 and "M-Q8_0.gguf" in body["error"]
    status, body = pull_from_hub(engine, monkeypatch, 200, info, "hf.co/Someone/M-GGUF:Q2_K")
    assert status == 501 and "split" in body["error"]


def raw_hub(answer_for_path) -> socketserver.TCPServer:
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            request_line = self.rfile.readline().decode("latin-1")
            while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                pass
            self.wfile.write(answer_for_path(request_line.split(" ")[1] if " " in request_line else ""))

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


GOOD_SHA = hashlib.sha256(b"weights").hexdigest()
GOOD_INFO = json.dumps({"siblings": [{"rfilename": "M-Q8_0.gguf", "lfs": {"sha256": GOOD_SHA, "size": 7}}]}).encode()


def ok(body: bytes) -> bytes:
    return b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\nConnection: close\r\n\r\n" + body


BROKEN_ANSWERS = {
    "short body": lambda path: b"HTTP/1.1 200 OK\r\nContent-Length: 500\r\nConnection: close\r\n\r\n{\"siblings\": [",
    "not http": lambda path: b"SSH-2.0-OpenSSH_9.6\r\n",
    "bad chunks": lambda path: b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\nzz\r\n",
    "deep json": lambda path: ok(b"[" * 200000),
    "blob not http": lambda path: ok(GOOD_INFO) if path.startswith("/api/") else b"garbage\r\n\r\n",
    "blob short": lambda path: ok(GOOD_INFO) if path.startswith("/api/") else
                  b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\nConnection: close\r\n\r\nwei",
}


@pytest.mark.parametrize("case", sorted(BROKEN_ANSWERS))
def test_a_broken_answer_from_hugging_face_is_a_502_never_a_500(engine, monkeypatch, case):
    hub = raw_hub(BROKEN_ANSWERS[case])
    point_at(monkeypatch, f"http://127.0.0.1:{hub.server_address[1]}")
    try:
        status, body = pull_answer(engine, "hf.co/Someone/M-GGUF:Q8_0")
    finally:
        hub.shutdown()
    assert status == 502, body
    assert "engine error" not in body["error"]


@pytest.mark.parametrize("name", ["bad name", "example.invalid/a/b:c", "nothere:1"])
def test_a_streamed_pull_of_a_name_it_refuses_answers_400(engine, name):
    import urllib.request
    request = urllib.request.Request(f"{engine.url}/api/pull", data=json.dumps({"model": name}).encode(), method="POST")
    status, body = error_of(lambda: urllib.request.urlopen(request, timeout=30))
    assert status == 400 and body["error"]


def test_a_repo_hugging_face_hides_without_a_token_is_a_404(engine, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    status, body = pull_from_hub(engine, monkeypatch, 401, {"error": "Invalid username or password."}, "hf.co/Someone/Hidden-GGUF:Q8_0")
    assert status == 404 and "HF_TOKEN" in body["error"]


def test_a_refused_token_is_still_a_502(engine, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "wrong-token-value")
    status, body = pull_from_hub(engine, monkeypatch, 401, {"error": "Invalid credentials"}, "hf.co/Someone/Hidden-GGUF:Q8_0")
    assert status == 502 and "wrong-token-value" not in body["error"]


def tiny_puller(tmp_path, **settings):
    return pull.Puller(tmp_path / "own", 0, short_names={"tiny:1": pull.ShortName("hf.co/Someone/Tiny-GGUF:Q8_0")}, **settings)


def test_an_installed_model_the_engine_cannot_run_is_refused_not_reported_as_pulled(tmp_path, fake_hf):
    from fake_store import INLINE_VISION, add_model
    root = tmp_path / "own"
    add_model(root, "tiny", "1", INLINE_VISION, {"blk.0.w": (4, 4), "v.blk.0.w": (4, 4)})
    puller = tiny_puller(tmp_path, store=ModelStore([root]))
    with pytest.raises(pull.PullError, match="delete it first") as refused:
        puller.pull("tiny:1", lambda update: None)
    assert refused.value.status == 409 and "not supported" in str(refused.value)
    assert fake_hf["paths"] == []


def test_an_installed_model_whose_manifest_is_unreadable_is_refused(tmp_path, fake_hf):
    puller = tiny_puller(tmp_path, store=ModelStore([tmp_path / "own"]))
    puller.pull("tiny:1", lambda update: None)
    (tmp_path / "own" / "catalog" / "local" / "library" / "tiny" / "1.json").write_text("{")
    with pytest.raises(pull.PullError, match="delete it first") as refused:
        puller.pull("tiny:1", lambda update: None)
    assert refused.value.status == 409 and "can't be read" in str(refused.value)


def test_a_pinned_short_name_refuses_a_file_that_changed_on_the_hub(tmp_path, fake_hf):
    puller = pull.Puller(tmp_path / "own", 0, short_names={"tiny:1": pull.ShortName("hf.co/Someone/Tiny-GGUF:Q8_0", "1" * 64)})
    with pytest.raises(pull.PullError, match="changed") as refused:
        puller.pull("tiny:1", lambda update: None)
    assert refused.value.status == 502 and fake_hf["ranges"] == []


def test_a_pulled_manifest_records_its_repo_and_revision_and_downloads_from_that_revision(tmp_path, fake_hf):
    pinned = pull.ShortName("hf.co/Someone/Tiny-GGUF:Q8_0", fake_hf["sha"])
    pull.Puller(tmp_path / "own", 0, short_names={"tiny:1": pinned}).pull("tiny:1", lambda update: None)
    manifest = ModelStore([tmp_path / "own"]).get("tiny:1").manifest
    assert manifest["source"] == {"name": "hf.co/Someone/Tiny-GGUF:Q8_0", "revision": fake_hf["revision"]}
    assert any(f"/resolve/{fake_hf['revision']}/Tiny-Q8_0.gguf" in path for path in fake_hf["paths"])


def test_the_projector_quant_matches_a_whole_word():
    siblings = [{"rfilename": "M-Q4_K_M.gguf"}, {"rfilename": "mmproj-M-bf16.gguf"}, {"rfilename": "mmproj-M-F16.gguf"}]
    assert pull.choose_hf_files(siblings, "Q4_K_M")[1]["rfilename"] == "mmproj-M-F16.gguf"
    assert pull.choose_hf_files(siblings[:2], "Q4_K_M")[1]["rfilename"] == "mmproj-M-bf16.gguf"


@pytest.mark.parametrize("projector", ["mmproj.gguf", "mmproj-F32.gguf", "mmproj-fp16.gguf", "mmproj-Q4_K_M.gguf"])
def test_a_projector_named_another_way_is_still_taken(projector):
    siblings = [{"rfilename": "M-Q4_K_M.gguf"}, {"rfilename": projector}]
    assert pull.choose_hf_files(siblings, "Q4_K_M")[1]["rfilename"] == projector


def test_a_pinned_short_name_without_a_projector_pin_takes_no_projector(tmp_path, fake_hf):
    fake_hf["more_files"] = [{"rfilename": "mmproj-F16.gguf", "lfs": {"sha256": "2" * 64, "size": 10}}]
    pinned = pull.ShortName("hf.co/Someone/Tiny-GGUF:Q8_0", fake_hf["sha"])
    pull.Puller(tmp_path / "own", 0, short_names={"tiny:1": pinned}).pull("tiny:1", lambda update: None)
    manifest = ModelStore([tmp_path / "own"]).get("tiny:1").manifest
    assert [layer["mediaType"] for layer in manifest["layers"]] == [MODEL_LAYER]
    assert not any("mmproj" in path for path in fake_hf["paths"])


def test_a_file_listed_without_a_size_is_refused_before_downloading(tmp_path, fake_hf):
    fake_hf["listed_size"] = 0
    with pytest.raises(pull.PullError, match="size") as refused:
        pull.Puller(tmp_path / "own", 0).pull("hf.co/Someone/Tiny-GGUF:Q8_0", lambda update: None)
    assert refused.value.status == 502 and fake_hf["ranges"] == []


def test_a_download_longer_than_listed_is_cut_off_and_thrown_away(tmp_path, fake_hf):
    fake_hf["listed_size"] = 10
    with pytest.raises(pull.PullError, match="more than"):
        pull.Puller(tmp_path / "own", 0).pull("hf.co/Someone/Tiny-GGUF:Q8_0", lambda update: None)
    assert not list((tmp_path / "own" / "blobs").iterdir())


def test_a_pull_that_would_fill_the_disk_is_refused_before_downloading(tmp_path, fake_hf, monkeypatch):
    monkeypatch.setattr(pull.shutil, "disk_usage", lambda path: pull.shutil._ntuple_diskusage(10**9, 10**9 - 100, 100))
    with pytest.raises(pull.PullError, match="free") as refused:
        pull.Puller(tmp_path / "own", 0, keep_free_bytes=50).pull("hf.co/Someone/Tiny-GGUF:Q8_0", lambda update: None)
    assert refused.value.status == 507 and fake_hf["ranges"] == []


def partial_of(tmp_path, fake_hf):
    blob = tmp_path / "own" / "blobs" / f"sha256-{fake_hf['sha']}"
    blob.parent.mkdir(parents=True)
    return blob, blob.with_name(blob.name + "-partial")


def test_a_finished_partial_is_checked_and_kept_without_asking_the_hub(tmp_path, fake_hf):
    blob, partial = partial_of(tmp_path, fake_hf)
    partial.write_bytes(fake_hf["data"])
    pull.Puller(tmp_path / "own", 0).pull("hf.co/Someone/Tiny-GGUF:Q8_0", lambda update: None)
    assert fake_hf["ranges"] == [] and blob.read_bytes() == fake_hf["data"] and not partial.exists()


def test_a_partial_longer_than_the_file_starts_over(tmp_path, fake_hf):
    blob, partial = partial_of(tmp_path, fake_hf)
    partial.write_bytes(fake_hf["data"] + b"extra")
    pull.Puller(tmp_path / "own", 0).pull("hf.co/Someone/Tiny-GGUF:Q8_0", lambda update: None)
    assert fake_hf["ranges"] == [0] and blob.read_bytes() == fake_hf["data"]


def test_a_refused_range_starts_the_download_over(tmp_path, fake_hf):
    blob, partial = partial_of(tmp_path, fake_hf)
    partial.write_bytes(fake_hf["data"][:10])
    fake_hf["refuse_ranges"] = 1
    pull.Puller(tmp_path / "own", 0).pull("hf.co/Someone/Tiny-GGUF:Q8_0", lambda update: None)
    assert fake_hf["ranges"] == [10, 0] and blob.read_bytes() == fake_hf["data"]


def test_an_installed_short_name_that_is_not_supported_answers_409_through_the_api(engine):
    from fake_store import INLINE_VISION, add_model
    add_model(engine.config.roots[0], "qwen2.5vl", "7b", INLINE_VISION, {"blk.0.w": (4, 4), "v.blk.0.w": (4, 4)})
    status, body = pull_answer(engine, "qwen2.5vl:7b")
    assert status == 409 and "delete it first" in body["error"]
