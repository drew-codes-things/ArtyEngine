import json
import logging
import socket
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from arty_engine import rpc
from arty_engine.config import Config
from arty_engine.gpu import GpuMemory
from arty_engine.runner import FOREVER, RunnerPool, WorkersDown, build_arguments
from arty_engine.server import Engine, make_handler
from arty_engine.store import ModelStore
from engine_harness import error_of, http_json
from fake_runner import FakeLauncher
from fake_store import build_store

TOO_SMALL_BUDGET_MB = 150
ROOMY_BUDGET_MB = 100000
CHAT_REPLY = {"json": {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}]}}


class Listener:
    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind(("127.0.0.1", 0))
        self.socket.listen(16)
        self.address = f"127.0.0.1:{self.socket.getsockname()[1]}"

    def close(self):
        self.socket.close()


def closed_address() -> str:
    listener = Listener()
    listener.close()
    return listener.address


@pytest.fixture
def worker():
    listener = Listener()
    yield listener
    listener.close()


def make_pool(tmp_path, launcher=None, budget=TOO_SMALL_BUDGET_MB, **settings):
    build_store(tmp_path / "other")
    config = Config(store=str(tmp_path / "own"), model_roots=[str(tmp_path / "other")], state_dir=str(tmp_path / "state"),
                    runner_binary="/opt/runtime/server", vram_budget_mb=budget, load_overhead_mb=100, **settings)
    store = ModelStore(config.roots)
    launcher = launcher or FakeLauncher()
    pool = RunnerPool(config, store, launcher=launcher, memory=lambda: GpuMemory(total_mb=12288, used_mb=0))
    return pool, store, launcher


def value_after(arguments: list[str], flag: str) -> str:
    return arguments[arguments.index(flag) + 1]


@pytest.mark.parametrize("workers, problem", [
    ("192.168.1.196:50052", "must be a list"),
    (["no-port"], "host:port"),
    (["host:99999"], "outside 1 to 65535"),
    (["a b:1"], "no spaces or commas"),
    ([{"address": "h:1", "label": "pc"}, {"address": "h:2", "label": "pc"}], "label pc is listed more than once"),
    ([{"address": "h:1", "password": "x"}], "unknown settings password"),
    ([{"address": "h:1", "memory_mb": -1}], "memory_mb"),
])
def test_bad_rpc_workers_stop_the_engine(tmp_path, workers, problem):
    with pytest.raises(ValueError, match=problem):
        Config(rpc_workers=workers)


@pytest.mark.parametrize("settings, problem", [
    ({"m": {"tensor_splits": [1]}}, "unknown setting tensor_splits"),
    ({"m": {"rpc": "sometimes"}}, "rpc must be one of auto, always, never"),
    ({"m": {"rpc_workers": ["elsewhere"]}}, "names elsewhere, which rpc_workers does not list"),
    ({"m": {"tensor_overrides": ["no-equals"]}}, "pattern"),
    ({"m": {"tensor_split": [1, "x"]}}, "numbers"),
    ({"m": {"with_workers": {"parallel": 2}}}, "unknown setting parallel"),
    ({"m": {"gpu_layers": "most"}}, "auto, all"),
    ({"m": {"devices": ["CUDA0,RPC0"]}}, "comma"),
    ({"m": {"lora": [{"path": "/a.gguf", "scale": "big"}]}}, "scale must be a number"),
])
def test_bad_model_settings_stop_the_engine(settings, problem):
    with pytest.raises(ValueError, match=problem):
        Config(rpc_workers=[{"address": "h:1", "label": "pc"}], model_settings=settings)


def test_rpc_always_needs_a_worker():
    with pytest.raises(ValueError, match="lists no workers"):
        Config(model_settings={"m": {"rpc": "always"}})


def test_layout_and_run_settings_become_runtime_arguments(tmp_path):
    settings = {"qwen3:14b": {"gpu_layers": 30, "devices": ["CUDA0"], "tensor_split": [3, 1.5], "split_mode": "layer",
                              "main_gpu": 0, "tensor_overrides": ["blk\\.(1[0-9])\\.ffn_.*_exps\\.=CPU"], "cpu_moe_layers": 4,
                              "fit": False, "parallel": 2, "kv_cache_type": "f16", "flash_attn": "on", "cache_reuse": 256,
                              "rope_scaling": "yarn", "rope_scale": 2, "yarn_orig_ctx": 32768, "threads": 6,
                              "load_mode": "mmap+mlock", "lora": [{"path": "/lora/a.gguf", "scale": 0.5}]}}
    pool, store, _ = make_pool(tmp_path, budget=ROOMY_BUDGET_MB, model_settings=settings)
    arguments = build_arguments(pool.config, pool.plan(store.get("qwen3:14b"), None), 1)
    expected = {"-ngl": "30", "-dev": "CUDA0", "-ts": "3,1.5", "-sm": "layer", "-mg": "0",
                "-ot": "blk\\.(1[0-9])\\.ffn_.*_exps\\.=CPU", "-ncmoe": "4", "--fit": "off", "-np": "2", "-ctk": "f16",
                "-ctv": "f16", "-fa": "on", "--cache-reuse": "256", "--rope-scaling": "yarn", "--rope-scale": "2",
                "--yarn-orig-ctx": "32768", "-t": "6", "-lm": "mmap+mlock", "--lora-scaled": "/lora/a.gguf:0.5"}
    assert {flag: value_after(arguments, flag) for flag in expected} == expected
    assert "--rpc" not in arguments


def test_a_cpu_request_keeps_the_layout_off(tmp_path):
    pool, store, _ = make_pool(tmp_path, model_settings={"qwen3:14b": {"gpu_layers": 30, "devices": ["CUDA0"]}})
    arguments = build_arguments(pool.config, pool.plan(store.get("qwen3:14b"), None, cpu_only=True), 1)
    assert value_after(arguments, "-ngl") == "0" and value_after(arguments, "--device") == "none" and "-dev" not in arguments


def test_request_load_options_win_over_the_model_settings(tmp_path):
    pool, store, _ = make_pool(tmp_path, budget=ROOMY_BUDGET_MB, model_settings={"qwen3:14b": {"gpu_layers": 30, "threads": 4}})
    plan = pool.plan(store.get("qwen3:14b"), None, load_options={"gpu_layers": 12, "threads": 8, "batch": 64})
    arguments = build_arguments(pool.config, plan, 1)
    assert value_after(arguments, "-ngl") == "12" and value_after(arguments, "-t") == "8" and value_after(arguments, "-b") == "64"


def test_a_model_that_fits_never_asks_the_workers(tmp_path):
    asked = []
    pool, store, launcher = make_pool(tmp_path, budget=ROOMY_BUDGET_MB, rpc_workers=[closed_address()])
    pool.health.prober = lambda worker, timeout: asked.append(worker) or ""
    runner, _ = pool.acquire(store.get("qwen3:14b"), 1024)
    pool.release(runner, FOREVER)
    assert asked == [] and "--rpc" not in launcher.launched[0].arguments
    pool.shutdown()


def test_a_model_too_big_for_the_card_loads_with_the_workers(tmp_path, worker):
    workers = [{"address": worker.address, "label": "pc-1060", "memory_mb": 50}]
    settings = {"qwen3:14b": {"gpu_layers": 10, "with_workers": {"gpu_layers": "all", "tensor_split": [23, 18, 0],
                                                                 "tensor_overrides": ["blk\\.2[0-9]\\.ffn_.*_exps\\.=RPC1"]}}}
    pool, store, launcher = make_pool(tmp_path, rpc_workers=workers, model_settings=settings)
    model = store.get("qwen3:14b")
    runner, _ = pool.acquire(model, 1024)
    pool.release(runner, FOREVER)
    arguments = launcher.launched[0].arguments
    assert value_after(arguments, "--rpc") == worker.address
    assert value_after(arguments, "-ngl") == "all" and value_after(arguments, "-ts") == "23,18,0"
    full = pool.full_estimate_mb(pool.plan(model, 1024))
    assert runner.estimate_mb == min(full - 50, TOO_SMALL_BUDGET_MB)
    assert [worker.label for worker in runner.plan.workers] == ["pc-1060"]
    pool.shutdown()


def test_a_worker_that_is_down_means_a_quick_local_load(tmp_path, caplog):
    settings = {"qwen3:14b": {"gpu_layers": 10, "with_workers": {"tensor_split": [1, 1]}}}
    pool, store, launcher = make_pool(tmp_path, rpc_workers=[closed_address()], model_settings=settings)
    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        runner, _ = pool.acquire(store.get("qwen3:14b"), 1024)
    pool.release(runner, FOREVER)
    arguments = launcher.launched[0].arguments
    assert time.monotonic() - started < 5
    assert "--rpc" not in arguments and "-ts" not in arguments and value_after(arguments, "-ngl") == "10"
    assert "did not answer; loading it on this machine instead" in caplog.text
    pool.shutdown()


def test_a_worker_that_is_down_fails_clearly_when_asked_to(tmp_path):
    down = closed_address()
    pool, store, launcher = make_pool(tmp_path, rpc_workers=[{"address": down, "label": "pc"}],
                                      model_settings={"qwen3:14b": {"rpc_when_down": "fail"}})
    started = time.monotonic()
    with pytest.raises(WorkersDown, match=f"pc \\({down}\\) did not answer"):
        pool.acquire(store.get("qwen3:14b"), 1024)
    assert time.monotonic() - started < 5 and launcher.launched == []


def test_every_named_worker_must_answer(tmp_path, worker):
    workers = [{"address": worker.address, "label": "up"}, {"address": closed_address(), "label": "down"}]
    pool, store, launcher = make_pool(tmp_path, rpc_workers=workers)
    runner, _ = pool.acquire(store.get("qwen3:14b"), 1024)
    pool.release(runner, FOREVER)
    assert "--rpc" not in launcher.launched[0].arguments
    pool.shutdown()


def test_rpc_always_uses_the_named_workers_even_for_a_small_model(tmp_path, worker):
    workers = [{"address": worker.address, "label": "pc"}, {"address": closed_address(), "label": "laptop"}]
    pool, store, launcher = make_pool(tmp_path, budget=ROOMY_BUDGET_MB, rpc_workers=workers,
                                      model_settings={"qwen3:14b": {"rpc": "always", "rpc_workers": ["pc"]}})
    runner, _ = pool.acquire(store.get("qwen3:14b"), 1024)
    pool.release(runner, FOREVER)
    assert value_after(launcher.launched[0].arguments, "--rpc") == worker.address
    pool.shutdown()


def test_rpc_never_keeps_a_big_model_local(tmp_path, worker):
    pool, store, launcher = make_pool(tmp_path, rpc_workers=[worker.address], model_settings={"qwen3:14b": {"rpc": "never"}})
    runner, _ = pool.acquire(store.get("qwen3:14b"), 1024)
    pool.release(runner, FOREVER)
    assert "--rpc" not in launcher.launched[0].arguments
    pool.shutdown()


def test_a_cpu_request_never_uses_the_workers(tmp_path, worker):
    pool, store, launcher = make_pool(tmp_path, rpc_workers=[worker.address], model_settings={"qwen3:14b": {"rpc": "always"}})
    runner, _ = pool.acquire(store.get("qwen3:14b"), 1024, cpu_only=True)
    pool.release(runner, FOREVER)
    assert "--rpc" not in launcher.launched[0].arguments
    pool.shutdown()


def test_a_worker_that_drops_during_the_load_ends_the_wait_and_loads_locally(tmp_path, worker, monkeypatch):
    monkeypatch.setattr("arty_engine.runner.WORKER_RECHECK_SECONDS", 0.2)
    launcher = FakeLauncher(loading_when=lambda arguments: "--rpc" in arguments)
    pool, store, _ = make_pool(tmp_path, launcher=launcher, rpc_workers=[worker.address], load_timeout_seconds=600)
    threading.Timer(0.5, worker.close).start()
    started = time.monotonic()
    runner, _ = pool.acquire(store.get("qwen3:14b"), 1024)
    pool.release(runner, FOREVER)
    assert time.monotonic() - started < 10
    assert "--rpc" in launcher.launched[0].arguments and launcher.launched[0].returncode == -15
    assert "--rpc" not in launcher.launched[1].arguments and runner.plan.workers == ()
    pool.shutdown()


def test_a_worker_that_drops_during_the_load_fails_clearly_when_asked_to(tmp_path, worker, monkeypatch):
    monkeypatch.setattr("arty_engine.runner.WORKER_RECHECK_SECONDS", 0.2)
    launcher = FakeLauncher(loading_when=lambda arguments: "--rpc" in arguments)
    pool, store, _ = make_pool(tmp_path, launcher=launcher, rpc_workers=[worker.address],
                               model_settings={"qwen3:14b": {"rpc_when_down": "fail"}})
    threading.Timer(0.5, worker.close).start()
    with pytest.raises(WorkersDown, match="stopped answering while qwen3:14b was loading"):
        pool.acquire(store.get("qwen3:14b"), 1024)
    assert len(launcher.launched) == 1 and launcher.launched[0].returncode == -15


def test_a_loaded_model_is_unloaded_after_its_worker_stays_down(tmp_path, worker):
    pool, store, _ = make_pool(tmp_path, rpc_workers=[worker.address])
    runner, _ = pool.acquire(store.get("qwen3:14b"), 1024)
    pool.release(runner, FOREVER)
    pool.watch_workers_once()
    assert pool.loaded() == [runner]
    worker.close()
    pool.watch_workers_once()
    assert pool.loaded() == [runner]
    pool.watch_workers_once()
    assert pool.loaded() == []


def test_worker_health_is_described_with_the_models_that_use_it(tmp_path, worker):
    down = closed_address()
    pool, store, _ = make_pool(tmp_path, rpc_workers=[{"address": worker.address, "label": "pc", "memory_mb": 5800}, down])
    runner, _ = pool.acquire(store.get("qwen3:14b"), 1024)
    pool.release(runner, FOREVER)
    pool.watch_workers_once()
    described = {entry["label"]: entry for entry in pool.workers_status()}
    assert described["pc"]["up"] is True and described["pc"]["models"] == [] and described["pc"]["memory_mb"] == 5800
    assert described[down]["up"] is False and described[down]["error"]
    pool.shutdown()


def test_probe_reports_a_refused_port():
    assert rpc.probe(rpc.RpcWorker(address=closed_address(), label="x"), timeout=1)


class ApiEngine:
    def __init__(self, tmp_path, **settings):
        build_store(tmp_path / "other")
        self.config = Config(store=str(tmp_path / "own"), model_roots=[str(tmp_path / "other")], state_dir=str(tmp_path / "state"),
                             runner_binary="/opt/runtime/server", load_overhead_mb=100, vram_budget_mb=TOO_SMALL_BUDGET_MB,
                             **settings)
        store = ModelStore(self.config.roots)
        self.launcher = FakeLauncher({"/v1/chat/completions": CHAT_REPLY})
        pool = RunnerPool(self.config, store, launcher=self.launcher, memory=lambda: GpuMemory(total_mb=12288, used_mb=0))
        self.engine = Engine(self.config, pool=pool, store=store)
        self.http = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.engine))
        threading.Thread(target=self.http.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.http.server_address[1]}"

    def close(self):
        self.http.shutdown()
        self.engine.pool.shutdown()


def test_the_status_api_shows_workers_and_the_models_on_them(tmp_path, worker):
    running = ApiEngine(tmp_path, rpc_workers=[{"address": worker.address, "label": "pc-1060"}])
    try:
        http_json(f"{running.url}/api/chat", {"model": "qwen3:14b", "messages": [{"role": "user", "content": "hi"}],
                                              "stream": False, "options": {"num_ctx": 1024}})
        workers = http_json(f"{running.url}/api/workers")["workers"]
        assert workers[0]["label"] == "pc-1060" and workers[0]["up"] is True and workers[0]["models"] == ["qwen3:14b"]
        status = http_json(f"{running.url}/api/status")
        assert status["workers"] == workers and status["loaded"][0]["rpc_workers"] == [{"label": "pc-1060", "address": worker.address}]
        assert http_json(f"{running.url}/api/ps")["models"][0]["rpc_workers"][0]["label"] == "pc-1060"
    finally:
        running.close()


def test_a_down_worker_in_fail_mode_answers_503(tmp_path):
    running = ApiEngine(tmp_path, rpc_workers=[closed_address()], model_settings={"qwen3:14b": {"rpc_when_down": "fail"}})
    try:
        status, body = error_of(lambda: http_json(f"{running.url}/api/chat", {
            "model": "qwen3:14b", "messages": [{"role": "user", "content": "hi"}], "stream": False, "options": {"num_ctx": 1024}}))
        assert status == 503 and "did not answer" in body["error"]
    finally:
        running.close()


def test_status_without_workers_is_an_empty_list(tmp_path):
    running = ApiEngine(tmp_path)
    try:
        request = urllib.request.Request(f"{running.url}/api/status")
        with urllib.request.urlopen(request, timeout=10) as response:
            status = json.loads(response.read())
        assert status["workers"] == [] and status["loaded"] == [] and status["gpu_backend"] == "cuda"
    finally:
        running.close()
