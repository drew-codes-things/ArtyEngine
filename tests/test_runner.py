import time

import pytest

from arty_engine import gpu
from arty_engine.backend import Backend
from arty_engine.config import Config
from arty_engine.gpu import GpuMemory
from arty_engine.runner import FOREVER, LoadError, RunnerPool, build_arguments, compact_tool_definitions, kv_cache_mb
from arty_engine.store import ModelStore
from fake_runner import FakeLauncher
from fake_store import add_model, build_store


def make_pool(tmp_path, launcher=None, budget=11500, free_mb=12000, **settings):
    config = Config(store=str(tmp_path / "own"), model_roots=[str(tmp_path / "other")], state_dir=str(tmp_path / "state"),
                    runner_binary="/opt/runtime/server", vram_budget_mb=budget, load_overhead_mb=100, **settings)
    store = ModelStore(config.roots)
    build_store(tmp_path / "other")
    launcher = launcher or FakeLauncher()
    pool = RunnerPool(config, store, launcher=launcher, memory=lambda: GpuMemory(total_mb=12288, used_mb=12288 - free_mb))
    return pool, store, launcher


def test_loads_once_and_reuses_the_running_model(tmp_path):
    pool, store, launcher = make_pool(tmp_path)
    model = store.get("qwen3:14b")
    first, load_seconds = pool.acquire(model, 12288)
    pool.release(first, FOREVER)
    second, second_load = pool.acquire(model, 12288)
    pool.release(second, FOREVER)
    assert first is second and len(launcher.launched) == 1
    assert load_seconds > 0 and second_load == 0.0
    pool.shutdown()


def test_a_different_context_reloads_the_model(tmp_path):
    pool, store, launcher = make_pool(tmp_path)
    model = store.get("qwen3:14b")
    runner, _ = pool.acquire(model, 12288)
    pool.release(runner, FOREVER)
    runner, _ = pool.acquire(model, 4096)
    pool.release(runner, FOREVER)
    assert len(launcher.launched) == 2
    assert launcher.launched[0].returncode == -15 and launcher.launched[1].returncode is None
    pool.shutdown()


def test_keep_alive_zero_unloads_after_the_request(tmp_path):
    pool, store, launcher = make_pool(tmp_path)
    runner, _ = pool.acquire(store.get("qwen3:14b"), None)
    pool.release(runner, 0)
    assert pool.loaded() == [] and launcher.launched[0].returncode == -15


def test_expired_keep_alive_is_reaped(tmp_path):
    pool, store, launcher = make_pool(tmp_path)
    runner, _ = pool.acquire(store.get("qwen3:14b"), None)
    pool.release(runner, 0.05)
    assert pool.loaded() == [runner]
    time.sleep(0.1)
    pool.reap_once()
    assert pool.loaded() == []


def test_a_model_in_use_is_not_reaped(tmp_path):
    pool, store, _ = make_pool(tmp_path)
    runner, _ = pool.acquire(store.get("qwen3:14b"), None)
    runner.expires_at = time.monotonic() - 1
    pool.reap_once()
    assert pool.loaded() == [runner]
    pool.shutdown()


def test_budget_unloads_the_least_recently_used_idle_model(tmp_path):
    pool, store, launcher = make_pool(tmp_path, budget=150)
    first, _ = pool.acquire(store.get("qwen3:14b"), 1024)
    pool.release(first, FOREVER)
    second, _ = pool.acquire(store.get("mimo:9b"), 1024)
    pool.release(second, FOREVER)
    assert [runner.name for runner in pool.loaded()] == ["mimo:9b"]
    assert launcher.launched[0].returncode == -15
    pool.shutdown()


def test_both_models_stay_when_they_fit(tmp_path):
    pool, store, _ = make_pool(tmp_path, budget=100000)
    for name in ("qwen3:14b", "mimo:9b"):
        runner, _ = pool.acquire(store.get(name), 1024)
        pool.release(runner, FOREVER)
    assert sorted(runner.name for runner in pool.loaded()) == ["mimo:9b", "qwen3:14b"]
    pool.shutdown()


def test_the_pool_holds_a_model_only_while_its_runner_is_running(tmp_path):
    pool, store, launcher = make_pool(tmp_path)
    assert not pool.holds("qwen3:14b")
    runner, _ = pool.acquire(store.get("qwen3:14b"), None)
    pool.release(runner, FOREVER)
    assert pool.holds("qwen3:14b") and not pool.holds("mimo:9b")
    launcher.launched[0].returncode = 1
    assert not pool.holds("qwen3:14b")
    pool.shutdown()


def test_a_crashing_runner_reports_its_log(tmp_path):
    pool, store, _ = make_pool(tmp_path, launcher=FakeLauncher(fail=True))
    with pytest.raises(LoadError, match="exited with 1"):
        pool.acquire(store.get("qwen3:14b"), None)
    assert pool.loaded() == []


def test_arguments_for_a_chat_model(tmp_path):
    pool, store, _ = make_pool(tmp_path)
    plan = pool.plan(store.get("qwen3:14b"), None)
    arguments = build_arguments(pool.config, plan, 4321)
    assert arguments[0] == "/opt/runtime/server"
    for pair in (["--port", "4321"], ["-c", "12288"], ["-ctk", "q8_0"], ["--reasoning-format", "deepseek"], ["-a", "qwen3:14b"]):
        assert arguments[arguments.index(pair[0]) + 1] == pair[1]
    assert "--jinja" in arguments and "--mmproj" not in arguments and "--model-draft" not in arguments


def test_vision_model_gets_its_projector(tmp_path):
    pool, store, _ = make_pool(tmp_path)
    model = store.get("mimo:9b")
    arguments = build_arguments(pool.config, pool.plan(model, None), 1)
    assert arguments[arguments.index("--mmproj") + 1] == str(model.projector_path)


def test_embedding_model_uses_its_own_context_and_batch(tmp_path):
    pool, store, _ = make_pool(tmp_path)
    plan = pool.plan(store.get("nomic-embed-text"), 12288)
    arguments = build_arguments(pool.config, plan, 1)
    assert plan.ctx == 2048 and "--embedding" in arguments and arguments[arguments.index("-ub") + 1] == "2048"


def test_model_without_a_template_borrows_its_twin(tmp_path):
    pool, store, _ = make_pool(tmp_path, compact_tool_json=False)
    plan = pool.plan(store.get("untemplated"), None)
    assert plan.template_file is not None and plan.template_file.read_text() == store.get("qwen3:14b").gguf.chat_template
    assert "--chat-template-file" in build_arguments(pool.config, plan, 1)


def test_a_borrowed_template_also_gets_compact_tool_definitions(tmp_path):
    pool, store, _ = make_pool(tmp_path)
    plan = pool.plan(store.get("untemplated"), None)
    assert plan.template_file.read_text() == compact_tool_definitions(store.get("qwen3:14b").gguf.chat_template)


def test_tool_definitions_are_written_without_spaces(tmp_path):
    pool, store, _ = make_pool(tmp_path)
    plan = pool.plan(store.get("qwen3:14b"), None)
    template = plan.template_file.read_text()
    assert '{{- tool | tojson(separators=(",", ":")) }}' in template and "{{- tool | tojson }}" not in template
    arguments = build_arguments(pool.config, plan, 1)
    assert arguments[arguments.index("--chat-template-file") + 1] == str(plan.template_file)


def test_compact_tool_definitions_keep_the_template_s_own_tojson_arguments(tmp_path):
    pool, store, _ = make_pool(tmp_path)
    template = pool.plan(store.get("mimo:9b"), None).template_file.read_text()
    assert '(tool | tojson(ensure_ascii=False, separators=(",", ":")))' in template


def test_compact_tool_definitions_leave_other_tojson_calls_and_existing_separators_alone():
    template = '{{ tool | tojson(separators=(",", ": ")) }}{{ tool_call.arguments | tojson }}'
    assert compact_tool_definitions(template) == template


def test_compact_tool_json_off_runs_the_model_s_own_template(tmp_path):
    pool, store, _ = make_pool(tmp_path, compact_tool_json=False)
    plan = pool.plan(store.get("qwen3:14b"), None)
    assert plan.template_file is None and "--chat-template-file" not in build_arguments(pool.config, plan, 1)


def test_context_is_capped_at_what_the_model_supports(tmp_path):
    pool, store, _ = make_pool(tmp_path)
    assert pool.plan(store.get("qwen3:14b"), 100000).ctx == 40960


def test_speculative_settings_add_draft_arguments(tmp_path):
    pool, store, _ = make_pool(tmp_path, speculative={"qwen3:14b": {"draft": "untemplated", "draft_max": 8}})
    plan = pool.plan(store.get("qwen3:14b"), None)
    arguments = build_arguments(pool.config, plan, 1)
    assert arguments[arguments.index("--model-draft") + 1] == str(store.get("untemplated").model_path)
    assert arguments[arguments.index("--spec-draft-n-max") + 1] == "8"


def test_ngram_speculation_needs_no_draft_model(tmp_path):
    pool, store, _ = make_pool(tmp_path, speculative={"qwen3:14b": {"spec_type": "ngram-mod"}})
    arguments = build_arguments(pool.config, pool.plan(store.get("qwen3:14b"), None), 1)
    assert arguments[arguments.index("--spec-type") + 1] == "ngram-mod" and "--model-draft" not in arguments


def test_kv_cache_estimate_for_qwen3_14b_at_12k(tmp_path):
    _, store, _ = make_pool(tmp_path)
    assert kv_cache_mb(store.get("qwen3:14b"), 12288, "q8_0") == 1020


KV_CONTEXT = 32768


@pytest.mark.parametrize("counts, megabytes", [
    ({"attention.head_count": [8, 32, 16]}, 1024),
    ({"attention.head_count": 32, "attention.head_count_kv": [2, 8, 4]}, 256),
    ({"attention.head_count": 32, "attention.head_count_kv": 8, "attention.key_length": 64}, 192),
], ids=["kv heads fall back to the largest head count", "the largest of a list of kv heads", "value length falls back to the head width"])
def test_kv_cache_estimate_from_lists_and_missing_counts(tmp_path, counts, megabytes):
    metadata = {"general.architecture": "qwen3", "qwen3.block_count": 2, "qwen3.embedding_length": 4096,
                **{f"qwen3.{name}": value for name, value in counts.items()}}
    add_model(tmp_path / "own", "kv", "1", metadata, {"blk.0.w": (2, 2)})
    assert kv_cache_mb(ModelStore([tmp_path / "own"]).get("kv:1"), KV_CONTEXT, "f16") == megabytes


def test_num_gpu_zero_runs_on_the_cpu_outside_the_budget(tmp_path):
    pool, store, launcher = make_pool(tmp_path)
    model = store.get("nomic-embed-text")
    plan = pool.plan(model, None, cpu_only=True)
    arguments = build_arguments(pool.config, plan, 1)
    assert arguments[arguments.index("-ngl") + 1] == "0" and arguments[arguments.index("--device") + 1] == "none"
    assert pool.estimate_mb(plan) == 0
    runner, _ = pool.acquire(model, None, cpu_only=True)
    pool.release(runner, FOREVER)
    gpu_runner, _ = pool.acquire(model, None)
    pool.release(gpu_runner, FOREVER)
    assert len(launcher.launched) == 2 and launcher.launched[0].returncode == -15
    pool.shutdown()


def test_an_unload_asked_for_during_a_request_happens_when_the_request_ends(tmp_path):
    pool, store, launcher = make_pool(tmp_path)
    runner, _ = pool.acquire(store.get("qwen3:14b"), None)
    assert pool.unload("qwen3:14b")
    assert pool.loaded() == [runner]
    pool.release(runner, FOREVER)
    assert pool.loaded() == [] and launcher.launched[0].returncode == -15


def test_a_new_request_after_an_unload_keeps_the_model(tmp_path):
    pool, store, launcher = make_pool(tmp_path)
    model = store.get("qwen3:14b")
    first, _ = pool.acquire(model, None)
    pool.unload("qwen3:14b")
    second, _ = pool.acquire(model, None)
    pool.release(first, FOREVER)
    pool.release(second, FOREVER)
    assert pool.loaded() == [first] and len(launcher.launched) == 1
    pool.shutdown()


def test_making_room_on_the_card_never_evicts_a_model_running_on_the_cpu(tmp_path):
    pool, store, launcher = make_pool(tmp_path)
    chat_estimates = [pool.estimate_mb(pool.plan(store.get(name), 1024)) for name in ("qwen3:14b", "untemplated")]
    pool.config.vram_budget_mb = sum(chat_estimates) - 1
    embedder, _ = pool.acquire(store.get("nomic-embed-text"), None, cpu_only=True)
    pool.release(embedder, FOREVER)
    for name in ("qwen3:14b", "untemplated"):
        runner, _ = pool.acquire(store.get(name), 1024)
        pool.release(runner, FOREVER)
    assert [process.returncode for process in launcher.launched] == [None, -15, None]
    assert sorted(runner.name for runner in pool.loaded()) == ["nomic-embed-text:latest", "untemplated:latest"]
    pool.shutdown()


def test_a_model_still_answering_is_not_killed_to_load_it_with_another_context(tmp_path, monkeypatch):
    monkeypatch.setattr("arty_engine.runner.WAIT_FOR_IDLE_SECONDS", 0.5)
    pool, store, launcher = make_pool(tmp_path)
    model = store.get("qwen3:14b")
    busy, _ = pool.acquire(model, 12288)
    with pytest.raises(LoadError, match="still answering"):
        pool.acquire(model, 4096)
    assert busy.alive and launcher.launched[0].returncode is None and len(launcher.launched) == 1
    pool.release(busy, FOREVER)
    pool.shutdown()


def backend_pool(tmp_path, name, runtimes=None, memory=None):
    config = Config(store=str(tmp_path / "own"), model_roots=[str(tmp_path / "other")], state_dir=str(tmp_path / "state"),
                    runner_binary="/opt/runtime/server", library_path=["/opt/cuda/lib"], load_overhead_mb=100,
                    runtimes=runtimes or {})
    build_store(tmp_path / "other")
    runtime = config.runtime_for(name)
    backend = Backend(name=name, reason="test", runner_binary=runtime.runner_binary, library_path=tuple(runtime.library_path))
    store = ModelStore(config.roots)
    launcher = FakeLauncher()
    return RunnerPool(config, store, launcher=launcher, memory=memory, backend=backend), store, launcher


HIDING_VARIABLES = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")
HIDE_ALL = {"CUDA_VISIBLE_DEVICES": "", "ROCR_VISIBLE_DEVICES": ""}


def hidden_in(environment: dict) -> dict:
    return {key: environment[key] for key in HIDING_VARIABLES if key in environment}


@pytest.mark.parametrize("name, hidden", [("cuda", {"CUDA_VISIBLE_DEVICES": ""}), ("rocm", {"ROCR_VISIBLE_DEVICES": ""}),
                                          ("vulkan", {}), ("cpu", HIDE_ALL)])
def test_a_cpu_only_request_hides_the_cards_of_the_chosen_backend(tmp_path, monkeypatch, name, hidden):
    for variable in HIDING_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    pool, store, launcher = backend_pool(tmp_path, name)
    runner, _ = pool.acquire(store.get("nomic-embed-text"), None, cpu_only=True)
    pool.release(runner, FOREVER)
    assert hidden_in(launcher.environments[0]) == hidden
    arguments = launcher.launched[0].arguments
    assert arguments[arguments.index("--device") + 1] == "none"
    pool.shutdown()


def test_a_gpu_request_hides_no_card(tmp_path, monkeypatch):
    for variable in HIDING_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    pool, store, launcher = backend_pool(tmp_path, "rocm")
    runner, _ = pool.acquire(store.get("nomic-embed-text"), None)
    pool.release(runner, FOREVER)
    assert hidden_in(launcher.environments[0]) == {}
    pool.shutdown()


def test_the_cpu_backend_keeps_every_request_off_the_cards(tmp_path, monkeypatch):
    for variable in HIDING_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    pool, store, launcher = backend_pool(tmp_path, "cpu")
    runner, _ = pool.acquire(store.get("qwen3:14b"), 4096)
    pool.release(runner, FOREVER)
    arguments = launcher.launched[0].arguments
    assert arguments[arguments.index("-ngl") + 1] == "0" and arguments[arguments.index("--device") + 1] == "none"
    assert hidden_in(launcher.environments[0]) == HIDE_ALL and runner.estimate_mb == 0
    pool.shutdown()


def test_the_chosen_backend_s_runtime_and_libraries_are_launched(tmp_path):
    runtimes = {"rocm": {"runner_binary": "/opt/rocm-build/server", "library_path": ["/opt/rocm-libs/lib"]}}
    pool, store, launcher = backend_pool(tmp_path, "rocm", runtimes)
    runner, _ = pool.acquire(store.get("qwen3:14b"), 4096)
    pool.release(runner, FOREVER)
    assert launcher.launched[0].arguments[0] == "/opt/rocm-build/server"
    assert launcher.environments[0]["LD_LIBRARY_PATH"].split(":") == ["/opt/rocm-libs/lib", "/opt/rocm-build"]
    pool.shutdown()


def test_the_cuda_backend_launches_the_top_level_runtime_as_before(tmp_path):
    pool, store, launcher = backend_pool(tmp_path, "cuda", {"rocm": {"runner_binary": "/opt/rocm-build/server"}})
    runner, _ = pool.acquire(store.get("qwen3:14b"), 4096)
    pool.release(runner, FOREVER)
    assert launcher.launched[0].arguments[0] == "/opt/runtime/server"
    assert launcher.environments[0]["LD_LIBRARY_PATH"].split(":") == ["/opt/cuda/lib", "/opt/runtime"]
    pool.shutdown()


def test_the_pool_reads_card_memory_the_backend_s_way(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(gpu, "card_memory", lambda backend="cuda": seen.append(backend) or GpuMemory(total_mb=8192, used_mb=0))
    monkeypatch.setattr(gpu, "memory_by_process", lambda backend="cuda": seen.append(f"processes {backend}") or {})
    pool, store, _ = backend_pool(tmp_path, "rocm")
    runner, _ = pool.acquire(store.get("qwen3:14b"), 4096)
    pool.release(runner, FOREVER)
    assert runner.vram_mb() == 0
    assert seen[0] == "rocm" and "processes rocm" in seen
    pool.shutdown()


@pytest.mark.parametrize("name, leaks", [("qwen3:14b", False), ("untemplated", False), ("mimo:9b", True), ("nomic-embed-text", False)])
def test_whether_a_model_can_leak_thinking_is_read_from_its_template_at_load(tmp_path, name, leaks):
    pool, store, launcher = make_pool(tmp_path)
    runner, _ = pool.acquire(store.get(name), None)
    pool.release(runner, FOREVER)
    assert runner.plan.may_leak_thinking is leaks
    pool.shutdown()


def test_a_model_named_as_a_leaker_holds_its_content_though_its_template_reads_the_switch(tmp_path):
    pool, store, launcher = make_pool(tmp_path, thinking_leakers=["qwen3:14b"])
    runner, _ = pool.acquire(store.get("qwen3:14b"), None)
    pool.release(runner, FOREVER)
    assert runner.plan.may_leak_thinking is True
    pool.shutdown()


@pytest.mark.parametrize("value", ["mimo:9b", [5], None])
def test_thinking_leakers_must_be_a_list_of_names(value):
    with pytest.raises(ValueError, match="thinking_leakers"):
        Config(thinking_leakers=value)


def test_a_thinking_leaker_name_that_matches_no_model_is_warned_about(tmp_path, caplog):
    caplog.set_level("WARNING")
    make_pool(tmp_path, thinking_leakers=["mimo:9b", "mimo-v2.6"])
    assert "mimo-v2.6:latest" in caplog.text and "thinking_leakers" in caplog.text
    assert "mimo:9b" not in caplog.text
