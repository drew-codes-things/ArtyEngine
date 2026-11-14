import time

import pytest

from arty_engine.config import Config
from arty_engine.gpu import GpuMemory
from arty_engine.runner import FOREVER, LoadError, RunnerPool, build_arguments, compact_tool_definitions, kv_cache_mb
from arty_engine.store import ModelStore
from fake_runner import FakeLauncher
from fake_store import build_store


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
