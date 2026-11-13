import json
import sys
from pathlib import Path

import pytest

from bench import bench

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def bench_arguments(tmp_path, *extra):
    prompt = tmp_path / "prompt.md"
    prompt.write_text("You are Arty.")
    return ["bench", "http://127.0.0.1:9", "test", "--parts", "code", "--code-models", "qwen3:14b", "--support", str(tmp_path),
            "--system-prompt", str(prompt), "--tools", str(FIXTURES / "brain_tools.json"),
            "--edit-tools", str(FIXTURES / "cli_edit_tools.json"), "--output", str(tmp_path / "bench.json"), *extra]


@pytest.fixture
def answers_without_edits(monkeypatch):
    bodies = []

    def fake_call(base, path, body=None, timeout=bench.TIMEOUT_SECONDS):
        bodies.append(body)
        return {"message": {"content": "Nothing to change."}, "eval_count": 3, "eval_duration": 100_000_000}

    monkeypatch.setattr(bench, "call", fake_call)
    monkeypatch.setattr(bench, "sandbox_problem", lambda: "")
    return bodies


def test_requests_use_the_context_size_the_brain_sends_by_default(tmp_path, monkeypatch, answers_without_edits):
    monkeypatch.setattr(sys, "argv", bench_arguments(tmp_path))
    assert bench.main() == 0
    assert {body["options"]["num_ctx"] for body in answers_without_edits} == {bench.DEFAULT_CONTEXT_TOKENS}
    assert bench.DEFAULT_CONTEXT_TOKENS == 24576
    assert json.loads((tmp_path / "bench.json").read_text())["num_ctx"] == 24576


def test_a_chosen_context_size_reaches_every_request(tmp_path, monkeypatch, answers_without_edits):
    monkeypatch.setattr(sys, "argv", bench_arguments(tmp_path, "--num-ctx", "12288"))
    assert bench.main() == 0
    assert {body["options"]["num_ctx"] for body in answers_without_edits} == {12288}


def test_the_stop_file_ends_the_run_after_the_case_in_progress(tmp_path, monkeypatch, answers_without_edits):
    stop_file = tmp_path / "stop"
    real_call = bench.call

    def call_then_ask_to_stop(*arguments, **keywords):
        stop_file.touch()
        return real_call(*arguments, **keywords)

    monkeypatch.setattr(bench, "call", call_then_ask_to_stop)
    monkeypatch.setattr(sys, "argv", bench_arguments(tmp_path, "--stop-file", str(stop_file)))
    assert bench.main() == 0
    report = json.loads((tmp_path / "bench.json").read_text())
    assert len(answers_without_edits) == 1
    assert report["code:qwen3:14b"]["cases"] == 1 and report["code:qwen3:14b"]["stopped_early"] is True
    assert report["stopped"] == f"stop file {stop_file} appeared"


def test_a_stop_file_present_at_the_start_runs_nothing(tmp_path, monkeypatch, answers_without_edits):
    stop_file = tmp_path / "stop"
    stop_file.touch()
    monkeypatch.setattr(sys, "argv", bench_arguments(tmp_path, "--stop-file", str(stop_file)))
    assert bench.main() == 0
    assert answers_without_edits == []
    assert "code:qwen3:14b" not in json.loads((tmp_path / "bench.json").read_text())
