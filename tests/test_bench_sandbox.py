import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from bench import bench

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def property_values(command):
    return [command[index + 1] for index, part in enumerate(command) if part == "-p"]


def test_model_written_code_runs_as_a_throwaway_user_without_network_or_secrets():
    command = bench.check_command("/run/arty-bench-checks/abc")
    assert command[:2] == ["systemd-run", "--wait"] and "--pipe" in command
    properties = property_values(command)
    for expected in ("DynamicUser=yes", "PrivateNetwork=yes", "ProtectSystem=strict", "ProtectHome=yes",
                     "InaccessiblePaths=/srv/ai", "NoNewPrivileges=yes",
                     f"BindReadOnlyPaths=/run/arty-bench-checks/abc:{bench.SANDBOX_FOLDER}"):
        assert expected in properties
    assert command[-4:] == [sys.executable, "-E", "-s", "check.py"]


@pytest.fixture
def recorded_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "CHECK_ROOT", str(tmp_path / "checks"))
    seen = []
    answers = []

    def fake_run(command, **options):
        bound = next(value for value in property_values(command) if value.startswith("BindReadOnlyPaths="))
        folder = Path(bound.split("=", 1)[1].split(":")[0])
        seen.append({"mode": stat.S_IMODE(folder.stat().st_mode), "files": {p.name: p.read_text() for p in folder.iterdir()}})
        return answers.pop(0)

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    return seen, answers


def test_the_check_folder_is_readable_by_the_sandbox_user(recorded_runs):
    seen, answers = recorded_runs
    answers.append(subprocess.CompletedProcess([], 0, stdout="", stderr=""))
    assert bench.run_check("tool.py", "VALUE = 1\n", "import tool\n") == ""
    assert seen == [{"mode": 0o755, "files": {"tool.py": "VALUE = 1\n", "check.py": "import tool\n"}}]


def test_a_failed_check_reports_its_last_error_line(recorded_runs):
    _, answers = recorded_runs
    answers.append(subprocess.CompletedProcess([], 1, stdout="", stderr="Traceback\nAssertionError: wrong total"))
    assert bench.run_check("tool.py", "", "") == "AssertionError: wrong total"


def test_the_coding_part_is_skipped_when_the_sandbox_does_not_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "sandbox_problem", lambda: "network is reachable")
    monkeypatch.setattr(bench, "code_eval", lambda *arguments: pytest.fail("model code ran without a sandbox"))
    output = tmp_path / "bench.json"
    prompt = tmp_path / "prompt.md"
    prompt.write_text("You are Arty.")
    monkeypatch.setattr(sys, "argv", ["bench", "http://127.0.0.1:9", "test", "--parts", "code", "--support", str(tmp_path),
                                      "--system-prompt", str(prompt), "--tools", str(FIXTURES / "brain_tools.json"),
                                      "--edit-tools", str(FIXTURES / "cli_edit_tools.json"), "--output", str(output)])
    assert bench.main() == 0
    assert json.loads(output.read_text())["code"] == {"skipped": "sandbox check failed: network is reachable"}
