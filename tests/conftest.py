import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from arty_engine import backend, runner, server


@pytest.fixture(autouse=True)
def an_nvidia_machine_unless_a_test_names_another(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("nvidia-machine")
    (root / "dev").mkdir()
    (root / "dev" / "nvidia0").write_text("")
    machine = backend.Machine(root=root)
    resolve = backend.resolve_backend

    def resolve_on_that_machine(config, chosen=None):
        return resolve(config, chosen or machine)
    monkeypatch.setattr(runner, "resolve_backend", resolve_on_that_machine)
    monkeypatch.setattr(server, "resolve_backend", resolve_on_that_machine)
