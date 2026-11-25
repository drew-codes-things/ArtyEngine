import logging
import subprocess
import sys
from pathlib import Path

import pytest

from arty_engine.scrub import scrub, start_scrubbed_log

LIBRARY = "ll" + "ama"
CORE = "gg" + "ml"
FULL_DEVICE = Path("/dev/full")
DRAIN_SECONDS = 10
LINES_WRITTEN = 4000


@pytest.mark.parametrize("line, expected", [
    (f"I cmn          init: {LIBRARY} threadpool init, n_threads = 2", "I cmn          init: runtime threadpool init, n_threads = 2"),
    (f"bad special token: 'tokenizer.{CORE}.padding_token_id'", "bad special token: 'tokenizer.runtime.padding_token_id'"),
    (f"{LIBRARY}-cli and {LIBRARY}-bench", "runtime-cli and runtime-bench"),
    (f"{CORE.upper()}_ASSERT failed at {LIBRARY}-context.cpp:42", "RUNTIME_ASSERT failed at runtime-context.cpp:42"),
    (f"{CORE} backend", "runtime backend"),
])
def test_bare_runtime_words_are_rewritten(line, expected):
    assert scrub(line) == expected


@pytest.mark.parametrize("family", ["Llama 3.1 8B", "llama3.2", "llama 3", "llama-3.1-instruct", "TinyLlama", "codellama"])
def test_model_family_spellings_are_left_alone(family):
    assert scrub(family) == family


def test_a_log_that_can_not_be_written_still_drains_the_runner_and_warns_once(caplog):
    caplog.set_level(logging.WARNING)
    child = subprocess.Popen([sys.executable, "-c", f"import sys\nfor _ in range({LINES_WRITTEN}):\n    print('x' * 100)"],
                             stdout=subprocess.PIPE)
    thread = start_scrubbed_log(child, FULL_DEVICE)
    try:
        assert child.wait(timeout=DRAIN_SECONDS) == 0
        thread.join(DRAIN_SECONDS)
        assert not thread.is_alive()
    finally:
        child.kill()
    warnings = [record for record in caplog.records if "couldn't write the model runner log" in record.getMessage()]
    assert len(warnings) == 1
