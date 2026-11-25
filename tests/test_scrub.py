import logging
import subprocess
import sys
from pathlib import Path

import pytest

from arty_engine.compat import PASS_THROUGH_CHUNK_BYTES, event_line_scrubbed
from arty_engine.scrub import scrub, start_scrubbed_log

LIBRARY = "ll" + "ama"
CORE = "gg" + "ml"
FULL_DEVICE = Path("/dev/full")
DRAIN_SECONDS = 10
LINES_WRITTEN = 4000


@pytest.mark.parametrize("line, expected", [
    (f"I cmn          init: {LIBRARY} threadpool init, n_threads = 2", "I cmn          init: runtime threadpool init, n_threads = 2"),
    (f"{LIBRARY}-cli and {LIBRARY}-bench", "runtime-cli and runtime-bench"),
    (f"{CORE.upper()}_ASSERT failed at {LIBRARY}-context.cpp:42", "RUNTIME_ASSERT failed at runtime-context.cpp:42"),
    (f"{CORE}_backend_init: no device", "runtime_backend_init: no device"),
    (f"{LIBRARY}: failed to load", "runtime: failed to load"),
    (f"cmake -D{LIBRARY.upper()}_CUDA=ON -D{CORE.upper()}_NATIVE=OFF", "cmake -DRUNTIME_CUDA=ON -DRUNTIME_NATIVE=OFF"),
    (f"lib{LIBRARY} not found", "libruntime not found"),
    (f"{LIBRARY}-context:error in decode", "runtime-context:error in decode"),
    (f"{LIBRARY}:8080 refused", "runtime:8080 refused"),
    (f"see https://github.com/{CORE}-org/{LIBRARY}.c{'p'}p/pull/1", "see (link removed)"),
])
def test_bare_runtime_words_are_rewritten(line, expected):
    assert scrub(line) == expected


@pytest.mark.parametrize("text", [
    "Llama 3.1 8B", f"{LIBRARY}3.2", f"{LIBRARY} 3", f"{LIBRARY}-3.1-instruct", "TinyLlama", f"code{LIBRARY}",
    f"hf.co/meta-{LIBRARY}/Llama-3.1-8B", f"pull https://huggingface.co/meta-{LIBRARY}/Llama-3.2-1B first",
    f"{LIBRARY}-guard3:8b", f"{LIBRARY}-guard:latest", f"{LIBRARY}:latest", f"{LIBRARY}:8b", f"{LIBRARY}_3_instruct",
    f"general.architecture str = {LIBRARY}", f'"general.architecture": "{LIBRARY}"', f"tokenizer.{CORE}.model str = {LIBRARY}",
    f"bad special token: 'tokenizer.{CORE}.padding_token_id'", f"{CORE} backend",
])
def test_model_names_architectures_and_the_core_library_are_left_alone(text):
    assert scrub(text) == text


def test_an_event_line_is_passed_on_byte_for_byte_unless_its_error_changes():
    cut_inside_a_character = b'data: {"error": {"message": "' + "\u00e9".encode() * 40000
    cut_inside_a_character = cut_inside_a_character[:PASS_THROUGH_CHUNK_BYTES - 1]
    untouched = b'data:  {"error" : {"message": "out of memory"} }  \n'
    assert event_line_scrubbed(cut_inside_a_character) == cut_inside_a_character
    assert event_line_scrubbed(untouched) == untouched
    changed = event_line_scrubbed(f'data: {{"error": {{"message": "{LIBRARY}_decode failed"}}}}\n'.encode())
    assert changed == b'data: {"error": {"message": "runtime_decode failed"}}\n'


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
