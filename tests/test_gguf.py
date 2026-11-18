import struct

from arty_engine.gguf import ARRAY, STRING, UINT32, GgufError, SkippedArray, read_gguf
from gguf_writer import break_first_key_length, write_gguf
import pytest


def test_reads_metadata_tensors_and_parameter_count(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {
        "general.architecture": "qwen3", "qwen3.context_length": 40960, "tokenizer.chat_template": "{{ tools }} <think>",
        "tokenizer.vocab.tokens": [f"t{index}" for index in range(100)], "qwen3.attention.head_count_kv": 8,
    }, {"token_embd.weight": (10, 20), "blk.0.attn_q.weight": (20, 20)})
    info = read_gguf(path)
    assert info.architecture == "qwen3"
    assert info.arch_value("context_length") == 40960
    assert info.chat_template == "{{ tools }} <think>"
    assert info.metadata["tokenizer.vocab.tokens"] == SkippedArray(100)
    assert info.tensor_names == ["token_embd.weight", "blk.0.attn_q.weight"]
    assert info.parameter_count == 600
    assert not info.has_inline_vision


def test_vision_tensors_inside_the_model_are_noticed(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "qwen25vl"}, {"v.blk.0.attn_q.weight": (4, 4), "blk.0.x": (4,)})
    assert read_gguf(path).has_inline_vision


def test_rejects_files_that_are_not_gguf(tmp_path):
    path = tmp_path / "other.bin"
    path.write_bytes(b"NOPE" + bytes(20))
    with pytest.raises(GgufError):
        read_gguf(path)


def test_rejects_truncated_headers(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "qwen3"}, {})
    path.write_bytes(path.read_bytes()[:30])
    with pytest.raises(GgufError):
        read_gguf(path)


GGUF_START = b"GGUF" + struct.pack("<IQQ", 3, 0, 1)
KEY = struct.pack("<Q", 4) + b"name"
BROKEN_LENGTHS = {
    "key longer than memory": GGUF_START + struct.pack("<Q", 2 ** 40) + b"name",
    "key longer than a file can be": GGUF_START + struct.pack("<Q", 2 ** 63) + b"name",
    "string value longer than memory": GGUF_START + KEY + struct.pack("<IQ", STRING, 2 ** 40) + b"text",
    "skipped string longer than a file can be": GGUF_START + KEY + struct.pack("<IIQQ", ARRAY, STRING, 65, 2 ** 63) + b"text",
    "skipped numbers longer than a file can be": GGUF_START + KEY + struct.pack("<IIQ", ARRAY, UINT32, 2 ** 62) + bytes(8),
}


@pytest.mark.parametrize("case", sorted(BROKEN_LENGTHS))
def test_a_length_past_the_end_of_the_file_is_a_gguf_error(tmp_path, case):
    path = tmp_path / "model.gguf"
    path.write_bytes(BROKEN_LENGTHS[case])
    with pytest.raises(GgufError, match="larger than the rest of the file"):
        read_gguf(path)


def test_a_model_file_whose_first_key_length_was_broken_is_a_gguf_error(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "qwen3"}, {"blk.0.w": (4, 4)})
    with pytest.raises(GgufError, match="larger than the rest of the file"):
        read_gguf(break_first_key_length(path, 2 ** 40))
