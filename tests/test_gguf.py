from arty_engine.gguf import GgufError, SkippedArray, read_gguf
from gguf_writer import write_gguf
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
