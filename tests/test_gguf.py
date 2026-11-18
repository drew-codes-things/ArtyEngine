import json
import os
import struct
import time

from arty_engine.gguf import ARRAY, STRING, UINT32, UINT64, GgufError, SkippedArray, plain_metadata, read_gguf
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


KEY_CAP = 64 * 1024
STRING_CAP = 64 * 1024 * 1024
TEXT_CAP = 128 * 1024 * 1024
KEY_COUNT_CAP = 65536
LARGEST_STRING_UNDER_THE_TEXT_CAP = 50 * 1024 * 1024
ONE_VALUE = GGUF_START + KEY


def sparse_file(path, head: bytes, size: int):
    path.write_bytes(head)
    os.truncate(path, size)
    return path


def test_a_tensor_with_more_than_eight_dimensions_is_a_gguf_error(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "qwen3"}, {"blk.0.w": (2 ** 64 - 1,) * 9})
    with pytest.raises(GgufError, match="9 dimensions"):
        read_gguf(path)


def test_a_tensor_with_eight_dimensions_reads(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "qwen3"}, {"blk.0.w": (2,) * 8})
    assert read_gguf(path).parameter_count == 256


def test_thousands_of_dimensions_are_refused_without_multiplying_them(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "qwen3"}, {"blk.0.w": (2 ** 64 - 1,) * 40000})
    started = time.process_time()
    with pytest.raises(GgufError, match="40000 dimensions"):
        read_gguf(path)
    assert time.process_time() - started < 1


def test_a_key_longer_than_64_kib_is_a_gguf_error(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(GGUF_START + struct.pack("<Q", KEY_CAP + 1) + b"k" * (KEY_CAP + 1) + struct.pack("<IB", 0, 1))
    with pytest.raises(GgufError, match="key"):
        read_gguf(path)


def test_a_key_of_64_kib_reads(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(GGUF_START + struct.pack("<Q", KEY_CAP) + b"k" * KEY_CAP + struct.pack("<IB", 0, 1))
    assert read_gguf(path).metadata == {"k" * KEY_CAP: 1}


def test_a_string_value_longer_than_64_mib_is_a_gguf_error(tmp_path):
    head = ONE_VALUE + struct.pack("<IQ", STRING, STRING_CAP + 1)
    path = sparse_file(tmp_path / "model.gguf", head, len(head) + STRING_CAP + 1)
    with pytest.raises(GgufError, match="string"):
        read_gguf(path)


def test_a_string_value_of_64_mib_reads(tmp_path):
    head = ONE_VALUE + struct.pack("<IQ", STRING, STRING_CAP)
    path = sparse_file(tmp_path / "model.gguf", head, len(head) + STRING_CAP)
    assert len(read_gguf(path).metadata["name"]) == STRING_CAP


def test_more_than_128_mib_of_header_text_is_a_gguf_error(tmp_path):
    path = tmp_path / "model.gguf"
    with open(path, "wb") as handle:
        handle.write(b"GGUF" + struct.pack("<IQQ", 3, 0, 3))
        for index in range(3):
            handle.write(struct.pack("<Q", 1) + str(index).encode() + struct.pack("<IQ", STRING, LARGEST_STRING_UNDER_THE_TEXT_CAP))
            handle.seek(LARGEST_STRING_UNDER_THE_TEXT_CAP, 1)
        handle.truncate()
    with pytest.raises(GgufError, match="text"):
        read_gguf(path)


def test_more_than_65536_keys_is_a_gguf_error(tmp_path):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, KEY_COUNT_CAP + 1) + bytes(64))
    with pytest.raises(GgufError, match="keys"):
        read_gguf(path)


def test_65536_keys_read(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {f"k{index}": index for index in range(KEY_COUNT_CAP)}, {})
    assert len(read_gguf(path).metadata) == KEY_COUNT_CAP


START_OF_ONE_SHORT_KEY = b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1) + b"k"
EDGE_FILES_THAT_READ = {
    "a string value that is exactly the rest of the file": START_OF_ONE_SHORT_KEY + struct.pack("<IQ", STRING, 7) + b"exactly",
    "a long array of 65 uint64 that is exactly the rest of the file":
        START_OF_ONE_SHORT_KEY + struct.pack("<IIQ", ARRAY, UINT64, 65) + bytes(520),
}
EDGE_FILES_ONE_BYTE_PAST_THE_END = {
    "a string value one byte longer than the rest of the file": START_OF_ONE_SHORT_KEY + struct.pack("<IQ", STRING, 8) + b"exactly",
    "a long array of 65 uint64 with one byte missing at the end":
        START_OF_ONE_SHORT_KEY + struct.pack("<IIQ", ARRAY, UINT64, 65) + bytes(519),
    "a long array of strings whose last string runs one byte past the end":
        START_OF_ONE_SHORT_KEY + struct.pack("<IIQ", ARRAY, STRING, 65) + struct.pack("<Q", 0) * 64 + struct.pack("<Q", 4) + b"abc",
}


@pytest.mark.parametrize("case", sorted(EDGE_FILES_THAT_READ))
def test_a_length_that_is_exactly_the_rest_of_the_file_reads(tmp_path, case):
    path = tmp_path / "model.gguf"
    path.write_bytes(EDGE_FILES_THAT_READ[case])
    assert "k" in read_gguf(path).metadata


@pytest.mark.parametrize("case", sorted(EDGE_FILES_ONE_BYTE_PAST_THE_END))
def test_a_length_one_byte_past_the_end_of_the_file_is_a_gguf_error(tmp_path, case):
    path = tmp_path / "model.gguf"
    path.write_bytes(EDGE_FILES_ONE_BYTE_PAST_THE_END[case])
    with pytest.raises(GgufError, match="larger than the rest of the file"):
        read_gguf(path)


@pytest.mark.parametrize("data", [b"", b"GGUF", b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + b"\x01\x00"],
                         ids=["empty", "four letters", "cut inside a fixed-size number"])
def test_a_file_cut_where_no_length_was_read_says_the_file_ended(tmp_path, data):
    path = tmp_path / "model.gguf"
    path.write_bytes(data)
    with pytest.raises(GgufError, match="file ended inside the GGUF header"):
        read_gguf(path)


def test_values_json_cannot_hold_are_left_out_of_the_plain_metadata(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "qwen3", "a": float("nan"), "b": float("inf"),
                                                "c": [1.5, float("-inf")], "d": 0.5}, {})
    plain = plain_metadata(read_gguf(path))
    assert json.loads(json.dumps(plain, allow_nan=False)) == {"general.architecture": "qwen3", "a": None, "b": None,
                                                              "c": [1.5, None], "d": 0.5}


@pytest.mark.parametrize("value", ["many", [4096], float("inf"), float("nan"), True])
def test_a_count_of_the_wrong_type_reads_as_zero(tmp_path, value):
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "qwen3", "qwen3.context_length": value}, {})
    assert read_gguf(path).arch_number("context_length") == 0


def test_a_count_reads_as_a_whole_number(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "qwen3", "qwen3.context_length": 40960,
                                                "qwen3.rope.freq_base": 1000000.0}, {})
    info = read_gguf(path)
    assert (info.arch_number("context_length"), info.arch_number("rope.freq_base"), info.arch_number("missing")) == (40960, 1000000, 0)


def test_a_tensor_name_longer_than_64_kib_is_a_gguf_error(tmp_path):
    path = write_gguf(tmp_path / "model.gguf", {"general.architecture": "qwen3"}, {"w" * (KEY_CAP + 1): (2, 2)})
    with pytest.raises(GgufError, match="tensor name"):
        read_gguf(path)
