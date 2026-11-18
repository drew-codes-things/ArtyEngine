import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = b"GGUF"
LARGE_ARRAY_LENGTH = 64
MOST_TENSOR_DIMENSIONS = 8
MOST_KEYS = 65536
MOST_KEY_BYTES = 64 * 1024
MOST_STRING_BYTES = 64 * 1024 * 1024
MOST_HEADER_TEXT_BYTES = 128 * 1024 * 1024
MOST_HEADER_VALUES = 1_000_000
MOST_ARRAY_DEPTH = 8
LARGEST_WHOLE_NUMBER = 2 ** 63
INLINE_VISION_PREFIXES = ("v.", "mm.")
TENSOR_TYPE_AND_OFFSET_BYTES = 4 + 8

UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)
SCALAR_FORMATS = {
    UINT8: "<B", INT8: "<b", UINT16: "<H", INT16: "<h", UINT32: "<I", INT32: "<i",
    FLOAT32: "<f", BOOL: "<?", UINT64: "<Q", INT64: "<q", FLOAT64: "<d",
}


class GgufError(ValueError):
    pass


@dataclass
class SkippedArray:
    length: int


@dataclass
class GgufInfo:
    version: int
    metadata: dict = field(default_factory=dict)
    parameter_count: int = 0
    has_inline_vision: bool = False

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    def arch_value(self, suffix: str, default=None):
        return self.metadata.get(f"{self.architecture}.{suffix}", default)

    def arch_number(self, suffix: str) -> int:
        return whole_number(self.arch_value(suffix))

    @property
    def chat_template(self) -> str:
        template = self.metadata.get("tokenizer.chat_template", "")
        return template if isinstance(template, str) else ""


def whole_number(value) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float) and -LARGEST_WHOLE_NUMBER <= value < LARGEST_WHOLE_NUMBER:
        return int(value)
    return 0


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, SkippedArray):
        return None
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


class _Reader:
    def __init__(self, handle):
        self.handle = handle
        self.size = os.fstat(handle.fileno()).st_size
        self.text_bytes = 0
        self.values = 0

    def within_file(self, count: int) -> int:
        if count > self.size - self.handle.tell():
            raise GgufError("a length in the GGUF header is larger than the rest of the file")
        return count

    def exact(self, count: int) -> bytes:
        data = self.handle.read(count)
        if len(data) != count:
            raise GgufError("file ended inside the GGUF header")
        return data

    def scalar(self, kind: int):
        fmt = SCALAR_FORMATS[kind]
        return struct.unpack(fmt, self.exact(struct.calcsize(fmt)))[0]

    def string(self, what: str = "string", most_bytes: int = MOST_STRING_BYTES) -> str:
        length = self.within_file(self.scalar(UINT64))
        if length > most_bytes:
            raise GgufError(f"a {what} in the GGUF header is {length} bytes long; the engine reads at most {most_bytes}")
        self.text_bytes += length
        if self.text_bytes > MOST_HEADER_TEXT_BYTES:
            raise GgufError(f"the GGUF header holds more than {MOST_HEADER_TEXT_BYTES} bytes of text")
        return self.exact(length).decode("utf-8", errors="replace")

    def count_values(self, count: int) -> None:
        self.values += count
        if self.values > MOST_HEADER_VALUES:
            raise GgufError(f"the GGUF header holds more than {MOST_HEADER_VALUES} values and tensors; "
                            f"the engine reads at most {MOST_HEADER_VALUES}")

    def skip_string(self) -> None:
        self.handle.seek(self.within_file(self.scalar(UINT64)), 1)

    def value(self, kind: int, depth: int = 0):
        self.count_values(1)
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            return self.array(depth + 1)
        if kind in SCALAR_FORMATS:
            return self.scalar(kind)
        raise GgufError(f"unknown GGUF value type {kind}")

    def array(self, depth: int):
        if depth > MOST_ARRAY_DEPTH:
            raise GgufError(f"arrays in the GGUF header are nested more than {MOST_ARRAY_DEPTH} deep")
        element_kind = self.scalar(UINT32)
        length = self.scalar(UINT64)
        if length <= LARGE_ARRAY_LENGTH:
            return [self.value(element_kind, depth) for _ in range(length)]
        if element_kind == STRING:
            for _ in range(length):
                self.skip_string()
        elif element_kind in SCALAR_FORMATS:
            self.handle.seek(self.within_file(struct.calcsize(SCALAR_FORMATS[element_kind]) * length), 1)
        else:
            for _ in range(length):
                self.value(element_kind, depth)
        return SkippedArray(length)


def read_gguf(path: Path, opened: BinaryIO | None = None) -> GgufInfo:
    if opened is None:
        with open(path, "rb") as handle:
            return read_gguf(path, handle)
    try:
        return _read_header(path, opened)
    except RecursionError as error:
        raise GgufError("arrays in the GGUF header are nested too deep") from error


def _read_header(path: Path, handle: BinaryIO) -> GgufInfo:
    reader = _Reader(handle)
    if reader.exact(4) != GGUF_MAGIC:
        raise GgufError(f"{path} is not a GGUF file")
    version = reader.scalar(UINT32)
    tensor_count = reader.scalar(UINT64)
    kv_count = reader.scalar(UINT64)
    if kv_count > MOST_KEYS:
        raise GgufError(f"the GGUF header has {kv_count} keys; the engine reads at most {MOST_KEYS}")
    reader.count_values(tensor_count)
    info = GgufInfo(version=version)
    for _ in range(kv_count):
        key = reader.string("key", MOST_KEY_BYTES)
        info.metadata[key] = reader.value(reader.scalar(UINT32))
    for _ in range(tensor_count):
        name = reader.string("tensor name", MOST_KEY_BYTES)
        info.has_inline_vision = info.has_inline_vision or name.startswith(INLINE_VISION_PREFIXES)
        dimensions = reader.scalar(UINT32)
        if dimensions > MOST_TENSOR_DIMENSIONS:
            raise GgufError(f"a tensor in the GGUF header has {dimensions} dimensions; "
                            f"the engine reads at most {MOST_TENSOR_DIMENSIONS}")
        elements = 1
        for _ in range(dimensions):
            elements *= reader.scalar(UINT64)
        info.parameter_count += elements
        handle.seek(TENSOR_TYPE_AND_OFFSET_BYTES, 1)
    return info


def plain_metadata(info: GgufInfo) -> dict:
    return {key: json_safe(value) for key, value in info.metadata.items() if not isinstance(value, SkippedArray)}
