import struct
from dataclasses import dataclass, field
from pathlib import Path

GGUF_MAGIC = b"GGUF"
LARGE_ARRAY_LENGTH = 64

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
    tensor_names: list = field(default_factory=list)
    parameter_count: int = 0

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    def arch_value(self, suffix: str, default=None):
        return self.metadata.get(f"{self.architecture}.{suffix}", default)

    @property
    def chat_template(self) -> str:
        template = self.metadata.get("tokenizer.chat_template", "")
        return template if isinstance(template, str) else ""

    @property
    def has_inline_vision(self) -> bool:
        return any(name.startswith(("v.", "mm.")) for name in self.tensor_names)


class _Reader:
    def __init__(self, handle):
        self.handle = handle

    def exact(self, count: int) -> bytes:
        data = self.handle.read(count)
        if len(data) != count:
            raise GgufError("file ended inside the GGUF header")
        return data

    def scalar(self, kind: int):
        fmt = SCALAR_FORMATS[kind]
        return struct.unpack(fmt, self.exact(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        length = self.scalar(UINT64)
        return self.exact(length).decode("utf-8", errors="replace")

    def skip_string(self) -> None:
        self.handle.seek(self.scalar(UINT64), 1)

    def value(self, kind: int):
        if kind == STRING:
            return self.string()
        if kind == ARRAY:
            return self.array()
        if kind in SCALAR_FORMATS:
            return self.scalar(kind)
        raise GgufError(f"unknown GGUF value type {kind}")

    def array(self):
        element_kind = self.scalar(UINT32)
        length = self.scalar(UINT64)
        if length <= LARGE_ARRAY_LENGTH:
            return [self.value(element_kind) for _ in range(length)]
        if element_kind == STRING:
            for _ in range(length):
                self.skip_string()
        elif element_kind in SCALAR_FORMATS:
            self.handle.seek(struct.calcsize(SCALAR_FORMATS[element_kind]) * length, 1)
        else:
            for _ in range(length):
                self.value(element_kind)
        return SkippedArray(length)


def read_gguf(path: Path) -> GgufInfo:
    with open(path, "rb") as handle:
        reader = _Reader(handle)
        if reader.exact(4) != GGUF_MAGIC:
            raise GgufError(f"{path} is not a GGUF file")
        version = reader.scalar(UINT32)
        tensor_count = reader.scalar(UINT64)
        kv_count = reader.scalar(UINT64)
        info = GgufInfo(version=version)
        for _ in range(kv_count):
            key = reader.string()
            info.metadata[key] = reader.value(reader.scalar(UINT32))
        for _ in range(tensor_count):
            info.tensor_names.append(reader.string())
            elements = 1
            for _ in range(reader.scalar(UINT32)):
                elements *= reader.scalar(UINT64)
            info.parameter_count += elements
            handle.seek(4 + 8, 1)
        return info


def plain_metadata(info: GgufInfo) -> dict:
    return {key: value for key, value in info.metadata.items() if not isinstance(value, SkippedArray)}
