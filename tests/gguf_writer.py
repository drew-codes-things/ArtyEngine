import os
import struct
from pathlib import Path

STRING, ARRAY, UINT32, FLOAT32, BOOL = 8, 9, 4, 6, 7
FIRST_KEY_LENGTH_OFFSET = 24
LATER_CHANGE_TIME_NS = 2_000_000_000


def _string(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def _value(value) -> tuple[int, bytes]:
    if isinstance(value, bool):
        return BOOL, struct.pack("<?", value)
    if isinstance(value, int):
        return UINT32, struct.pack("<I", value)
    if isinstance(value, float):
        return FLOAT32, struct.pack("<f", value)
    if isinstance(value, str):
        return STRING, _string(value)
    if isinstance(value, list):
        kind = _value(value[0])[0] if value else UINT32
        return ARRAY, struct.pack("<IQ", kind, len(value)) + b"".join(_value(item)[1] for item in value)
    raise TypeError(value)


def write_gguf(path: Path, metadata: dict, tensors: dict[str, tuple[int, ...]]) -> Path:
    parts = [b"GGUF", struct.pack("<IQQ", 3, len(tensors), len(metadata))]
    for key, value in metadata.items():
        kind, data = _value(value)
        parts += [_string(key), struct.pack("<I", kind), data]
    for name, shape in tensors.items():
        parts += [_string(name), struct.pack("<I", len(shape)), b"".join(struct.pack("<Q", size) for size in shape),
                  struct.pack("<IQ", 0, 0)]
    path.write_bytes(b"".join(parts))
    return path


def break_first_key_length(path: Path, length: int) -> Path:
    before = path.stat()
    data = bytearray(path.read_bytes())
    data[FIRST_KEY_LENGTH_OFFSET:FIRST_KEY_LENGTH_OFFSET + 8] = struct.pack("<Q", length)
    path.write_bytes(bytes(data))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + LATER_CHANGE_TIME_NS))
    return path
