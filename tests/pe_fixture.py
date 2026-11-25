import struct
from pathlib import Path

MACHINE_X64 = 0x8664
OPTIONAL_HEADER_SIZE = 240
PE32_PLUS = 0x20B
EXECUTABLE_IMAGE = 0x0002
LARGE_ADDRESS_AWARE = 0x0020
DLL_IMAGE = 0x2000
CONSOLE_SUBSYSTEM = 3
DLL_CHARACTERISTICS = 0x8160
IMAGE_BASE = 0x140000000
SECTION_ALIGNMENT = 0x1000
FILE_ALIGNMENT = 0x200
HEADERS_SIZE = 0x200
CODE_RVA = 0x1000
SECTION_RVA = 0x2000
RETURN_INSTRUCTION = b"\xc3"
CODE_FLAGS = 0x60000020
DATA_DIRECTORIES = 16
IMPORT_DIRECTORY = 1
IAT_DIRECTORY = 12
DESCRIPTOR_SIZE = 20
THUNK_SIZE = 8
SECTION_FLAGS = 0xC0000040
NEW_HEADER_OFFSET = 0x40
STACK_AND_HEAP = (0x100000, 0x1000, 0x100000, 0x1000)


def aligned(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def import_section(imports: list[str]) -> tuple[bytes, int, int, int]:
    descriptors_size = (len(imports) + 1) * DESCRIPTOR_SIZE
    thunk_tables = len(imports) * 2 * THUNK_SIZE
    strings_start = descriptors_size + 2 * thunk_tables
    strings = bytearray()
    hint_names, dll_names = [], []
    for name in imports:
        hint_names.append(strings_start + len(strings))
        strings += struct.pack("<H", 0) + b"Entry\x00"
        strings += b"\x00" * (len(strings) % 2)
        dll_names.append(strings_start + len(strings))
        strings += name.encode() + b"\x00"
        strings += b"\x00" * (len(strings) % 2)
    lookup_start = descriptors_size
    address_start = descriptors_size + thunk_tables
    descriptors = bytearray()
    lookups = bytearray()
    for index in range(len(imports)):
        thunk_offset = index * 2 * THUNK_SIZE
        descriptors += struct.pack("<IIIII", SECTION_RVA + lookup_start + thunk_offset, 0, 0, SECTION_RVA + dll_names[index],
                                   SECTION_RVA + address_start + thunk_offset)
        lookups += struct.pack("<QQ", SECTION_RVA + hint_names[index], 0)
    descriptors += b"\x00" * DESCRIPTOR_SIZE
    return bytes(descriptors + lookups + lookups + strings), descriptors_size, address_start, thunk_tables


def write_pe(path: Path, imports: list[str], dll: bool = False) -> Path:
    content, imports_size, address_start, address_size = import_section(imports)
    raw_size = aligned(len(content), FILE_ALIGNMENT)
    image_size = SECTION_RVA + aligned(len(content), SECTION_ALIGNMENT)
    code_offset = HEADERS_SIZE + raw_size
    characteristics = EXECUTABLE_IMAGE | LARGE_ADDRESS_AWARE | (DLL_IMAGE if dll else 0)
    dos_header = b"MZ" + b"\x00" * (0x3C - 2) + struct.pack("<I", NEW_HEADER_OFFSET)
    coff = struct.pack("<HHIIIHH", MACHINE_X64, 2, 0, 0, 0, OPTIONAL_HEADER_SIZE, characteristics)
    directories = [(0, 0)] * DATA_DIRECTORIES
    directories[IMPORT_DIRECTORY] = (SECTION_RVA, imports_size)
    directories[IAT_DIRECTORY] = (SECTION_RVA + address_start, address_size)
    optional = struct.pack("<HBBIIIII", PE32_PLUS, 14, 0, FILE_ALIGNMENT, raw_size, 0, CODE_RVA, CODE_RVA)
    optional += struct.pack("<QIIHHHHHHIIIIHH", IMAGE_BASE, SECTION_ALIGNMENT, FILE_ALIGNMENT, 6, 0, 0, 0, 6, 0, 0, image_size,
                            HEADERS_SIZE, 0, CONSOLE_SUBSYSTEM, DLL_CHARACTERISTICS)
    optional += struct.pack("<QQQQII", *STACK_AND_HEAP, 0, DATA_DIRECTORIES)
    optional += b"".join(struct.pack("<II", *directory) for directory in directories)
    sections = struct.pack("<8sIIIIIIHHI", b".text", len(RETURN_INSTRUCTION), CODE_RVA, FILE_ALIGNMENT, code_offset, 0, 0, 0, 0,
                           CODE_FLAGS)
    sections += struct.pack("<8sIIIIIIHHI", b".idata", len(content), SECTION_RVA, raw_size, HEADERS_SIZE, 0, 0, 0, 0, SECTION_FLAGS)
    headers = dos_header + b"PE\x00\x00" + coff + optional + sections
    code = RETURN_INSTRUCTION.ljust(FILE_ALIGNMENT, b"\x00")
    path.write_bytes(headers.ljust(HEADERS_SIZE, b"\x00") + content.ljust(raw_size, b"\x00") + code)
    return path
