import argparse
import base64
import json
import re
import socket
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arty_engine import gpu
from arty_engine.runtime import ReleaseInfo, unpack

REQUEST_TIMEOUT_SECONDS = 600
PULL_TIMEOUT_SECONDS = 3600
WAIT_POLL_SECONDS = 0.5
CHAT_QUESTION = "What is the capital of France? Answer with one word."
CHAT_ANSWER = "PARIS"
EMBED_LENGTH = 768
PICTURE_TEXT = "ROCK 42"
PICTURE_QUESTION = "What text is written in this picture? Reply with the text only."
PICTURE_SCALE = 16
PICTURE_MARGIN = 48
GLYPH_WIDTH = 5
GLYPH_GAP = 1
GLYPHS = {
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    " ": ("00000",) * 7,
}
GLYPH_HEIGHT = len(GLYPHS["R"])
WHITE, BLACK = 255, 0
BUDGET_HEADROOM_MB = 512
SMALLEST_BUDGET_MB = 1024
UNKNOWN_CARD_BUDGET_MB = 8192
CHECK_CONTEXT = 4096
CHECK_PULL_RESERVE_MB = 512
OFFLOAD_LINE = re.compile(r"offloaded (\d+)/(\d+) layers to GPU")
ON_THE_CARD, ON_THE_CPU = "card", "cpu"


class CheckFailed(Exception):
    pass


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def published_digest(release_file: Path, asset: str) -> str:
    try:
        release = json.loads(release_file.read_text())
    except (OSError, ValueError) as error:
        raise CheckFailed(f"can't read the release listing: {error}") from error
    for entry in release.get("assets", []) if isinstance(release, dict) else []:
        digest = entry.get("digest") or ""
        if entry.get("name") == asset and digest.startswith("sha256:"):
            return digest[len("sha256:"):]
    raise CheckFailed(f"the release listing gives no sha256 for {asset}")


def budget_mb(memory: gpu.GpuMemory | None) -> int:
    if memory is None:
        return UNKNOWN_CARD_BUDGET_MB
    return max(memory.total_mb - BUDGET_HEADROOM_MB, SMALLEST_BUDGET_MB)


def write_config(out: Path, port: int, work: Path, backend: str, runner: str, libraries: list[str]) -> dict:
    settings = {
        "listen": f"127.0.0.1:{port}",
        "store": str(work / "models"),
        "state_dir": str(work / "state"),
        "gpu_backend": backend,
        "runtimes": {backend: {"runner_binary": runner, "library_path": libraries}},
        "vram_budget_mb": budget_mb(gpu.card_memory(backend)),
        "default_ctx": CHECK_CONTEXT,
        "default_keep_alive": "5m",
        "pull_rate_bytes": 0,
        "pull_keep_free_mb": CHECK_PULL_RESERVE_MB,
    }
    out.write_text(json.dumps(settings, indent=2) + "\n")
    return settings


def call(url: str, path: str, body: dict | None = None, timeout: int = REQUEST_TIMEOUT_SECONDS) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{url}{path}", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as error:
        raise CheckFailed(f"{path} answered {error.code}: {error.read()[:300].decode(errors='replace')}") from error
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise CheckFailed(f"{path} failed: {error}") from error


def wait_for_engine(url: str, seconds: float) -> str:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            return f"version {call(url, '/api/version', timeout=2).get('version', '?')}"
        except CheckFailed:
            time.sleep(WAIT_POLL_SECONDS)
    raise CheckFailed(f"the engine did not answer on {url} within {seconds:.0f} s")


def pull(url: str, model: str) -> str:
    answer = call(url, "/api/pull", {"model": model, "stream": False}, timeout=PULL_TIMEOUT_SECONDS)
    if answer.get("status") != "success":
        raise CheckFailed(f"pull of {model} answered {answer}")
    return f"{model} pulled"


def chat(url: str, model: str, cpu_only: bool) -> str:
    options = {"temperature": 0, **({"num_gpu": 0} if cpu_only else {})}
    answer = call(url, "/api/chat", {"model": model, "stream": False, "think": False, "options": options,
                                     "messages": [{"role": "user", "content": CHAT_QUESTION}]})
    content = (answer.get("message") or {}).get("content", "")
    if CHAT_ANSWER not in content.upper():
        raise CheckFailed(f"{model} answered {content[:200]!r}, expected {CHAT_ANSWER.title()}")
    return f"{model} answered {content.strip()[:60]!r}"


def embed(url: str, model: str) -> str:
    vectors = call(url, "/api/embed", {"model": model, "input": "a short test sentence"}).get("embeddings") or [[]]
    if len(vectors[0]) != EMBED_LENGTH:
        raise CheckFailed(f"{model} gave a vector of {len(vectors[0])} numbers, expected {EMBED_LENGTH}")
    return f"{model} gave {EMBED_LENGTH} numbers"


def text_rows(text: str) -> list[list[int]]:
    rows = []
    for row in range(GLYPH_HEIGHT):
        pixels = []
        for character in text:
            pixels += [BLACK if bit == "1" else WHITE for bit in GLYPHS[character][row]] + [WHITE] * GLYPH_GAP
        rows.append(pixels[:-GLYPH_GAP])
    return rows


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def text_picture(text: str) -> bytes:
    glyph_rows = text_rows(text)
    width = len(glyph_rows[0]) * PICTURE_SCALE + 2 * PICTURE_MARGIN
    blank = bytes([WHITE] * width)
    lines = [blank] * PICTURE_MARGIN
    for row in glyph_rows:
        scaled = bytes([WHITE] * PICTURE_MARGIN + [pixel for pixel in row for _ in range(PICTURE_SCALE)] + [WHITE] * PICTURE_MARGIN)
        lines += [scaled] * PICTURE_SCALE
    lines += [blank] * PICTURE_MARGIN
    raw = b"".join(b"\x00" + line for line in lines)
    header = struct.pack(">IIBBBBB", width, len(lines), 8, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", header) + png_chunk(b"IDAT", zlib.compress(raw)) + png_chunk(b"IEND", b"")


def picture(url: str, model: str, image_out: Path) -> str:
    image = text_picture(PICTURE_TEXT)
    image_out.write_bytes(image)
    answer = call(url, "/api/chat", {"model": model, "stream": False, "think": False, "options": {"temperature": 0},
                                     "messages": [{"role": "user", "content": PICTURE_QUESTION,
                                                   "images": [base64.b64encode(image).decode()]}]})
    content = (answer.get("message") or {}).get("content", "")
    words = re.findall(r"[A-Z0-9]+", content.upper())
    if not all(word in words for word in PICTURE_TEXT.split()):
        raise CheckFailed(f"{model} read {content[:200]!r}, expected {PICTURE_TEXT!r}")
    return f"{model} read {content.strip()[:60]!r}"


def loaded(url: str) -> str:
    models = call(url, "/api/ps").get("models", [])
    if not models:
        return "no model loaded"
    return ", ".join(f"{model['name']} size_vram {model.get('size_vram', 0) // gpu.MIB} MiB" for model in models)


def log_sizes(logs: Path) -> str:
    sizes = {path.name: path.stat().st_size for path in sorted(logs.glob("*.log"))} if logs.is_dir() else {}
    return json.dumps(sizes)


def new_log_text(logs: Path, since: Path) -> str:
    try:
        before = json.loads(since.read_text())
    except (OSError, ValueError) as error:
        raise CheckFailed(f"can't read the log sizes taken before the step: {error}") from error
    text = ""
    for path in sorted(logs.glob("*.log")) if logs.is_dir() else []:
        with open(path, "rb") as log:
            log.seek(before.get(path.name, 0))
            text += log.read().decode(errors="replace")
    return text


def offload(logs: Path, since: Path, expect: str) -> str:
    loads = [(int(on_card), int(total)) for on_card, total in OFFLOAD_LINE.findall(new_log_text(logs, since))]
    if not loads:
        raise CheckFailed("the runtime logged no 'offloaded N/M layers to GPU' line for this step, so where the model "
                          "ran is unknown")
    described = ", ".join(f"{on_card}/{total} layers on the card" for on_card, total in loads)
    if expect == ON_THE_CARD and not all(on_card > 0 for on_card, _ in loads):
        raise CheckFailed(f"{described}: the model ran on the CPU, not on the card")
    if expect == ON_THE_CPU and any(on_card > 0 for on_card, _ in loads):
        raise CheckFailed(f"{described}: num_gpu 0 still put layers on the card")
    return described


def card_memory(backend: str) -> str:
    memory = gpu.card_memory(backend)
    if memory is None:
        return "card memory not readable"
    return f"{memory.free_mb} MiB free of {memory.total_mb} MiB ({memory.source})"


def parser() -> argparse.ArgumentParser:
    commands = argparse.ArgumentParser(description="Steps of tools/amd-check.sh, each printing one line")
    steps = commands.add_subparsers(dest="step", required=True)
    steps.add_parser("free-port")
    digest = steps.add_parser("published-digest")
    digest.add_argument("release_file", type=Path)
    digest.add_argument("asset")
    steps.add_parser("pinned-digest").add_argument("platform")
    unpacking = steps.add_parser("unpack")
    unpacking.add_argument("archive", type=Path)
    unpacking.add_argument("folder", type=Path)
    config = steps.add_parser("write-config")
    for name in ("out", "work"):
        config.add_argument(f"--{name}", type=Path, required=True)
    config.add_argument("--port", type=int, required=True)
    config.add_argument("--backend", required=True, choices=("rocm", "vulkan"))
    config.add_argument("--runner", required=True)
    config.add_argument("--library", action="append", default=[])
    for name in ("wait", "pull", "chat", "embed", "picture", "loaded"):
        step = steps.add_parser(name)
        step.add_argument("--url", required=True)
        if name in ("pull", "chat", "embed", "picture"):
            step.add_argument("--model", required=True)
    steps.choices["wait"].add_argument("--seconds", type=float, default=60)
    steps.choices["chat"].add_argument("--cpu-only", action="store_true")
    steps.choices["picture"].add_argument("--image-out", type=Path, required=True)
    memory = steps.add_parser("card-memory")
    memory.add_argument("--backend", required=True)
    sizes = steps.add_parser("log-sizes")
    sizes.add_argument("--logs", type=Path, required=True)
    layers = steps.add_parser("offload")
    layers.add_argument("--logs", type=Path, required=True)
    layers.add_argument("--since", type=Path, required=True)
    layers.add_argument("--expect", required=True, choices=(ON_THE_CARD, ON_THE_CPU))
    return commands


def run(arguments: argparse.Namespace) -> str:
    step = arguments.step
    if step == "free-port":
        return str(free_port())
    if step == "published-digest":
        return published_digest(arguments.release_file, arguments.asset)
    if step == "pinned-digest":
        return ReleaseInfo.read().sha256.get(arguments.platform, "")
    if step == "unpack":
        unpack(arguments.archive, arguments.folder)
        return str(arguments.folder)
    if step == "write-config":
        settings = write_config(arguments.out, arguments.port, arguments.work, arguments.backend, arguments.runner, arguments.library)
        return f"{arguments.out} (listen {settings['listen']}, vram_budget_mb {settings['vram_budget_mb']})"
    if step == "wait":
        return wait_for_engine(arguments.url, arguments.seconds)
    if step == "pull":
        return pull(arguments.url, arguments.model)
    if step == "chat":
        return chat(arguments.url, arguments.model, arguments.cpu_only)
    if step == "embed":
        return embed(arguments.url, arguments.model)
    if step == "picture":
        return picture(arguments.url, arguments.model, arguments.image_out)
    if step == "loaded":
        return loaded(arguments.url)
    if step == "log-sizes":
        return log_sizes(arguments.logs)
    if step == "offload":
        return offload(arguments.logs, arguments.since, arguments.expect)
    return card_memory(arguments.backend)


def main(argv: list[str] | None = None) -> int:
    try:
        print(run(parser().parse_args(argv)))
    except CheckFailed as error:
        print(error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
