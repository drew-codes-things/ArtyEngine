import re
import subprocess
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
UNWANTED_NAMES = re.compile(r"llama[._ -]?cpp|llama[-_]server|[o]llama|ggml[-]org", re.IGNORECASE)
APPEND_ONLY_HISTORY = re.compile(r"(^|/)history(\.md|/)")
RUNTIME_LITERALS = ("LL" + "AMA_CACHE", "gg" + "ml-rpc-server")


def tracked_files() -> list[Path]:
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=REPOSITORY,
                            capture_output=True, check=True).stdout.decode().split("\0")
    return [REPOSITORY / name for name in listed if name and not APPEND_ONLY_HISTORY.search(name)]


def mentions_in(path: Path) -> list[str]:
    try:
        text = path.read_text()
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return []
    found = [f"{path.relative_to(REPOSITORY)}:{number}: {line.strip()[:120]}"
             for number, line in enumerate(text.splitlines(), 1) if UNWANTED_NAMES.search(line)]
    if UNWANTED_NAMES.search(path.name):
        found.append(f"{path.relative_to(REPOSITORY)} (file name)")
    return found


def test_the_tree_never_names_the_runtime_s_source_projects():
    found = [mention for path in tracked_files() for mention in mentions_in(path)]
    assert found == []


def test_the_scan_would_catch_each_spelling():
    pieces = [("LLAMA", ".", "CPP"), ("llama", "-", "cpp"), ("llama", "_", "cpp"), ("llama", "", "cpp"),
              ("llama", "-", "server"), ("llama", "_", "server"), ("O", "llama", ""), ("O", "LLAMA", "_HOST"),
              ("github.com/ggml", "-", "org/x")]
    spellings = ["".join(parts) for parts in pieces]
    assert all(UNWANTED_NAMES.search(spelling) for spelling in spellings)
    assert not any(UNWANTED_NAMES.search(fine) for fine in ["Llama 3", "llama3.2", "gguf", "The ggml authors", "libggml-rpc"])


def test_the_runtime_s_own_variable_and_binary_names_are_built_not_written():
    found = [f"{path.relative_to(REPOSITORY)}: {literal}" for path in tracked_files() if path.suffix in (".py", ".md", ".json", ".service")
             for literal in RUNTIME_LITERALS if literal in read_text_or_empty(path)]
    assert found == []


def read_text_or_empty(path: Path) -> str:
    try:
        return path.read_text()
    except (UnicodeDecodeError, FileNotFoundError, IsADirectoryError):
        return ""
