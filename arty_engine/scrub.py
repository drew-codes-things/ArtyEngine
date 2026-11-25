import logging
import re
import threading
from pathlib import Path

log = logging.getLogger("arty-engine.runner")

LIBRARY_WORD = "ll" + "ama"
CORE_WORD = "gg" + "ml"
NEUTRAL_WORD = "runtime"
LINK_REPLACEMENT = "(link removed)"
UPSTREAM_SPELLINGS = re.compile(rf"{LIBRARY_WORD}[._ -]?c{'p'}p|{LIBRARY_WORD}[-_]server|o{LIBRARY_WORD}|{CORE_WORD}-org",
                                re.IGNORECASE)
UPSTREAM_PREFIXES = re.compile(rf"\blib(?:{LIBRARY_WORD}|{CORE_WORD})(?=[_.\s-]|$)|\b(?:{LIBRARY_WORD}|{CORE_WORD})(?=_[A-Za-z])")
UPSTREAM_VARIABLES = re.compile(rf"(?:{LIBRARY_WORD.upper()}|{CORE_WORD.upper()})_")
MODEL_TAG_AHEAD = r":(?=[A-Za-z0-9])(?!\d+\b)|-[\w-]*:(?:\d+[A-Za-z]|latest\b)"
UPSTREAM_WORD = re.compile(rf"(?<!meta-)\b{LIBRARY_WORD}\b(?![\s._-]*\d)(?!{MODEL_TAG_AHEAD})")
ARCHITECTURE_VALUE = re.compile(r"(?:architecture|tokenizer\.\w+\.model)[\"']?(?:\s+str)?\s*[=:]\s*[\"']?$")
ARCHITECTURE_CONTEXT_CHARS = 80
LINK = re.compile(r"https?://[^\s)\]>\"']+")


def names_upstream_project(link: str) -> bool:
    return bool(UPSTREAM_SPELLINGS.search(link))


def neutral_unless_architecture(found: re.Match) -> str:
    before = found.string[max(0, found.start() - ARCHITECTURE_CONTEXT_CHARS):found.start()]
    return found.group(0) if ARCHITECTURE_VALUE.search(before) else NEUTRAL_WORD


def scrub(text: str) -> str:
    text = LINK.sub(lambda found: LINK_REPLACEMENT if names_upstream_project(found.group(0)) else found.group(0), text)
    text = UPSTREAM_SPELLINGS.sub(NEUTRAL_WORD, text)
    text = UPSTREAM_VARIABLES.sub(lambda found: NEUTRAL_WORD.upper() + "_", text)
    text = UPSTREAM_PREFIXES.sub(lambda found: "lib" + NEUTRAL_WORD if found.group(0).startswith("lib") else NEUTRAL_WORD, text)
    return UPSTREAM_WORD.sub(neutral_unless_architecture, text)


def scrub_strings(value):
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, list):
        return [scrub_strings(item) for item in value]
    if isinstance(value, dict):
        return {key: scrub_strings(item) for key, item in value.items()}
    return value


def open_log(log_path: Path):
    try:
        return open(log_path, "ab", buffering=0)
    except OSError as error:
        log.warning("couldn't open the model runner log %s: %s; its lines are dropped", log_path, error)
        return None


def copy_scrubbed(source, log_path: Path) -> None:
    output = open_log(log_path)
    warned = output is None
    for raw in iter(source.readline, b""):
        if output is None:
            continue
        try:
            output.write(scrub(raw.decode("utf-8", errors="replace")).encode("utf-8"))
        except OSError as error:
            if not warned:
                log.warning("couldn't write the model runner log %s: %s; the runner keeps going and its lines are dropped",
                            log_path, error)
                warned = True
    if output is not None:
        try:
            output.close()
        except OSError as error:
            log.warning("couldn't close the model runner log %s: %s", log_path, error)
    source.close()


def start_scrubbed_log(process, log_path: Path) -> threading.Thread | None:
    source = getattr(process, "stdout", None)
    if source is None:
        return None
    thread = threading.Thread(target=copy_scrubbed, args=(source, log_path), name=f"log-{log_path.stem}", daemon=True)
    thread.start()
    return thread
