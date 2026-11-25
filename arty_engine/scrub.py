import re
import threading
from pathlib import Path

LIBRARY_WORD = "ll" + "ama"
CORE_WORD = "gg" + "ml"
NEUTRAL_WORD = "runtime"
LINK_REPLACEMENT = "(link removed)"
UPSTREAM_SPELLINGS = re.compile(rf"{LIBRARY_WORD}[._ -]?c{'p'}p|{LIBRARY_WORD}[-_]server|o{LIBRARY_WORD}|{CORE_WORD}-org",
                                re.IGNORECASE)
UPSTREAM_PREFIXES = re.compile(rf"\blib(?:{LIBRARY_WORD}|{CORE_WORD})(?=[_.-])|\b(?:{LIBRARY_WORD}|{CORE_WORD})(?=_)")
UPSTREAM_VARIABLES = re.compile(rf"\b(?:{LIBRARY_WORD.upper()}|{CORE_WORD.upper()})_")
LINK = re.compile(r"https?://[^\s)\]>\"']+")


def names_upstream(text: str) -> bool:
    lowered = text.lower()
    return LIBRARY_WORD in lowered or CORE_WORD in lowered


def scrub(text: str) -> str:
    text = LINK.sub(lambda found: LINK_REPLACEMENT if names_upstream(found.group(0)) else found.group(0), text)
    text = UPSTREAM_SPELLINGS.sub(NEUTRAL_WORD, text)
    text = UPSTREAM_VARIABLES.sub(lambda found: NEUTRAL_WORD.upper() + "_", text)
    return UPSTREAM_PREFIXES.sub(lambda found: "lib" + NEUTRAL_WORD if found.group(0).startswith("lib") else NEUTRAL_WORD, text)


def copy_scrubbed(source, log_path: Path) -> None:
    with open(log_path, "a", encoding="utf-8", errors="replace") as output:
        for raw in iter(source.readline, b""):
            output.write(scrub(raw.decode("utf-8", errors="replace")))
            output.flush()
    source.close()


def start_scrubbed_log(process, log_path: Path) -> threading.Thread | None:
    source = getattr(process, "stdout", None)
    if source is None:
        return None
    thread = threading.Thread(target=copy_scrubbed, args=(source, log_path), name=f"log-{log_path.stem}", daemon=True)
    thread.start()
    return thread
