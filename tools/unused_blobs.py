import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arty_engine.store import CATALOG_FOLDER, MANIFEST_SUFFIX, OLDER_LAYOUT_FOLDERS, is_digest
from arty_engine.store import PARTIAL_SUFFIX as WRITE_IN_PROGRESS_SUFFIX
from migrate_store import (DEFAULT_ENGINE_URL, LARGEST_OLD_DOCUMENT, NO_ENGINE, PROBE_PREFIX, RESTORING_SUFFIX, MigrationError,
                           check_engine_stopped, is_old_manifest_document)

BLOB_NAME = re.compile(r"sha256-[0-9a-f]{64}")
PARTIAL_SUFFIX = "-partial"
REFUSED = 2
GGUF_MAGIC = b"GGUF"
PARTIAL_DOWNLOAD = "partial download"
OLD_DOCUMENT = "old manifest document"
MODEL_FILE = "model file"
OTHER_BLOB = "other"


class UnreadableManifest(Exception):
    pass


def catalog_manifests(root: Path) -> list[Path]:
    catalog = root / CATALOG_FOLDER
    return sorted(catalog.glob(f"*/*/*/*{MANIFEST_SUFFIX}")) if catalog.is_dir() else []


def named_blobs(root: Path) -> set[str]:
    named = set()
    for path in catalog_manifests(root):
        try:
            manifest = json.loads(path.read_text())
        except (OSError, ValueError, RecursionError) as error:
            raise UnreadableManifest(f"{path}: {error}") from error
        layers = manifest.get("layers") if isinstance(manifest, dict) else None
        entries = [*(layers if isinstance(layers, list) else []), manifest.get("config") if isinstance(manifest, dict) else None]
        named |= {entry["digest"].replace(":", "-") for entry in entries if isinstance(entry, dict) and is_digest(entry.get("digest"))}
    return named


def unused(root: Path) -> list[Path]:
    blobs = root / "blobs"
    if not blobs.is_dir():
        return []
    try:
        named = named_blobs(root)
    except UnreadableManifest:
        named = None
    found = []
    for path in sorted(blobs.iterdir()):
        is_partial = path.name.endswith(PARTIAL_SUFFIX) and BLOB_NAME.fullmatch(path.name.removesuffix(PARTIAL_SUFFIX))
        is_unnamed = named is not None and BLOB_NAME.fullmatch(path.name) and path.name not in named
        if path.is_file() and not path.is_symlink() and (is_partial or is_unnamed):
            found.append(path)
    return found


def is_old_document(path: Path) -> bool:
    if path.stat().st_size > LARGEST_OLD_DOCUMENT:
        return False
    try:
        return is_old_manifest_document(json.loads(path.read_bytes()))
    except (ValueError, UnicodeDecodeError, RecursionError):
        return False


def kind_of(path: Path) -> str:
    if path.name.endswith(PARTIAL_SUFFIX):
        return PARTIAL_DOWNLOAD
    if is_old_document(path):
        return OLD_DOCUMENT
    with open(path, "rb") as handle:
        if handle.read(len(GGUF_MAGIC)) == GGUF_MAGIC:
            return MODEL_FILE
    return OTHER_BLOB


def unfinished_writes(root: Path) -> list[Path]:
    catalog = root / CATALOG_FOLDER
    leftovers = list(catalog.rglob(f"*{WRITE_IN_PROGRESS_SUFFIX}")) if catalog.is_dir() else []
    return sorted([*leftovers, *root.glob(f"{PROBE_PREFIX}*"), *root.glob(f"*{RESTORING_SUFFIX}")])


def removal_refusal(root: Path, kinds: set[str], remove_unnamed: bool) -> str | None:
    older = [folder for folder in OLDER_LAYOUT_FOLDERS if (root / folder).exists()]
    if older:
        return (f"{root} still holds {', '.join(older)}: it is in the older layout or its migration stopped part way, and "
                f"those manifests name blobs this tool does not read. Run tools/migrate_store.py migrate (or undo) first")
    if not catalog_manifests(root):
        return f"{root} has no catalog manifest, so every blob would count as unused"
    leftovers = unfinished_writes(root)
    if leftovers:
        return (f"{leftovers[0]} is {leftover_kind(leftovers[0])}; finish or undo that migration, or remove it by hand if no "
                f"migration or pull is running")
    if OLD_DOCUMENT in kinds:
        return ("the list holds old manifest documents; move them with tools/migrate_store.py tidy --backup <that "
                "migration's backup folder>, so undo can still bring them back")
    if kinds & {MODEL_FILE, OTHER_BLOB} and not remove_unnamed:
        return ("the list holds model files or other blobs no manifest names; a model left out of a migration with --skip "
                "or an older copy in its backup still needs them (weights, template, system prompt, parameters). Add "
                "--remove-unnamed-blobs if they are really unwanted")
    return None


def leftover_kind(path: Path) -> str:
    if path.name.startswith(PROBE_PREFIX):
        return "the probe folder of a migration that stopped"
    if path.name.endswith(RESTORING_SUFFIX):
        return "the copy of an undo that stopped"
    return "the leftover of a manifest write that stopped"


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List the blobs no catalog manifest names and the partial downloads left "
                                                 "in a store. While a manifest can't be read, only partial downloads are "
                                                 "listed. Run it with no pull in progress.")
    parser.add_argument("--store", type=Path, required=True, help="the model store, such as /srv/ai/arty-engine/models")
    parser.add_argument("--remove", action="store_true", help="delete what is listed, once the whole list is printed")
    parser.add_argument("--remove-unnamed-blobs", action="store_true",
                        help="with --remove, also delete model files and other blobs no manifest names (without it only "
                             "partial downloads go)")
    parser.add_argument("--engine-url", default=DEFAULT_ENGINE_URL,
                        help=f"--remove refuses while anything listens here (default {DEFAULT_ENGINE_URL}, or {NO_ENGINE})")
    parsed = parser.parse_args(arguments)
    if parsed.remove_unnamed_blobs and not parsed.remove:
        parser.error("--remove-unnamed-blobs only goes with --remove")
    try:
        named_blobs(parsed.store)
    except UnreadableManifest as error:
        print(f"can't read {error}; its blobs count as in use", file=sys.stderr)
    found = [(path, kind_of(path)) for path in unused(parsed.store)]
    for path, kind in found:
        print(f"{path.stat().st_size:>14} {kind:<22} {path}")
    total = sum(path.stat().st_size for path, _ in found)
    print(f"{len(found)} files, {total} bytes")
    if not parsed.remove:
        return 0
    try:
        check_engine_stopped(parsed.engine_url)
    except MigrationError as error:
        print(f"nothing was removed: {error}", file=sys.stderr)
        return REFUSED
    refusal = removal_refusal(parsed.store, {kind for _, kind in found}, parsed.remove_unnamed_blobs)
    if refusal:
        print(f"nothing was removed: {refusal}", file=sys.stderr)
        return REFUSED
    for path, _ in found:
        path.unlink()
    print(f"{len(found)} files removed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
