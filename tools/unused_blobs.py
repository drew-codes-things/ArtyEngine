import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arty_engine.store import CATALOG_FOLDER, MANIFEST_SUFFIX, is_digest

BLOB_NAME = re.compile(r"sha256-[0-9a-f]{64}")
PARTIAL_SUFFIX = "-partial"


class UnreadableManifest(Exception):
    pass


def named_blobs(root: Path) -> set[str]:
    named = set()
    catalog = root / CATALOG_FOLDER
    for path in sorted(catalog.glob(f"*/*/*/*{MANIFEST_SUFFIX}")) if catalog.is_dir() else []:
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


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="List the blobs no catalog manifest names and the partial downloads left "
                                                 "in a store. While a manifest can't be read, only partial downloads are "
                                                 "listed. Run it with no pull in progress.")
    parser.add_argument("--store", type=Path, required=True, help="the model store, such as /srv/ai/arty-engine/models")
    parser.add_argument("--remove", action="store_true", help="delete what is listed")
    parsed = parser.parse_args(arguments)
    try:
        named_blobs(parsed.store)
    except UnreadableManifest as error:
        print(f"can't read {error}; its blobs count as in use", file=sys.stderr)
    found = unused(parsed.store)
    for path in found:
        print(f"{path.stat().st_size:>14} {path}")
        if parsed.remove:
            path.unlink()
    total = sum(path.stat().st_size for path in found if path.exists())
    print(f"{len(found)} files" + (" removed" if parsed.remove else f", {total} bytes"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
