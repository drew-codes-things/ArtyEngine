import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arty_engine.pull import HF_HOSTS
from arty_engine.store import (CATALOG_FOLDER, CONFIG_MEDIA_TYPE, DEFAULT_NAMESPACE, LOCAL_HOST, MEDIA_PREFIX, MODEL_LAYER,
                               ModelName, blob_path, engine_manifest, is_digest, write_manifest)

OLD_MANIFEST_FOLDERS = ("manifests-v2", "manifests")
OLD_CACHE_FOLDERS = ("metadata",)
OLD_FOLDERS = OLD_MANIFEST_FOLDERS + OLD_CACHE_FOLDERS
OLD_LAYER_TYPE = re.compile(r"application/vnd\.[a-z0-9-]+\.image\.([a-z0-9-]+)")
OLD_LIST_SUFFIX = ".manifest.list.v2+json"
GGUF_FORMAT = "gguf"
LAYOUT_BEFORE = "layout-before.json"
MIGRATION_RECORD = "migration.json"
COPY_FOLDER = "copy"
MOVED_FOLDER = "moved"


class MigrationError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def describe_entry(path: Path) -> dict:
    status = path.lstat()
    entry = {"mode": stat.S_IMODE(status.st_mode), "mtime_ns": status.st_mtime_ns}
    if stat.S_ISLNK(status.st_mode):
        return {**entry, "kind": "link", "target": os.readlink(path)}
    if stat.S_ISDIR(status.st_mode):
        return {**entry, "kind": "folder"}
    return {**entry, "kind": "file", "sha256": sha256_of(path), "size": status.st_size}


def describe_tree(root: Path, folders: tuple[str, ...]) -> dict:
    layout = {}
    for folder in folders:
        base = root / folder
        if not base.exists():
            continue
        layout[folder] = describe_entry(base)
        for path in sorted(base.rglob("*")):
            layout[path.relative_to(root).as_posix()] = describe_entry(path)
    return layout


def old_name(host: str, namespace: str, model: str, tag: str) -> ModelName:
    if host.lower() in HF_HOSTS or namespace != DEFAULT_NAMESPACE:
        return ModelName(host, namespace, model, tag)
    return ModelName(LOCAL_HOST, DEFAULT_NAMESPACE, model, tag)


def old_manifests(root: Path) -> dict[str, tuple[ModelName, Path]]:
    found = {}
    for folder in OLD_MANIFEST_FOLDERS:
        base = root / folder
        if not base.is_dir():
            continue
        for path in sorted(base.glob("*/*/*/*")):
            if not path.is_file():
                continue
            name = old_name(*path.relative_to(base).parts)
            found.setdefault(name.display, (name, path))
    return found


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise MigrationError(f"can't read {path}: {error}") from error
    if not isinstance(value, dict):
        raise MigrationError(f"{path} does not hold a JSON object")
    return value


def single_build(root: Path, path: Path, manifest: dict) -> dict:
    if not str(manifest.get("mediaType", "")).endswith(OLD_LIST_SUFFIX):
        return manifest
    builds = [entry for entry in manifest.get("manifests", []) if entry.get("format") == GGUF_FORMAT]
    if len(builds) != 1 or not is_digest(builds[0].get("digest")):
        raise MigrationError(f"{path} lists {len(builds)} GGUF builds; the engine can take exactly one")
    return read_json(blob_path(root, builds[0]["digest"]))


def converted_layers(root: Path, path: Path, manifest: dict) -> list[dict]:
    layers = []
    for layer in manifest.get("layers", []):
        kind = OLD_LAYER_TYPE.fullmatch(str(layer.get("mediaType", "")))
        if not kind:
            continue
        if not is_digest(layer.get("digest")) or not blob_path(root, layer["digest"]).is_file():
            raise MigrationError(f"{path} names the blob {layer.get('digest')!r}, which the store does not hold")
        layers.append({**layer, "mediaType": f"{MEDIA_PREFIX}{kind.group(1)}"})
    if not any(layer["mediaType"] == MODEL_LAYER for layer in layers):
        raise MigrationError(f"{path} has no model layer")
    return layers


def converted_config(manifest: dict) -> dict | None:
    config = manifest.get("config") or {}
    if not is_digest(config.get("digest")):
        return None
    return {"mediaType": CONFIG_MEDIA_TYPE, "digest": config["digest"], "size": int(config.get("size", 0))}


def converted_manifest(root: Path, path: Path) -> dict:
    manifest = single_build(root, path, read_json(path))
    return engine_manifest(converted_layers(root, path, manifest), converted_config(manifest))


def plan_migration(root: Path) -> list[tuple[ModelName, Path, dict]]:
    found = old_manifests(root)
    if not found:
        raise MigrationError(f"{root} holds no manifests in the old layout ({', '.join(OLD_MANIFEST_FOLDERS)})")
    planned = []
    for name, path in found.values():
        if name.manifest_path(root).exists():
            raise MigrationError(f"{name.manifest_path(root)} already exists; nothing was changed")
        planned.append((name, path, converted_manifest(root, path)))
    return planned


def check_backup_folder(root: Path, backup: Path) -> None:
    if backup.exists() and any(backup.iterdir()):
        raise MigrationError(f"{backup} is not empty; pick a new backup folder")
    parent = backup if backup.exists() else backup.parent
    if not parent.is_dir():
        raise MigrationError(f"{parent} does not exist")
    if parent.stat().st_dev != root.stat().st_dev:
        raise MigrationError(f"{backup} is on another file system than {root}; the old folders are moved, never copied across")


def copy_old_folders(root: Path, backup: Path, layout: dict) -> None:
    for folder in OLD_FOLDERS:
        if (root / folder).exists():
            shutil.copytree(root / folder, backup / COPY_FOLDER / folder, symlinks=True)
    copied = describe_tree(backup / COPY_FOLDER, OLD_FOLDERS)
    if strip_times(copied) != strip_times(layout):
        raise MigrationError(f"the copy in {backup / COPY_FOLDER} does not match the store; nothing in the store was changed")


def strip_times(layout: dict) -> dict:
    return {path: {key: value for key, value in entry.items() if key != "mtime_ns"} for path, entry in layout.items()}


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest).encode()).hexdigest()


def migrate(root: Path, backup: Path) -> list[str]:
    root, backup = root.resolve(), backup.absolute()
    check_backup_folder(root, backup)
    planned = plan_migration(root)
    backup.mkdir(exist_ok=True)
    layout = describe_tree(root, OLD_FOLDERS)
    (backup / LAYOUT_BEFORE).write_text(json.dumps(layout, indent=1))
    copy_old_folders(root, backup, layout)
    record = {"store": str(root), "catalog_before": sorted(catalog_files(root)),
              "written": {name.manifest_path(root).relative_to(root).as_posix(): manifest_digest(manifest)
                          for name, _, manifest in planned},
              "moved": [folder for folder in OLD_FOLDERS if (root / folder).exists()]}
    (backup / MIGRATION_RECORD).write_text(json.dumps(record, indent=1))
    for name, old_path, manifest in planned:
        path = write_manifest(root, name, manifest)
        old_time = old_path.lstat().st_mtime_ns
        os.utime(path, ns=(old_time, old_time))
    (backup / MOVED_FOLDER).mkdir()
    for folder in record["moved"]:
        os.rename(root / folder, backup / MOVED_FOLDER / folder)
    return [f"{name.display}: {old_path.relative_to(root)} -> {name.manifest_path(root).relative_to(root)}"
            for name, old_path, _ in planned]


def catalog_files(root: Path) -> set[str]:
    catalog = root / CATALOG_FOLDER
    if not catalog.is_dir():
        return set()
    return {path.relative_to(root).as_posix() for path in catalog.rglob("*") if path.is_file() or path.is_symlink()}


def check_undo(root: Path, backup: Path, record: dict) -> None:
    changed = [path for path, digest in record["written"].items() if (root / path).is_file() and sha256_of(root / path) != digest]
    if changed:
        raise MigrationError("these manifests changed since the migration; nothing was undone:\n  " + "\n  ".join(changed))
    extra = sorted(catalog_files(root) - set(record["written"]) - set(record["catalog_before"]))
    if extra:
        raise MigrationError("these manifests were added after the migration and the old layout can't hold them; "
                             "move or delete them first, nothing was undone:\n  " + "\n  ".join(extra))
    for folder in record["moved"]:
        if (root / folder).exists() and (backup / MOVED_FOLDER / folder).exists():
            raise MigrationError(f"{root / folder} exists again; nothing was undone")


def prune_empty(folder: Path, stop: Path) -> None:
    while folder != stop.parent and folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
        folder = folder.parent


def move_back(root: Path, backup: Path, folder: str, before: dict) -> None:
    if not (backup / MOVED_FOLDER / folder).exists():
        return
    os.rename(backup / MOVED_FOLDER / folder, root / folder)
    old_time = before[folder]["mtime_ns"]
    os.utime(root / folder, ns=(old_time, old_time), follow_symlinks=False)


def undo(root: Path, backup: Path) -> list[str]:
    root, backup = root.resolve(), backup.absolute()
    record = read_json(backup / MIGRATION_RECORD)
    if Path(record["store"]) != root:
        raise MigrationError(f"{backup} belongs to the store {record['store']}, not {root}")
    check_undo(root, backup, record)
    before = read_json(backup / LAYOUT_BEFORE)
    for folder in record["moved"]:
        move_back(root, backup, folder, before)
    for relative in record["written"]:
        path = root / relative
        if path.is_file():
            path.unlink()
            prune_empty(path.parent, root / CATALOG_FOLDER)
    after = describe_tree(root, OLD_FOLDERS)
    differences = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    if differences:
        raise MigrationError("the old layout came back with differences:\n  " + "\n  ".join(differences))
    return [f"restored {folder}" for folder in record["moved"]] + [f"removed {relative}" for relative in record["written"]]


def dry_run(root: Path) -> list[str]:
    return [f"{name.display}: {path.relative_to(root)} -> {name.manifest_path(root).relative_to(root)}"
            for name, path, _ in plan_migration(root.resolve())]


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Move a model store from the old manifest layout to ArtyEngine's catalog, "
                                                 "or back. Blobs are never touched.")
    parser.add_argument("action", choices=("plan", "migrate", "undo"))
    parser.add_argument("--store", type=Path, required=True, help="the model store, such as /srv/ai/arty-engine/models")
    parser.add_argument("--backup", type=Path, help="an empty folder on the store's file system (migrate), or the one "
                                                    "a migration filled (undo)")
    parsed = parser.parse_args(arguments)
    if parsed.action != "plan" and parsed.backup is None:
        parser.error(f"{parsed.action} needs --backup")
    try:
        if parsed.action == "plan":
            lines = dry_run(parsed.store)
        elif parsed.action == "migrate":
            lines = migrate(parsed.store, parsed.backup)
        else:
            lines = undo(parsed.store, parsed.backup)
    except (MigrationError, OSError) as error:
        print(f"{parsed.action}: {error}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
