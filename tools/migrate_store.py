import argparse
import hashlib
import json
import os
import re
import shutil
import socket
import stat
import sys
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arty_engine.pull import HF_HOSTS
from arty_engine.store import (CATALOG_FOLDER, CONFIG_MEDIA_TYPE, LOCAL_HOST, MEDIA_PREFIX, MODEL_LAYER, OLDER_LAYOUT_FOLDERS,
                               PARTIAL_SUFFIX,
                               ModelName, ModelNotFound, blob_path, engine_manifest, is_digest, is_engine_manifest,
                               parse_name, sync_folder, write_manifest)

OLD_MANIFEST_FOLDERS = OLDER_LAYOUT_FOLDERS
OLD_CACHE_FOLDERS = ("metadata",)
OLD_FOLDERS = OLD_MANIFEST_FOLDERS + OLD_CACHE_FOLDERS
OLD_LAYER_TYPE = re.compile(r"application/vnd\.[a-z0-9-]+\.image\.([a-z0-9-]+)")
OLD_LIST_SUFFIX = ".manifest.list.v2+json"
BLOB_NAME = re.compile(r"sha256-[0-9a-f]{64}")
BLOBS_FOLDER = "blobs"
GGUF_FORMAT = "gguf"
LAYOUT_BEFORE = "layout-before.json"
MIGRATION_RECORD = "migration.json"
COPY_FOLDER = "copy"
MOVED_FOLDER = "moved"
DOCUMENTS_FOLDER = "old-manifest-blobs"
PROBE_PREFIX = ".migrate-probe-"
RESTORING_SUFFIX = ".restoring"
ENGINE_URL_SCHEMES = ("http://", "https://")
LARGEST_OLD_DOCUMENT = 1024 * 1024
DEFAULT_ENGINE_URL = "http://127.0.0.1:11438"
NO_ENGINE = "none"
ENGINE_CHECK_SECONDS = 3


class MigrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class OldManifest:
    folder: str
    path: Path
    name: ModelName


@dataclass
class Plan:
    convert: list
    older_copies: list
    skipped: list
    unread_copies: list = field(default_factory=list)
    notes: list = field(default_factory=list)


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


def part_of(layout: dict, folder: str) -> dict:
    return {path: entry for path, entry in layout.items() if path == folder or path.startswith(f"{folder}/")}


def strip_times(layout: dict) -> dict:
    return {path: {key: value for key, value in entry.items() if key != "mtime_ns"} for path, entry in layout.items()}


def write_synced(path: Path, text: str) -> None:
    partial = path.with_name(path.name + PARTIAL_SUFFIX)
    with open(partial, "w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)
    sync_folder(path.parent)


def default_host_of(base: Path, chosen: tuple[str, ...]) -> str | None:
    hosts = sorted(path.name.lower() for path in base.iterdir() if path.is_dir() and path.name.lower() not in HF_HOSTS)
    named = sorted(set(hosts) & {host.lower() for host in chosen})
    if len(named) > 1:
        raise MigrationError(f"{base} holds several of the hosts given with --default-host ({', '.join(named)}); name one")
    if chosen:
        return named[0] if named else None
    if len(hosts) > 1:
        raise MigrationError(f"{base} holds models from several hosts ({', '.join(hosts)}); name the one its short names "
                             f"came from with --default-host; nothing was changed")
    return hosts[0] if hosts else None


def new_name(path: Path, host: str, namespace: str, model: str, tag: str, default_host: str | None) -> ModelName:
    lowered = host.lower()
    if lowered in HF_HOSTS:
        name = ModelName(lowered, namespace, model, tag)
    elif lowered == default_host:
        name = ModelName(LOCAL_HOST, namespace.lower(), model.lower(), tag)
    elif lowered == LOCAL_HOST:
        raise MigrationError(f"{path} sits under a host called {host!r}, which the engine keeps for its own short names")
    else:
        name = ModelName(lowered, namespace, model, tag)
    try:
        if parse_name(name.display) != name:
            raise ModelNotFound(name.display)
    except ModelNotFound as error:
        raise MigrationError(f"{path} would become {name.display!r}, a name the engine can't look up; nothing was changed") from error
    return name


def old_manifests(root: Path, default_hosts: tuple[str, ...]) -> tuple[list[OldManifest], list[tuple[OldManifest, OldManifest]]]:
    kept: dict[str, OldManifest] = {}
    older_copies = []
    for folder in OLD_MANIFEST_FOLDERS:
        base = root / folder
        if not base.is_dir():
            continue
        default_host = default_host_of(base, default_hosts)
        in_folder: dict[str, Path] = {}
        for path in sorted(base.glob("*/*/*/*")):
            if not path.is_file():
                continue
            name = new_name(path, *path.relative_to(base).parts, default_host)
            if name.display in in_folder:
                raise MigrationError(f"{in_folder[name.display].relative_to(root)} and {path.relative_to(root)} would both become "
                                     f"{name.display}; rename or remove one of them first; nothing was changed")
            in_folder[name.display] = path
            entry = OldManifest(folder, path, name)
            if name.display in kept:
                older_copies.append((entry, kept[name.display]))
            else:
                kept[name.display] = entry
    return list(kept.values()), older_copies


def model_file_of(root: Path, path: Path) -> str | None:
    try:
        manifest = single_build(root, path, read_object(path))
    except MigrationError:
        return None
    layers = manifest.get("layers")
    for layer in layers if isinstance(layers, list) else []:
        kind = OLD_LAYER_TYPE.fullmatch(str(layer.get("mediaType", ""))) if isinstance(layer, dict) else None
        if kind and kind.group(1) == "model":
            return layer.get("digest")
    return None


def sort_older_copies(root: Path, older_copies: list, skip: tuple[str, ...]) -> tuple[list, list, list, list]:
    same, unread, skipped, problems = [], [], [], []
    for older, kept in older_copies:
        older_model = model_file_of(root, older.path)
        if older_model is None:
            unread.append((older, kept))
            continue
        if kept.name.display in skip:
            skipped.append(older)
            continue
        kept_model = model_file_of(root, kept.path)
        if kept_model is not None and kept_model != older_model:
            problems.append(f"{kept.name.display}: {kept.path.relative_to(root)} and {older.path.relative_to(root)} hold "
                            f"different model files; move one of them out of the store first")
        else:
            same.append((older, kept))
    return same, unread, skipped, problems


def host_notes(root: Path, default_hosts: tuple[str, ...]) -> list[str]:
    if not default_hosts:
        return []
    chosen = {host.lower() for host in default_hosts}
    notes = []
    for folder in OLD_MANIFEST_FOLDERS:
        base = root / folder
        if not base.is_dir():
            continue
        hosts = sorted(path.name.lower() for path in base.iterdir() if path.is_dir() and path.name.lower() not in HF_HOSTS)
        if hosts and not chosen & set(hosts):
            notes.append(f"note: {folder} holds none of the hosts given with --default-host, so its models keep their host "
                         f"({', '.join(hosts)}) in the name; give --default-host once per folder for short names there")
    return notes


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError, RecursionError) as error:
        raise MigrationError(f"can't read {path}: {error}") from error


def read_object(path: Path) -> dict:
    value = read_json(path)
    if not isinstance(value, dict):
        raise MigrationError(f"{path} does not hold a JSON object")
    return value


def single_build(root: Path, path: Path, manifest: dict) -> dict:
    if not str(manifest.get("mediaType", "")).endswith(OLD_LIST_SUFFIX):
        return manifest
    entries = manifest.get("manifests")
    if not isinstance(entries, list) or not all(isinstance(entry, dict) for entry in entries):
        raise MigrationError(f"{path} is a manifest list whose entries can't be read")
    builds = [entry for entry in entries if entry.get("format") == GGUF_FORMAT]
    if len(builds) != 1 or not is_digest(builds[0].get("digest")):
        raise MigrationError(f"{path} lists {len(builds)} GGUF builds; the engine can take exactly one")
    return read_object(blob_path(root, builds[0]["digest"]))


def converted_layers(root: Path, path: Path, manifest: dict) -> list[dict]:
    layers = manifest.get("layers")
    if not isinstance(layers, list) or not all(isinstance(layer, dict) for layer in layers):
        raise MigrationError(f"{path} has layers the engine can't read")
    converted = []
    for layer in layers:
        kind = OLD_LAYER_TYPE.fullmatch(str(layer.get("mediaType", "")))
        if not kind:
            continue
        if not is_digest(layer.get("digest")) or not blob_path(root, layer["digest"]).is_file():
            raise MigrationError(f"{path} names the blob {layer.get('digest')!r}, which the store does not hold")
        converted.append({**layer, "mediaType": f"{MEDIA_PREFIX}{kind.group(1)}"})
    if not any(layer["mediaType"] == MODEL_LAYER for layer in converted):
        raise MigrationError(f"{path} has no model layer")
    return converted


def converted_config(path: Path, manifest: dict) -> dict | None:
    config = manifest.get("config") or {}
    if not isinstance(config, dict):
        raise MigrationError(f"{path} has a config the engine can't read")
    if not is_digest(config.get("digest")):
        return None
    return {"mediaType": CONFIG_MEDIA_TYPE, "digest": config["digest"], "size": config.get("size", 0)}


def converted_manifest(root: Path, path: Path) -> dict:
    manifest = single_build(root, path, read_object(path))
    converted = engine_manifest(converted_layers(root, path, manifest), converted_config(path, manifest))
    if not is_engine_manifest(converted):
        raise MigrationError(f"{path} has a size or digest of the wrong type")
    return converted


def plan_migration(root: Path, default_hosts: tuple[str, ...] = (), skip: tuple[str, ...] = ()) -> Plan:
    found, older_copies = old_manifests(root, default_hosts)
    if not found:
        raise MigrationError(f"{root} holds no manifests in the old layout ({', '.join(OLD_MANIFEST_FOLDERS)})")
    unknown_skips = sorted(set(skip) - {entry.name.display for entry in found})
    if unknown_skips:
        raise MigrationError(f"--skip names models the old layout doesn't hold: {', '.join(unknown_skips)}")
    existing = [entry.name.manifest_path(root) for entry in found if entry.name.manifest_path(root).exists()]
    if existing:
        raise MigrationError(f"{existing[0]} already exists, so an earlier migrate stopped part way or finished: run undo with "
                             f"that migrate's backup folder first; nothing was changed")
    same_copies, unread_copies, skipped, problems = sort_older_copies(root, older_copies, skip)
    convert = []
    for entry in found:
        if entry.name.display in skip:
            skipped.append(entry)
            continue
        try:
            convert.append((entry, converted_manifest(root, entry.path)))
        except MigrationError as error:
            problems.append(f"{entry.name.display}: {error}")
    if problems:
        raise MigrationError("these old manifests can't be converted; nothing was changed. Leave a model out with "
                             "--skip <name>:\n  " + "\n  ".join(problems))
    return Plan(convert, same_copies, skipped, unread_copies, host_notes(root, default_hosts))


def check_backup_folder(root: Path, backup: Path) -> None:
    if backup.exists() and any(backup.iterdir()):
        raise MigrationError(f"{backup} is not empty; pick a new backup folder")
    parent = backup if backup.exists() else backup.parent
    if not parent.is_dir():
        raise MigrationError(f"{parent} does not exist")
    if parent.stat().st_dev != root.stat().st_dev:
        raise MigrationError(f"{backup} is on another file system than {root}; the old folders are moved, never copied across")


def check_folders_can_move(root: Path, backup: Path) -> None:
    probe = root / f"{PROBE_PREFIX}{os.getpid()}"
    probe.mkdir()
    try:
        os.rename(probe, backup / MOVED_FOLDER / probe.name)
    except OSError as error:
        probe.rmdir()
        raise MigrationError(f"can't move folders from {root} into {backup} ({error.strerror}); pick a backup folder a rename "
                             f"can reach, such as one inside the same mount; nothing in the store was changed") from error
    (backup / MOVED_FOLDER / probe.name).rmdir()


def copy_old_folders(root: Path, backup: Path, layout: dict) -> None:
    for folder in OLD_FOLDERS:
        if (root / folder).exists():
            shutil.copytree(root / folder, backup / COPY_FOLDER / folder, symlinks=True)
    copied = describe_tree(backup / COPY_FOLDER, OLD_FOLDERS)
    if strip_times(copied) != strip_times(layout):
        raise MigrationError(f"the copy in {backup / COPY_FOLDER} does not match the store; nothing in the store was changed")


def manifest_digest(manifest: dict) -> str:
    return hashlib.sha256(json.dumps(manifest).encode()).hexdigest()


def digests_named_by(manifest) -> set[str]:
    if not isinstance(manifest, dict):
        return set()
    layers = manifest.get("layers")
    entries = [*(layers if isinstance(layers, list) else []), manifest.get("config")]
    return {entry["digest"] for entry in entries if isinstance(entry, dict) and is_digest(entry.get("digest"))}


def catalog_files(root: Path) -> set[str]:
    catalog = root / CATALOG_FOLDER
    if not catalog.is_dir():
        return set()
    return {path.relative_to(root).as_posix() for path in catalog.rglob("*")
            if (path.is_file() or path.is_symlink()) and not path.name.endswith(PARTIAL_SUFFIX)}


def blob_names(digests: set[str]) -> set[str]:
    return {digest.replace(":", "-") for digest in digests}


def blobs_in_use(root: Path) -> set[str]:
    used = set()
    for relative in catalog_files(root):
        try:
            used |= digests_named_by(json.loads((root / relative).read_text()))
        except (OSError, ValueError, RecursionError):
            continue
    return blob_names(used)


def is_old_manifest_document(document) -> bool:
    if not isinstance(document, dict) or "schemaVersion" not in document:
        return False
    if str(document.get("mediaType", "")).endswith(OLD_LIST_SUFFIX):
        return True
    layers = document.get("layers")
    return isinstance(layers, list) and any(
        isinstance(layer, dict) and OLD_LAYER_TYPE.fullmatch(str(layer.get("mediaType", ""))) for layer in layers)


def old_documents(root: Path, in_use: set[str]) -> dict[str, str]:
    found = {}
    blobs = root / BLOBS_FOLDER
    if not blobs.is_dir():
        return found
    for path in sorted(blobs.iterdir()):
        if not BLOB_NAME.fullmatch(path.name) or path.name in in_use or path.is_symlink() or not path.is_file():
            continue
        if path.stat().st_size > LARGEST_OLD_DOCUMENT:
            continue
        try:
            document = json.loads(path.read_bytes())
        except (ValueError, UnicodeDecodeError, RecursionError):
            continue
        if is_old_manifest_document(document):
            found[path.name] = sha256_of(path)
    return found


def move_documents_out(root: Path, backup: Path, documents: dict[str, str]) -> None:
    (backup / DOCUMENTS_FOLDER).mkdir(exist_ok=True)
    for name in documents:
        os.rename(root / BLOBS_FOLDER / name, backup / DOCUMENTS_FOLDER / name)


def move_documents_back(root: Path, backup: Path, documents: dict[str, str]) -> None:
    for name in documents:
        if (backup / DOCUMENTS_FOLDER / name).exists():
            os.rename(backup / DOCUMENTS_FOLDER / name, root / BLOBS_FOLDER / name)


def check_documents(root: Path, backup: Path, documents: dict[str, str]) -> None:
    for name, digest in documents.items():
        place = root / BLOBS_FOLDER / name
        if not place.exists():
            place = backup / DOCUMENTS_FOLDER / name
        if not place.is_file() or sha256_of(place) != digest:
            raise MigrationError(f"the old manifest blob {name} is missing or changed in {backup / DOCUMENTS_FOLDER}; "
                                 f"nothing was undone")


def migrate(root: Path, backup: Path, default_hosts: tuple[str, ...] = (), skip: tuple[str, ...] = ()) -> list[str]:
    root, backup = root.resolve(), backup.absolute()
    check_backup_folder(root, backup)
    plan = plan_migration(root, default_hosts, skip)
    remove_empty_probes(root)
    store_time = root.lstat().st_mtime_ns
    blobs_time = (root / BLOBS_FOLDER).lstat().st_mtime_ns
    (backup / MOVED_FOLDER).mkdir(parents=True)
    check_folders_can_move(root, backup)
    layout = describe_tree(root, OLD_FOLDERS)
    write_synced(backup / LAYOUT_BEFORE, json.dumps(layout, indent=1))
    copy_old_folders(root, backup, layout)
    written = {entry.name.manifest_path(root).relative_to(root).as_posix(): manifest_digest(manifest)
               for entry, manifest in plan.convert}
    in_use = blobs_in_use(root) | blob_names(set().union(*(digests_named_by(manifest) for _, manifest in plan.convert)))
    record = {"store": str(root), "catalog_before": sorted(catalog_files(root)), "written": written,
              "moved": [folder for folder in OLD_FOLDERS if (root / folder).exists()], "store_mtime_ns": store_time,
              "blobs_mtime_ns": blobs_time, "documents": old_documents(root, in_use), "finished": False}
    write_synced(backup / MIGRATION_RECORD, json.dumps(record, indent=1))
    for entry, manifest in plan.convert:
        path = write_manifest(root, entry.name, manifest)
        old_time = entry.path.lstat().st_mtime_ns
        os.utime(path, ns=(old_time, old_time))
    sync_folder(root)
    for folder in record["moved"]:
        os.rename(root / folder, backup / MOVED_FOLDER / folder)
    move_documents_out(root, backup, record["documents"])
    record["finished"] = True
    write_synced(backup / MIGRATION_RECORD, json.dumps(record, indent=1))
    return plan_lines(root, plan) + [f"moved {len(record['documents'])} old manifest blobs to {backup / DOCUMENTS_FOLDER}"]


def plan_lines(root: Path, plan: Plan) -> list[str]:
    lines = [f"{entry.name.display}: {entry.path.relative_to(root)} -> {entry.name.manifest_path(root).relative_to(root)}"
             for entry, _ in plan.convert]
    lines += [f"{older.name.display}: {older.path.relative_to(root)} is an older copy of {kept.path.relative_to(root)} "
              f"(same model file); it stays in the backup" for older, kept in plan.older_copies]
    lines += [f"{older.name.display}: {older.path.relative_to(root)} is an older copy of {kept.path.relative_to(root)} "
              f"that can't be read; it stays in the backup, unread" for older, kept in plan.unread_copies]
    lines += [f"{entry.name.display}: {entry.path.relative_to(root)} left out (--skip); it stays in the backup"
              for entry in plan.skipped]
    return lines + plan.notes


def dry_run(root: Path, default_hosts: tuple[str, ...] = (), skip: tuple[str, ...] = ()) -> list[str]:
    root = root.resolve()
    return plan_lines(root, plan_migration(root, default_hosts, skip))


def is_torn(path: Path) -> bool:
    try:
        json.loads(path.read_text())
    except (OSError, ValueError, RecursionError):
        return True
    return False


def check_catalog(root: Path, record: dict) -> None:
    changed = [path for path, digest in record["written"].items()
               if (root / path).is_file() and not is_torn(root / path) and sha256_of(root / path) != digest]
    if changed:
        raise MigrationError("these manifests changed since the migration; nothing was undone:\n  " + "\n  ".join(changed))
    extra = sorted(catalog_files(root) - set(record["written"]) - set(record["catalog_before"]))
    if extra:
        raise MigrationError("these manifests were added after the migration and the old layout can't hold them; "
                             "move or delete them first, nothing was undone:\n  " + "\n  ".join(extra))


def is_earlier_record(record: dict) -> bool:
    return "finished" not in record


def earlier_record_needs_moved(backup: Path) -> str:
    return (f"{backup} was made by this tool before 15 Nov 2026: its record does not say whether the migration finished, "
            f"so only its {MOVED_FOLDER}/ folder is trusted to bring the old layout back, and that folder is gone")


def run_finished(record: dict, backup: Path) -> bool:
    if "finished" in record:
        return bool(record["finished"])
    return (backup / MOVED_FOLDER).is_dir()


def source_of(root: Path, backup: Path, record: dict, before: dict, folder: str) -> Path | None:
    expected = strip_times(part_of(before, folder))
    in_store, moved, copy = root / folder, backup / MOVED_FOLDER / folder, backup / COPY_FOLDER / folder
    if in_store.exists() and moved.exists():
        raise MigrationError(f"{in_store} exists again; nothing was undone")
    candidates = [(in_store, root), (moved, backup / MOVED_FOLDER)]
    if run_finished(record, backup):
        candidates.append((copy, backup / COPY_FOLDER))
    for place, base in candidates:
        if not place.exists():
            continue
        if strip_times(describe_tree(base, (folder,))) != expected:
            raise MigrationError(f"{place} does not match what the migration moved; nothing was undone")
        return None if place == in_store else place
    if is_earlier_record(record):
        raise MigrationError(earlier_record_needs_moved(backup) + "; nothing was undone")
    if not run_finished(record, backup):
        raise MigrationError(f"{folder} is in neither {root} nor {backup / MOVED_FOLDER}: this backup belongs to a migrate that "
                             f"stopped before it moved the old folders, so they are in a later run's backup; undo with that "
                             f"one; nothing was undone")
    raise MigrationError(f"{folder} is in neither {root} nor {backup / MOVED_FOLDER} nor {backup / COPY_FOLDER}; "
                         f"nothing was undone")


def restore_folder(root: Path, source: Path, folder: str, before: dict) -> None:
    if source.parent.name == MOVED_FOLDER:
        os.rename(source, root / folder)
    else:
        restoring = root / f"{folder}{RESTORING_SUFFIX}"
        shutil.copytree(source, restoring, symlinks=True)
        os.rename(restoring, root / folder)
    for relative in sorted(part_of(before, folder), key=lambda path: path.count("/"), reverse=True):
        old_time = before[relative]["mtime_ns"]
        os.utime(root / relative, ns=(old_time, old_time), follow_symlinks=False)


def remove_written(root: Path, record: dict) -> None:
    catalog = root / CATALOG_FOLDER
    leftovers = [path.relative_to(root).as_posix() for path in catalog.rglob(f"*{PARTIAL_SUFFIX}")] if catalog.is_dir() else []
    for relative in [*record["written"], *leftovers]:
        path = root / relative
        if path.is_file():
            path.unlink()
            prune_empty(path.parent, catalog)


def prune_empty(folder: Path, stop: Path) -> None:
    while folder != stop.parent and folder.is_dir() and not any(folder.iterdir()):
        folder.rmdir()
        folder = folder.parent


def models_missing_blobs(root: Path, folders: list[str]) -> list[str]:
    lines = []
    for folder in folders:
        for path in sorted((root / folder).glob("*/*/*/*")):
            try:
                manifest = single_build(root, path, read_object(path))
            except MigrationError as error:
                lines.append(f"warning: {path.relative_to(root)} can't be read ({error}), so that model won't load; a later "
                             f"migrate needs --skip for it")
                continue
            missing = sorted(digest for digest in digests_named_by(manifest) if not blob_path(root, digest).is_file())
            if missing:
                lines.append(f"warning: {path.relative_to(root)} names {len(missing)} blobs the store no longer holds, so that "
                             f"model won't load; a later migrate needs --skip for it")
    return lines


def store_record(root: Path, backup: Path) -> dict:
    record = read_object(backup / MIGRATION_RECORD)
    if Path(record["store"]) != root:
        raise MigrationError(f"{backup} belongs to the store {record['store']}, not {root}")
    return record


def undo(root: Path, backup: Path) -> list[str]:
    root, backup = root.resolve(), backup.absolute()
    record = store_record(root, backup)
    before = read_object(backup / LAYOUT_BEFORE)
    if record.get("undone") and not any((root / folder).exists() for folder in record["moved"]):
        raise MigrationError(f"{backup} was already undone and its old folders are not in the store, so a later migrate "
                             f"holds it; undo with that migrate's backup; nothing was undone")
    remove_restoring_leftovers(root, record)
    remove_empty_probes(root)
    check_catalog(root, record)
    sources = {folder: source_of(root, backup, record, before, folder) for folder in record["moved"]}
    documents = record.get("documents", {})
    check_documents(root, backup, documents)
    for folder, source in sources.items():
        if source is not None:
            restore_folder(root, source, folder, before)
    move_documents_back(root, backup, documents)
    remove_written(root, record)
    for key, folder in (("blobs_mtime_ns", root / BLOBS_FOLDER), ("store_mtime_ns", root)):
        if key in record:
            os.utime(folder, ns=(record[key], record[key]))
    after = describe_tree(root, OLD_FOLDERS)
    differences = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    if differences:
        raise MigrationError("the old layout came back with differences:\n  " + "\n  ".join(differences))
    record["undone"] = True
    write_synced(backup / MIGRATION_RECORD, json.dumps(record, indent=1))
    return ([f"restored {folder}" for folder, source in sources.items() if source is not None]
            + [f"removed {relative}" for relative in record["written"]]
            + models_missing_blobs(root, [folder for folder in record["moved"] if folder in OLD_MANIFEST_FOLDERS]))


def remove_empty_probes(root: Path) -> None:
    for probe in root.glob(f"{PROBE_PREFIX}*"):
        if probe.is_dir() and not probe.is_symlink() and not any(probe.iterdir()):
            probe.rmdir()


def remove_restoring_leftovers(root: Path, record: dict) -> None:
    for folder in record["moved"]:
        leftover = root / f"{folder}{RESTORING_SUFFIX}"
        if leftover.is_dir() and not leftover.is_symlink():
            shutil.rmtree(leftover)


def finished_record(root: Path, backup: Path) -> dict:
    record = store_record(root, backup)
    if record.get("undone"):
        raise MigrationError(f"the migration recorded in {backup} was undone; nothing was changed")
    if not run_finished(record, backup):
        if is_earlier_record(record):
            raise MigrationError(earlier_record_needs_moved(backup) + "; nothing was changed")
        raise MigrationError(f"{backup} belongs to a migrate that did not finish; undo it first")
    back = [folder for folder in record["moved"] if (root / folder).exists()]
    if back:
        raise MigrationError(f"{', '.join(back)} is back in {root}, so the migration recorded in {backup} is not in force "
                             f"(was it undone by an earlier version of this tool?); nothing was changed")
    return record


def tidy(root: Path, backup: Path) -> list[str]:
    root, backup = root.resolve(), backup.absolute()
    record = finished_record(root, backup)
    if record.get("documents"):
        check_documents(root, backup, record["documents"])
        left = {name: digest for name, digest in record["documents"].items() if (root / BLOBS_FOLDER / name).exists()}
        move_documents_out(root, backup, left)
        if not left:
            return [f"the old manifest blobs of this migration are already in {backup / DOCUMENTS_FOLDER}"]
        return [f"moved {name} to {backup / DOCUMENTS_FOLDER}" for name in left]
    record["blobs_mtime_ns"] = (root / BLOBS_FOLDER).lstat().st_mtime_ns
    record["documents"] = old_documents(root, blobs_in_use(root))
    write_synced(backup / MIGRATION_RECORD, json.dumps(record, indent=1))
    move_documents_out(root, backup, record["documents"])
    return [f"moved {name} to {backup / DOCUMENTS_FOLDER}" for name in record["documents"]]


def untidy(root: Path, backup: Path) -> list[str]:
    root, backup = root.resolve(), backup.absolute()
    record = finished_record(root, backup)
    documents = record.get("documents", {})
    check_documents(root, backup, documents)
    move_documents_back(root, backup, documents)
    if "blobs_mtime_ns" in record:
        os.utime(root / BLOBS_FOLDER, ns=(record["blobs_mtime_ns"], record["blobs_mtime_ns"]))
    record["documents"] = {}
    write_synced(backup / MIGRATION_RECORD, json.dumps(record, indent=1))
    return [f"put {name} back in {root / BLOBS_FOLDER}" for name in documents]


def engine_address(url: str) -> str:
    if "://" not in url:
        return f"http://{url}"
    if not url.startswith(ENGINE_URL_SCHEMES):
        raise MigrationError(f"--engine-url {url} is not an http:// or https:// address (or none); nothing was changed")
    return url


def host_and_port(address: str) -> tuple[str, int]:
    parts = urllib.parse.urlsplit(address)
    try:
        port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError as error:
        raise MigrationError(f"--engine-url {address} has a port that isn't a number; nothing was changed") from error
    if not parts.hostname:
        raise MigrationError(f"--engine-url {address} names no host; nothing was changed")
    return parts.hostname, port


def check_engine_stopped(url: str) -> None:
    if url == NO_ENGINE:
        return
    address = engine_address(url)
    host, port = host_and_port(address)
    try:
        connection = socket.create_connection((host, port), timeout=ENGINE_CHECK_SECONDS)
    except (ConnectionRefusedError, TimeoutError):
        return
    except OSError as error:
        raise MigrationError(f"can't tell whether an engine listens at {address} ({error}); nothing was changed") from error
    connection.close()
    raise MigrationError(f"an engine (or something else) listens at {address}; stop it first (or pass --engine-url none "
                         f"when no engine reads this store)")


ACTIONS = {"undo": undo, "tidy": tidy, "untidy": untidy}


def main(arguments: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Move a model store from the old manifest layout to ArtyEngine's catalog, "
                                                 "or back. Model blobs are never touched; the old manifest documents in "
                                                 "blobs/ move into the backup.")
    parser.add_argument("action", choices=("plan", "migrate", *ACTIONS))
    parser.add_argument("--store", type=Path, required=True, help="the model store, such as /srv/ai/arty-engine/models")
    parser.add_argument("--backup", type=Path, help="an empty folder a rename can reach (migrate), or the one a migration "
                                                    "filled (undo, tidy, untidy)")
    parser.add_argument("--default-host", action="append", default=[],
                        help="the host whose models were reached by short names; once given, only the hosts given become "
                             "short names, so give it once per old folder (plan names a folder it leaves out)")
    parser.add_argument("--skip", action="append", default=[], help="a model to leave out (it stays in the backup)")
    parser.add_argument("--engine-url", default=DEFAULT_ENGINE_URL,
                        help=f"refuse to change the store while anything listens here (default {DEFAULT_ENGINE_URL}, or none)")
    parsed = parser.parse_args(arguments)
    if parsed.action != "plan" and parsed.backup is None:
        parser.error(f"{parsed.action} needs --backup")
    try:
        if parsed.action == "plan":
            lines = dry_run(parsed.store, tuple(parsed.default_host), tuple(parsed.skip))
        else:
            check_engine_stopped(parsed.engine_url)
            if parsed.action == "migrate":
                lines = migrate(parsed.store, parsed.backup, tuple(parsed.default_host), tuple(parsed.skip))
            else:
                lines = ACTIONS[parsed.action](parsed.store, parsed.backup)
    except (MigrationError, OSError) as error:
        print(f"{parsed.action}: {error}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
