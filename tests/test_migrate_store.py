import errno
import hashlib
import json
import os
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import migrate_store
from arty_engine.store import MANIFEST_MEDIA_TYPE, MEDIA_PREFIX, ModelStore
from fake_store import QWEN3, add_model
from old_layout_store import EXPECTED_NAMES, OLD_TIME_NS, REGISTRY_HOST, SHORT_HOST, VENDOR, add_old_model, build_old_store, write_old


@pytest.fixture
def old_store(tmp_path):
    return build_old_store(tmp_path / "models")


def blob_listing(root: Path) -> dict:
    return {path.name: (path.stat().st_ino, path.stat().st_mtime_ns) for path in (root / "blobs").iterdir()}


def model_blob_listing(root: Path) -> dict:
    return {name: entry for name, entry in blob_listing(root).items() if not is_old_document(root / "blobs" / name)}


def is_old_document(path: Path) -> bool:
    return path.stat().st_size < 4096 and VENDOR.encode() in path.read_bytes() and b"schemaVersion" in path.read_bytes()


def snapshot(root: Path) -> dict:
    entries = {}
    for folder, folders, files in os.walk(root):
        for name in [*folders, *files, ""] if folder == str(root) else [*folders, *files]:
            path = Path(folder) / name
            status = path.lstat()
            entry = (status.st_mode, status.st_uid, status.st_mtime_ns, status.st_size)
            if path.is_symlink():
                entry += (os.readlink(path),)
            elif path.is_file():
                entry += (hashlib.sha256(path.read_bytes()).hexdigest(),)
            entries[path.relative_to(root).as_posix()] = entry
    return entries


def test_the_engine_does_not_read_the_old_layout(old_store):
    assert ModelStore([old_store]).names() == []


def test_migration_keeps_every_name_and_leaves_the_model_blobs_alone(old_store, tmp_path):
    blobs_before = model_blob_listing(old_store)
    migrate_store.migrate(old_store, tmp_path / "backup")
    store = ModelStore([old_store])
    assert set(store.names()) == EXPECTED_NAMES
    assert model_blob_listing(old_store) == blobs_before
    assert not (old_store / "manifests").exists() and not (old_store / "manifests-v2").exists()
    assert not (old_store / "metadata").exists()


def test_migrated_manifests_use_only_the_engines_media_types(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    for path in (old_store / "catalog").rglob("*.json"):
        text = path.read_text()
        manifest = json.loads(text)
        assert manifest["mediaType"] == MANIFEST_MEDIA_TYPE
        assert all(layer["mediaType"].startswith(MEDIA_PREFIX) for layer in manifest["layers"])
        assert VENDOR not in text and "docker" not in text


def test_migrated_models_keep_their_parts_sizes_and_times(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    store = ModelStore([old_store])
    qwen3 = store.get("qwen3:14b")
    assert qwen3.capabilities == ["completion", "tools", "thinking"]
    assert qwen3.details["parameter_size"] == "14.8B" and qwen3.params == {"temperature": 0.6}
    assert qwen3.size == sum(layer["size"] for layer in qwen3.manifest["layers"]) + qwen3.manifest["config"]["size"]
    assert int(qwen3.modified_at.timestamp() * 1e9) == pytest.approx(OLD_TIME_NS, abs=1000)
    mimo = store.get("mimo-v2.6:9b")
    assert mimo.projector_path is not None and mimo.projector_path.is_file() and "vision" in mimo.capabilities
    vision = store.get("qwen2.5vl:7b")
    assert not vision.runnable and vision.system == "You are helpful."
    assert "not supported by ArtyEngine" in vision.unrunnable_reason
    assert store.get("nomic-embed-text").capabilities == ["embedding"]


def test_the_newer_layout_folder_wins_when_a_name_is_in_both_old_folders(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    layers = ModelStore([old_store]).get("qwen3:14b").manifest["layers"]
    assert [layer["mediaType"].removeprefix(MEDIA_PREFIX) for layer in layers] == ["model", "template", "params"]


def test_undo_restores_the_old_layout_byte_for_byte(old_store, tmp_path):
    before = snapshot(old_store)
    blobs_before = blob_listing(old_store)
    migrate_store.migrate(old_store, tmp_path / "backup")
    assert snapshot(old_store) != before
    migrate_store.undo(old_store, tmp_path / "backup")
    assert snapshot(old_store) == before
    assert blob_listing(old_store) == blobs_before
    assert not (old_store / "catalog").exists()
    assert (tmp_path / "backup" / "copy" / "manifests-v2").is_dir()


def test_migration_runs_again_after_an_undo(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "first")
    migrate_store.undo(old_store, tmp_path / "first")
    migrate_store.migrate(old_store, tmp_path / "second")
    assert set(ModelStore([old_store]).names()) == EXPECTED_NAMES


def test_undo_refuses_when_a_model_was_pulled_after_the_migration(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    add_model(old_store, "later", "1", QWEN3, {"blk.0.w": (4, 4)})
    with pytest.raises(migrate_store.MigrationError, match="added after the migration"):
        migrate_store.undo(old_store, tmp_path / "backup")
    assert not (old_store / "manifests-v2").exists()


def test_undo_refuses_when_a_migrated_manifest_changed(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    path = old_store / "catalog" / "local" / "library" / "qwen3" / "14b.json"
    path.write_text(path.read_text() + " ")
    with pytest.raises(migrate_store.MigrationError, match="changed since the migration"):
        migrate_store.undo(old_store, tmp_path / "backup")


def test_migration_needs_an_empty_backup_folder_and_changes_nothing_otherwise(old_store, tmp_path):
    (tmp_path / "backup").mkdir()
    (tmp_path / "backup" / "something").write_text("x")
    before = migrate_store.describe_tree(old_store, migrate_store.OLD_FOLDERS)
    with pytest.raises(migrate_store.MigrationError, match="not empty"):
        migrate_store.migrate(old_store, tmp_path / "backup")
    assert migrate_store.describe_tree(old_store, migrate_store.OLD_FOLDERS) == before
    assert not (old_store / "catalog").exists()


def test_a_manifest_naming_a_missing_blob_stops_the_migration_before_any_change(old_store, tmp_path):
    path = old_store / "manifests-v2" / "hf.co" / "Someone" / "Tiny-GGUF" / "Q8_0"
    manifest = json.loads(path.read_text())
    manifest["layers"][0]["digest"] = "sha256:" + "1" * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(migrate_store.MigrationError, match="does not hold"):
        migrate_store.migrate(old_store, tmp_path / "backup")
    assert not (old_store / "catalog").exists() and not (tmp_path / "backup").exists()


def test_the_command_line_plans_migrates_and_undoes(old_store, tmp_path, capsys):
    closed = ["--engine-url", "http://127.0.0.1:9"]
    assert migrate_store.main(["plan", "--store", str(old_store)]) == 0
    assert "qwen3:14b: manifests-v2/" in capsys.readouterr().out
    assert migrate_store.main(["migrate", "--store", str(old_store), "--backup", str(tmp_path / "backup"), *closed]) == 0
    assert migrate_store.main(["undo", "--store", str(old_store), "--backup", str(tmp_path / "backup"), *closed]) == 0
    assert migrate_store.main(["undo", "--store", str(old_store), "--backup", str(tmp_path / "other"), *closed]) == 1


def catalog_paths(root: Path) -> list[str]:
    return [path.relative_to(root).as_posix() for path in (root / "catalog").rglob("*")]


def test_names_outside_the_library_keep_the_form_clients_use(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    store = ModelStore([old_store])
    assert store.get("someone/custom:1").name == "someone/custom:1"
    assert not any(SHORT_HOST in name or REGISTRY_HOST in name for name in store.names())
    assert not any(SHORT_HOST in path or REGISTRY_HOST in path for path in catalog_paths(old_store))


def test_a_folder_with_two_registries_needs_the_default_named_and_never_merges_them(old_store, tmp_path):
    own = add_old_model(old_store, "manifests-v2", SHORT_HOST, "library", "plain", "1", rows=12)
    other = add_old_model(old_store, "manifests-v2", "other.example", "library", "plain", "1", rows=14)
    with pytest.raises(migrate_store.MigrationError, match="--default-host"):
        migrate_store.migrate(old_store, tmp_path / "refused")
    assert not (old_store / "catalog").exists()
    migrate_store.migrate(old_store, tmp_path / "backup", default_hosts=(SHORT_HOST,))
    store = ModelStore([old_store])
    assert store.get("plain:1").manifest["layers"][0]["digest"] == own["layers"][0]["digest"]
    assert store.get("other.example/library/plain:1").manifest["layers"][0]["digest"] == other["layers"][0]["digest"]


def test_two_old_names_that_would_become_one_are_refused_before_any_change(old_store, tmp_path):
    add_old_model(old_store, "manifests-v2", SHORT_HOST, "library", "Plain", "1", rows=12)
    add_old_model(old_store, "manifests-v2", SHORT_HOST, "library", "plain", "1", rows=14)
    with pytest.raises(migrate_store.MigrationError, match="would both become plain:1"):
        migrate_store.migrate(old_store, tmp_path / "backup")
    assert not (old_store / "catalog").exists() and not (tmp_path / "backup").exists()


def test_plan_names_the_older_copy_it_leaves_in_the_backup(old_store):
    lines = migrate_store.dry_run(old_store)
    assert any("qwen3:14b" in line and "older copy" in line and "manifests/" in line for line in lines)


def test_the_old_manifest_documents_leave_the_blobs_folder_and_come_back_on_undo(old_store, tmp_path):
    documents = [name for name in blob_listing(old_store) if is_old_document(old_store / "blobs" / name)]
    assert documents
    migrate_store.migrate(old_store, tmp_path / "backup")
    assert not any(is_old_document(path) for path in (old_store / "blobs").iterdir())
    assert all((tmp_path / "backup" / "old-manifest-blobs" / name).is_file() for name in documents)
    assert set(ModelStore([old_store]).names()) == EXPECTED_NAMES


def test_tidy_moves_the_documents_of_an_earlier_migration_and_untidy_puts_them_back(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    migrate_store.untidy(old_store, tmp_path / "backup")
    restored = snapshot(old_store)
    assert any(is_old_document(path) for path in (old_store / "blobs").iterdir())
    migrate_store.tidy(old_store, tmp_path / "backup")
    assert not any(is_old_document(path) for path in (old_store / "blobs").iterdir())
    migrate_store.untidy(old_store, tmp_path / "backup")
    assert snapshot(old_store) == restored


def test_a_tidied_backup_is_refused_before_anything_changes(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    shutil.rmtree(tmp_path / "backup" / "moved")
    shutil.rmtree(tmp_path / "backup" / "copy")
    before = snapshot(old_store)
    with pytest.raises(migrate_store.MigrationError, match="nothing was undone"):
        migrate_store.undo(old_store, tmp_path / "backup")
    assert snapshot(old_store) == before
    assert set(ModelStore([old_store]).names()) == EXPECTED_NAMES


def test_undo_takes_the_old_folders_from_the_copy_when_the_moved_ones_are_gone(old_store, tmp_path):
    before = snapshot(old_store)
    migrate_store.migrate(old_store, tmp_path / "backup")
    shutil.rmtree(tmp_path / "backup" / "moved")
    migrate_store.undo(old_store, tmp_path / "backup")
    assert snapshot(old_store) == before


def test_a_backup_that_does_not_match_is_refused_before_anything_changes(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    (tmp_path / "backup" / "moved" / "metadata" / "cache.json").write_text("{}")
    before = snapshot(old_store)
    with pytest.raises(migrate_store.MigrationError, match="does not match"):
        migrate_store.undo(old_store, tmp_path / "backup")
    assert snapshot(old_store) == before


def interrupt_writes(monkeypatch, after: int):
    real = migrate_store.write_manifest
    written = []

    def failing(root, name, manifest):
        if len(written) == after:
            name.manifest_path(root).parent.mkdir(parents=True, exist_ok=True)
            name.manifest_path(root).with_name(name.manifest_path(root).name + ".partial").write_text("{")
            raise OSError(errno.ENOSPC, "No space left on device")
        written.append(name)
        return real(root, name, manifest)

    monkeypatch.setattr(migrate_store, "write_manifest", failing)


def test_a_migration_stopped_part_way_is_undone_with_its_own_backup(old_store, tmp_path, monkeypatch):
    before = snapshot(old_store)
    interrupt_writes(monkeypatch, 2)
    with pytest.raises(OSError):
        migrate_store.migrate(old_store, tmp_path / "first")
    monkeypatch.undo()
    with pytest.raises(migrate_store.MigrationError, match="run undo with"):
        migrate_store.migrate(old_store, tmp_path / "second")
    migrate_store.undo(old_store, tmp_path / "first")
    assert snapshot(old_store) == before


def test_the_backup_of_a_run_that_never_moved_the_folders_is_refused(old_store, tmp_path, monkeypatch):
    interrupt_writes(monkeypatch, 0)
    with pytest.raises(OSError):
        migrate_store.migrate(old_store, tmp_path / "first")
    monkeypatch.undo()
    migrate_store.undo(old_store, tmp_path / "first")
    migrate_store.migrate(old_store, tmp_path / "second")
    before = snapshot(old_store)
    with pytest.raises(migrate_store.MigrationError, match="nothing was undone"):
        migrate_store.undo(old_store, tmp_path / "first")
    assert snapshot(old_store) == before


def test_a_stop_between_the_folder_moves_is_undone(old_store, tmp_path, monkeypatch):
    before = snapshot(old_store)
    real = os.rename
    moves = []

    def failing(source, target):
        if Path(target).parent.name == "moved" and not Path(source).name.startswith(".migrate-probe"):
            if moves:
                raise OSError(errno.EIO, "stopped")
            moves.append(source)
        return real(source, target)

    monkeypatch.setattr(migrate_store.os, "rename", failing)
    with pytest.raises(OSError):
        migrate_store.migrate(old_store, tmp_path / "backup")
    monkeypatch.undo()
    assert set(ModelStore([old_store]).names()) == EXPECTED_NAMES
    migrate_store.undo(old_store, tmp_path / "backup")
    assert snapshot(old_store) == before


def test_a_store_the_backup_cannot_be_renamed_into_is_refused_before_writing(old_store, tmp_path, monkeypatch):
    def cross_device(source, target):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(migrate_store.os, "rename", cross_device)
    with pytest.raises(migrate_store.MigrationError, match="can't move folders"):
        migrate_store.migrate(old_store, tmp_path / "backup")
    monkeypatch.undo()
    assert not (old_store / "catalog").exists()
    assert not any(path.name.startswith(".migrate-probe") for path in old_store.iterdir())


def test_a_torn_migrated_manifest_does_not_block_undo(old_store, tmp_path):
    before = snapshot(old_store)
    migrate_store.migrate(old_store, tmp_path / "backup")
    (old_store / "catalog" / "local" / "library" / "qwen3" / "14b.json").write_text("")
    migrate_store.undo(old_store, tmp_path / "backup")
    assert snapshot(old_store) == before


DAMAGED_OLD = {
    "layers not a list": {"schemaVersion": 2, "layers": 5},
    "layer not an object": {"schemaVersion": 2, "layers": [5]},
    "size not a number": None,
    "list entry not an object": {"schemaVersion": 2, "mediaType": f"{VENDOR}.manifest.list.v2+json", "manifests": [5]},
    "two builds": None,
}


@pytest.mark.parametrize("damage", sorted(DAMAGED_OLD))
def test_a_damaged_old_manifest_is_named_and_can_be_skipped(old_store, tmp_path, damage):
    manifest = DAMAGED_OLD[damage]
    if damage == "size not a number":
        good = json.loads((old_store / "manifests" / REGISTRY_HOST / "library" / "nomic-embed-text" / "latest").read_text())
        manifest = {**good, "layers": [{**layer, "size": "12"} for layer in good["layers"]]}
    if damage == "two builds":
        good = json.loads((old_store / "manifests" / REGISTRY_HOST / "library" / "nomic-embed-text" / "latest").read_text())
        digest = "sha256:" + hashlib.sha256(json.dumps(good).encode()).hexdigest()
        (old_store / "blobs" / digest.replace(":", "-")).write_text(json.dumps(good))
        manifest = {"schemaVersion": 2, "mediaType": f"{VENDOR}.manifest.list.v2+json",
                    "manifests": [{"digest": digest, "format": "gguf"}, {"digest": digest, "format": "gguf"}]}
    write_old(old_store, "manifests-v2", SHORT_HOST, "library", "broken", "1", manifest)
    with pytest.raises(migrate_store.MigrationError, match="broken:1"):
        migrate_store.migrate(old_store, tmp_path / "refused")
    assert not (old_store / "catalog").exists()
    migrate_store.migrate(old_store, tmp_path / "backup", skip=("broken:1",))
    assert set(ModelStore([old_store]).names()) == EXPECTED_NAMES


def test_undo_after_a_delete_names_the_model_that_lost_its_blobs(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    ModelStore([old_store]).remove_owned("nomic-embed-text")
    lines = migrate_store.undo(old_store, tmp_path / "backup")
    assert any("nomic-embed-text" in line and "no longer holds" in line for line in lines)
    migrate_store.migrate(old_store, tmp_path / "again", skip=("nomic-embed-text:latest",))
    assert "nomic-embed-text:latest" not in ModelStore([old_store]).names()


def answering_engine():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def test_the_command_line_refuses_while_an_engine_answers(old_store, tmp_path, capsys):
    engine = answering_engine()
    try:
        url = f"http://127.0.0.1:{engine.server_address[1]}"
        assert migrate_store.main(["migrate", "--store", str(old_store), "--backup", str(tmp_path / "b"), "--engine-url", url]) == 1
    finally:
        engine.shutdown()
    assert "stop it first" in capsys.readouterr().err
    assert not (old_store / "catalog").exists()
    assert migrate_store.main(["migrate", "--store", str(old_store), "--backup", str(tmp_path / "b"), "--engine-url", "none"]) == 0
