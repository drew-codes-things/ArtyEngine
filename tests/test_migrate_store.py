import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import migrate_store
from arty_engine.store import MANIFEST_MEDIA_TYPE, MEDIA_PREFIX, ModelStore
from fake_store import QWEN3, add_model
from old_layout_store import EXPECTED_NAMES, OLD_TIME_NS, VENDOR, build_old_store


@pytest.fixture
def old_store(tmp_path):
    return build_old_store(tmp_path / "models")


def blob_listing(root: Path) -> dict:
    return {path.name: (path.stat().st_ino, path.stat().st_mtime_ns) for path in (root / "blobs").iterdir()}


def test_the_engine_does_not_read_the_old_layout(old_store):
    assert ModelStore([old_store]).names() == []


def test_migration_keeps_every_name_and_leaves_the_blobs_alone(old_store, tmp_path):
    blobs_before = blob_listing(old_store)
    migrate_store.migrate(old_store, tmp_path / "backup")
    store = ModelStore([old_store])
    assert set(store.names()) == EXPECTED_NAMES
    assert blob_listing(old_store) == blobs_before
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


def test_the_newer_manifest_wins_when_a_name_is_in_both_old_folders(old_store, tmp_path):
    migrate_store.migrate(old_store, tmp_path / "backup")
    layers = ModelStore([old_store]).get("qwen3:14b").manifest["layers"]
    assert [layer["mediaType"].removeprefix(MEDIA_PREFIX) for layer in layers] == ["model", "template", "params"]


def test_undo_restores_the_old_layout_byte_for_byte(old_store, tmp_path):
    before = migrate_store.describe_tree(old_store, migrate_store.OLD_FOLDERS)
    blobs_before = blob_listing(old_store)
    store_time_before = old_store.lstat().st_mtime_ns
    migrate_store.migrate(old_store, tmp_path / "backup")
    migrate_store.undo(old_store, tmp_path / "backup")
    assert migrate_store.describe_tree(old_store, migrate_store.OLD_FOLDERS) == before
    assert old_store.lstat().st_mtime_ns == store_time_before
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
    assert migrate_store.main(["plan", "--store", str(old_store)]) == 0
    assert "qwen3:14b: manifests-v2/" in capsys.readouterr().out
    assert migrate_store.main(["migrate", "--store", str(old_store), "--backup", str(tmp_path / "backup")]) == 0
    assert migrate_store.main(["undo", "--store", str(old_store), "--backup", str(tmp_path / "backup")]) == 0
    assert migrate_store.main(["undo", "--store", str(old_store), "--backup", str(tmp_path / "other")]) == 1
