import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import migrate_store
import unused_blobs
from arty_engine.store import ModelStore
from fake_store import QWEN3, add_model, build_store
from old_layout_store import EXPECTED_NAMES, REGISTRY_HOST, _layer, build_old_store, write_old

NO_ENGINE = ["--engine-url", "none"]


def test_lists_partials_and_blobs_no_manifest_names(tmp_path):
    root = build_store(tmp_path / "own")
    (root / "blobs" / ("sha256-" + "a" * 64)).write_bytes(b"left over")
    (root / "blobs" / ("sha256-" + "b" * 64 + "-partial")).write_bytes(b"half")
    found = unused_blobs.unused(root)
    assert sorted(path.name for path in found) == ["sha256-" + "a" * 64, "sha256-" + "b" * 64 + "-partial"]


def test_the_blobs_of_an_unreadable_manifest_are_never_listed(tmp_path):
    root = build_store(tmp_path / "own")
    path = add_model(root, "broken", "1", QWEN3, {"blk.0.w": (4, 4)})
    path.write_text("{")
    assert unused_blobs.unused(root) == []


def test_remove_deletes_only_what_it_lists_and_the_models_still_load(tmp_path, capsys):
    root = build_store(tmp_path / "own")
    leftover = root / "blobs" / ("sha256-" + "a" * 64)
    leftover.write_bytes(b"left over")
    assert unused_blobs.main(["--store", str(root)]) == 0
    assert leftover.name in capsys.readouterr().out and leftover.exists()
    assert unused_blobs.main(["--store", str(root), "--remove", *NO_ENGINE]) == unused_blobs.REFUSED
    assert leftover.exists()
    assert unused_blobs.main(["--store", str(root), "--remove", "--remove-unnamed-blobs", *NO_ENGINE]) == 0
    assert not leftover.exists()
    store = ModelStore([root])
    assert all(model.model_path.is_file() for model in store.models())


def blob_count(root: Path) -> int:
    return len(list((root / "blobs").iterdir()))


def test_a_store_in_the_older_layout_is_never_cleaned(tmp_path, capsys):
    root = build_old_store(tmp_path / "models")
    before = blob_count(root)
    assert unused_blobs.main(["--store", str(root), "--remove", *NO_ENGINE]) == unused_blobs.REFUSED
    assert blob_count(root) == before
    assert "migrate_store.py" in capsys.readouterr().err


def test_a_store_whose_migration_stopped_part_way_is_never_cleaned(tmp_path, monkeypatch, capsys):
    root = build_old_store(tmp_path / "models")
    written = []

    def write_one_then_stop(*arguments):
        if written:
            raise OSError("stopped")
        written.append(migrate_store_write(*arguments))
        return written[-1]

    migrate_store_write = migrate_store.write_manifest
    monkeypatch.setattr(migrate_store, "write_manifest", write_one_then_stop)
    with pytest.raises(OSError):
        migrate_store.migrate(root, tmp_path / "backup")
    before = blob_count(root)
    assert unused_blobs.main(["--store", str(root), "--remove", *NO_ENGINE]) == unused_blobs.REFUSED
    assert blob_count(root) == before


def test_the_model_file_of_a_skipped_model_is_kept_unless_asked_for(tmp_path, capsys):
    root = build_old_store(tmp_path / "models")
    migrate_store.migrate(root, tmp_path / "backup", skip=("qwen2.5vl:7b",))
    before = blob_count(root)
    assert unused_blobs.main(["--store", str(root), "--remove", *NO_ENGINE]) == unused_blobs.REFUSED
    output = capsys.readouterr()
    assert blob_count(root) == before and "model file" in output.out and "--remove-unnamed-blobs" in output.err
    assert unused_blobs.main(["--store", str(root), "--remove", "--remove-unnamed-blobs", *NO_ENGINE]) == 0
    assert blob_count(root) < before
    assert {model.name for model in ModelStore([root]).models()} >= EXPECTED_NAMES - {"qwen2.5vl:7b"}


def test_old_manifest_documents_are_named_and_left_for_tidy(tmp_path, capsys):
    root = build_old_store(tmp_path / "models")
    migrate_store.migrate(root, tmp_path / "backup")
    migrate_store.untidy(root, tmp_path / "backup")
    before = blob_count(root)
    assert unused_blobs.main(["--store", str(root), "--remove", *NO_ENGINE]) == unused_blobs.REFUSED
    output = capsys.readouterr()
    assert blob_count(root) == before and "old manifest document" in output.out and "tidy" in output.err


def test_a_store_without_a_catalog_is_never_cleaned(tmp_path):
    root = tmp_path / "models"
    (root / "blobs").mkdir(parents=True)
    (root / "blobs" / ("sha256-" + "a" * 64)).write_bytes(b"GGUF weights")
    assert unused_blobs.main(["--store", str(root), "--remove", *NO_ENGINE]) == unused_blobs.REFUSED
    assert blob_count(root) == 1


def test_the_leftover_of_a_stopped_undo_blocks_removal(tmp_path):
    root = build_store(tmp_path / "own")
    (root / "manifests-v2.restoring").mkdir()
    (root / "blobs" / ("sha256-" + "a" * 64)).write_bytes(b"left over")
    assert unused_blobs.main(["--store", str(root), "--remove", *NO_ENGINE]) == unused_blobs.REFUSED
    assert (root / "blobs" / ("sha256-" + "a" * 64)).exists()


def test_the_own_layers_of_a_skipped_model_that_shares_weights_are_kept(tmp_path, capsys):
    root = build_old_store(tmp_path / "models")
    nomic = json.loads((root / "manifests" / REGISTRY_HOST / "library" / "nomic-embed-text" / "latest").read_text())
    variant = {**nomic, "layers": nomic["layers"] + [_layer(root, "system", "You are a variant.")]}
    write_old(root, "manifests", REGISTRY_HOST, "library", "variant", "1", variant)
    migrate_store.migrate(root, tmp_path / "backup", skip=("variant:1",))
    before = blob_count(root)
    assert unused_blobs.main(["--store", str(root), "--remove", *NO_ENGINE]) == unused_blobs.REFUSED
    assert blob_count(root) == before and "other" in capsys.readouterr().out
    lines = migrate_store.undo(root, tmp_path / "backup")
    assert not any("variant" in line for line in lines)


def test_a_partial_download_is_never_removed_while_an_engine_listens(tmp_path, capsys):
    import socket
    root = build_store(tmp_path / "own")
    partial = root / "blobs" / ("sha256-" + "b" * 64 + "-partial")
    partial.write_bytes(b"half")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    try:
        url = f"127.0.0.1:{listener.getsockname()[1]}"
        assert unused_blobs.main(["--store", str(root), "--remove", "--engine-url", url]) == unused_blobs.REFUSED
    finally:
        listener.close()
    assert partial.exists() and "listens" in capsys.readouterr().err
    assert unused_blobs.main(["--store", str(root), "--remove", *NO_ENGINE]) == 0
    assert not partial.exists()


def test_a_stray_write_leftover_is_named_with_the_way_out(tmp_path, capsys):
    root = build_store(tmp_path / "own")
    manifest = next((root / "catalog").rglob("*.json"))
    manifest.with_name(manifest.name + ".partial").write_text("{")
    assert unused_blobs.main(["--store", str(root), "--remove", *NO_ENGINE]) == unused_blobs.REFUSED
    assert "remove it by hand if no migration or pull is running" in capsys.readouterr().err


def test_the_unnamed_blobs_flag_needs_remove(tmp_path):
    root = build_store(tmp_path / "own")
    with pytest.raises(SystemExit):
        unused_blobs.main(["--store", str(root), "--remove-unnamed-blobs"])
