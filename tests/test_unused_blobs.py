import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import unused_blobs
from arty_engine.store import ModelStore
from fake_store import QWEN3, add_model, build_store


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
    assert unused_blobs.main(["--store", str(root), "--remove"]) == 0
    assert not leftover.exists()
    store = ModelStore([root])
    assert all(model.model_path.is_file() for model in store.models())
