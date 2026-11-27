import hashlib
import json
import os
import shutil
import struct
import threading
import time

import pytest

from arty_engine.store import (TEMPLATE_LAYER, ModelNotFound, ModelStore, blob_path, engine_manifest, parse_name,
                               write_manifest)
from arty_engine.gguf import GgufError
from fake_store import QWEN3, QWEN3_TEMPLATE, add_model, build_store
from gguf_writer import LATER_CHANGE_TIME_NS, break_first_key_length, write_gguf


@pytest.fixture
def store(tmp_path):
    return ModelStore([tmp_path / "own", build_store(tmp_path / "other")])


def test_lists_every_model_in_the_catalog(store):
    assert sorted(store.names()) == ["looker:9b", "nomic-embed-text:latest", "qwen2.5vl:7b", "qwen3:14b", "untemplated:latest"]


def test_short_and_full_names_find_the_same_model(store):
    assert store.get("qwen3:14b").digest == store.get("library/qwen3:14b").digest
    assert store.get("nomic-embed-text").name == "nomic-embed-text:latest"


def test_digest_is_the_manifest_digest_and_size_counts_layers_and_config(store):
    model = store.get("qwen3:14b")
    manifest = model.root / "catalog" / "local" / "library" / "qwen3" / "14b.json"
    assert model.digest == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert model.size == sum(layer["size"] for layer in model.manifest["layers"]) + model.manifest["config"]["size"]


def test_capabilities_come_from_the_template_and_projector(store):
    assert store.get("qwen3:14b").capabilities == ["completion", "tools", "thinking"]
    assert store.get("looker:9b").capabilities == ["completion", "vision", "tools", "thinking"]
    assert store.get("nomic-embed-text").capabilities == ["embedding"]


def test_a_vision_model_runs_with_its_projector(store):
    model = store.get("looker:9b")
    assert model.projector_path is not None and model.projector_path.exists()
    assert model.runnable


def test_vision_weights_inside_the_model_file_are_listed_but_not_supported(store):
    model = store.get("qwen2.5vl:7b")
    assert not model.runnable
    assert "not supported by ArtyEngine" in model.unrunnable_reason and "projector file" in model.unrunnable_reason
    assert model.system == "You are helpful."


def test_model_params_are_read(store):
    assert store.get("qwen3:14b").params == {"temperature": 0.6, "stop": ["<|im_end|>"]}


def test_template_donor_shares_the_stored_template(store):
    untemplated = store.get("untemplated")
    donor = store.template_donor(untemplated)
    assert donor is not None and donor.name == "qwen3:14b" and donor.gguf.chat_template == QWEN3_TEMPLATE
    assert store.template_donor(store.get("qwen3:14b")) is None


def test_details_match_the_tags_shape(store):
    described = store.get("qwen3:14b").describe()
    assert set(described) == {"name", "model", "modified_at", "size", "digest", "details", "capabilities"}
    assert described["details"]["quantization_level"] == "Q4_K_M"
    assert described["details"]["parameter_size"] == "1M"
    assert described["modified_at"].endswith("Z")


def test_parameter_size_is_the_one_stored_with_the_model(tmp_path):
    add_model(tmp_path, "looker-v2", "9b", QWEN3, {"token_embd.weight": (4, 4)}, config_fields={"model_type": "9.0B"})
    assert ModelStore([tmp_path]).get("looker-v2:9b").details["parameter_size"] == "9.0B"


@pytest.mark.parametrize("name", ["", "../etc/passwd:1", "a/b/c/d:e", "bad name"])
def test_bad_names_are_refused(store, name):
    with pytest.raises(ModelNotFound):
        store.get(name)


def test_hugging_face_names_keep_their_case():
    parsed = parse_name("hf.co/Unsloth/Qwen3-0.6B-GGUF:Q8_0")
    assert (parsed.host, parsed.namespace, parsed.model, parsed.tag) == ("hf.co", "Unsloth", "Qwen3-0.6B-GGUF", "Q8_0")
    assert parsed.display == "hf.co/Unsloth/Qwen3-0.6B-GGUF:Q8_0"


@pytest.mark.parametrize("name", ["example.com/../..:probe", "library/..:14b", "qwen3:..", "./qwen3:14b", "a/./b:c"])
def test_names_that_walk_out_of_the_manifest_tree_are_refused(name):
    with pytest.raises(ModelNotFound, match="invalid model name"):
        parse_name(name)


@pytest.mark.parametrize("name", ["qwen3:14b\n", "qwen3:14b\r\n", "qwen3:14b ", " qwen3:14b", "\tqwen3:14b", "qwen3 :14b", "qwen3:14\nb",
                                  "qwen3:14b\x00", "qwen3\x7f:14b", "qwen3:14b\u2028", "q\u00e9n3:14b"])
def test_names_with_whitespace_or_control_characters_are_refused(name):
    with pytest.raises(ModelNotFound, match="invalid model name"):
        parse_name(name)


def test_names_too_long_for_a_file_name_are_refused():
    assert parse_name("a" * 255).model == "a" * 255
    with pytest.raises(ModelNotFound, match="invalid model name"):
        parse_name("a" * 256)
    with pytest.raises(ModelNotFound, match="invalid model name"):
        parse_name("qwen3:" + "b" * 256)


@pytest.mark.parametrize("name", [None, 5, ["qwen3:14b"], {"model": "qwen3:14b"}])
def test_names_that_are_not_text_are_refused(name):
    with pytest.raises(ModelNotFound, match="invalid model name"):
        parse_name(name)


@pytest.mark.parametrize("digest", ["../../victim.txt", "sha256:../../victim", "", "sha256:" + "a" * 63, "sha256:" + "a" * 64 + "/x"])
def test_blob_paths_need_a_sha256_digest(tmp_path, digest):
    with pytest.raises(ModelNotFound, match="digest"):
        blob_path(tmp_path, digest)


def test_a_manifest_pointing_outside_the_store_is_not_read_or_deleted(tmp_path):
    root = tmp_path / "store"
    victim = tmp_path / "victim.txt"
    victim.write_text("precious")
    path = add_model(root, "evil", "1", QWEN3, {"blk.0.w": (4, 4)})
    manifest = json.loads(path.read_text())
    manifest["layers"].append({"mediaType": TEMPLATE_LAYER, "digest": "../../victim.txt", "size": 8})
    path.write_text(json.dumps(manifest))
    store = ModelStore([root])
    with pytest.raises(ModelNotFound):
        store.get("evil:1")
    store.remove_owned("evil:1")
    assert victim.read_text() == "precious"
    assert not path.exists()


def test_a_store_moved_into_the_engines_store_keeps_every_name_and_digest(tmp_path):
    read_in_place = ModelStore([tmp_path / "own", build_store(tmp_path / "other")])
    before = {model.name: (model.digest, model.size) for model in read_in_place.models()}
    (tmp_path / "other").rename(tmp_path / "moved")
    own_store_only = ModelStore([tmp_path / "moved"])
    assert {model.name: (model.digest, model.size) for model in own_store_only.models()} == before
    assert own_store_only.owned_manifest("qwen3:14b").is_file()


def test_short_names_live_in_the_local_catalog_and_keep_their_display():
    parsed = parse_name("qwen3:14b")
    assert (parsed.host, parsed.namespace, parsed.model, parsed.tag) == ("local", "library", "qwen3", "14b")
    assert parse_name("someone/custom:1").display == "someone/custom:1"


def test_a_manifest_in_another_format_is_not_read(tmp_path):
    path = add_model(tmp_path, "odd", "1", QWEN3, {"blk.0.w": (4, 4)})
    manifest = json.loads(path.read_text())
    manifest["mediaType"] = "application/vnd.docker.distribution.manifest.v2+json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ModelNotFound, match="not an ArtyEngine manifest"):
        ModelStore([tmp_path]).get("odd:1")


def test_written_manifests_are_found_by_name(tmp_path):
    write_manifest(tmp_path, parse_name("hf.co/Someone/M-GGUF:Q8_0"), engine_manifest([]))
    assert ModelStore([tmp_path]).names() == ["hf.co/Someone/M-GGUF:Q8_0"]


@pytest.mark.parametrize("name", ["a//b", "a/", "x/:t", "hf.co//x", "/a", "a//b/c:d"])
def test_names_with_an_empty_part_are_refused(name):
    with pytest.raises(ModelNotFound, match="invalid model name"):
        parse_name(name)


DAMAGED_MANIFESTS = {
    "text media type": lambda good: {**good, "layers": good["layers"] + [{**good["layers"][-1], "mediaType": 5}]},
    "list media type": lambda good: {**good, "layers": good["layers"] + [{**good["layers"][-1], "mediaType": ["x"]}]},
    "layer not an object": lambda good: {**good, "layers": good["layers"] + [5]},
    "layers not a list": lambda good: {**good, "layers": "x"},
    "config not an object": lambda good: {**good, "config": 5},
    "manifest not an object": lambda good: [good],
    "list entries": lambda good: {"manifests": [5]},
    "size not a number": lambda good: {**good, "layers": [{**layer, "size": "big"} for layer in good["layers"]]},
}


@pytest.mark.parametrize("damage", sorted(DAMAGED_MANIFESTS))
def test_a_damaged_manifest_hides_only_its_own_model_and_can_be_deleted(tmp_path, damage, caplog):
    root = build_store(tmp_path / "own")
    path = add_model(root, "broken", "1", QWEN3, {"blk.0.w": (4, 4)})
    path.write_text(json.dumps(DAMAGED_MANIFESTS[damage](json.loads(path.read_text()))))
    store = ModelStore([root])
    assert {model.name for model in store.models()} == set(store.names()) - {"broken:1"}
    assert "broken:1" in caplog.text
    store.remove_owned("broken:1")
    assert "broken:1" not in store.names()
    store.remove_owned("untemplated:latest")
    assert store.get("qwen3:14b").name == "qwen3:14b"


def test_deleting_works_while_another_manifest_is_damaged(tmp_path):
    root = build_store(tmp_path / "own")
    path = add_model(root, "broken", "1", QWEN3, {"blk.0.w": (4, 4)})
    path.write_text(json.dumps({"layers": [5, {"mediaType": ["x"], "digest": 5}], "config": 5}))
    removed = ModelStore([root]).remove_owned("untemplated:latest")
    assert removed and "untemplated:latest" not in ModelStore([root]).names()


@pytest.mark.parametrize("text", ["", "{", "[" * 200000])
def test_a_manifest_that_is_not_json_is_skipped_deleted_and_never_blocks_another_delete(tmp_path, text, caplog):
    root = build_store(tmp_path / "own")
    path = add_model(root, "broken", "1", QWEN3, {"blk.0.w": (4, 4)})
    path.write_text(text)
    store = ModelStore([root])
    assert "broken:1" not in {model.name for model in store.models()}
    assert store.remove_owned("untemplated:latest")
    assert "couldn't read" in caplog.text
    store.remove_owned("broken:1")
    assert not path.exists() and "broken:1" not in ModelStore([root]).names()
    assert store.get("qwen3:14b").name == "qwen3:14b"


READ_WAIT_SECONDS = 0.3
READERS_AT_ONCE = 4


def counted_reads(monkeypatch, wait: float = 0.0) -> list:
    from arty_engine import store as store_module
    real_read = store_module.read_gguf
    reads = []

    def counting_read(path, *rest):
        reads.append(str(path))
        time.sleep(wait)
        return real_read(path, *rest)

    monkeypatch.setattr(store_module, "read_gguf", counting_read)
    return reads


def broken_model_file(root):
    add_model(root, "broken", "1", QWEN3, {"blk.0.w": (4, 4)})
    path = ModelStore([root]).get("broken:1").model_path
    return break_first_key_length(path, 2 ** 40)


def test_a_failed_read_is_kept_so_every_listing_does_not_read_the_file_again(tmp_path, monkeypatch, caplog):
    root = build_store(tmp_path / "own")
    path = broken_model_file(root)
    store = ModelStore([root])
    reads = counted_reads(monkeypatch)
    for _ in range(3):
        assert "broken:1" not in {model.name for model in store.models()}
    assert reads.count(str(path)) == 1
    assert caplog.text.count("left broken:1 out of the model list") == 3
    assert caplog.text.count("larger than the rest of the file") == 3


def test_a_failed_read_is_read_again_once_the_file_changes(tmp_path, monkeypatch):
    root = build_store(tmp_path / "own")
    path = broken_model_file(root)
    store = ModelStore([root])
    with pytest.raises(GgufError):
        store.get("broken:1")
    write_gguf(path, QWEN3, {"blk.0.w": (4, 4)})
    os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + 2 * LATER_CHANGE_TIME_NS))
    assert store.get("broken:1").gguf.architecture == "qwen3"


@pytest.mark.parametrize("name", ["qwen3:14b", "broken:1"])
def test_requests_at_once_read_one_model_file_once(tmp_path, monkeypatch, name):
    root = build_store(tmp_path / "own")
    paths = {"qwen3:14b": str(ModelStore([root]).get("qwen3:14b").model_path), "broken:1": str(broken_model_file(root))}
    store = ModelStore([root])
    reads = counted_reads(monkeypatch, READ_WAIT_SECONDS)
    start = threading.Barrier(READERS_AT_ONCE)
    outcomes = []

    def read_once():
        start.wait()
        try:
            outcomes.append(store.get(name).name)
        except GgufError as error:
            outcomes.append(type(error).__name__)

    readers = [threading.Thread(target=read_once) for _ in range(READERS_AT_ONCE)]
    for reader in readers:
        reader.start()
    for reader in readers:
        reader.join()
    assert reads.count(paths[name]) == 1
    assert len(outcomes) == READERS_AT_ONCE and len(set(outcomes)) == 1


DEEP_ARRAY_LEVELS = 3000
LISTINGS_AT_ONCE = 3
SLOW_READ_SECONDS = 2.0
UINT8, ARRAY = 0, 9


def deeply_nested_model_file(root):
    add_model(root, "deep", "1", QWEN3, {"blk.0.w": (4, 4)})
    path = ModelStore([root]).get("deep:1").model_path
    path.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 1) + b"k" + struct.pack("<I", ARRAY)
                     + struct.pack("<IQ", ARRAY, 1) * (DEEP_ARRAY_LEVELS - 1) + struct.pack("<IQ", UINT8, 0))
    return path


def kept_refusals(store) -> list:
    found = []
    for entry in store._gguf_cache.values():
        found += [item for item in (entry if isinstance(entry, tuple) else (entry,)) if isinstance(item, GgufError)]
    return found


def test_listings_at_once_read_a_file_nested_too_deep_once(tmp_path, monkeypatch, caplog):
    root = build_store(tmp_path / "own")
    path = deeply_nested_model_file(root)
    store = ModelStore([root])
    reads = counted_reads(monkeypatch, READ_WAIT_SECONDS)
    start = threading.Barrier(LISTINGS_AT_ONCE)
    listed = []

    def list_once():
        start.wait()
        listed.append({model.name for model in store.models()})

    listings = [threading.Thread(target=list_once) for _ in range(LISTINGS_AT_ONCE)]
    for listing in listings:
        listing.start()
    for listing in listings:
        listing.join()
    assert len(listed) == LISTINGS_AT_ONCE and all("deep:1" not in names and "qwen3:14b" in names for names in listed)
    assert reads.count(str(path)) == 1
    assert "left deep:1 out of the model list: arrays in the GGUF header are nested" in caplog.text


def test_a_kept_refusal_holds_no_traceback(tmp_path):
    root = build_store(tmp_path / "own")
    broken_model_file(root)
    store = ModelStore([root])
    for _ in range(2):
        with pytest.raises(GgufError):
            store.get("broken:1")
    refusals = kept_refusals(store)
    assert len(refusals) == 1
    assert refusals[0].__traceback__ is None and refusals[0].__context__ is None and refusals[0].__cause__ is None


def test_an_os_error_while_reading_is_not_kept(tmp_path, monkeypatch):
    from arty_engine import store as store_module
    root = build_store(tmp_path / "own")
    store = ModelStore([root])
    real_read = store_module.read_gguf
    failures = [OSError("the disk could not be read")]

    def read_that_fails_once(path, *rest):
        if failures:
            raise failures.pop()
        return real_read(path, *rest)

    monkeypatch.setattr(store_module, "read_gguf", read_that_fails_once)
    with pytest.raises(OSError, match="could not be read"):
        store.get("qwen3:14b")
    assert store.get("qwen3:14b").gguf.architecture == "qwen3"


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a file without read permission")
def test_a_file_without_read_permission_is_read_again_once_permission_is_back(tmp_path):
    root = build_store(tmp_path / "own")
    path = ModelStore([root]).get("qwen3:14b").model_path
    store = ModelStore([root])
    before = path.stat()
    path.chmod(0)
    try:
        for _ in range(2):
            with pytest.raises(PermissionError):
                store.get("qwen3:14b")
    finally:
        path.chmod(0o644)
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert store.get("qwen3:14b").gguf.architecture == "qwen3"


def test_a_file_replaced_just_before_the_read_is_kept_under_its_own_stamp(tmp_path, monkeypatch):
    from arty_engine import gguf as gguf_module, store as store_module
    root = build_store(tmp_path / "own")
    path = ModelStore([root]).get("qwen3:14b").model_path
    good = tmp_path / "good.gguf"
    shutil.copy2(path, good)
    broken = tmp_path / "broken.gguf"
    shutil.copy2(path, broken)
    break_first_key_length(broken, 2 ** 40)
    store = ModelStore([root])
    real_open = open
    swaps = [broken]

    def open_after_a_swap(file, *rest, **named):
        if str(file) == str(path) and swaps:
            os.replace(swaps.pop(), path)
        return real_open(file, *rest, **named)

    monkeypatch.setattr(gguf_module, "open", open_after_a_swap, raising=False)
    monkeypatch.setattr(store_module, "open", open_after_a_swap, raising=False)
    with pytest.raises(GgufError):
        store.get("qwen3:14b")
    os.replace(good, path)
    assert store.get("qwen3:14b").gguf.architecture == "qwen3"


def test_a_file_renamed_into_place_with_the_old_size_and_change_time_is_read_again(tmp_path):
    root = build_store(tmp_path / "own")
    path = ModelStore([root]).get("qwen3:14b").model_path
    store = ModelStore([root])
    before = path.stat()
    assert store.get("qwen3:14b").gguf.architecture == "qwen3"
    copy = tmp_path / "copy.gguf"
    shutil.copy2(path, copy)
    break_first_key_length(copy, 2 ** 40)
    os.utime(copy, ns=(before.st_atime_ns, before.st_mtime_ns))
    os.replace(copy, path)
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (before.st_size, before.st_mtime_ns)
    with pytest.raises(GgufError):
        store.get("qwen3:14b")


def test_one_entry_is_kept_per_model_file_however_often_it_changes(tmp_path):
    root = build_store(tmp_path / "own")
    path = broken_model_file(root)
    store = ModelStore([root])
    for change in range(5):
        os.utime(path, ns=(path.stat().st_atime_ns, path.stat().st_mtime_ns + (change + 1) * LATER_CHANGE_TIME_NS))
        with pytest.raises(GgufError):
            store.get("broken:1")
    assert len(store._gguf_cache) == 1


def test_a_slow_read_of_one_file_does_not_hold_up_another(tmp_path, monkeypatch):
    from arty_engine import store as store_module
    root = build_store(tmp_path / "own")
    slow_path = str(broken_model_file(root))
    store = ModelStore([root])
    real_read = store_module.read_gguf
    slow_read_started = threading.Event()

    def read_slowly_for_one_file(path, *rest):
        if str(path) == slow_path:
            slow_read_started.set()
            time.sleep(SLOW_READ_SECONDS)
        return real_read(path, *rest)

    monkeypatch.setattr(store_module, "read_gguf", read_slowly_for_one_file)

    outcomes = []

    def read_the_slow_file():
        try:
            store.get("broken:1")
        except GgufError as error:
            outcomes.append(type(error).__name__)

    slow = threading.Thread(target=read_the_slow_file)
    slow.start()
    assert slow_read_started.wait(SLOW_READ_SECONDS)
    started = time.monotonic()
    assert store.get("qwen3:14b").gguf.architecture == "qwen3"
    took = time.monotonic() - started
    slow.join()
    assert outcomes == ["GgufError"] and took < SLOW_READ_SECONDS / 2
