import json

import pytest

from arty_engine.store import ModelNotFound, ModelStore, blob_path, parse_name
from fake_store import QWEN3, QWEN3_TEMPLATE, add_v2_model, build_store


@pytest.fixture
def store(tmp_path):
    return ModelStore([tmp_path / "own", build_store(tmp_path / "ollama")])


def test_lists_every_model_from_both_manifest_layouts(store):
    assert sorted(store.names()) == ["mimo:9b", "nomic-embed-text:latest", "qwen2.5vl:7b", "qwen3:14b", "untemplated:latest"]


def test_short_and_full_names_find_the_same_model(store):
    assert store.get("qwen3:14b").digest == store.get("ollama.com/library/qwen3:14b").digest
    assert store.get("nomic-embed-text").name == "nomic-embed-text:latest"


def test_digest_is_the_manifest_digest_and_size_counts_layers_and_config(store):
    model = store.get("qwen3:14b")
    assert model.digest == model.root.joinpath("manifests-v2/ollama.com/library/qwen3/14b").resolve().name.removeprefix("sha256-")
    assert model.size == sum(layer["size"] for layer in model.manifest["layers"]) + model.manifest["config"]["size"]


def test_capabilities_follow_ollama(store):
    assert store.get("qwen3:14b").capabilities == ["completion", "tools", "thinking"]
    assert store.get("mimo:9b").capabilities == ["completion", "vision", "tools", "thinking"]
    assert store.get("nomic-embed-text").capabilities == ["embedding"]


def test_manifest_list_picks_the_gguf_build_and_its_projector(store):
    model = store.get("mimo:9b")
    assert model.projector_path is not None and model.projector_path.exists()
    assert model.runnable


def test_ollama_engine_vision_models_are_listed_but_not_runnable(store):
    model = store.get("qwen2.5vl:7b")
    assert not model.runnable
    assert "Ollama's own engine format" in model.unrunnable_reason
    assert model.system == "You are helpful."


def test_model_params_are_read(store):
    assert store.get("qwen3:14b").params == {"temperature": 0.6, "stop": ["<|im_end|>"]}


def test_template_donor_shares_the_ollama_template(store):
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


def test_parameter_size_is_the_one_ollama_stored_with_the_model(tmp_path):
    add_v2_model(tmp_path, "mimo-v2.6", "9b", QWEN3, {"token_embd.weight": (4, 4)}, config_fields={"model_type": "9.0B"})
    assert ModelStore([tmp_path]).get("mimo-v2.6:9b").details["parameter_size"] == "9.0B"


@pytest.mark.parametrize("name", ["", "../etc/passwd:1", "a/b/c/d:e", "bad name"])
def test_bad_names_are_refused(store, name):
    with pytest.raises(ModelNotFound):
        store.get(name)


def test_hugging_face_names_keep_their_case():
    parsed = parse_name("hf.co/Unsloth/Qwen3-0.6B-GGUF:Q8_0")
    assert (parsed.host, parsed.namespace, parsed.model, parsed.tag) == ("hf.co", "Unsloth", "Qwen3-0.6B-GGUF", "Q8_0")
    assert parsed.display == "hf.co/Unsloth/Qwen3-0.6B-GGUF:Q8_0"


@pytest.mark.parametrize("name", ["ollama.com/../..:probe", "library/..:14b", "qwen3:..", "./qwen3:14b", "a/./b:c"])
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
    add_v2_model(root, "evil", "1", QWEN3, {"blk.0.w": (4, 4)})
    path = root / "manifests-v2" / "ollama.com" / "library" / "evil" / "1"
    manifest = json.loads(path.read_text())
    manifest["layers"].append({"mediaType": "application/vnd.ollama.image.template", "digest": "../../victim.txt", "size": 8})
    path.unlink()
    path.write_text(json.dumps(manifest))
    store = ModelStore([root])
    with pytest.raises(ModelNotFound):
        store.get("evil:1")
    store.remove_owned("evil:1")
    assert victim.read_text() == "precious"
    assert not path.exists()
