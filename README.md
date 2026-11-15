<div align="center">

# ArtyEngine

**Arty's own model engine. It serves models from its own store, keeps them inside a VRAM budget, pulls new ones
from Hugging Face and runs speculative decoding, behind one HTTP API.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-standard%20library-brightgreen?style=flat-square)](#install)
[![License](https://img.shields.io/badge/license-PolyForm%20Strict%201.0.0-orange?style=flat-square)](LICENSE)

</div>

---

## Overview

Arty is a self-hosted assistant. ArtyEngine is the server its brain, its tool hub and its web chat send every model
request to, through the engine's HTTP API (`/api/chat`, `/api/generate`, `/api/embed` and the rest, listed below).

- **Its own model store.** Every model has one manifest in the store's `catalog` folder, naming its weights and
  extras (projector, template, parameters, system prompt, license) by sha256; the files themselves sit once each in
  `blobs`. Models keep short names such as `qwen3:14b`, and pulled ones are named after their Hugging Face repository.
- **One runtime process per model.** For each loaded model the engine starts a prebuilt model runtime (see
  [NOTICE](NOTICE)) on a local port, so a crash takes down one model, not the server. The runtime is downloaded
  separately and never copied into this repository.
- **VRAM budget swapping.** Before a model loads, the engine estimates what it needs (weights, KV cache, overhead)
  and unloads the least recently used idle models until it fits `vram_budget_mb` and the card's free memory.
  `keep_alive` says how long a model stays loaded.
- **Hugging Face pulls.** `hf.co/<owner>/<repo>:<quant>` picks the matching GGUF (and a vision projector when the
  repository has one, F16 first), checks its sha256, resumes partial downloads and keeps to a bandwidth cap. Short names such
  as `qwen3:14b` or `nomic-embed-text` are pulled from the Hugging Face repository `arty_engine/short_names.json`
  lists for them and keep the short name. Pulled models go into the engine's own store. Hugging Face is the only
  source.
- **Speculative decoding.** Per model: a small draft model or n-gram speculation (`spec_type`, for example
  `ngram-mod`), which needs no draft model at all.
- **Answers the way Arty expects.** Thinking comes back as `message.thinking`, tool call arguments come back as
  objects, tool definitions are written into the prompt compactly, and `/api/generate` with an empty prompt loads or
  unloads a model.
- **Optional trainer hand-off.** With `trainer_url` set, the engine tells a GPU-sharing controller `/busy` before a
  request and `/done` after it, so something else (such as a training job) can use the card while the engine is idle.

The engine is Python standard library only. It reads the card's free memory with `nvidia-smi` when it is there.

---

## Requirements

- **Linux with an NVIDIA GPU**, with `nvidia-smi` on the `PATH`. Built and tested on an RTX 3060 (12 GB) in a Debian 13
  container.
- **Python 3.10 or newer.** No packages to install. The tests need `pytest`.
- **The model runtime**, a prebuilt release for your GPU (see Install). If the CUDA runtime libraries it needs
  (`libcudart`, `libcublas`) are not on the system, put them in a folder and list it in `library_path`.

---

## Install

The defaults expect everything under `/srv/ai/arty-engine`, with the models in `/srv/ai/arty-engine/models`, the
layout Arty uses. Any other folders work; set them in the config.

```bash
sudo mkdir -p /srv/ai/arty-engine/runtime
sudo cp -r . /srv/ai/arty-engine/src
```

Download the runtime's official CUDA build from https://github.com/ggml-org/llama.cpp/releases (the engine was
tested with release b12110 for CUDA 12.8), unpack it into a folder of its own under `/srv/ai/arty-engine/runtime/`,
and link `/srv/ai/arty-engine/runtime/server` to the model server binary in that folder (not the RPC server). The
engine runs the binary through that link, so a new release is one changed link.

Then write the config the service reads, starting from the shipped one:

```bash
sudo cp /srv/ai/arty-engine/src/deploy/engine.json /srv/ai/arty-engine/engine.json
sudo nano /srv/ai/arty-engine/engine.json
```

Set at least `vram_budget_mb` to a little under your card's memory, and remove the `speculative` and
`internal_models` entries for models you do not have. Then install and start the service:

```bash
sudo cp /srv/ai/arty-engine/src/deploy/arty-engine.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now arty-engine
curl -s localhost:11438/api/version
curl -s localhost:11438/api/tags
```

To run it in a terminal instead:

```bash
cd /srv/ai/arty-engine/src
python3 -m arty_engine --config /srv/ai/arty-engine/engine.json --verbose
```

`--listen host:port` overrides `listen`. Without `--config`, the engine reads the file named by
`ARTY_ENGINE_CONFIG`, and without either it uses the built-in defaults below.

The unit runs the engine as root, and its `MemoryMax=5G` also covers the runtime processes the engine starts,
because they run in the unit's control group.

The engine listens on `127.0.0.1:11438` and has no authentication of its own. Keep it on loopback, or put a reverse
proxy with authentication in front of it. Requests that carry an `Origin` header from a web page other than
localhost are refused with 403, and so are requests for a `Host` other than localhost while the engine listens on
loopback, which stops other web sites from reaching it through your browser. A proxy that passes its own host name,
or a web app you trust, goes in `allowed_hosts` or `allowed_origins`.

### Stores in the older layout

Earlier versions kept manifests in `manifests` and `manifests-v2` folders beside `blobs`. The engine no longer reads
those; `tools/migrate_store.py` converts such a store in place, with the engine stopped:

```bash
python3 tools/migrate_store.py plan --store /srv/ai/arty-engine/models
python3 tools/migrate_store.py migrate --store /srv/ai/arty-engine/models --backup /srv/ai/arty-engine/store-layout-backup
python3 tools/migrate_store.py undo --store /srv/ai/arty-engine/models --backup /srv/ai/arty-engine/store-layout-backup
```

`plan` lists every model name, where its manifest will go and which old copies stay behind, and changes nothing.
`migrate` writes one catalog manifest per name (with the old manifest's time, so `/api/tags` keeps its order), keeps
a copy of the old folders in the backup folder, then moves the old folders and the old manifest documents in `blobs/`
there. Model files are never touched, copied or renamed. Short names stay short, and a name with a namespace keeps
it (`someone/model:tag`); a model from another registry keeps that registry's host in its name. The one registry of
an old folder becomes the short names; when a folder holds several, or when you want to choose, `--default-host`
names it (only the hosts given then become short names). Two old names that would become one are refused, and a
name in both old folders counts as an older copy only when both hold the same model file. A model whose old manifest can't be converted stops the run until it is left out with `--skip
<name>`. The backup folder must be empty and reachable by a rename (the same mount). Every command that changes the
store refuses while an engine answers on `--engine-url` (`http://127.0.0.1:11438` unless given, `http://` added
when the address has no scheme, `none` to skip), and when it can't tell.

`undo` first checks that the backup can bring the old layout back (from `moved/`, or from `copy/` when `moved/` is
gone) and changes nothing when it can't; then it moves everything back, removes the catalog manifests the migration
wrote and checks the old layout byte for byte against the record taken before the migration. It refuses when a
migrated manifest changed or a model was pulled since, and names any model whose blobs were deleted in between or
whose old manifest can't be read. An undo that stopped while copying can be run again. `tidy` moves the old manifest
documents of a store migrated by an earlier version of this tool into its backup (and finishes a move that stopped);
`untidy` puts them back. Both refuse once the migration was undone. At startup the engine names every root that is still in the older layout.

`tools/unused_blobs.py --store <store>` lists what no catalog manifest names: partial downloads, model files, old
manifest documents and anything else in `blobs/`, each with its kind and size. `--remove` deletes the list once it
is printed, and refuses (exit 2, nothing removed) while the store still holds an old manifest folder, has no catalog
manifest, holds the leftovers of a stopped migration or manifest write, or lists old manifest documents (use `tidy`
for those, so undo can bring them back). Model files no manifest names, such as the weights of a model left out of a
migration with `--skip`, are removed only with `--remove-model-files` as well. Run it with no pull in progress, and
never on a store in the older layout: there every blob counts as unused.

### Upgrading an install from before 14 Nov 2026

- `engine.json` names the runtime binary as `runner_binary` now, by default `/srv/ai/arty-engine/runtime/server`
  (make that a link to the runtime binary). A config that still names it under the older key stops the engine with
  "unknown settings"; rewrite it with `tools/write_config.py` (`arty-backend.sh engine-config` on a Proxmox
  install) or rename the key by hand.
- Arty's parts read the engine's address from `ARTY_ENGINE_URL`, and the trainer controller from its
  `arty_engine_url` setting.
- Bench configs need a `listen` of their own (the shipped `engine-test.json` uses 11439).
- Convert the store as above.

---

## Configuration

`engine.json` is a JSON object. Every key is optional; an unknown key stops the engine with an error.

| Key | Default | What it does |
|---|---|---|
| `listen` | `127.0.0.1:11438` | Address and port the API listens on |
| `store` | `/srv/ai/arty-engine/models` | The engine's own store: pulls go here, and only these models can be deleted |
| `model_roots` | `[]` | Other engine stores, read in place, read only |
| `runner_binary` | `/srv/ai/arty-engine/runtime/server` | The model runtime binary the engine starts for each model |
| `library_path` | `["/srv/ai/arty-engine/cuda-12.8/lib"]` | Folders added to `LD_LIBRARY_PATH` for the runtime (the folder its binary is in is always added) |
| `state_dir` | `/srv/ai/arty-engine/state` | Logs of each runtime process and borrowed chat templates |
| `vram_budget_mb` | `11500` | Models are unloaded, least recently used first, to keep the loaded ones under this |
| `load_overhead_mb` | `700` | Added to each model's estimate for buffers |
| `default_ctx` | `12288` | Context size when a request and the model's own parameters give none |
| `kv_cache_type` | `q8_0` | KV cache type |
| `flash_attn` | `auto` | Flash attention |
| `batch`, `ubatch` | `1024`, `512` | Batch sizes |
| `parallel` | `1` | Slots per model |
| `cache_ram_mb` | `1024` | Prompt cache kept in RAM |
| `fit_target_mb` | `256` | Headroom the runtime leaves when it fits layers on the card |
| `compact_tool_json` | `true` | Writes tool definitions into the prompt without spaces (fewer prompt tokens) |
| `default_keep_alive` | `5m` | How long a model stays loaded after a request that names no `keep_alive`; `-1` keeps it loaded |
| `load_timeout_seconds` | `600` | How long a model may take to load |
| `trainer_url` | empty | A controller that gets `/busy` and `/done` around GPU requests; empty for none |
| `pull_rate_bytes` | `4000000` | Download cap for pulls in bytes per second; `0` for no cap |
| `pull_keep_free_mb` | `2048` | A pull is refused (507) before it downloads anything when it would leave less than this free on the store's disk |
| `speculative` | `{}` | Per model: `{"draft": "<model>", "draft_max": 16, "draft_min": 0, "draft_p_min": 0.75}` or `{"spec_type": "ngram-mod", "ngram_min": 0, "ngram_max": 0}` |
| `internal_models` | `[]` | Models kept out of `/api/tags` (such as a draft model); they still answer by name |
| `extra_args` | `{}` | Per model: extra runtime arguments, as a list |
| `allowed_hosts` | `[]` | Host names besides localhost, 127.0.0.1 and ::1 that requests may name in `Host` while the engine listens on loopback |
| `allowed_origins` | `[]` | Host names of web pages besides localhost whose requests (with an `Origin` header) are served |

The shipped `deploy/engine.json` is Arty's setup, not these defaults: a 24576 context, models kept loaded
(`default_keep_alive` `-1`), n-gram speculation for `qwen3:14b` and `mimo-v2.6:9b`, and the Qwen3-0.6B draft listed
as an internal model.

`draft_ctx` is accepted in `speculative` but not passed to the runtime yet. Pulls from Hugging Face send `HF_TOKEN`
from the environment when it is set, for gated repositories. Without it, Hugging Face answers 401 for a repository
that is missing, private or gated, and the engine passes that on as 404; a refused token is a 502.

---

## The engine's HTTP API

| Endpoint | In the engine |
|---|---|
| `GET /`, `HEAD /` | Answers that the engine is running |
| `GET /api/version` | The engine's own version |
| `GET /api/tags` | Every model in the store and in `model_roots`, minus `internal_models` |
| `GET /api/ps` | Loaded models, their context and VRAM |
| `POST /api/show` | Details, parameters, template, capabilities, model info, and `runnable` |
| `POST /api/chat` | Streamed or not; tools; `think` (true, false, unset, a level); `format` (`"json"`, a schema); images; `keep_alive`; sampling; `num_ctx`; `num_gpu: 0`; `_debug_render_only` |
| `POST /api/generate` | Prompts, streamed and not; an empty prompt loads a model, and with `keep_alive: 0` unloads it; `raw` prompts |
| `POST /api/embed`, `POST /api/embeddings` | Embeddings |
| `POST /api/pull` | Hugging Face names (`hf.co/<owner>/<repo>:<quant>`) and the short names in `arty_engine/short_names.json`, with progress; one pull at a time (more below) |
| `DELETE /api/delete` | Models in the engine's own store. Models read in place from `model_roots` answer 405, internal models and deletes during a pull answer 409 |
| `POST /api/generate` with `suffix` | **501** |
| `POST /api/generate` raw with images or a `format` | **501** |
| `POST /api/create`, `/api/copy`, `/api/push` | **501** |
| Models whose file keeps the vision weights inside it (such as `qwen2.5vl:7b`) | Listed, but chats answer **501**: not supported, pull a build with a separate projector file instead |
| Split GGUF files on Hugging Face | Refused by `/api/pull` |
| Anything else (`/api/blobs`, the `/v1` OpenAI-style routes) | 404 |

Every answer carries `Server: arty-engine/<version>`.

Pulls: a short name already in the store is not downloaded again; one the engine can't run or read answers 409
(delete it, then pull). Each short name is pinned to the sha256 of its files, so a file changed on Hugging Face is
refused (502), and a short name that pins no projector takes none. A file listed without a size is refused (502).
Pulled manifests record the repo and revision. Any other name answers 400 with the form to use. With
`"stream": false` a pull that would break `pull_keep_free_mb` answers 507; a streamed pull reports that and a pin
mismatch inside the stream.

---

## How Arty uses it

Arty's brain, its tool hub and its web chat send their model requests to `ARTY_ENGINE_URL`
(`http://127.0.0.1:11438`). `deploy/brain-through-engine.conf` and `deploy/tool-hub-through-engine.conf` are the
systemd drop-ins that set it, and `deploy/arty-backend.env` (installed as `/etc/default/arty-backend`) sets it for
Arty's `ai-pull` and `ai-models` scripts.

On a Proxmox install, `tools/host/arty-backend.sh` sets this up from the host, in an order that never leaves two
servers on the card: `engine` starts the engine, points the trainer controller, the brain, the tool hub and the web
chat's route at it and loads the main model; `status` shows where each part points; `settings` prints what it
resolved and changes nothing; `engine-config` only rewrites the installed `engine.json`, which the engine reads when
it next starts. It reads `site.env` from the folder above `src` (`/srv/ai/arty-engine/site.env` in the container),
copied from `deploy/site.env.example`:

| Setting | What it is |
|---|---|
| `ARTY_CT` | The id of Arty's container |
| `ARTY_CONTROLLER_URL` | The GPU-sharing trainer controller, or `none` |
| `ARTY_CONTROLLER_SETTINGS`, `ARTY_CONTROLLER_PYTHON`, `ARTY_CONTROLLER_CODE` | The controller's settings file, its Python and its code (only with a controller) |
| `ARTY_CONTROLLER_UNIT`, `ARTY_CONTROLLER_MODULE` | The controller's systemd unit and the Python module it runs (only with a controller) |
| `ARTY_CONTROLLER_OLDER_SETTINGS` | Optional: older names of the controller's `arty_engine_url` setting, space separated, which the switch removes |
| `ARTY_ENGINE_ALLOWED_HOSTS`, `ARTY_ENGINE_ALLOWED_ORIGINS` | Space separated names for the config's `allowed_hosts` and `allowed_origins` (such as the `Host` a proxy passes) |
| `ARTY_ENGINE_MODEL_ROOTS` | Folders, separated by spaces, written into the installed config's `model_roots` (other engine stores read in place); empty means the engine's own store only |
| `ARTY_MAIN_MODEL` | The model loaded after a switch, `qwen3:14b` by default |

The script stops before changing anything when a setting it needs is missing. When a controller is set, the switch
writes it into the installed `engine.json` as `trainer_url`, with the allowed hosts and origins and the model roots
(`tools/write_config.py`), and sets the controller's `arty_engine_url` setting (`tools/host/controller_backend.py`).

`tools/container/` holds the bench helpers that run inside the container: `slice.sh` takes the GPU lock, asks the
controller for the card, unloads the engine's models, runs one bench and puts Arty's main model back; `slice.sh
--show-settings` prints what it would use. They expect to be copied into `/srv/ai/arty-engine`, beside `src`, with
`engine-test.json` from `deploy/` (the bench engine's config) and a `bench-support/` folder holding Arty's
`eval_scoring.py`, `e2e_battery.py` and `eval_images/` from the brain's `tests/` (the folder `bench.py --support`
reads). `bench-args.sh` also points at the brain's `/srv/ai/brain/system-prompt.md` and at the tool lists in this
repository's `tests/fixtures/`. `bench/bench.py` measures a server on load time, first token, generation speed,
Arty's chat battery, coding edits and vision.

`slice-engine.sh` starts the bench engine as `arty-engine-test` through `engine-restart.sh` and benches the address
in that config's `listen` (`127.0.0.1:11439` in `engine-test.json`, so the live engine on 11438 keeps serving). It
stops without benching when that port already answers before the start, when the unit is not running, or when the
unit never answers there; every bench config needs its own port and its own `state_dir`. The result file records
which models the live engine held before and after the bench (`live_engine`); `busy: true` means the two engines
shared the card and the numbers are not clean.

A Qwen3-0.6B draft model for `qwen3:14b` was slower than no speculation at a 24576 context on a 12 GB card, because
it pushes some of the 14B's layers off the GPU; n-gram speculation needs no extra memory and is what Arty runs.

---

## Development

```bash
python3 -m pytest -q tests
```

The tests run without a GPU or models: they build small GGUF files and fake stores, start a fake model runtime, and
replay answers recorded from real runs (`tests/fixtures/`).

---

## License

[PolyForm Strict 1.0.0](LICENSE). In short: you may use the engine for any noncommercial purpose (personal use,
research, hobby projects, and use by charities, schools, public research and government bodies), but you may not
distribute it or make changes or new works based on it. It comes as is, without warranty. See [NOTICE](NOTICE) for
the model runtime the engine starts, which is MIT licensed.
