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

The engine is Python standard library only. It reads the card's free memory with `nvidia-smi` on an NVIDIA card, and
with `amd-smi`, `rocm-smi` or the amdgpu driver's own files on an AMD card ([untested](#amd-gpus-untested)).

---

## Requirements

- **Linux with an NVIDIA GPU**, with `nvidia-smi` on the `PATH`. Built and tested on an RTX 3060 (12 GB) in a Debian 13
  container. AMD cards have a code path that has never run on a real AMD card; see [AMD GPUs (untested)](#amd-gpus-untested).
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
an old folder becomes the short names. When a folder holds several, or when you want to choose, `--default-host`
names it; once given, only the hosts given become short names in every folder, so give it once per old folder (`plan`
adds a note for a folder it leaves out). Two old names that would become one are refused, and a name in both old
folders counts as an older copy only when both hold the same model file; an older copy that can't be read stays in
the backup unread. A model whose old manifest can't be converted, or whose two copies hold different model files,
stops the run until it is left out with `--skip <name>`. The backup folder must be empty and reachable by a rename
(the same mount). Every command that changes the store refuses while anything accepts a connection on `--engine-url`
(`http://127.0.0.1:11438` unless given, `http://` added when the address has no scheme, `none` to skip), and when it
can't tell; a stalled engine counts as running.

`undo` first checks that the backup can bring the old layout back (from `moved/`, or from `copy/` when `moved/` is
gone) and changes nothing when it can't; then it moves everything back, removes the catalog manifests the migration
wrote and checks the old layout byte for byte against the record taken before the migration. It refuses when a
migrated manifest changed or a model was pulled since, or when the backup was already undone and a later migrate
holds the store, and names any model whose blobs were deleted in between or whose old manifest can't be read. An undo
that stopped while copying can be run again. A backup made before 15 Nov 2026 needs its `moved/` folder: its record
does not say whether the run finished, so its `copy/` alone is not trusted. `tidy` moves the old manifest documents
of a store migrated by an earlier version of this tool into its backup (and finishes a move that stopped); `untidy`
puts them back. Both refuse once the migration was undone. Empty probe folders (`.migrate-probe-*`) left by a stopped
run are removed by the next migrate or undo. At startup the engine names every root that is still in the older layout.

`tools/unused_blobs.py --store <store>` lists what no catalog manifest names: partial downloads, model files, old
manifest documents and anything else in `blobs/`, each with its kind and size. `--remove` deletes the list once it
is printed, and refuses (exit 2, nothing removed) while anything listens on `--engine-url` (the engine's address
unless given, `none` to skip), while the store still holds an old manifest folder, has no catalog manifest, holds the
leftovers of a stopped migration, undo or manifest write (the refusal names which; remove it by hand if nothing is
running), or lists old manifest documents (use `tidy` for those, so undo can bring them back). Plain `--remove`
deletes only partial downloads; model files and other blobs no manifest names, such as the weights, template, system
prompt or parameters of a model left out of a migration with `--skip` or of an older copy in its backup, go only with
`--remove-unnamed-blobs` as well. Never run it on a store in the older layout: there every blob counts as unused.

### Upgrading an install from before 14 Nov 2026

- `engine.json` names the runtime binary as `runner_binary` now, by default `/srv/ai/arty-engine/runtime/server`
  (make that a link to the runtime binary). A config that still names it under the older key stops the engine with
  "unknown settings"; rewrite it with `tools/write_config.py` (`arty-backend.sh engine-config` on a Proxmox
  install) or rename the key by hand.
- Arty's parts read the engine's address from `ARTY_ENGINE_URL`, and the trainer controller from its
  `arty_engine_url` setting.
- Bench configs need a `listen` of their own (the shipped `engine-test.json` uses 11439).
- Convert the store as above.
- Rolling back to a version from before `gpu_backend` existed: that version stops with "unknown settings
  gpu_backend" on the newer `engine.json`, so put back the older `engine.json` (or rewrite it with the older
  `tools/write_config.py`) together with the older code, before the engine restarts.

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
| `thinking_leakers` | `[]` | Thinking models that write reasoning into their answer even when told not to; their streamed replies are held back until a leak is ruled out (below) |
| `extra_args` | `{}` | Per model: extra runtime arguments, as a list |
| `allowed_hosts` | `[]` | Host names besides localhost, 127.0.0.1 and ::1 that requests may name in `Host` while the engine listens on loopback |
| `allowed_origins` | `[]` | Host names of web pages besides localhost whose requests (with an `Origin` header) are served |
| `gpu_backend` | `auto` | `cuda`, `rocm`, `vulkan` or `cpu`; `auto` picks one at start (below) |
| `runtimes` | `{}` | Per backend: `{"runner_binary": "<path>", "library_path": ["<folder>"]}`; a backend not listed uses `runner_binary` and `library_path` above |

The shipped `deploy/engine.json` is Arty's setup, not these defaults: a 24576 context, models kept loaded
(`default_keep_alive` `-1`), n-gram speculation for `qwen3:14b` and `mimo-v2.6:9b`, the Qwen3-0.6B draft listed
as an internal model, `mimo-v2.6:9b` as a thinking leaker, and `gpu_backend` `cuda`.

`auto` picks `cuda` when `/dev/nvidia0` is there (`nvidia-smi` alone, without the device, does not count); otherwise
`rocm` when `/dev/kfd` and a render node of the amdgpu driver are there, `runtimes` lists `rocm` and the card's gfx
target (read from `/sys/class/kfd/kfd/topology/nodes/*/properties`) is one the ROCm build carries code for (the list
under AMD GPUs below); otherwise `vulkan` when a render node (amdgpu first) is there and `runtimes` lists `vulkan`;
otherwise `cpu`. The engine logs its choice, the reason and the card's free memory once at start. A config without
`gpu_backend` or `runtimes` works as before: on an NVIDIA machine it resolves to `cuda` with the top-level
`runner_binary` and `library_path`. A request with `num_gpu: 0` runs on the CPU with `-ngl 0 --device none` and the
devices hidden: `CUDA_VISIBLE_DEVICES` empty for `cuda`, `ROCR_VISIBLE_DEVICES` empty for `rocm` (an empty
`HIP_VISIBLE_DEVICES` hides nothing), nothing more for `vulkan`. With `cpu` every request runs that way, with both
variables empty.

`draft_ctx` is accepted in `speculative` but not passed to the runtime yet. Pulls from Hugging Face send `HF_TOKEN`
from the environment when it is set, for gated repositories. Without it, Hugging Face answers 401 for a repository
that is missing, private or gated, and the engine passes that on as 404; a refused token is a 502.

---

## AMD GPUs (untested)

**No real AMD run has been done.** The AMD path is written and tested only against fakes (fake `amd-smi`, `rocm-smi`,
sysfs and device nodes) on machines without an AMD card. Treat it as a starting point; `tools/amd-check.sh`
([docs/amd-check.md](docs/amd-check.md)) checks it on a machine that has one.

**Which runtime build.** Two official release builds of the same runtime as the CUDA one (release b12110, the one the
engine was tested with), each unpacked into a folder of its own:

- ROCm: https://github.com/ggml-org/llama.cpp/releases/download/b12110/llama-b12110-bin-ubuntu-rocm-10.0-x64.tar.gz
- Vulkan: https://github.com/ggml-org/llama.cpp/releases/download/b12110/llama-b12110-bin-ubuntu-vulkan-x64.tar.gz

Their published sha256 digests are `664b5c218ec9a38d1e7b543bae7407dc62a4051ba09ddc51bd29f68ee77da674` (ROCm) and
`a5d741940956ec62e8783aee7e5a4c7c761af867d6f027c3277a51e1a3ac59b0` (Vulkan). Link a `server` in each folder to its
model server binary (not the RPC server), as for CUDA.

- The **ROCm** build loads ROCm 10.0's `libamdhip64.so.7`, `libhipblas.so.3` and `librocblas.so.5`, which it does not
  include. Either install ROCm 10.0 on the system, or install AMD's own Python packages of it into a folder, the way
  the build itself was made. Name the card's kernel package as an extra, `device-<gfx target>` (such as
  `device-gfx1100` for an RX 7900 XTX; `rocminfo` or the `gfx_target_version` in
  `/sys/class/kfd/kfd/topology/nodes/*/properties` gives it), never the plain `device` extra, which falls back to
  gfx1010 on a machine without AMD's tools:
  `python3 -m venv /srv/ai/arty-engine/rocm-10.0 && /srv/ai/arty-engine/rocm-10.0/bin/pip install --no-cache-dir --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[libraries,device-gfx1100]==10.0.0"`.
  The libraries land in two folders, `_rocm_sdk_core/lib` (`libamdhip64.so.7`) and `_rocm_sdk_libraries/lib`
  (`libhipblas.so.3`, `librocblas.so.5`): list both in `runtimes.rocm.library_path`
  (`find /srv/ai/arty-engine/rocm-10.0 -name 'libamdhip64.so.7*' -o -name 'libhipblas.so.3*' -o -name 'librocblas.so.5*'`
  shows them). The build carries GPU code for gfx908, gfx90a, gfx942, gfx950, gfx1010 to gfx1012,
  gfx1030 to gfx1036, gfx1100 to gfx1103, gfx1150 to gfx1152, gfx1200 and gfx1201.
- The **Vulkan** build needs the Vulkan loader (`libvulkan.so.1`) and a Vulkan driver for the card, on Debian
  `apt install libvulkan1 mesa-vulkan-drivers` (Mesa's RADV driver for AMD).

**Which cards ROCm supports.** AMD's own list is the ROCm compatibility matrix,
https://rocm.docs.amd.com/en/latest/compatibility/compatibility-matrix.html. On 15 Nov 2026 it covered ROCm 10.1.0 and
listed these Radeon cards: RX 9070 XT, RX 9070 GRE, RX 9070, RX 9060 XT LP, RX 9060 XT, RX 9060, RX 9050, RX 7900 XTX,
RX 7900 XT, RX 7900 GRE, RX 7800 XT, RX 7700 XT, RX 7700 and RX 7600, plus Radeon PRO (such as the W7900, W7800, W6800
and V620) and Instinct cards. Consumer RX 6000 cards are not on it. Check the page before choosing a card or a
rented machine; it changes between releases.

**Older cards need Vulkan.** Cards such as the RX 570 (Polaris, gfx803) are neither on AMD's list nor among the GPU
code the ROCm build carries, so they run on the Vulkan build, or on the CPU.

**Config.** Name both runtimes and let `auto` choose (it takes ROCm only for a card the ROCm build carries code for,
so an RX 570 lands on Vulkan), or set `gpu_backend` to `rocm` or `vulkan`. A card made to pass as another target with
`HSA_OVERRIDE_GFX_VERSION` needs `gpu_backend` set to `rocm`, since `auto` reads the card's own target:

```json
"gpu_backend": "auto",
"runtimes": {
  "rocm": {"runner_binary": "/srv/ai/arty-engine/runtime-rocm/server",
           "library_path": ["<venv>/lib/python3.X/site-packages/_rocm_sdk_core/lib",
                            "<venv>/lib/python3.X/site-packages/_rocm_sdk_libraries/lib"]},
  "vulkan": {"runner_binary": "/srv/ai/arty-engine/runtime-vulkan/server"}
}
```

**Container devices.** ROCm needs `/dev/kfd` and the card's render node `/dev/dri/renderD*`; Vulkan needs only the
render node. Pass the card node `/dev/dri/card*` of the same GPU too. AMD's docs give GPU access to the `render` and
`video` groups, so give the nodes the container's own group ids: `render` for `/dev/kfd` and the render node, `video`
for the card node (`getent group render video` inside the container shows them). On Proxmox, for example:
`pct set <ct> -dev0 /dev/kfd,gid=<render> -dev1 /dev/dri/renderD128,gid=<render> -dev2 /dev/dri/card1,gid=<video>`.
The shipped unit runs the engine as root, which can open the nodes either way.

**Not done for AMD.** `/api/ps` shows the engine's estimate as `size_vram` (there is no per-process reading as with
`nvidia-smi`). A `num_gpu: 0` request on the Vulkan build relies on `--device none` alone. When `amd-smi` lists several
cards, the engine reads the one with the most memory.

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
| `POST /api/pull` | Hugging Face names (`hf.co/<owner>/<repo>:<quant>`) and the short names in `arty_engine/short_names.json`, or a short name with its own `source`, with progress; one pull at a time (more below) |
| `DELETE /api/delete` | Models in the engine's own store. Models read in place from `model_roots` answer 405, internal models and deletes during a pull answer 409 |
| `POST /api/generate` with `suffix` | **501** |
| `POST /api/generate` raw with images or a `format` | **501** |
| `POST /api/create`, `/api/copy`, `/api/push` | **501** |
| Models whose file keeps the vision weights inside it (such as `qwen2.5vl:7b`) | Listed, but chats answer **501**: not supported, pull a build with a separate projector file instead |
| Split GGUF files on Hugging Face | Refused by `/api/pull` |
| Anything else (`/api/blobs`, the `/v1` OpenAI-style routes) | 404 |

Every answer carries `Server: arty-engine/<version>`.

Thinking that leaks into the answer: some models (seen with `mimo-v2.6:9b`) write their reasoning and a closing
`</think>` into the reply even with `think` false. Whether a model can leak is decided once when it loads: a thinking
model can when it is named in `thinking_leakers` or its chat template does not read `enable_thinking`. `mimo-v2.6:9b`
needs naming: its template does read the switch, yet the model still writes its reasoning into the answer. Any other
model (such as `qwen3:14b`) streams its content at once and exactly as written, think tags included. For a model that
can leak, chat and generate replies (not raw ones) are read the same way streamed and not streamed, so both give the
same answer. Until the first closing tag, reasoning from the model or the end of the reply, content is held back, so a
streamed reply without a closing tag arrives in one piece at the end, however long it is. The text before the first
closing tag goes into `thinking`, or is dropped when the request has `think` false, and the answer starts after it,
without the blank lines that follow the tag. After that, a `<think>` ... `</think>` block in the answer is a second
block of reasoning and goes into `thinking` too, a lone closing tag is taken out, and both are logged as a warning;
an opening tag there that is never closed stays in the answer as written, unless the reply was cut by the length
limit, when the text after it is thinking.
The same characters mean both things, so an answer from such a model that writes the tags itself keeps only what
follows the first closing tag, without any later tagged block. A reply with no closing tag at all is delivered
exactly as written, unless it opens with `<think>` and was cut by the length limit: that reply is all thinking, with
no content. A reply cut by the length limit before any tag can't be told from a cut answer, so it is delivered as
written and logged as a warning, since it may be reasoning. Once the model's runner has delivered reasoning of its own,
a model that is held only because its template does not read `enable_thinking` has its answer passed on exactly as
written; one named in `thinking_leakers` still has tagged blocks taken out of the answer. A name in
`thinking_leakers` that matches no installed model is logged as a warning at start.

Pulls: a short name already in the store is not downloaded again; one the engine can't run or read answers 409
(delete it, then pull). Each short name is pinned to the sha256 of its files, so a file changed on Hugging Face is
refused (502), and a short name that pins no projector takes none. A file listed without a size is refused (502).
Pulled manifests record the repo and revision. Any other name answers 400 with the form to use. With
`"stream": false` a pull that would break `pull_keep_free_mb` answers 507; a streamed pull reports that and a pin
mismatch inside the stream.

A pull can also name its own source: `{"model": "qwen2.5vl:7b", "source": "hf.co/<owner>/<repo>:Q4_K_M"}` fetches that
repository and quant and installs it under the short name given, and the manifest records the source. Without `source`
the short names in `arty_engine/short_names.json` decide, as above. Pulling a name already installed from the same
source downloads nothing; `hf.co` and `huggingface.co` are the same host, and the owner, repository and quant are
compared without regard to case. From a different source the new files are downloaded and checked first, and only then
does the new manifest replace the old one, and the old copy is unloaded. A replace keeps only the new weights: the old
copy's template, system prompt and parameters are not carried over (the status line names the ones dropped), and the
old copy's files stay in the store, since a migration backup or a model left out with `--skip` may still need them.
The status line says how much space they hold and prints the cleanup command in full, with the store's path; it runs
only with the engine stopped, and it removes every blob no model names, not only the old copy's, so run it without
`--remove` first to see the list. Until then the old files count against `pull_keep_free_mb`, so a second replace of a
large model may answer 507. Like any pull that writes a manifest, a replace means a migration made before it can no
longer be undone. A new source whose files are the old copy's is not a replace: only the recorded source changes,
nothing is unloaded, and the loaded copy keeps answering (the recorded source is not part of what decides whether a
loaded model can be reused). A name installed only in another store root gets its new copy in the engine's own store
and the other root's copy is left as it is, no longer used. Every pull that names a source is logged as a warning with
the caller's address, the name and the source, and every replace with the old and the new source; `/api/show` shows a
model's recorded source (`name` and `revision`) when it has one. Projector rules, size checks and `pull_keep_free_mb`
apply as for any pull, and a source that matches a short name's catalogue entry, in any of the spellings above, keeps
its sha256 pins. `source` must be a Hugging Face name (400 otherwise), only a short name or that same Hugging Face
name can take it (400), and an internal model's name answers 409.

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
A failed `engine` run ends with one line that says how far it got: the config could not be written (nothing started
or moved), the engine did not start, the engine started but Arty was not moved (with the reason), the move did not
finish (some parts may already point at the engine), or Arty uses the engine but the main model did not load or the
controller is not running. When the engine did not start or the move did not finish it prints `status` first, so the
parts that already point at the engine can be seen.

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
