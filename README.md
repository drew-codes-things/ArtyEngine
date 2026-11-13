<div align="center">

# Arty engine

**An Ollama-compatible model server built on llama.cpp's `llama-server`. It reads the models you already have in
Ollama's store, keeps them inside a VRAM budget, pulls new ones from Hugging Face, and runs speculative decoding.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![llama.cpp](https://img.shields.io/badge/llama.cpp-llama--server-000000?style=flat-square)](https://github.com/ggml-org/llama.cpp)
[![Dependencies](https://img.shields.io/badge/dependencies-standard%20library-brightgreen?style=flat-square)](#install)
[![License](https://img.shields.io/badge/license-PolyForm%20Strict%201.0.0-orange?style=flat-square)](LICENSE)

</div>

---

## Overview

Arty is a self-hosted assistant that talks to its models through Ollama's API. The engine is a drop-in server for
that API: any client that speaks to Ollama (Arty's brain, the `ollama` CLI with `OLLAMA_HOST`, other apps) can talk
to the engine instead, without downloading its models again.

- **Reuses Ollama's model store in place.** It reads Ollama's manifests and blobs (read only) and lists the same
  models with the same names, digests, sizes and capabilities. Nothing is copied or converted.
- **One `llama-server` per model.** Each loaded model runs in its own llama.cpp process on a local port, so a crash
  takes down one model, not the server, and llama.cpp is never built into this repository.
- **VRAM budget swapping.** Before a model loads, the engine estimates what it needs (weights, KV cache, overhead)
  and unloads the least recently used idle models until it fits `vram_budget_mb` and the card's free memory.
  `keep_alive` works as in Ollama.
- **Hugging Face pulls.** `hf.co/<owner>/<repo>:<quant>` picks the matching GGUF (and a vision projector when the
  repository has one), checks its sha256, resumes partial downloads and keeps to a bandwidth cap. Names from
  ollama.com are pulled from Ollama's registry. Pulled models go into the engine's own store.
- **Speculative decoding.** Per model: a small draft model (`--model-draft`) or llama.cpp's n-gram speculation
  (`spec_type`, for example `ngram-mod`), with no draft model at all.
- **Behaves like Ollama where clients notice.** Thinking comes back as `message.thinking`, tool call arguments come
  back as objects, tool definitions are written into the prompt as compactly as Ollama writes them, and
  `/api/generate` with an empty prompt loads or unloads a model.
- **Optional trainer hand-off.** With `trainer_url` set, the engine tells a GPU-sharing controller `/busy` before a
  request and `/done` after it, so something else (such as a training job) can use the card while the engine is idle.

The engine is Python standard library only. It needs a `llama-server` build that suits your GPU, and reads the card's
free memory with `nvidia-smi` when it is there.

---

## Requirements

- **Linux with an NVIDIA GPU**, with `nvidia-smi` on the `PATH`. Built and tested on an RTX 3060 (12 GB) in a Debian 13
  container.
- **Python 3.10 or newer.** No packages to install. The tests need `pytest`.
- **A llama.cpp `llama-server` build** from [llama.cpp's releases](https://github.com/ggml-org/llama.cpp/releases).
  The engine was tested with release b12110 built for CUDA 12.8. If the CUDA runtime libraries the build needs
  (`libcudart`, `libcublas`) are not on the system, put them in a folder and list it in `library_path`.
- **Optional: an Ollama model store** to read from. Ollama itself does not have to be running.

---

## Install

The defaults expect everything under `/srv/ai/arty-engine` and Ollama's models in `/srv/ai/ollama`, the layout Arty
uses. Any other folders work; set them in the config.

```bash
sudo mkdir -p /srv/ai/arty-engine
sudo cp -r . /srv/ai/arty-engine/src
```

Put `llama-server` (and the libraries that came with it) in `/srv/ai/arty-engine/llama.cpp/current/`, then write the
config the service reads, starting from the shipped one:

```bash
sudo cp /srv/ai/arty-engine/src/deploy/engine.json /srv/ai/arty-engine/engine.json
sudo nano /srv/ai/arty-engine/engine.json
```

Set at least `model_roots` to your Ollama store (whatever `OLLAMA_MODELS` points at; Ollama's Linux service keeps
it in `/usr/share/ollama/.ollama/models` by default), `vram_budget_mb` to a little under your card's memory, and
remove the `speculative` and `internal_models` entries for models you do not have.

Then install and start the service:

```bash
sudo cp /srv/ai/arty-engine/src/deploy/arty-engine.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now arty-engine
curl -s localhost:11438/api/version
OLLAMA_HOST=127.0.0.1:11438 ollama list
```

To run it in a terminal instead:

```bash
cd /srv/ai/arty-engine/src
python3 -m arty_engine --config /srv/ai/arty-engine/engine.json --verbose
```

`--listen host:port` overrides `listen`. Without `--config`, the engine reads the file named by
`ARTY_ENGINE_CONFIG`, and without either it uses the built-in defaults below.

The unit runs the engine as root, and its `MemoryMax=5G` also covers the `llama-server` processes the engine starts,
because they run in the unit's control group.

The engine listens on `127.0.0.1:11438` and has no authentication of its own, like Ollama. Keep it on loopback, or
put a reverse proxy with authentication in front of it. Requests that carry an `Origin` header from a web page other
than localhost are refused with 403, and so are requests for a `Host` other than localhost while the engine listens
on loopback, which stops other web sites from reaching it through your browser. A proxy that passes its own host
name, or a web app you trust, goes in `allowed_hosts` or `allowed_origins`.

---

## Configuration

`engine.json` is a JSON object. Every key is optional; an unknown key stops the engine with an error.

| Key | Default | What it does |
|---|---|---|
| `listen` | `127.0.0.1:11438` | Address and port the API listens on |
| `store` | `/srv/ai/arty-engine/models` | The engine's own store: pulls go here, and only these models can be deleted |
| `model_roots` | `["/srv/ai/ollama"]` | Ollama stores read in place, read only |
| `llama_server` | `/srv/ai/arty-engine/llama.cpp/current/llama-server` | The `llama-server` binary |
| `library_path` | `["/srv/ai/arty-engine/cuda-12.8/lib"]` | Folders added to `LD_LIBRARY_PATH` for `llama-server` (its own folder is always added) |
| `state_dir` | `/srv/ai/arty-engine/state` | Logs of each `llama-server` and borrowed chat templates |
| `vram_budget_mb` | `11500` | Models are unloaded, least recently used first, to keep the loaded ones under this |
| `load_overhead_mb` | `700` | Added to each model's estimate for buffers |
| `default_ctx` | `12288` | Context size when a request and the model's own parameters give none |
| `kv_cache_type` | `q8_0` | KV cache type (`-ctk`, `-ctv`) |
| `flash_attn` | `auto` | Flash attention (`-fa`) |
| `batch`, `ubatch` | `1024`, `512` | Batch sizes (`-b`, `-ub`) |
| `parallel` | `1` | Slots per model (`-np`) |
| `cache_ram_mb` | `1024` | Prompt cache kept in RAM (`--cache-ram`) |
| `fit_target_mb` | `256` | Headroom `llama-server` leaves when it fits layers on the card (`--fit-target`) |
| `compact_tool_json` | `true` | Writes tool definitions into the prompt without spaces, as Ollama does (fewer prompt tokens) |
| `default_keep_alive` | `5m` | How long a model stays loaded after a request that names no `keep_alive`; `-1` keeps it loaded |
| `load_timeout_seconds` | `600` | How long a model may take to load |
| `trainer_url` | empty | A controller that gets `/busy` and `/done` around GPU requests; empty for none |
| `pull_rate_bytes` | `4000000` | Download cap for pulls in bytes per second; `0` for no cap |
| `speculative` | `{}` | Per model: `{"draft": "<model>", "draft_max": 16, "draft_min": 0, "draft_p_min": 0.75}` or `{"spec_type": "ngram-mod", "ngram_min": 0, "ngram_max": 0}` |
| `internal_models` | `[]` | Models kept out of `/api/tags` (such as a draft model); they still answer by name |
| `extra_args` | `{}` | Per model: extra `llama-server` arguments, as a list |
| `allowed_hosts` | `[]` | Host names besides localhost, 127.0.0.1 and ::1 that requests may name in `Host` while the engine listens on loopback |
| `allowed_origins` | `[]` | Host names of web pages besides localhost whose requests (with an `Origin` header) are served |

The shipped `deploy/engine.json` is Arty's setup, not these defaults: a 24576 context, models kept loaded
(`default_keep_alive` `-1`), n-gram speculation for `qwen3:14b` and `mimo-v2.6:9b`, and the Qwen3-0.6B draft listed
as an internal model.

`draft_ctx` is accepted in `speculative` but not passed to `llama-server` yet. Pulls from Hugging Face send
`HF_TOKEN` from the environment when it is set, for gated repositories.

---

## Ollama API coverage

| Endpoint | In the engine |
|---|---|
| `GET /`, `HEAD /` | Answers that the engine is running |
| `GET /api/version` | The engine's own version |
| `GET /api/tags` | Every model in the store and the Ollama stores, minus `internal_models` |
| `GET /api/ps` | Loaded models, their context and VRAM |
| `POST /api/show` | Details, parameters, template, capabilities, model info |
| `POST /api/chat` | Streamed and not; tools; `think` true, false, unset or a level; `format` `"json"` or a JSON schema; images on vision models; `keep_alive`; sampling options; `num_ctx`; `num_gpu: 0` runs on the CPU; `_debug_render_only` renders the prompt only |
| `POST /api/generate` | Prompts, streamed and not; an empty prompt loads a model, and with `keep_alive: 0` unloads it; `raw` prompts |
| `POST /api/embed`, `POST /api/embeddings` | Embeddings |
| `POST /api/pull` | Hugging Face (`hf.co/...`) and ollama.com names, with progress; one pull at a time |
| `DELETE /api/delete` | Models the engine pulled itself. Ollama's models answer 405 (use `ollama rm`), internal models and deletes during a pull answer 409 |
| `POST /api/generate` with `suffix` | **501** |
| `POST /api/generate` raw with images or a `format` | **501** |
| `POST /api/create`, `/api/copy`, `/api/push` | **501**, use Ollama |
| Vision models stored in Ollama's own engine format (such as `qwen2.5vl:7b`) | **501**, naming a llama.cpp build with a separate projector to pull instead |
| Split GGUF files on Hugging Face | Refused by `/api/pull` |
| Anything else (`/api/blobs`, the `/v1` OpenAI-style routes) | 404 |

---

## How Arty uses it

Arty's brain, its tool hub and its web chat send their model requests to `OLLAMA_URL`. Pointing that at the engine
(`http://127.0.0.1:11438`) is the whole switch; `deploy/brain-through-engine.conf` and
`deploy/tool-hub-through-engine.conf` are the systemd drop-ins for it, and `deploy/arty-backend.env` points Arty's
`ai-pull` and `ai-models` scripts at the engine. Ollama can stay installed, empty, as the way back.

On a Proxmox install, `tools/host/arty-backend.sh` does the switch in both directions from the host, in an order that
never leaves two servers on the card: `engine` starts the engine, moves the brain, the tool hub and the web chat's
route over and loads the main model; `ollama` puts everything back; `status` shows where each part points;
`settings` prints what it resolved and changes nothing; `engine-config` only rewrites the installed `engine.json`,
which the engine reads when it next starts. It reads `site.env` from the folder above `src`
(`/srv/ai/arty-engine/site.env` in the container), copied from `deploy/site.env.example`:

| Setting | What it is |
|---|---|
| `ARTY_CT` | The id of Arty's container |
| `ARTY_CONTROLLER_URL` | The GPU-sharing trainer controller, or `none` |
| `ARTY_CONTROLLER_SETTINGS`, `ARTY_CONTROLLER_PYTHON`, `ARTY_CONTROLLER_CODE` | The controller's settings file, its Python and its code (only with a controller) |
| `ARTY_CONTROLLER_UNIT`, `ARTY_CONTROLLER_MODULE` | The controller's systemd unit and the Python module it runs (only with a controller) |
| `ARTY_ENGINE_ALLOWED_HOSTS`, `ARTY_ENGINE_ALLOWED_ORIGINS` | Names, separated by spaces, written into the installed config's `allowed_hosts` and `allowed_origins` (for example the address a reverse proxy passes as `Host`) |
| `ARTY_MAIN_MODEL` | The model loaded after a switch, `qwen3:14b` by default |

The script stops before changing anything when a setting it needs is missing. When a controller is set, the switch
writes it into the installed `engine.json` as `trainer_url`, with the allowed hosts and origins
(`tools/write_config.py`).

`tools/container/` holds the bench helpers that run inside the container: `slice.sh` takes the GPU lock, asks the
controller for the card, unloads both servers, runs one bench and puts Arty's main model back; `slice.sh
--show-settings` prints what it would use. They expect to be copied into `/srv/ai/arty-engine`, beside `src`, with
`engine-test.json` from `deploy/` (the bench engine's config) and a `bench-support/` folder holding Arty's
`eval_scoring.py`, `e2e_battery.py` and `eval_images/` from the brain's `tests/` (the folder `bench.py --support`
reads). `bench-args.sh` also points at the brain's `/srv/ai/brain/system-prompt.md` and at the tool lists in this
repository's `tests/fixtures/`. `bench/bench.py` compares the engine with Ollama on load time, first token,
generation speed, Arty's chat battery, coding edits and vision.

### Measured on an RTX 3060 (12 GB), 24576 context, q8_0 KV cache on both

| | Ollama 0.40.16 | Engine |
|---|---|---|
| `qwen3:14b` load | 47.7 s | 41.5 s |
| `qwen3:14b` generation | 33.8 tok/s | 34.1 tok/s plain, 36.9 tok/s with n-gram speculation |
| `mimo-v2.6:9b` load | 17.4 s | 29.5 s, slower (19.3 to 22.6 s once the unit's memory cap no longer evicted the page cache) |
| `mimo-v2.6:9b` generation | 54.4 tok/s | 55.5 tok/s plain, 73.8 tok/s with n-gram speculation |
| Arty's chat battery (19 cases) | 16/19 | 16/19, the same misses |
| Vision (`mimo-v2.6:9b`, 4 pictures) | 4/4 | 4/4 |

A Qwen3-0.6B draft model for `qwen3:14b` was slower at this context on a 12 GB card, because it pushes some of the
14B's layers off the GPU; n-gram speculation needs no extra memory and is what Arty runs.

---

## Development

```bash
python3 -m pytest -q tests
```

The tests run without a GPU or models: they build small GGUF files and fake Ollama stores, start a fake
`llama-server`, and replay answers recorded from real `llama-server` and Ollama runs (`tests/fixtures/`).

---

## License

[PolyForm Strict 1.0.0](LICENSE). In short: you may use the engine for any noncommercial purpose (personal use,
research, hobby projects, and use by charities, schools, public research and government bodies), but you may not
distribute it or make changes or new works based on it. It comes as is, without warranty. See [NOTICE](NOTICE) for
llama.cpp, which the engine runs and which is MIT licensed.
