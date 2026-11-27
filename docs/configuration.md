# Configuration

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
| `thinking_leakers` | `[]` | Thinking models that write reasoning into their answer even when told not to; their streamed replies are held back until a leak is ruled out ([how](api.md#thinking-that-leaks-into-the-answer)) |
| `extra_args` | `{}` | Per model: extra runtime arguments, as a list |
| `rpc_workers` | `[]` | Other machines whose GPUs a model can be spread over: `"host:port"` or `{"address", "label", "memory_mb"}` ([rpc.md](rpc.md)) |
| `model_settings` | `{}` | Per model: when to use the workers, how to place the layers, and runtime settings such as `parallel`, `kv_cache_type`, `cache_reuse`, rope scaling and LoRA adapters ([rpc.md](rpc.md#placing-a-models-layers)). A model may be named any way it can be asked for (`qwen3`, `qwen3:latest` and `library/qwen3` are one model; naming it twice is refused), and a name that matches no installed model is warned about at start |
| `runtime_url` | empty | A runtime zip at another address for `arty-engine runtime install` (which then needs `--sha256`); empty for ArtyEngine's own release ([install.md](install.md#the-model-runtime)) |
| `allowed_hosts` | `[]` | Host names besides localhost, 127.0.0.1 and ::1 that requests may name in `Host` while the engine listens on loopback |
| `allowed_origins` | `[]` | Host names of web pages besides localhost whose requests (with an `Origin` header) are served |
| `gpu_backend` | `auto` | `cuda`, `rocm`, `vulkan` or `cpu`; `auto` picks one at start ([how](#choosing-a-gpu-backend)) |
| `runtimes` | `{}` | Per backend: `{"runner_binary": "<path>", "library_path": ["<folder>"]}`; a backend not listed uses `runner_binary` and `library_path` above |
| `remote_models` | `{}` | Outside OpenAI-compatible servers whose models the engine lists and forwards to ([below](#remote-models-outside-openai-compatible-servers)); never holds a key |

## The shipped config

The shipped `deploy/engine.json` is Arty's setup, not these defaults: a 24576 context, models kept loaded
(`default_keep_alive` `-1`), n-gram speculation for `qwen3:14b`, the Qwen3-0.6B draft listed as an internal model,
no thinking leakers, and `gpu_backend` `cuda`.

## Choosing a GPU backend

`auto` picks `cuda` when `/dev/nvidia0` is there (`nvidia-smi` alone, without the device, does not count); otherwise
`rocm` when `/dev/kfd` and a render node of the amdgpu driver are there, `runtimes` lists `rocm` and the card's gfx
target (read from `/sys/class/kfd/kfd/topology/nodes/*/properties`) is one the ROCm build carries code for (the list
in [AMD GPUs](../README.md#amd-gpus-untested)); otherwise `vulkan` when a render node (amdgpu first) is there and `runtimes` lists `vulkan`;
otherwise `cpu`. The engine logs its choice, the reason and the card's free memory once at start. A config without
`gpu_backend` or `runtimes` works as before: on an NVIDIA machine it resolves to `cuda` with the top-level
`runner_binary` and `library_path`. A request with `num_gpu: 0` runs on the CPU with `-ngl 0 --device none` and the
devices hidden: `CUDA_VISIBLE_DEVICES` empty for `cuda`, `ROCR_VISIBLE_DEVICES` empty for `rocm` (an empty
`HIP_VISIBLE_DEVICES` hides nothing), nothing more for `vulkan`. With `cpu` every request runs that way, with both
variables empty.

## Remote models (outside OpenAI-compatible servers)

`remote_models` names servers that speak the OpenAI chat API (`GET <url>/models`, `POST <url>/chat/completions`), such
as another person's model server shared through an app key. Their models appear in `/api/tags` as
`remote/<server>/<model>` (the server's model name in lower case, other characters turned into `-`), and `/api/chat`
and `/api/generate` with that name are forwarded there with `Authorization: Bearer <key>`, streamed or not, with tools,
images (when `vision` is listed) and thinking (when `thinking` is listed). They use no card memory: nothing is loaded,
budgeted or unloaded for them, the trainer controller is not told, and they never show in `/api/ps`. An empty prompt
answers `load` or `unload` straight away.

```json
"remote_models": {
  "remote": {
    "url": "https://models.example.com/v1",
    "key_file": "/srv/ai/arty-engine/secrets/remote.key",
    "capabilities": ["tools"],
    "timeout_seconds": 300
  }
}
```

| Setting | What it does |
|---|---|
| `url` | The server's OpenAI-compatible address, usually ending in `/v1`; `http://` or `https://`, with no user or password in it |
| `key_file` | A file holding only the key. It must be readable by its owner alone (`chmod 600`), or requests are refused with a message saying so |
| `key_env` | Or the name of an environment variable holding the key (one of the two, or neither for a server without keys) |
| `ca_file` | A PEM certificate to trust for a server with its own certificate |
| `models` | The server's model names to list; when left out the engine asks `<url>/models` (at most once a minute, again 15 s after a failure) |
| `capabilities` | Any of `tools`, `vision`, `thinking`; default `["tools"]`. `thinking` also sends the thinking switch the way the engine's own runtime takes it (`chat_template_kwargs`) |
| `timeout_seconds` | How long to wait for the server to start answering, and between streamed pieces; default 300 |

A `key` (or `api_key`, `token`, `secret`) in the config stops the engine at start: the key lives in its file or
variable only. The engine reads it for each request, so a new key needs no restart, and never writes it to its log,
`/api/tags` or `/api/show` (which names the server and the model, not the address). Only the standard sampling options
(`temperature`, `top_p`, `presence_penalty`, `frequency_penalty`, `seed`, `stop`, `num_predict` as `max_tokens`) are
sent; `num_ctx` and the runtime's own options are dropped, since the server decides those. Refusals become readable
errors: a refused key (401) and refused access (403) answer 502 saying so, a rate limit answers 429 with the wait the
server asked for, a model the server doesn't have answers 404, no answer within `timeout_seconds` answers 504, and a
key the server echoes back is replaced by `[key hidden]`.

A server whose ids are `<12 hex connection id>/<model>` (starfire's app keys) gets short names: the connection id is
left out when only one connection offers that model (`remote/starfire/qwen3-14b-q4_k_m`), and two connections offering
the same model get the first 6 hex of their id added (`remote/starfire/same-0123ab`). The full form
(`remote/starfire/<id>-<model>`) is always accepted too, for a name that never changes when a connection is added.

Every remote model in `/api/tags` and `/api/show` carries `available`. The engine checks each server's `<url>/models`
every 60 s (5 s timeout) and logs once when a server goes offline and once when it is back. While a server can't be
reached its last-known models stay listed with `available: false` (a server never reached since the engine started
is left out with a warning), and a server that marks a model `"available": false` in its own list (starfire does for
a connection that is down) makes that model unavailable too. A chat to a model the server answers with 503
`model_unavailable` gets 503 "this model is offline right now". Embeddings, `raw` prompts and `_debug_render_only` answer 400 or 501 for remote models, and names
starting `remote/` are never pulled or deleted.

`tools/write_config.py --remote-models <file>` puts a JSON file of these settings into the installed config;
`tools/host/arty-backend.sh engine-config` passes `ARTY_ENGINE_REMOTE_MODELS` from `site.env` (a path inside the
container) that way.

## Other notes

`draft_ctx` is accepted in `speculative` but not passed to the runtime yet. Pulls from Hugging Face send `HF_TOKEN`
from the environment when it is set, for gated repositories. Without it, Hugging Face answers 401 for a repository
that is missing, private or gated, and the engine passes that on as 404; a refused token is a 502.
