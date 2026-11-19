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
| `allowed_hosts` | `[]` | Host names besides localhost, 127.0.0.1 and ::1 that requests may name in `Host` while the engine listens on loopback |
| `allowed_origins` | `[]` | Host names of web pages besides localhost whose requests (with an `Origin` header) are served |
| `gpu_backend` | `auto` | `cuda`, `rocm`, `vulkan` or `cpu`; `auto` picks one at start ([how](#choosing-a-gpu-backend)) |
| `runtimes` | `{}` | Per backend: `{"runner_binary": "<path>", "library_path": ["<folder>"]}`; a backend not listed uses `runner_binary` and `library_path` above |

## The shipped config

The shipped `deploy/engine.json` is Arty's setup, not these defaults: a 24576 context, models kept loaded
(`default_keep_alive` `-1`), n-gram speculation for `qwen3:14b` and `mimo-v2.6:9b`, the Qwen3-0.6B draft listed
as an internal model, `mimo-v2.6:9b` as a thinking leaker, and `gpu_backend` `cuda`.

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

## Other notes

`draft_ctx` is accepted in `speculative` but not passed to the runtime yet. Pulls from Hugging Face send `HF_TOKEN`
from the environment when it is set, for gated repositories. Without it, Hugging Face answers 401 for a repository
that is missing, private or gated, and the engine passes that on as 404; a refused token is a 502.
