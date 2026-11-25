# What the engine covers

ArtyEngine is compatible with clients that speak the common `/api` local-model protocol and the `/v1` OpenAI-style
protocol, and it exposes the model runtime's own routes for each model. This page lists what is there, what is not, and
why. [api.md](api.md) has the details of each route.

## The `/api` protocol

| Feature | State |
|---|---|
| `/api/chat`: streaming (NDJSON), tools, images, thinking (on, off, levels), `format` (`"json"` or a schema), `keep_alive`, `options` | Yes |
| `/api/chat` `truncate` (drop the oldest turns when the chat outgrows the context; `false` keeps the error) | Yes |
| `/api/generate`: prompts, `system`, `raw`, an empty prompt to load or unload, streaming | Yes |
| `/api/generate` `suffix` (fill in the middle, for models with fill-in tokens) | Yes |
| `/api/generate` raw with `format` | Yes |
| `/api/embed` (with `dimensions`) and the older `/api/embeddings` | Yes |
| `/api/tags`, `/api/show`, `/api/ps`, `/api/version`, `/api/delete` | Yes |
| `/api/pull` with progress | Yes, from Hugging Face and the engine's short names |
| `/api/create` with a recipe (`FROM`, `PARAMETER`, `TEMPLATE`, `SYSTEM`, `LICENSE`, `ADAPTER`, `MESSAGE`), from a model, a blob or files | Yes |
| `/api/copy` | Yes |
| `HEAD`/`POST /api/blobs/<digest>` | Yes |
| Options that shape a load: `num_ctx`, `num_gpu` (0 for CPU, above 0 for that many layers), `main_gpu`, `num_thread`, `num_batch`, `use_mmap`, `use_mlock`, `numa`, `low_vram` | Yes; `num_gpu` up to 1024 and `main_gpu` up to 63, `num_thread` is cut to the CPU count and `num_batch` to the context |
| Sampling options: `temperature`, `top_k`, `top_p`, `min_p`, `typical_p`, `repeat_penalty`, `repeat_last_n`, `presence_penalty`, `frequency_penalty`, `seed`, `stop`, `mirostat` and its two settings, `num_predict`, `num_keep` | Yes |
| `/api/push` | No: there is no registry to push to (501) |
| `/api/create` `quantize` | No: pull a quantized build instead (501) |
| `/api/generate` `template` override | No (501): the runtime renders with the Jinja template inside the model file, and a template in the other protocol's own template language can't be run; `raw` with a rendered prompt does the same job |
| `/api/generate` raw with images | No (501): use `/api/chat` |
| `/api/generate` returning `context` token lists | No: an older field that clients have moved off; chats carry their history instead |
| Recipe `TEMPLATE` | Stored and shown by `/api/show`, but not used for rendering, for the reason above |

## The `/v1` protocol

| Feature | State |
|---|---|
| `/v1/models`, `/v1/models/<name>` | Yes |
| `/v1/chat/completions`, streamed as server-sent events, with the model's own parameters as defaults; remote models too | Yes |
| `/v1/completions`, `/v1/embeddings`, `/v1/rerank` | Yes |
| `/v1/responses`, `/v1/messages`, `/v1/messages/count_tokens` | Passed to the runtime as they are |
| Runtime fields in a `/v1` body (`grammar`, `json_schema`, `response_format`, `logit_bias`, `n_probs`, `logprobs`, `cache_prompt`, `samplers` and the rest) | Passed to the runtime untouched |
| Errors in the OpenAI shape | Yes |
| `/v1/audio/transcriptions`, `/v1/realtime` | No: the engine runs text, vision and embedding models only |

## The runtime's own routes

| Feature | State |
|---|---|
| `/completion`, `/infill`, `/tokenize`, `/detokenize`, `/apply-template` | Yes, for the model named in the body |
| `/props`, `/slots`, slot `save`, `restore`, `erase` | Yes; each model saves its slots in `state_dir/slots/<model>` |
| `/lora-adapters` (read and set the scales) | Yes |
| `/metrics` | Yes, every loaded model's with a `model` label, plus the engine's own |
| `/health` | Yes |
| Changing `/props` with a POST | No: the runtime only allows it when started with a flag the engine does not pass, since it changes every caller's settings |

## Runtime features

| Feature | How |
|---|---|
| Parallel slots, KV cache type, flash attention, batch sizes | `parallel`, `kv_cache_type`, `flash_attn`, `batch`, `ubatch`, for all models and per model in `model_settings` |
| Speculative decoding (draft model or n-gram) | `speculative` |
| Vision projector files | Pulled with the model and passed at load |
| LoRA adapters | `lora` in `model_settings`, or `ADAPTER` in a recipe |
| Rope and YaRN scaling | `rope_scaling`, `rope_scale`, `rope_freq_base`, `rope_freq_scale`, `yarn_orig_ctx` in `model_settings` |
| Prompt cache reuse | `cache_reuse` in `model_settings`; `cache_ram_mb` for the RAM prompt cache |
| Grammar and JSON schema | `format` on `/api`, or `grammar`, `json_schema` and `response_format` on `/v1` and `/completion` |
| Reranking | Models with rank pooling load in rerank mode |
| Spreading a model over other machines' GPUs | `rpc_workers` ([rpc.md](rpc.md)) |
| Layer placement: `gpu_layers`, `devices`, `tensor_split`, `split_mode`, `main_gpu`, `tensor_overrides`, `cpu_moe`, `cpu_moe_layers`, `fit` | `model_settings` |
| Memory mapping, locking, KV offload, NUMA, threads | `load_mode`, `kv_offload`, `numa`, `threads` in `model_settings` |
| Anything else | `extra_args` per model |
