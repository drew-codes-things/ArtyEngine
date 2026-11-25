# Sharing another machine's GPU (RPC workers)

A model too big for the engine's own card can be spread over the card of another machine on the network. That
machine runs `arty-engine worker`, which serves its GPU (and, if asked, its CPU) to the runtime the engine starts. The
engine lists its workers in `engine.json` and decides per model whether to use them.

## The engine side

```json
"rpc_workers": [
  {"address": "192.168.1.196:50052", "label": "pc-1060", "memory_mb": 5800}
],
"model_settings": {
  "hf.co/unsloth/Qwen3.6-35B-A3B-GGUF:UD-IQ4_XS": {"rpc": "always", "rpc_when_down": "fail"}
}
```

`rpc_workers` is a list of `"host:port"` strings or objects with `address`, an optional `label` (shown in logs and the
status API; the address when left out) and an optional `memory_mb`, the memory the worker gives. With `memory_mb` the
engine counts that much of a model as living on the worker when it budgets the local card; without it the whole model
is counted locally.

Per model, in `model_settings`:

| Setting | Values | What it does |
|---|---|---|
| `rpc` | `auto` (default), `always`, `never` | `auto` uses the workers only for a model whose estimate does not fit `vram_budget_mb`; `always` for every load; `never` keeps the model local |
| `rpc_workers` | labels | Only these workers (all of `rpc_workers` when left out) |
| `rpc_when_down` | `local` (default), `fail` | What to do when a worker does not answer: load the model on this machine alone, or answer 503 at once |
| `with_workers` | an object of layout settings | Layout settings used only when the workers are in use (below) |

A request with `num_gpu: 0` never uses workers.

### Health checks

Before every load that would use workers, each worker no loaded model holds gets a TCP connect (2 seconds). A worker
that refuses, can't be reached or doesn't answer in time counts as down (an idle worker answers at once, so silence
means it is out of reach or serving another client): it is logged, and the load goes on locally
(`rpc_when_down: "local"`) or fails with 503 naming the worker (`"fail"`), at once, never after the 600 second load
timeout. During a load the workers are checked again every 5 seconds. There a connect that times out counts as unknown,
not down, because a bare runtime worker serves one client at a time with a backlog of one: while the loading runner
holds its connection, further connects wait and time out though the worker is fine. A worker that is refused or
unreachable stops the load, and if the runner exits while loading and a worker's last check was not a clear answer,
the load falls back the same way.

A worker serves one model at a time. When a model needs a worker another loaded model holds, the holder is unloaded
first if it is idle, as it would be to make room on the GPU. If the holder is answering a request, the new model loads
on this machine when it fits (`rpc_when_down: "local"`), and otherwise the request is answered with 503 at once.

While one of the engine's own loaded models uses a worker, that worker is never probed: its health is that model
runner's process. If the worker goes away, the runner fails its next request and exits; the engine drops it at once,
even with requests still running, loads the model again (locally or with the worker, whichever answers) and asks once
more, so the request that found the dead runner still gets its answer. Workers no loaded model uses are probed every
15 seconds for `/api/workers`.

All the named workers must answer: the layout settings below name devices by position, so a load with fewer workers
than planned would put layers in the wrong place.

`GET /api/workers` lists each worker with `label`, `address`, `memory_mb`, `up` (`true`, `false`, or `null` when the
last check timed out), `in_use` (a loaded model holds it), `checked_at`, `latency_ms`, `error` and `models` (the
loaded models using it). `GET /api/status` has the same list next to the loaded models, the runtime release, the
backend and the card. `/api/ps` shows `rpc_workers` for each model, and `/metrics` has `arty_engine_rpc_worker_up`.

## Placing a model's layers

These per-model settings sit in `model_settings` next to the RPC ones, or inside `with_workers` to apply only when
the workers are used (`with_workers` takes the layout settings only).

| Setting | Runtime meaning |
|---|---|
| `gpu_layers` | Layers on the GPUs: a number, `"auto"` or `"all"` |
| `devices` | Devices to use, such as `["CUDA0", "RPC0"]` |
| `tensor_split` | Share of the model per device, such as `[23, 18, 0]` |
| `split_mode` | `none`, `layer` or `row` |
| `main_gpu` | The device for intermediate results |
| `tensor_overrides` | `"<tensor name pattern>=<device>"` items, such as `"blk\\.(2[3-9])\\.ffn_.*_exps\\.=RPC0"` |
| `cpu_moe` | `true` keeps every expert tensor on the CPU |
| `cpu_moe_layers` | Keeps the expert tensors of the first N layers on the CPU |
| `fit` | `false` turns off the runtime's own fitting, needed when the layout is given by hand |

Other per-model settings: `parallel`, `kv_cache_type`, `flash_attn`, `batch`, `ubatch`, `threads`, `cache_reuse`,
`load_mode` (`auto`, `none`, `mmap`, `mlock`, `mmap+mlock`, `dio`), `kv_offload`, `numa`, `rope_scaling`,
`rope_scale`, `rope_freq_base`, `rope_freq_scale`, `yarn_orig_ctx` and `lora` (a list of `{"path", "scale"}`). A
request's `options` can set `num_gpu` (above 0), `main_gpu`, `num_thread`, `num_batch`, `use_mmap`, `use_mlock`, `numa`
and `low_vram`, which win over the model's settings for that load. Unknown keys stop the engine at start with a
message naming them.

## The worker side

```
python3 -m arty_engine worker --allow 192.168.1.16 --devices CUDA0,CPU --cache-dir /var/cache/arty-engine-worker
```

or with a settings file, `--config worker.json`:

```json
{"listen": "0.0.0.0:50052", "allowed_clients": ["192.168.1.16"], "devices": ["CUDA0", "CPU"],
 "cache_dir": "/var/cache/arty-engine-worker", "worker_binary": "/opt/arty-engine/runtime"}
```

| Setting | Flag | What it does |
|---|---|---|
| `listen` | `--listen` | `host:port` to serve on; default `0.0.0.0:50052` |
| `open` | `--open` | Allow a network `listen` with no `allowed_clients`; without it the worker refuses to start that way |
| `allowed_clients` | `--allow` (repeat) | Addresses or networks (`192.168.1.0/24`) allowed in; others are refused and logged |
| `devices` | `--devices` | Devices to share; all of them when left out |
| `cache_dir` | `--cache-dir` | Keeps the model weights sent to it on disk, so a reload sends far less |
| `worker_binary` | `--worker-binary` | The runtime worker, or the runtime folder; `python3 -m arty_engine runtime find` shows what is found |
| `threads` | `--threads` | CPU threads for the CPU device |
| `library_path` | | Folders added to the library path |

The runtime worker has no authentication of its own: whoever can reach its port can run work on the GPU and read
what is sent to it. With `allowed_clients` the runtime worker listens on 127.0.0.1 only and `arty-engine worker`
stands in front of it, passing on connections from the allowed addresses and refusing the rest (at most 16 at once,
with TCP keepalive; if it ever stops taking connections the worker exits with code 3 so its service restarts it).
A network `listen` without `allowed_clients` is refused unless `open` (`--open`) is set; then the worker warns at
start (three times on `0.0.0.0`), and the firewall is the only guard. Keep workers on the home network, never
forwarded from the internet.

### Linux

`deploy/arty-engine-worker.service` runs a worker as a systemd service. Copy the source and the runtime as for the
engine ([install.md](install.md)), put the settings in `/etc/arty-engine/worker.json`, then:

```
sudo cp deploy/arty-engine-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now arty-engine-worker
sudo ufw allow from 192.168.1.16 to any port 50052 proto tcp
```

### Windows

Install Python 3.12 from python.org, unpack the source (for example to `D:\arty-engine\src`), and install the
runtime with `python -m arty_engine runtime install --platform windows-cuda-12.4-x64 --dir D:\arty-engine\runtime`.
Windows has no links, so the runtime stays in its release folder. Put the settings in `D:\arty-engine\worker.json`
with `"worker_binary": "D:\\arty-engine\\runtime\\runtime-b12110-windows-cuda-12.4-x64"` and a `start-worker.cmd`
beside it:

```
cd /d D:\arty-engine\src
"C:\Path\To\Python312\python.exe" -m arty_engine worker --config D:\arty-engine\worker.json > D:\arty-engine\worker.log 2>&1
```

Register it as a scheduled task that starts at boot, from an administrator PowerShell:

```powershell
$action = New-ScheduledTaskAction -Execute "D:\arty-engine\start-worker.cmd"
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) -ExecutionTimeLimit 0
Register-ScheduledTask -TaskName "arty-engine-worker" -Action $action -Trigger $trigger -Settings $settings -User "SYSTEM" -RunLevel Highest
New-NetFirewallRule -DisplayName "arty-engine-worker" -Direction Inbound -Protocol TCP -LocalPort 50052 -RemoteAddress 192.168.1.16 -Action Allow
```

`Unregister-ScheduledTask -TaskName "arty-engine-worker"` and `Remove-NetFirewallRule -DisplayName "arty-engine-worker"`
take it away again. A worker started the older way, with the runtime worker binary straight from a script, keeps
working: the engine only needs the address.
