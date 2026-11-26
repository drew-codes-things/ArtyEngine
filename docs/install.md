# Installing ArtyEngine

The short version is in the [README](../README.md#install). This page is the whole walk-through.

The defaults expect everything under `/srv/ai/arty-engine`, with the models in `/srv/ai/arty-engine/models`, the
layout Arty uses. Any other folders work; set them in the config.

```bash
sudo mkdir -p /srv/ai/arty-engine/runtime
sudo cp -r . /srv/ai/arty-engine/src
```

## The model runtime

The engine starts a prebuilt model runtime for each model. ArtyEngine publishes it as zips on its own GitHub
releases (release `runtime-b12110`, tested with the CUDA 12.8 build), one per platform, each holding
`arty-runtime-server`, `arty-runtime-worker` (the RPC worker), their libraries, the runtime's `LICENSE` and
[THIRD_PARTY_NOTICES](../THIRD_PARTY_NOTICES):

```bash
cd /srv/ai/arty-engine/src
sudo python3 -m arty_engine runtime install --platform linux-cuda-12.8-x64
python3 -m arty_engine runtime find
```

`install` downloads `arty-runtime-<platform>.zip`, checks its SHA-256 against the digest pinned in
`arty_engine/runtime_release.json`, or the one given with `--sha256` (take it from the release's `SHA256SUMS` after
checking it). A platform with no pin and no `--sha256` is refused; `--allow-unpinned` takes the digest the release's
`SHA256SUMS` lists instead, which only guards against a broken download. It unpacks the zip into
`/srv/ai/arty-engine/runtime/runtime-b12110-<platform>/` and points the `runtime/server` and `runtime/worker` links
there, so a new release is one changed link; `--replace` moves the old folder aside before the new one goes in and
puts it back if that fails. Platforms: `linux-x64` (CPU only), `linux-cuda-12.8-x64`,
`linux-cuda-13.4-x64`, `linux-vulkan-x64`, `linux-rocm-10.0-x64`, `windows-x64`, `windows-cuda-12.4-x64`,
`windows-cuda-13.4-x64`, `windows-vulkan-x64`. The CUDA builds need NVIDIA's CUDA runtime libraries; list their folder
in `library_path`.

To use a runtime zip from somewhere else, set `runtime_url` in `engine.json` (or pass `--url`) together with
`--sha256`, the zip's SHA-256. `runner_binary` may name the binary, its link, or the runtime folder; in a folder the
engine looks for `arty-runtime-server` and then `server`, and a folder unpacked from an earlier official build zip
works as it is.

`tools/package_runtime.py` builds these zips from the official build archives (it checks the archive's SHA-256, renames
the binaries, adds the licence files and writes `SHA256SUMS`); publishing them is a separate, manual step. For Linux
builds it ships only the libraries the two binaries load (and the backend modules the runtime opens by name, such as
`libggml-cpu-*.so`), gives the runtime's own libraries `libarty-runtime*` names and rewrites the links to them with
`patchelf` (`apt install patchelf`, or `--patchelf <path>`). For Windows builds it ships only the DLLs the two `.exe`
files import (and the backend modules), renames the runtime's own DLLs to `arty-runtime*.dll` and rewrites the import
tables with [lief](https://lief.re) (`pip install 'arty-engine[packaging]'`, needed only to package Windows builds). A
Windows archive without a LICENSE file gets the MIT text from THIRD_PARTY_NOTICES, and the OpenMP runtime ships with
its own licence file.

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
`ARTY_ENGINE_CONFIG`, and without either it uses the built-in defaults in [Configuration](configuration.md).

The unit runs the engine as root, and its `MemoryMax=5G` also covers the runtime processes the engine starts,
because they run in the unit's control group.

Before putting the engine on any address other than loopback, read [Security](security.md).

## Upgrading an install from before 14 Nov 2026

- `engine.json` names the runtime binary as `runner_binary` now, by default `/srv/ai/arty-engine/runtime/server`
  (make that a link to the runtime binary). A config that still names it under the older key stops the engine with
  "unknown settings"; rewrite it with `tools/write_config.py` (`arty-backend.sh engine-config` on a Proxmox
  install) or rename the key by hand.
- Arty's parts read the engine's address from `ARTY_ENGINE_URL`, and the trainer controller from its
  `arty_engine_url` setting.
- Bench configs need a `listen` of their own (the shipped `engine-test.json` uses 11439).
- Convert the store ([Stores in the older layout](store-migration.md)).
- Rolling back to a version from before `gpu_backend` existed: that version stops with "unknown settings
  gpu_backend" on the newer `engine.json`, so put back the older `engine.json` (or rewrite it with the older
  `tools/write_config.py`) together with the older code, before the engine restarts.
