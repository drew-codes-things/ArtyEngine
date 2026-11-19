# Installing ArtyEngine

The short version is in the [README](../README.md#install). This page is the whole walk-through.

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
