# How Arty uses the engine

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
(`tools/write_config.py`), and sets the controller's `arty_engine_url` setting (`tools/host/controller_backend.py`). A
failed `engine` run ends with one line that says how far it got: the config could not be written or the engine's unit
file could not be installed (nothing started or moved), the engine did not start, the engine started but Arty was not
moved (with the reason), the move did not finish (some parts may already point at the engine), or Arty uses the engine
but the main model did not load or the controller is not running. When the engine did not start or the move did not
finish it prints `status` first, so the parts that already point at the engine can be seen.
