#!/bin/bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
SITE=${ARTY_ENGINE_SITE:-$(cd "$HERE/../../.." && pwd)/site.env}
[ ! -f "$SITE" ] || . "$SITE"
CT=${ARTY_CT:-}
MAIN_MODEL=${ARTY_MAIN_MODEL:-qwen3:14b}
CONTROLLER_URL=${ARTY_CONTROLLER_URL:-}
CONTROLLER_SETTINGS=${ARTY_CONTROLLER_SETTINGS:-}
CONTROLLER_PYTHON=${ARTY_CONTROLLER_PYTHON:-}
CONTROLLER_CODE=${ARTY_CONTROLLER_CODE:-}
CONTROLLER_UNIT=${ARTY_CONTROLLER_UNIT:-}
CONTROLLER_MODULE=${ARTY_CONTROLLER_MODULE:-}
ALLOWED_HOSTS=${ARTY_ENGINE_ALLOWED_HOSTS:-}
ALLOWED_ORIGINS=${ARTY_ENGINE_ALLOWED_ORIGINS:-}
MODEL_ROOTS=${ARTY_ENGINE_MODEL_ROOTS:-}
CONTROLLER_TOOL=$HERE/controller_backend.py
NO_CONTROLLER=none
LOCK_OWNER="arty-backend $(date +%s)"

usage() {
  echo "usage: $0 engine|status|settings|engine-config" >&2
  echo "  engine: Arty's brain, tool hub, the web chat's route and the trainer controller use arty-engine (port 11438)" >&2
  echo "  settings: show what $SITE sets, change nothing" >&2
  echo "  engine-config: write the engine's installed engine.json from deploy/ and $SITE; the engine reads it when it next starts" >&2
  exit 2
}

refuse_ollama() {
  echo "Ollama is not installed any more: arty-engine serves every model from its own store, so there is nothing to switch back to." >&2
  echo "Nothing was changed. To run Ollama again, reinstall it from its install archive first." >&2
  exit 2
}

require() {
  [ -n "$2" ] || { echo "set $1 in $SITE ($3); nothing was changed" >&2; exit 2; }
}

check_settings() {
  require ARTY_CT "$CT" "the id of Arty's container"
  require ARTY_CONTROLLER_URL "$CONTROLLER_URL" "the trainer controller's address, or $NO_CONTROLLER"
  has_controller || return 0
  require ARTY_CONTROLLER_SETTINGS "$CONTROLLER_SETTINGS" "the controller's JSON settings file"
  require ARTY_CONTROLLER_PYTHON "$CONTROLLER_PYTHON" "the Python that runs the controller"
  require ARTY_CONTROLLER_CODE "$CONTROLLER_CODE" "the controller's controller.py"
  require ARTY_CONTROLLER_UNIT "$CONTROLLER_UNIT" "the controller's systemd unit"
  require ARTY_CONTROLLER_MODULE "$CONTROLLER_MODULE" "the Python module that runs the controller"
}

has_controller() {
  [ "$CONTROLLER_URL" != "$NO_CONTROLLER" ]
}

trainer_url() {
  if has_controller; then echo "$CONTROLLER_URL"; fi
}

show_settings() {
  echo "site: $SITE"
  echo "container: $CT"
  echo "main model: $MAIN_MODEL"
  echo "controller: $CONTROLLER_URL"
  echo "controller settings: $CONTROLLER_SETTINGS"
  echo "controller python: $CONTROLLER_PYTHON"
  echo "controller code: $CONTROLLER_CODE"
  echo "controller unit: $CONTROLLER_UNIT"
  echo "controller module: $CONTROLLER_MODULE"
  echo "controller tool: $CONTROLLER_TOOL"
  local url
  url=$(trainer_url)
  echo "engine trainer_url: ${url:-(none)}"
  echo "engine allowed hosts: ${ALLOWED_HOSTS:-(none)}"
  echo "engine allowed origins: ${ALLOWED_ORIGINS:-(none)}"
  echo "engine model roots besides its own store: ${MODEL_ROOTS:-(none)}"
}

in_ct() {
  pct exec "$CT" -- bash -c "$1"
}

take_gpu_lock() {
  until in_ct "( set -o noclobber; echo '$LOCK_OWNER' > /root/gpu-lock ) 2>/dev/null"; do
    echo "waiting for the GPU lock, held by: $(in_ct 'cat /root/gpu-lock 2>/dev/null')"
    sleep 30
  done
  trap release_gpu_lock EXIT
}

release_gpu_lock() {
  in_ct "[ \"\$(cat /root/gpu-lock 2>/dev/null)\" = '$LOCK_OWNER' ] && rm -f /root/gpu-lock" || true
}

point_controller_at() {
  has_controller || return 0
  python3 "$CONTROLLER_TOOL" "$1" --settings "$CONTROLLER_SETTINGS" --python "$CONTROLLER_PYTHON" \
    --unit "$CONTROLLER_UNIT" --module "$CONTROLLER_MODULE"
  systemctl daemon-reload
  systemctl restart "$CONTROLLER_UNIT"
}

repeated_option() {
  local option=$1 name
  for name in $2; do printf " %s '%s'" "$option" "$name"; done
}

install_engine_config() {
  in_ct "python3 /srv/ai/arty-engine/src/tools/write_config.py /srv/ai/arty-engine/src/deploy/engine.json /srv/ai/arty-engine/engine.json --trainer-url '$(trainer_url)'$(repeated_option --allowed-host "$ALLOWED_HOSTS")$(repeated_option --allowed-origin "$ALLOWED_ORIGINS")$(repeated_option --model-root "$MODEL_ROOTS")"
}

start_engine() {
  install_engine_config
  in_ct 'set -e
install -m 644 /srv/ai/arty-engine/src/deploy/arty-engine.service /etc/systemd/system/arty-engine.service
systemctl daemon-reload
systemctl enable --now arty-engine
for attempt in $(seq 1 40); do curl -sf localhost:11438/api/version >/dev/null && exit 0; sleep 0.5; done
echo "arty-engine did not answer on 11438" >&2; exit 1'
}

move_ct_to_engine() {
  in_ct 'set -e
sed -i "s#reverse_proxy 127.0.0.1:11436 #reverse_proxy 127.0.0.1:11438 #" /etc/caddy/Caddyfile
if ! caddy validate --config /etc/caddy/Caddyfile --adapter caddyfile >/dev/null 2>&1; then
  sed -i "s#reverse_proxy 127.0.0.1:11438 #reverse_proxy 127.0.0.1:11436 #" /etc/caddy/Caddyfile
  echo "Caddy refused the change; nothing else in the container moved. The controller already points at the engine." >&2
  exit 1
fi
systemctl reload caddy
gate=/etc/systemd/system/brain.service.d/ollama-through-gate.conf
[ ! -f "$gate" ] || mv "$gate" "$gate.off"
install -d /etc/systemd/system/tool-hub.service.d
install -m 644 /srv/ai/arty-engine/src/deploy/brain-through-engine.conf /etc/systemd/system/brain.service.d/brain-through-engine.conf
install -m 644 /srv/ai/arty-engine/src/deploy/tool-hub-through-engine.conf /etc/systemd/system/tool-hub.service.d/tool-hub-through-engine.conf
install -m 644 /srv/ai/arty-engine/src/deploy/arty-backend.env /etc/default/arty-backend
systemctl daemon-reload
systemctl restart tool-hub brain'
  in_ct "curl -s -m 600 localhost:11438/api/generate -d '{\"model\":\"$MAIN_MODEL\",\"keep_alive\":-1}' >/dev/null"
}

show_controller_status() {
  has_controller || { echo "controller: none"; return 0; }
  echo "controller: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("ollama_url", "default (Ollama)"))' "$CONTROLLER_SETTINGS" 2>/dev/null || echo 'default (Ollama)')"
  echo "controller status: $(curl -s -m 5 "$CONTROLLER_URL/status" | head -c 200)"
}

show_status() {
  show_controller_status
  in_ct 'echo "brain: $(systemctl show brain -p Environment | grep -o "OLLAMA_URL=[^ ]*")"
echo "tool hub: $(systemctl show tool-hub -p Environment | grep -o "OLLAMA_URL=[^ ]*" || echo "OLLAMA_URL unset")"
echo "starfire route: $(grep -o "reverse_proxy 127.0.0.1:114[0-9]*" /etc/caddy/Caddyfile)"
echo "ai-pull and friends: $(cat /etc/default/arty-backend 2>/dev/null || echo "OLLAMA_HOST unset")"
echo "arty-engine: $(systemctl is-active arty-engine 2>/dev/null)"
echo "engine holds: $(curl -s localhost:11438/api/ps 2>/dev/null | python3 -c "import json,sys; print([m[\"name\"] for m in json.load(sys.stdin)[\"models\"]])" 2>/dev/null || echo "not running")"'
}

case "${1:-}" in
  ollama) refuse_ollama ;;
  engine|status|settings|engine-config) check_settings ;;
  *) usage ;;
esac

case "$1" in
  engine)
    if has_controller && ! grep -q reload_arty "$CONTROLLER_CODE"; then
      echo "the trainer controller in $CONTROLLER_CODE has no reload_arty yet; update it first" >&2
      exit 1
    fi
    take_gpu_lock
    start_engine
    point_controller_at engine
    move_ct_to_engine
    show_status
    ;;
  status) show_status ;;
  settings) show_settings ;;
  engine-config) install_engine_config ;;
esac
