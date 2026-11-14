#!/bin/bash
set -u
SITE=${ARTY_ENGINE_SITE:-/srv/ai/arty-engine/site.env}
[ ! -f "$SITE" ] || . "$SITE"
LOCK=${SLICE_LOCK:-/root/gpu-lock}
ENGINE_PORT=${SLICE_ENGINE_PORT:-11438}
RELOAD_PORT=${SLICE_RELOAD_PORT:-$ENGINE_PORT}
CONTROLLER=${SLICE_CONTROLLER:-${ARTY_CONTROLLER_URL:-}}
WORK_DIR=${SLICE_WORK_DIR:-/srv/ai/arty-engine/src}
MAIN_MODEL=${ARTY_MAIN_MODEL:-qwen3:14b}
BENCH_UNIT=arty-engine-test
SLICE_ID="slice-$(date +%s)-$$"
CONTROLLER_REFUSED=4
NOT_CONFIGURED=5
if [ -z "$CONTROLLER" ]; then
  echo "no trainer controller set: put ARTY_CONTROLLER_URL in $SITE (none when there is no controller), or set SLICE_CONTROLLER" >&2
  exit $NOT_CONFIGURED
fi
if [ "${1:-}" = --show-settings ]; then
  printf 'site: %s\ncontroller: %s\nlock: %s\nengine port: %s\nreload port: %s\nwork dir: %s\nmain model: %s\n' \
    "$SITE" "$CONTROLLER" "$LOCK" "$ENGINE_PORT" "$RELOAD_PORT" "$WORK_DIR" "$MAIN_MODEL"
  exit 0
fi
if ! ( set -o noclobber; echo "arty-engine $(date +%s)" > $LOCK ) 2>/dev/null; then exit 3; fi
drop_lock() {
  grep -q '^arty-engine ' $LOCK && rm -f $LOCK
}
tell_controller() {
  [ "$CONTROLLER" = none ] && { echo skipped; return 0; }
  local status
  status=$(curl -s -m 200 -o /dev/null -w '%{http_code}' -H 'Content-Type: application/json' -d "{\"id\":\"$SLICE_ID\"}" "$CONTROLLER/$1")
  echo "$status"
  [ "$status" = 200 ]
}
unload() {
  local loaded
  loaded=$(curl -sf localhost:$1/api/ps) || return 0
  for m in $(echo "$loaded" | python3 -c 'import json,sys; print(" ".join(x["name"] for x in json.load(sys.stdin)["models"]))'); do
    curl -s -m 120 localhost:$1/api/generate -d "{\"model\":\"$m\",\"keep_alive\":0}" > /dev/null
  done
}
release() {
  unload $ENGINE_PORT
  if systemctl is-active --quiet $BENCH_UNIT; then systemctl stop $BENCH_UNIT; fi
  sleep 2
  curl -s -m 600 localhost:$RELOAD_PORT/api/generate -d "{\"model\":\"$MAIN_MODEL\"}" > /dev/null
  echo "reloaded on $RELOAD_PORT: $(curl -s localhost:$RELOAD_PORT/api/ps | head -c 300)"
  echo "trainer controller /done: $(tell_controller done)"
  drop_lock
  echo "released at $(date +%s)"
}
if ! answer=$(tell_controller busy); then
  echo "trainer controller at $CONTROLLER did not hand over the GPU (HTTP $answer); not running. SLICE_CONTROLLER=none skips it"
  drop_lock
  exit $CONTROLLER_REFUSED
fi
trap release EXIT
echo "took lock at $(date +%s) (controller /busy: $answer)"
unload $ENGINE_PORT
sleep 3
cd $WORK_DIR
"$@"
