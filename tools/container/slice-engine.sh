#!/bin/bash
set -u
HOME_DIR=${BENCH_HOME:-/srv/ai/arty-engine}
LIVE_URL=${LIVE_ENGINE_URL:-http://127.0.0.1:11438}
. $HOME_DIR/bench-args.sh
CONFIG=${2:-$HOME_DIR/engine-test.json}
LABEL=${3:-engine}
OUTPUT=$HOME_DIR/runs/bench-$LABEL.json
mkdir -p $HOME_DIR/runs
live_models() {
  curl -s -m 5 "$LIVE_URL/api/ps" | python3 -c 'import json, sys; print(json.dumps([m["name"] for m in json.load(sys.stdin)["models"]]))' 2>/dev/null || echo null
}
url=$($HOME_DIR/engine-restart.sh $CONFIG) || {
  echo "the bench engine did not start on its own port; no bench was run" >&2
  exit 1
}
before=$(live_models)
timeout 840 python3 bench/bench.py $url $LABEL $BENCH_ARGS --output $OUTPUT --parts "${1:-speed,chat,code,vision}" ${4:-}
status=$?
after=$(live_models)
python3 - "$OUTPUT" "$before" "$after" <<'PY'
import json
import sys

path, before, after = sys.argv[1], json.loads(sys.argv[2]), json.loads(sys.argv[3])
try:
    result = json.load(open(path))
except (OSError, ValueError):
    sys.exit(0)
busy = before is None or after is None or bool(before) or bool(after)
result["live_engine"] = {"before": before, "after": after, "busy": busy}
json.dump(result, open(path, "w"), indent=1)
if busy:
    print(f"note: the live engine held {before} before and {after} after the bench, so it shared the card", file=sys.stderr)
PY
exit $status
