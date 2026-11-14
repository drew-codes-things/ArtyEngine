#!/bin/bash
set -u
HOME_DIR=${BENCH_HOME:-/srv/ai/arty-engine}
CONFIG=${1:-$HOME_DIR/engine-test.json}
BENCH_UNIT=arty-engine-test
READY_TRIES=20
listen=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["listen"])' "$CONFIG") || {
  echo "can't read listen from $CONFIG" >&2
  exit 1
}
url="http://$listen"
systemctl stop $BENCH_UNIT 2>/dev/null
sleep 1
if curl -s -m 2 -o /dev/null "$url/"; then
  echo "port ${listen##*:} from $CONFIG already answers: another engine holds it and the bench would measure that one; not starting $BENCH_UNIT" >&2
  exit 1
fi
systemd-run --unit=$BENCH_UNIT --collect -p MemoryMax=5G -p CPUQuota=500% --working-directory=$HOME_DIR/src \
  /usr/bin/python3 -m arty_engine --config "$CONFIG" >&2 || exit 1
for i in $(seq 1 $READY_TRIES); do
  if ! systemctl is-active --quiet $BENCH_UNIT; then
    echo "$BENCH_UNIT is not running, so nothing of the bench's answers on $url" >&2
    exit 1
  fi
  if curl -sf -m 2 "$url/api/version" >/dev/null; then
    echo "$url"
    exit 0
  fi
  sleep 0.5
done
echo "$BENCH_UNIT did not answer on $url" >&2
exit 1
