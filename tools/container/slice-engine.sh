#!/bin/bash
set -u
HOME_DIR=${BENCH_HOME:-/srv/ai/arty-engine}
. $HOME_DIR/bench-args.sh
CONFIG=${2:-$HOME_DIR/engine-test.json}
LABEL=${3:-engine}
mkdir -p $HOME_DIR/runs
url=$($HOME_DIR/engine-restart.sh $CONFIG) || {
  echo "the bench engine did not start on its own port; no bench was run" >&2
  exit 1
}
timeout 840 python3 bench/bench.py $url $LABEL $BENCH_ARGS --output $HOME_DIR/runs/bench-$LABEL.json --parts "${1:-speed,chat,code,vision}" ${4:-}
