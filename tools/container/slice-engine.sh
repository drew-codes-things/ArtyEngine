#!/bin/bash
. /srv/ai/arty-engine/bench-args.sh
CONFIG=${2:-/srv/ai/arty-engine/engine-test.json}
LABEL=${3:-engine}
mkdir -p /srv/ai/arty-engine/runs
/srv/ai/arty-engine/engine-restart.sh $CONFIG
timeout 840 python3 bench/bench.py http://127.0.0.1:11438 $LABEL $BENCH_ARGS --output /srv/ai/arty-engine/runs/bench-$LABEL.json --parts "${1:-speed,chat,code,vision}" ${4:-}
