#!/bin/bash
. /srv/ai/arty-engine/bench-args.sh
mkdir -p /srv/ai/arty-engine/runs
timeout 840 python3 bench/bench.py http://127.0.0.1:11434 ollama $BENCH_ARGS --output /srv/ai/arty-engine/runs/bench-ollama.json --parts "${1:-speed,chat,code,vision}"
