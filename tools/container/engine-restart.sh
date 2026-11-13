#!/bin/bash
systemctl stop arty-engine-test 2>/dev/null
sleep 1
systemd-run --unit=arty-engine-test --collect -p MemoryMax=5G -p CPUQuota=500% --working-directory=/srv/ai/arty-engine/src /usr/bin/python3 -m arty_engine --config ${1:-/srv/ai/arty-engine/engine-test.json}
for i in $(seq 1 20); do curl -sf localhost:11438/api/version >/dev/null && exit 0; sleep 0.5; done
echo "engine did not start"; exit 1
