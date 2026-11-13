#!/bin/bash
for i in $(seq 1 7200); do
  "$@"
  status=$?
  [ $status -ne 3 ] && exit $status
  sleep 1
done
echo "gave up waiting for the lock"
