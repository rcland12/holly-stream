#!/bin/bash
docker compose down
echo "Stopped Triton and other Docker services if they were running."

pid=$(cat .process.pid)
kill "$(($pid + 3))"
rm .process.pid nohup.out
echo "Stopped holly-stream."
