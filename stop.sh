#!/bin/bash
#
# Stops holly-stream on this device. Run it here, or for every camera at once with stop-all-cameras.sh.

cd "$(dirname "$0")"

if docker compose down; then
    echo "$(hostname): stopped"
else
    echo "$(hostname): could not stop holly-stream"
    exit 1
fi
