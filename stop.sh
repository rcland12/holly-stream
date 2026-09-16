#!/bin/bash
#
# Stops holly-stream on this Jetson. Run it here, or for every camera at once with stop-all-cameras.sh.

cd "$(dirname "$0")"
source .env

export PATH="${DOCKER_COMPOSE_PATH}:${PATH}"

if docker-compose down; then
    echo "$(hostname): stopped"
else
    echo "$(hostname): could not stop holly-stream"
    exit 1
fi
