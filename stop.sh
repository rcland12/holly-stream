#!/bin/bash
#
# Stops the camera on this machine. Run it here, or for every camera at once with stop-all-cameras.sh.

set -euo pipefail

cd "$(dirname "$0")/camera"
if docker compose --profile '*' down >/dev/null 2>&1; then
    echo "$(hostname): stopped"
else
    echo "$(hostname): could not stop holly-camera"
    exit 1
fi
