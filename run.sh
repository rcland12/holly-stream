#!/bin/bash
#
# Starts holly-stream on this device. It runs until stop.sh and never starts on boot.
# Run it here, or for every camera at once from another machine with run-all-cameras.sh.
# The last line printed is a one-line result, which run-all-cameras.sh shows for this camera.

cd "$(dirname "$0")"
source .env

if [[ ! "${OBJECT_DETECTION:-}" =~ ^(True|False)$ ]]; then
    echo "$(hostname): OBJECT_DETECTION in .env must be True or False (got '${OBJECT_DETECTION:-}')"
    exit 1
fi

if [[ "${OBJECT_DETECTION}" == "True" ]]; then
    docker compose up -d triton

    echo "Waiting to start Holly Stream until Triton is healthy."
    # Read the compose healthcheck instead of exec-ing into the container, which needs a terminal and fails
    # when run over SSH from run-all-cameras.sh
    for ((attempt = 1; attempt <= 60; attempt++)); do
        [[ "$(docker inspect -f '{{.State.Health.Status}}' holly-stream-triton 2>/dev/null)" == "healthy" ]] && break
        if [[ ${attempt} -eq 60 ]]; then
            docker compose down
            echo "$(hostname): Triton was not healthy after 60 seconds; stopped"
            exit 1
        fi
        sleep 1
    done
fi

docker compose up -d app
echo "Holly Stream has started. Performing health check..."
sleep 10

for i in {1..12}; do
    if [ "$(docker container inspect -f '{{.State.Running}}' holly-stream-app 2>/dev/null)" = "true" ]; then
        echo "$(hostname): started"
        exit 0
    fi
    echo "Health check attempt: $i/12"
    sleep 5
done

docker logs --tail 20 holly-stream-app
docker compose down
echo "$(hostname): the app did not stay running; stopped"
exit 1
