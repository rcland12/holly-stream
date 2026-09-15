#!/bin/bash

cd "$(dirname "$0")"
source .env

export PATH="${DOCKER_COMPOSE_PATH}:${PATH}"

# Publishing to this device means the local nginx (RTMP + web player) is the server.
if [[ "${STREAM_IP}" == "127.0.0.1" || "${STREAM_IP}" == "localhost" ]]; then
    docker-compose up -d nginx
    echo "Local nginx started. Watch at http://$(hostname -I | awk '{print $1}'):8080/?key=${STREAM_KEY}"
fi

docker-compose up -d app
echo "Holly Stream has started. Performing health check..."

# The first start of a new model builds its TensorRT engine (~10-15 minutes),
# so wait for the pipeline itself rather than just the container.
for i in {1..240}; do
    if [ "$(docker container inspect -f '{{.State.Running}}' holly-stream-app 2>/dev/null)" != "true" ]; then
        echo "Holly Stream STATUS: container is not running"
        docker logs --tail 30 holly-stream-app
        exit 1
    fi
    if docker logs holly-stream-app 2>&1 | grep -q "Pipeline is PLAYING"; then
        echo "Holly Stream STATUS: HEALTHY"
        echo "Follow performance with: docker logs -f holly-stream-app | grep STATS"
        exit 0
    fi
    if docker logs holly-stream-app 2>&1 | grep -q "Building TensorRT"; then
        [ $((i % 6)) -eq 1 ] && echo "Building TensorRT engine for the model (one-time)..."
    fi
    sleep 5
done

echo "Holly Stream STATUS: not playing after 20 minutes"
docker logs --tail 30 holly-stream-app
exit 1
