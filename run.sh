#!/bin/bash
#
# Starts holly-stream on this Jetson. It runs until stop.sh and never starts on boot; after errors or watchdog
# stalls Docker restarts it. Run it here, or for every camera at once from another machine with run-all-cameras.sh.
# The last line printed is a one-line result, which run-all-cameras.sh shows for this camera.

cd "$(dirname "$0")"
source .env

export PATH="${DOCKER_COMPOSE_PATH}:${PATH}"
# Recorded in the container so it stays stopped if Docker starts it again in a later boot (see app/entrypoint.sh)
export HOLLY_BOOT_ID="$(cat /proc/sys/kernel/random/boot_id)"

# Publishing to this device means the local nginx (RTMP + web player) is the server.
if [[ "${STREAM_IP}" == "127.0.0.1" || "${STREAM_IP}" == "localhost" ]]; then
    docker-compose up -d nginx
    echo "Local nginx started. Watch at http://$(hostname -I | awk '{print $1}'):8080/?key=${STREAM_KEY}"
fi

if ! docker-compose up -d app; then
    echo "$(hostname): docker-compose could not start the app"
    exit 1
fi
echo "Holly Stream has started. Performing health check..."

# The first start of a new model builds its TensorRT engine (~10-15 minutes),
# so wait for the pipeline itself rather than just the container.
started_at="$(docker inspect -f '{{.State.StartedAt}}' holly-stream-app)"
for i in {1..240}; do
    if [ "$(docker container inspect -f '{{.State.Running}}' holly-stream-app 2>/dev/null)" != "true" ]; then
        docker logs --tail 30 holly-stream-app
        echo "$(hostname): the app exited while starting (see the log above)"
        exit 1
    fi
    if docker logs --since "${started_at}" holly-stream-app 2>&1 | grep -q "Pipeline is PLAYING"; then
        echo "Follow performance with: docker logs -f holly-stream-app | grep STATS"
        echo "$(hostname): started"
        exit 0
    fi
    if docker logs --since "${started_at}" holly-stream-app 2>&1 | grep -q "Building TensorRT"; then
        [ $((i % 6)) -eq 1 ] && echo "Building TensorRT engine for the model (one-time)..."
    fi
    sleep 5
done

docker logs --tail 30 holly-stream-app
echo "$(hostname): not playing after 20 minutes"
exit 1
