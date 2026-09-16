#!/bin/bash
#
# Shows whether holly-stream is streaming on this Jetson. Used by status-all-cameras.sh.

cd "$(dirname "$0")"
source .env 2>/dev/null

target="rtmp://${STREAM_IP}:${STREAM_PORT:-1935}/${STREAM_APPLICATION}"

if [ "$(docker inspect -f '{{.State.Running}}' holly-stream-app 2>/dev/null)" != "true" ]; then
    echo "$(hostname): stopped"
    exit 0
fi

started_at="$(docker inspect -f '{{.State.StartedAt}}' holly-stream-app)"
restarts="$(docker inspect -f '{{.RestartCount}}' holly-stream-app)"
logs="$(docker logs --since "${started_at}" holly-stream-app 2>&1)"
if grep -q "Pipeline is PLAYING" <<< "${logs}"; then
    state="streaming"
elif grep -q "Building TensorRT" <<< "${logs}"; then
    state="building TensorRT engine"
else
    state="starting"
fi
[ "${restarts}" != "0" ] && state+=", restarted ${restarts} times"
echo "$(hostname): ${state}  -> ${target}  OBJECT_DETECTION=${OBJECT_DETECTION:-}"
