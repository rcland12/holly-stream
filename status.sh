#!/bin/bash
#
# Shows whether the camera on this machine is streaming, and its main settings. Used by status-all-cameras.sh.

cd "$(dirname "$0")/camera"
settings="$(grep -E '^(STREAM_NAME|DETECTION|ENCODER)=' camera.env 2>/dev/null | tr '\n' ' ')"

frames() { docker exec holly-camera cat /tmp/frame 2>/dev/null || echo 0; }

if ! state="$(docker inspect -f '{{.State.Status}} {{.RestartCount}}' holly-camera 2>/dev/null)"; then
    echo "$(hostname): stopped  ${settings}"
    exit 0
fi
if [[ "${state}" != running* ]]; then
    echo "$(hostname): ${state% *}  ${settings}"
    exit 0
fi

# Streaming means frames are going out right now, whatever happened before
before="$(frames)"
sleep 2.5
restarts="${state##* }"
if (( $(frames) > before )); then
    echo "$(hostname): streaming$( (( restarts > 0 )) && echo ", recovered after ${restarts} restarts")  ${settings}"
else
    echo "$(hostname): failing, restarted ${restarts} times, see docker logs holly-camera  ${settings}"
fi
