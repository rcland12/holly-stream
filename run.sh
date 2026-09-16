#!/bin/bash
#
# Starts the camera on this machine. It streams until stop.sh, restarting itself after errors, but it does not start
# on boot. Run it here, or for every camera at once from another machine with run-all-cameras.sh.
# The last line printed is a one-line result, which run-all-cameras.sh shows for this camera.

set -euo pipefail

cd "$(dirname "$0")/camera"
if [[ ! -f camera.env ]]; then
    echo "$(hostname): no camera/camera.env; copy camera/camera.env.example and set SERVER_HOST and STREAM_NAME" >&2
    exit 1
fi
# shellcheck source=/dev/null
source camera.env

# The GPU is only handed to the container when it can use it: an NVIDIA driver and Docker's NVIDIA runtime
profile=default
case "${ENCODER:-auto}" in
    nvenc) profile=nvidia ;;
    auto)
        if nvidia-smi -L >/dev/null 2>&1 && docker info 2>/dev/null | grep -qi nvidia; then
            profile=nvidia
        fi
        ;;
esac

# Recorded so the container stays stopped if Docker starts it again after a reboot (see holly-camera.sh)
HOLLY_BOOT_ID="$(cat /proc/sys/kernel/random/boot_id)"
export HOLLY_BOOT_ID
docker compose --profile '*' down >/dev/null 2>&1 || true
docker compose --profile "${profile}" up -d

# Wait for frames to flow: the camera, encoder and connection to the server are all checked by then
for ((attempt = 1; attempt <= 30; attempt++)); do
    sleep 1
    state="$(docker inspect -f '{{.State.Status}} {{.RestartCount}}' holly-camera 2>/dev/null || echo "missing 0")"
    frames="$(docker exec holly-camera cat /tmp/frame 2>/dev/null || echo 0)"
    if [[ "${state}" == "running 0" && "${frames}" -gt 0 ]]; then
        echo "$(hostname): started"
        exit 0
    fi
    if [[ "${state}" != running* || "${state##* }" != "0" ]]; then
        break
    fi
done

echo "$(hostname): the camera is not streaming:" >&2
docker logs --tail 5 holly-camera 2>&1 | sed 's/^/  /' >&2
echo "$(hostname): not streaming; see docker logs holly-camera (it keeps retrying until ./stop.sh)"
exit 1
