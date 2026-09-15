#!/bin/bash
#
# Shows whether the camera on this Pi is streaming, and its main settings.

if [[ ! -f /etc/systemd/system/holly-camera.service ]]; then
    echo "$(hostname): not installed"
    exit 1
fi

settings="$(grep -E '^(STREAM_NAME|DETECTION|ROTATION)=' /etc/holly-stream/camera.env 2>/dev/null | tr '\n' ' ')"
if ! systemctl is-active --quiet holly-camera; then
    echo "$(hostname): stopped  ${settings}"
elif [[ "$(systemctl show -p NRestarts --value holly-camera)" != "0" ]]; then
    echo "$(hostname): failing, restarted $(systemctl show -p NRestarts --value holly-camera) times  ${settings}"
else
    echo "$(hostname): streaming  ${settings}"
fi
