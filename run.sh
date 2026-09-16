#!/bin/bash
#
# Starts the camera on this Pi. It streams until stop.sh, restarting itself after errors, but it does not start
# on boot. Run it on the Pi, or for every camera at once from another machine with run-all-cameras.sh.

set -euo pipefail

if [[ ! -f /etc/systemd/system/holly-camera.service ]]; then
    echo "$(hostname): holly-camera is not installed; run: sudo ./camera/install.sh" >&2
    exit 1
fi

if ! sudo -n systemctl start holly-camera 2>/dev/null; then
    echo "$(hostname): could not start holly-camera without a password; re-run: sudo ./camera/install.sh" >&2
    exit 1
fi

# The service stays active while it retries, so also report a camera that is failing to stream
sleep 5
if systemctl is-active --quiet holly-camera && [[ "$(systemctl show -p NRestarts --value holly-camera)" == "0" ]]; then
    echo "$(hostname): started"
else
    echo "$(hostname): started, but the camera is failing:" >&2
    journalctl -u holly-camera -n 5 --no-pager -o cat >&2
    exit 1
fi
