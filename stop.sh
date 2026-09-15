#!/bin/bash
#
# Stops the camera on this Pi. Run it on the Pi, or for every camera at once with stop-all-cameras.sh.

set -euo pipefail

if [[ ! -f /etc/systemd/system/holly-camera.service ]]; then
    echo "$(hostname): holly-camera is not installed" >&2
    exit 1
fi

if ! sudo -n systemctl stop holly-camera 2>/dev/null; then
    echo "$(hostname): could not stop holly-camera without a password; re-run: sudo ./camera/install.sh" >&2
    exit 1
fi
echo "$(hostname): stopped"
