#!/bin/bash
#
# Removes the Holly Stream camera service from this Pi.
#
# Usage, on the Pi:  sudo ./uninstall.sh [--purge]
#
#   --purge   also delete /etc/holly-stream (this camera's settings) and the holly service user.
#
# The GStreamer packages are left installed.

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

systemctl disable --now holly-camera.service 2>/dev/null || true
rm -f /etc/systemd/system/holly-camera.service /usr/local/bin/holly-camera.sh
systemctl daemon-reload

if [[ "${1:-}" == "--purge" ]]; then
    rm -rf /etc/holly-stream
    id holly >/dev/null 2>&1 && userdel holly
    echo "Removed holly-camera, its settings and the holly user."
else
    echo "Removed holly-camera. Settings kept in /etc/holly-stream (delete with --purge)."
fi
