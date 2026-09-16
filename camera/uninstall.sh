#!/bin/bash
#
# Removes the Holly Stream camera service from this Pi.
#
# Usage, on the Pi from the repository:  sudo ./camera/uninstall.sh [--purge]
#
#   --purge   also delete /etc/holly-stream (this camera's settings) and the holly service user.
#
# The GStreamer packages are left installed.

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

systemctl stop holly-camera.service 2>/dev/null || true
rm -f /etc/systemd/system/holly-camera.service /etc/systemd/system/multi-user.target.wants/holly-camera.service \
    /usr/local/bin/holly-camera.sh /etc/sudoers.d/holly-camera
systemctl daemon-reload

if [[ "${1:-}" == "--purge" ]]; then
    rm -rf /etc/holly-stream
    id holly >/dev/null 2>&1 && userdel holly
    echo "Removed holly-camera, its settings and the holly user."
else
    echo "Removed holly-camera. Settings kept in /etc/holly-stream (delete with --purge)."
fi
