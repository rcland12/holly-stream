#!/bin/bash
#
# Installs or updates the Holly Stream camera service on a Raspberry Pi (Raspberry Pi OS Bookworm or Trixie).
#
# Usage, on the Pi from the repository:  sudo ./camera/install.sh
#
# Installs GStreamer from the Raspberry Pi OS repositories (their libcamera matches the kernel and firmware),
# creates a "holly" service user, installs the streaming script and the holly-camera systemd service, and lets
# the user running this start and stop the camera without a password (so run.sh and stop.sh work over SSH).
#
# The camera never starts on boot or on install; it runs only between run.sh and stop.sh.
# First install: creates /etc/holly-stream/camera.env for you to edit.
# Update (after git pull): installs the new script and restarts the camera if it was running.
# /etc/holly-stream/camera.env is never overwritten.

set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
    echo "Run with sudo: sudo $0" >&2
    exit 1
fi

cd "$(dirname "$0")"
ENV_FILE=/etc/holly-stream/camera.env
SUDOERS_FILE=/etc/sudoers.d/holly-camera

missing=()
for package in alsa-utils gstreamer1.0-alsa gstreamer1.0-libav gstreamer1.0-libcamera gstreamer1.0-plugins-bad \
               gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-tools; do
    [[ "$(dpkg-query -W -f='${Status}' "${package}" 2>/dev/null)" == "install ok installed" ]] || missing+=("${package}")
done
if [[ ${#missing[@]} -gt 0 ]]; then
    apt-get update
    apt-get install -y --no-install-recommends "${missing[@]}"
fi

id holly >/dev/null 2>&1 || useradd --system --no-create-home --shell /usr/sbin/nologin holly
usermod -aG video,audio,render holly

install -m 0755 holly-camera.sh /usr/local/bin/holly-camera.sh
install -m 0644 holly-camera.service /etc/systemd/system/holly-camera.service
# Earlier versions started on boot; this one only runs between run.sh and stop.sh
rm -f /etc/systemd/system/multi-user.target.wants/holly-camera.service
systemctl daemon-reload

# Passwordless start/stop for the installing user, and nothing else, so SSH and phone shortcuts can drive it
if [[ -n "${SUDO_USER:-}" && "${SUDO_USER}" != "root" ]]; then
    systemctl_path="$(command -v systemctl)"
    tmp="$(mktemp)"
    cat > "${tmp}" <<EOF
# Installed by holly-stream camera/install.sh: lets ${SUDO_USER} start and stop the camera without a password.
${SUDO_USER} ALL=(root) NOPASSWD: ${systemctl_path} start holly-camera, ${systemctl_path} stop holly-camera, ${systemctl_path} restart holly-camera
EOF
    if visudo -cf "${tmp}" >/dev/null; then
        install -m 0440 "${tmp}" "${SUDOERS_FILE}"
    else
        echo "Warning: could not validate the sudoers rule; run.sh and stop.sh will need a password." >&2
    fi
    rm -f "${tmp}"
fi

install -d -m 0755 /etc/holly-stream
if [[ ! -f "${ENV_FILE}" ]]; then
    install -m 0644 camera.env.example "${ENV_FILE}"
    cat <<EOF

Installed. Next:
  1. Set SERVER_HOST and STREAM_NAME (and DETECTION, ROTATION if needed) in ${ENV_FILE}
  2. Start the camera with ./run.sh here, or with ./run-all-cameras.sh from another machine
EOF
    exit 0
fi

# Settings added in newer versions get their defaults appended, so an old camera.env keeps working
while IFS= read -r line; do
    key="${line%%=*}"
    if [[ "${line}" =~ ^[A-Z_]+= ]] && ! grep -q "^${key}=" "${ENV_FILE}"; then
        echo "${line}" >> "${ENV_FILE}"
        echo "Added new setting to ${ENV_FILE}: ${line}"
    fi
done < camera.env.example

# The old Docker-based app holds the camera if it is still running
if command -v docker >/dev/null && docker ps --format '{{.Names}}' | grep -q '^holly-stream-app$'; then
    echo "Stopping the old holly-stream-app container, which would hold the camera."
    docker update --restart=no holly-stream-app >/dev/null
    docker stop holly-stream-app >/dev/null
fi

if systemctl is-active --quiet holly-camera; then
    systemctl restart holly-camera
    echo "Updated and restarted holly-camera."
else
    echo "Updated. Start the camera with ./run.sh"
fi
