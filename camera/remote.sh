#!/bin/bash
#
# Controls installed cameras from another machine (e.g. the server) over SSH.
#
# Usage: ./remote.sh <command> [user@]host [[user@]host ...]
#
#   start     start the camera now and on every boot
#   stop      stop the camera and keep it stopped across reboots
#   restart   restart the camera, e.g. after editing its camera.env
#   status    whether each camera is running, and its settings
#   logs      follow one camera's log (Ctrl+C to leave)
#
# Examples:
#   ./remote.sh start rustypi6
#   ./remote.sh status rustypi2 rustypi4 rustypi6
#
# Uses sudo on the Pi, which prompts for a password if the account needs one.

set -euo pipefail

usage() {
    sed -n '3,19p' "$0" | sed 's/^# \{0,1\}//'
    exit 1
}

[[ $# -ge 2 ]] || usage
command="$1"
shift

case "${command}" in
    start) remote='sudo systemctl enable --now holly-camera && sleep 3 && systemctl is-active holly-camera' ;;
    stop) remote='sudo systemctl disable --now holly-camera && echo stopped' ;;
    restart) remote='sudo systemctl restart holly-camera && sleep 3 && systemctl is-active holly-camera' ;;
    status)
        remote='printf "%s (%s): " "$(systemctl is-active holly-camera)" "$(systemctl is-enabled holly-camera 2>/dev/null)"
                grep -E "^(STREAM_NAME|DETECTION|ROTATION|SERVER_HOST)=" /etc/holly-stream/camera.env 2>/dev/null | tr "\n" " "
                echo
                systemctl is-active --quiet holly-camera || journalctl -u holly-camera -n 3 --no-pager -o cat 2>/dev/null' ;;
    logs)
        [[ $# -eq 1 ]] || { echo "logs takes one host" >&2; exit 1; }
        remote='journalctl -u holly-camera -f -n 30 -o cat' ;;
    *) usage ;;
esac

for host in "$@"; do
    echo "==> ${host}"
    # -t so sudo can prompt; errors on one camera don't stop the others
    ssh -t "${host}" "${remote}" || echo "(${host}: ${command} failed)"
done
