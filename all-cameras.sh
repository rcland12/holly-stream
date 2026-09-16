#!/bin/bash
#
# Runs run.sh, stop.sh or status.sh on every camera over SSH, all at once, and prints one result per camera.
# run-all-cameras.sh, stop-all-cameras.sh and status-all-cameras.sh call this. It is the same file on every branch.
#
# Usage: ./all-cameras.sh <run|stop|status>
#
# Cameras are listed in .env in the repository root (see .env.example):
#   CAMERA_HOSTNAMES=(rustynano rustypi2 rustypi6)
#   CAMERA_USERS=(russ russ russ)                                  optional, default: your SSH config / user
#   CAMERA_REPO_PATHS=(dev/holly-stream dev/holly-stream ...)       optional, default: dev/holly-stream
#
# Every camera only needs run.sh, stop.sh and status.sh at the top of its clone, so cameras running any branch
# (raspbian, jetson, linux) can be mixed. SSH must work without a password prompt (keys), as it does from a phone shortcut.

set -uo pipefail

action="${1:-}"
case "${action}" in
    run|stop|status) ;;
    *) sed -n '3,16p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
esac

root="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -f "${root}/.env" ]]; then
    echo "No ${root}/.env; copy .env.example and list your cameras in CAMERA_HOSTNAMES." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${root}/.env"
if [[ -z "${CAMERA_HOSTNAMES+x}" || ${#CAMERA_HOSTNAMES[@]} -eq 0 ]]; then
    echo "Set CAMERA_HOSTNAMES in ${root}/.env" >&2
    exit 1
fi

# Some run.sh scripts wait for their pipeline to come up, which can take a couple of minutes
TIMEOUT_SECONDS="${CAMERA_COMMAND_TIMEOUT:-150}"
results="$(mktemp -d)"
trap 'rm -rf "${results}"' EXIT

for i in "${!CAMERA_HOSTNAMES[@]}"; do
    host="${CAMERA_HOSTNAMES[$i]}"
    user="${CAMERA_USERS[$i]:-}"
    path="${CAMERA_REPO_PATHS[$i]:-dev/holly-stream}"
    target="${user:+${user}@}${host}"
    (
        output="$(timeout "${TIMEOUT_SECONDS}" ssh -n -o BatchMode=yes -o ConnectTimeout=5 "${target}" \
            "cd ${path@Q} && if [ -x ./${action}.sh ]; then ./${action}.sh; else echo 'no ${action}.sh in ~/${path}'; exit 3; fi" 2>&1)"
        code=$?
        last="$(printf '%s\n' "${output}" | grep -v '^\s*$' | tail -n 1)"
        case "${code}" in
            0) summary="ok${last:+: ${last#"${host}: "}}" ;;
            124) summary="still working after ${TIMEOUT_SECONDS}s (it carries on; check with status-all-cameras.sh)" ;;
            255) summary="unreachable" ;;
            *) summary="failed${last:+: ${last#"${host}: "}}" ;;
        esac
        printf '%s\t%s\n' "${host}" "${summary}" > "${results}/${i}"
    ) &
done
wait

for i in "${!CAMERA_HOSTNAMES[@]}"; do
    IFS=$'\t' read -r host summary < "${results}/${i}"
    printf '%-12s %s\n' "${host}" "${summary}"
done
