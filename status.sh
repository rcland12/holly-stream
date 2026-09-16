#!/bin/bash
#
# Shows whether holly-stream is running on this device. Used by status-all-cameras.sh.

cd "$(dirname "$0")"
source .env 2>/dev/null

state() {
    local status
    if status="$(docker inspect -f '{{.State.Status}}{{if .State.Health}} ({{.State.Health.Status}}){{end}}' "$1" 2>/dev/null)"; then
        echo "${status}"
    else
        echo "not running"
    fi
}

app="$(state holly-stream-app)"
details="app: ${app}"
[[ "${OBJECT_DETECTION:-}" == "True" ]] && details+=", triton: $(state holly-stream-triton)"

if [[ "${app}" == running* ]]; then
    echo "$(hostname): streaming  ${details}  OBJECT_DETECTION=${OBJECT_DETECTION:-}"
else
    echo "$(hostname): stopped  ${details}"
fi
