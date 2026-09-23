#!/bin/bash
#
# Forced command for the rustyserver-api key. This is what that key is allowed to do, and the only thing it can do:
# run, stop or status on this camera. Nothing else, no shell, no scp, no port forwarding.
#
# It is named in ~/.ssh/authorized_keys as the `command=` of one key, alongside `restrict`:
#
#   restrict,command="/home/russ/dev/holly-stream/remote-action.sh" ssh-ed25519 AAAA... rustyserver-api-cameras
#
# sshd ignores whatever the client asked to run and runs this instead, putting the client's request in
# $SSH_ORIGINAL_COMMAND. So the client cannot choose a command - it can only ask for one of the three below, and
# this script is what decides whether that word is acceptable. That check is the security boundary; the api
# container validates too, but only this side is enforced.
#
# Deployed by rustyserver's api/ssh/deploy-camera-key.sh, which copies this file to every camera and installs the
# authorized_keys line. Copy it by hand if you add a camera outside that script.

set -uo pipefail

action="${SSH_ORIGINAL_COMMAND:-}"

# Exactly three literal words, matched whole. Anything else - an argument, a flag, a second command after a
# semicolon, an empty request - is refused without being interpreted. There is no branch here that runs a string.
case "${action}" in
    run | stop | status) ;;
    *)
        logger -t holly-remote-action "refused: ${action:-<empty>} from ${SSH_CONNECTION%% *}"
        echo "$(hostname): this key may only run 'run', 'stop' or 'status'" >&2
        exit 2
        ;;
esac

logger -t holly-remote-action "${action} from ${SSH_CONNECTION%% *}"

cd "$(dirname "$0")" || {
    echo "$(hostname): cannot reach the holly-stream directory" >&2
    exit 1
}

if [[ ! -x "./${action}.sh" ]]; then
    echo "$(hostname): no ${action}.sh in $(pwd)" >&2
    exit 3
fi

exec "./${action}.sh"
