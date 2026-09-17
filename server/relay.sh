#!/bin/sh
#
# Run by the ingest (MediaMTX) for every connected camera, and stopped when the camera disconnects.
# Copies the camera to RELAY_URL, with {name} replaced by its stream name, e.g.
#   RELAY_URL=rtmp://nginx:1935/{name}  ->  rtmp://nginx:1935/hollystream4/hollyvideostream4
#
# If the detector is publishing annotated/<name> for this camera, that is relayed instead of the plain stream,
# and the relay switches back to plain whenever the annotated stream stops (e.g. the detector is down), so the
# destination only ever sees a gap of a few seconds. Nothing is re-encoded.
#
# With RELAY_URL unset, cameras are only served by the ingest itself (HLS, WebRTC, RTSP).

name="$1"
api="http://127.0.0.1:9997/v3/paths/get"

if [ -z "${RELAY_URL}" ]; then
    # Nothing to relay to; idle until MediaMTX stops this when the camera disconnects
    exec sleep 2147483647
fi

url="$(printf '%s' "${RELAY_URL}" | sed "s|{name}|${name}|g")"
pid=""

stop_ffmpeg() {
    if [ -n "${pid}" ]; then
        kill "${pid}" 2>/dev/null
        wait "${pid}" 2>/dev/null
        pid=""
    fi
}
trap 'stop_ffmpeg; exit 0' INT TERM

annotated_ready() {
    wget -q -O - "${api}/annotated/${name}" 2>/dev/null | grep -q '"ready":true'
}

source=""
while true; do
    if annotated_ready; then wanted="annotated/${name}"; else wanted="${name}"; fi

    if [ "${wanted}" != "${source}" ] || ! kill -0 "${pid}" 2>/dev/null; then
        if [ "${wanted}" = "${source}" ]; then
            # ffmpeg exited on its own (e.g. the destination is down); don't hammer it
            wait "${pid}" 2>/dev/null
            echo "relay: ${source} stopped, retrying in 5s"
            sleep 5
        fi
        stop_ffmpeg
        source="${wanted}"
        echo "relay: ${source} -> ${url%/*}/..."
        # Reading over RTSP gives ffmpeg the H.264 parameter sets up front, which nginx-rtmp needs in the FLV header
        ffmpeg -hide_banner -loglevel warning -nostdin \
            -rtsp_transport tcp -i "rtsp://127.0.0.1:8554/${source}" \
            -map 0 -c copy -f flv "${url}" &
        pid=$!
    fi

    sleep 2
done
