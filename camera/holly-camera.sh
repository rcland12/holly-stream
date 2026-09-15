#!/bin/bash
#
# Streams a Raspberry Pi camera and USB microphone to the Holly server over SRT.
#
# The Pi does no detection: libcamera captures, the VideoCore hardware encoder produces H.264, audio is
# encoded to AAC, and both are muxed into MPEG-TS and sent over SRT, which retransmits packets lost on WiFi.
# The server's ingest receives it under STREAM_NAME and relays it on (e.g. to nginx-rtmp at that application/key).
# With DETECTION=true, the server's GPU detector draws object detections on it first (see server/).
#
# Usage: holly-camera.sh [config]   (default /etc/holly-stream/camera.env)
# Exits on any error; the systemd unit restarts it until stop.sh. Started by run.sh.

set -euo pipefail

CONFIG="${1:-/etc/holly-stream/camera.env}"
if [[ -f "${CONFIG}" ]]; then
    # shellcheck source=/dev/null
    source "${CONFIG}"
fi

: "${SERVER_HOST:?Set SERVER_HOST in /etc/holly-stream/camera.env to the address of the Holly server}"
: "${STREAM_NAME:?Set STREAM_NAME in /etc/holly-stream/camera.env, e.g. hollystream4/hollyvideostream4}"
SERVER_PORT="${SERVER_PORT:-8890}"
SRT_LATENCY_MS="${SRT_LATENCY_MS:-300}"
# 0, or 180 for a camera mounted upside down. The server flips the frames, which costs the Pi nothing.
ROTATION="${ROTATION:-0}"
# true to have the server draw object detections on this camera (needs the detector running on the server)
DETECTION="${DETECTION:-false}"

WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-960}"
FPS="${FPS:-30}"
BITRATE_KBPS="${BITRATE_KBPS:-6000}"
KEYINT_SECONDS="${KEYINT_SECONDS:-2}"
# Extra libcamerasrc properties, e.g. "exposure-value=0.5 awb-mode=indoor". See: gst-inspect-1.0 libcamerasrc
CAMERA_OPTIONS="${CAMERA_OPTIONS:-}"

# "auto" picks the first ALSA capture card, "none" streams video only
AUDIO_DEVICE="${AUDIO_DEVICE:-auto}"
AUDIO_CHANNELS="${AUDIO_CHANNELS:-1}"
AUDIO_BITRATE_KBPS="${AUDIO_BITRATE_KBPS:-96}"

if [[ "${AUDIO_DEVICE}" == "auto" ]]; then
    card="$(arecord -l 2>/dev/null | sed -n 's/^card \([0-9]\+\):.*/\1/p' | head -n1)"
    AUDIO_DEVICE="${card:+plughw:${card},0}"
    AUDIO_DEVICE="${AUDIO_DEVICE:-none}"
fi

if [[ ! "${STREAM_NAME}" =~ ^[A-Za-z0-9_.~-]+(/[A-Za-z0-9_.~-]+)*$ ]]; then
    echo "STREAM_NAME may only contain letters, digits, _ . ~ - and / (got '${STREAM_NAME}')" >&2
    exit 1
fi
if [[ "${ROTATION}" != "0" && "${ROTATION}" != "180" ]]; then
    echo "ROTATION must be 0 or 180 (got '${ROTATION}')" >&2
    exit 1
fi
case "${DETECTION,,}" in
    true|yes|1) detect=1 ;;
    false|no|0) detect=0 ;;
    *) echo "DETECTION must be true or false (got '${DETECTION}')" >&2; exit 1 ;;
esac

srt_uri="srt://${SERVER_HOST}:${SERVER_PORT}?mode=caller"
# MediaMTX stream id: publish:<path>:<user>:<password>:<query>. The query carries options for the detector.
streamid="publish:${STREAM_NAME}:::rotate=${ROTATION}&detect=${detect}"

# shellcheck disable=SC2206
camera_options=(${CAMERA_OPTIONS})

pipeline=(
    libcamerasrc "${camera_options[@]}"
    ! "video/x-raw,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1,format=NV12"
    ! queue max-size-buffers=4
    # repeat_sequence_header puts SPS/PPS on every keyframe so the detector can join mid-stream
    ! v4l2h264enc "extra-controls=controls,video_bitrate=$((BITRATE_KBPS * 1000)),h264_i_frame_period=$((FPS * KEYINT_SECONDS)),repeat_sequence_header=1"
    ! "video/x-h264,profile=high,level=(string)4.1"
    ! h264parse config-interval=-1
    ! queue
    ! mux.
)

if [[ "${AUDIO_DEVICE}" != "none" ]]; then
    pipeline+=(
        alsasrc "device=${AUDIO_DEVICE}"
        ! queue
        ! audioconvert ! audioresample
        ! "audio/x-raw,rate=48000,channels=${AUDIO_CHANNELS}"
        ! avenc_aac "bitrate=$((AUDIO_BITRATE_KBPS * 1000))"
        ! aacparse
        ! queue
        ! mux.
    )
fi

pipeline+=(
    # alignment=7 packs 7 TS packets (1316 bytes) per SRT packet
    mpegtsmux name=mux alignment=7
    ! srtsink "uri=${srt_uri}" "streamid=${streamid}" "latency=${SRT_LATENCY_MS}" wait-for-connection=false
)

echo "Streaming ${WIDTH}x${HEIGHT}@${FPS} ${BITRATE_KBPS}k, audio=${AUDIO_DEVICE}, rotation=${ROTATION}, detection=${detect} -> ${srt_uri} as ${STREAM_NAME}"
exec gst-launch-1.0 -e "${pipeline[@]}"
