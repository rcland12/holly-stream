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
# Digital zoom: 1.0 uses the whole sensor, 1.3 crops 30% in from the edges. The ISP does it while it is already
# scaling the frame, so it is free, and it costs no detail until the crop falls below the output size.
ZOOM="${ZOOM:-1.0}"
# Sensor mode to capture in, as WxH, or auto to pick the smallest one that reads the whole sensor and can supply
# WIDTHxHEIGHT at FPS, or none to let libcamera choose. auto matters for a widescreen WIDTHxHEIGHT: left to itself
# libcamera picks a mode that is already cropped in on the sensor (on an OV5647, asking for 1280x720 selects a mode
# that sees only 74% of the sensor's width), whereas a full-sensor mode keeps the whole width and trims top and
# bottom instead. Note that this only ever narrows the view - no mode is wider than the lens.
SENSOR_MODE="${SENSOR_MODE:-auto}"
BITRATE_KBPS="${BITRATE_KBPS:-6000}"
KEYINT_SECONDS="${KEYINT_SECONDS:-2}"
# Extra libcamerasrc properties, e.g. "exposure-value=0.5 awb-mode=indoor". See: gst-inspect-1.0 libcamerasrc
CAMERA_OPTIONS="${CAMERA_OPTIONS:-}"
# Element between the camera and the hardware encoder: auto, none, videoconvert or v4l2convert.
# On GStreamer before 1.24 (Raspberry Pi OS Bookworm) the encoder cannot take the camera's buffers directly and the
# pipeline stalls with no error, so a converter is needed; from 1.24 on (Trixie) it is not, and costs ~25% of a core.
VIDEO_CONVERTER="${VIDEO_CONVERTER:-auto}"

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

if ! awk -v z="${ZOOM}" 'BEGIN { exit !(z >= 1.0 && z <= 10.0) }' 2>/dev/null; then
    echo "ZOOM must be between 1.0 and 10.0 (got '${ZOOM}')" >&2
    exit 1
fi

# Prints "<array_w> <array_h> <depth> <mode_w> <mode_h>" for the smallest sensor mode that reads the whole pixel
# array and can supply WIDTHxHEIGHT at FPS. Fails if the sensor cannot be listed or has no such mode.
sensor_geometry() {
    local listing line array_w=0 array_h=0 depth=10 best_w="" best_h=""
    # "0 : ov5647 [2592x1944 10-bit GBRG] (/base/soc/i2c0mux/i2c@1/ov5647@36)"
    local camera_re='^[0-9]+ : [^ ]+ \[([0-9]+)x([0-9]+) ([0-9]+)-bit'
    # "    Modes: 'SGBRG10_CSI2P' : 1296x972 [46.34 fps - (0, 0)/2592x1944 crop]"
    local mode_re='([0-9]+)x([0-9]+) \[([0-9]+)[.0-9]* fps - \([0-9]+, [0-9]+\)/([0-9]+)x([0-9]+) crop\]'
    listing="$(rpicam-hello --list-cameras 2>/dev/null)" || listing="$(libcamera-hello --list-cameras 2>/dev/null)" || return 1
    while IFS= read -r line; do
        if [[ "${line}" =~ ${camera_re} ]]; then
            if (( array_w != 0 )); then break; fi   # only the first camera
            array_w="${BASH_REMATCH[1]}"
            array_h="${BASH_REMATCH[2]}"
            depth="${BASH_REMATCH[3]}"
        elif (( array_w != 0 )) && [[ "${line}" =~ ${mode_re} ]]; then
            local mode_w="${BASH_REMATCH[1]}" mode_h="${BASH_REMATCH[2]}" mode_fps="${BASH_REMATCH[3]}"
            local crop_w="${BASH_REMATCH[4]}" crop_h="${BASH_REMATCH[5]}"
            # 3% of slack: some sensors trim a few columns even in their full-sensor modes
            (( crop_w * 100 >= array_w * 97 && crop_h * 100 >= array_h * 97 )) || continue
            (( mode_w >= WIDTH && mode_h >= HEIGHT && mode_fps >= FPS )) || continue
            if [[ -z "${best_w}" ]] || (( mode_w < best_w )); then
                best_w="${mode_w}"
                best_h="${mode_h}"
            fi
        fi
    done <<< "${listing}"
    [[ -n "${best_w}" ]] || return 1
    echo "${array_w} ${array_h} ${depth} ${best_w} ${best_h}"
}

array_w=0
sensor_config=""
if [[ "${SENSOR_MODE}" != "none" ]]; then
    if geometry="$(sensor_geometry)"; then
        read -r array_w array_h depth mode_w mode_h <<< "${geometry}"
        if [[ "${SENSOR_MODE}" != "auto" ]]; then
            if [[ ! "${SENSOR_MODE}" =~ ^([0-9]+)x([0-9]+)$ ]]; then
                echo "SENSOR_MODE must be auto, none or WxH (got '${SENSOR_MODE}')" >&2
                exit 1
            fi
            mode_w="${BASH_REMATCH[1]}"
            mode_h="${BASH_REMATCH[2]}"
        fi
        sensor_config="sensor/config,width=${mode_w},height=${mode_h},depth=${depth}"
    else
        echo "Could not find a full-sensor mode for ${WIDTH}x${HEIGHT}@${FPS}; letting libcamera choose one." >&2
        echo "The view may be cropped in; see SENSOR_MODE in camera.env." >&2
    fi
fi

# The largest rectangle with the output's shape that fits the sensor, divided by ZOOM and centred. Left unset at
# ZOOM=1.0, where libcamera already crops to the output's shape and keeps as much of the sensor as it can.
scaler_crop=""
if (( array_w != 0 )) && ! [[ "${ZOOM}" =~ ^1(\.0*)?$ ]]; then
    scaler_crop="$(awk -v aw="${array_w}" -v ah="${array_h}" -v ow="${WIDTH}" -v oh="${HEIGHT}" -v z="${ZOOM}" 'BEGIN {
        ar = ow / oh
        w = aw; h = aw / ar
        if (h > ah) { h = ah; w = ah * ar }
        w = int(w / z); h = int(h / z)
        printf "<%d,%d,%d,%d>", int((aw - w) / 2), int((ah - h) / 2), w, h
    }')"
fi

if [[ "${VIDEO_CONVERTER}" == "auto" ]]; then
    gst_version="$(gst-launch-1.0 --version | sed -n 's/^gst-launch-1.0 version \([0-9]*\.[0-9]*\).*/\1/p')"
    if [[ -n "${gst_version}" ]] && (( $(cut -d. -f1 <<< "${gst_version}") == 1 && $(cut -d. -f2 <<< "${gst_version}") < 24 )); then
        VIDEO_CONVERTER="videoconvert"
    else
        VIDEO_CONVERTER="none"
    fi
fi

srt_uri="srt://${SERVER_HOST}:${SERVER_PORT}?mode=caller"
# MediaMTX stream id: publish:<path>:<user>:<password>:<query>. The query carries options for the detector.
streamid="publish:${STREAM_NAME}:::rotate=${ROTATION}&detect=${detect}"

# shellcheck disable=SC2206
camera_options=(${CAMERA_OPTIONS})

pipeline=(
    libcamerasrc "${camera_options[@]}"
    ${sensor_config:+"sensor-config=${sensor_config}"}
    ${scaler_crop:+"scaler-crop=${scaler_crop}"}
    ! "video/x-raw,width=${WIDTH},height=${HEIGHT},framerate=${FPS}/1,format=NV12"
    ! queue max-size-buffers=4
)
if [[ "${VIDEO_CONVERTER}" != "none" ]]; then
    pipeline+=(! "${VIDEO_CONVERTER}")
fi
pipeline+=(
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

echo "Streaming ${WIDTH}x${HEIGHT}@${FPS} ${BITRATE_KBPS}k, sensor mode=${sensor_config:-chosen by libcamera}, zoom=${ZOOM}${scaler_crop:+ ${scaler_crop}}, audio=${AUDIO_DEVICE}, rotation=${ROTATION}, detection=${detect}, converter=${VIDEO_CONVERTER} -> ${srt_uri} as ${STREAM_NAME}"
exec gst-launch-1.0 -e "${pipeline[@]}"
