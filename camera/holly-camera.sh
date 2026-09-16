#!/bin/bash
#
# Streams a USB webcam (and its microphone) to the Holly server over SRT. Runs inside the holly-camera container.
#
# The camera sends MJPEG (or raw, or H.264 on cameras that have an encoder); this decodes and re-encodes it as H.264
# on the best hardware available, muxes it with AAC audio into MPEG-TS and sends it over SRT to the server's ingest.
# On an NVIDIA GPU the frames never touch the CPU: NVDEC decodes the JPEGs, NVENC encodes the H.264.
# The server's ingest receives it under STREAM_NAME, and with DETECTION=true its GPU detector draws detections on it.
#
# Settings come from the environment (camera.env, see camera.env.example). Exits on any error or stall; Docker
# restarts it until stop.sh.

set -euo pipefail

log() { echo "[holly-camera] $*" >&2; }

# Frame count for the watchdog, run.sh and status.sh; a count left from before a restart would look like progress
rm -f /tmp/frame
fail() { log "$*"; exit 1; }

# holly-stream only runs between run.sh and stop.sh. run.sh records the boot it was started in; if Docker restarts
# this container after a reboot or power cut, exit cleanly (0, so on-failure does not retry) until run.sh again.
if [[ -n "${HOLLY_BOOT_ID:-}" && "${HOLLY_BOOT_ID}" != "$(cat /proc/sys/kernel/random/boot_id)" ]]; then
    log "Not started by run.sh since this boot; staying stopped until run.sh."
    exit 0
fi

: "${SERVER_HOST:?Set SERVER_HOST in camera/camera.env to the address of the Holly server}"
: "${STREAM_NAME:?Set STREAM_NAME in camera/camera.env, e.g. hollystream2/hollyvideostream2}"
SERVER_PORT="${SERVER_PORT:-8890}"
SRT_LATENCY_MS="${SRT_LATENCY_MS:-200}"
DETECTION="${DETECTION:-false}"

CAMERA_DEVICE="${CAMERA_DEVICE:-auto}"
WIDTH="${WIDTH:-1280}"
HEIGHT="${HEIGHT:-720}"
FPS="${FPS:-30}"
INPUT_FORMAT="${INPUT_FORMAT:-auto}"
CAMERA_CONTROLS="${CAMERA_CONTROLS:-}"
ROTATION="${ROTATION:-0}"
HFLIP="${HFLIP:-false}"

ENCODER="${ENCODER:-auto}"
NVIDIA_GPU="${NVIDIA_GPU:-0}"
BITRATE_KBPS="${BITRATE_KBPS:-4000}"
KEYINT_SECONDS="${KEYINT_SECONDS:-2}"

AUDIO_DEVICE="${AUDIO_DEVICE:-auto}"
AUDIO_CHANNELS="${AUDIO_CHANNELS:-1}"
AUDIO_BITRATE_KBPS="${AUDIO_BITRATE_KBPS:-96}"

WATCHDOG_SECONDS="${WATCHDOG_SECONDS:-15}"

truthy() {
    case "${1,,}" in
        true|yes|1) return 0 ;;
        false|no|0) return 1 ;;
        *) fail "Expected true or false, got '$1'" ;;
    esac
}

[[ "${STREAM_NAME}" =~ ^[A-Za-z0-9_.~-]+(/[A-Za-z0-9_.~-]+)*$ ]] \
    || fail "STREAM_NAME may only contain letters, digits, _ . ~ - and / (got '${STREAM_NAME}')"
[[ "${NVIDIA_GPU}" =~ ^[0-9]+$ ]] || fail "NVIDIA_GPU must be a GPU index from nvidia-smi -L (got '${NVIDIA_GPU}')"
[[ "${ROTATION}" =~ ^(0|90|180|270)$ ]] || fail "ROTATION must be 0, 90, 180 or 270 (got '${ROTATION}')"
detect=0; truthy "${DETECTION}" && detect=1
hflip=0; truthy "${HFLIP}" && hflip=1

# ----- Camera -----

# Video capture nodes; UVC cameras also create a metadata node per camera, which lists no formats
formats_of() { v4l2-ctl -d "$1" --list-formats-ext 2>/dev/null || true; }

if [[ "${CAMERA_DEVICE}" == "auto" ]]; then
    CAMERA_DEVICE=""
    for dev in /dev/video*; do
        if [[ -c "${dev}" ]] && formats_of "${dev}" | grep -q "Size:"; then
            CAMERA_DEVICE="${dev}"
            break
        fi
    done
    [[ -n "${CAMERA_DEVICE}" ]] || fail "No camera found; is it plugged in? (looked at /dev/video*)"
elif [[ "${CAMERA_DEVICE}" != /dev/* ]]; then
    # A stable name from /dev/v4l/by-id, which survives the camera being plugged into another port
    CAMERA_DEVICE="/dev/v4l/by-id/${CAMERA_DEVICE}"
fi
[[ -c "$(readlink -f "${CAMERA_DEVICE}")" ]] || fail "Camera ${CAMERA_DEVICE} does not exist"
camera_name="$(v4l2-ctl -d "${CAMERA_DEVICE}" --info 2>/dev/null | sed -n 's/^\s*Card type\s*:\s*//p' | head -n1)"
formats="$(formats_of "${CAMERA_DEVICE}")"

# Does the camera offer this pixel format (fourcc) at WIDTHxHEIGHT and FPS?
offers() {
    awk -v want="'$1'" -v size="${WIDTH}x${HEIGHT}" -v fps="(${FPS}." '
        /^[ \t]*\[[0-9]+\]:/ { current = $2 }
        /Size:/ { in_size = (current == want && $NF == size) }
        /Interval:/ && in_size && index($0, fps) { found = 1 }
        END { exit !found }' <<< "${formats}"
}

transform=0
[[ "${ROTATION}" != "0" || "${hflip}" == "1" ]] && transform=1

if [[ "${INPUT_FORMAT}" == "auto" ]]; then
    # H.264 straight from the camera costs nothing to send, but it can only be copied when no rotation is needed.
    # Otherwise MJPEG, which USB 2.0 cameras deliver at full frame rate; raw YUYV is a last resort (often 5-10 fps).
    if offers H264 && (( ! transform )); then INPUT_FORMAT=h264
    elif offers MJPG; then INPUT_FORMAT=mjpeg
    elif offers YUYV; then INPUT_FORMAT=yuyv422
    else
        log "The camera offers these formats:"
        echo "${formats}" >&2
        fail "${camera_name:-The camera} has no H264, MJPG or YUYV mode at ${WIDTH}x${HEIGHT}@${FPS}; pick one listed above"
    fi
fi

# Controls such as exposure_dynamic_framerate=0 (see camera.env.example). A control the camera lacks is only a warning.
for control in ${CAMERA_CONTROLS}; do
    v4l2-ctl -d "${CAMERA_DEVICE}" --set-ctrl "${control}" 2>/dev/null || log "Warning: could not set camera control ${control}"
done

# ----- Audio -----

# The ALSA card on the same USB device as the camera, i.e. the webcam's own microphone
camera_audio_card() {
    local video_usb card card_usb
    video_usb="$(readlink -f "/sys/class/video4linux/$(basename "$(readlink -f "${CAMERA_DEVICE}")")/device" 2>/dev/null)" || return 1
    video_usb="${video_usb%:*}"   # .../usb3/3-6/3-6:1.0 -> .../usb3/3-6/3-6
    for card in /sys/class/sound/card*; do
        card_usb="$(readlink -f "${card}/device" 2>/dev/null)" || continue
        if [[ "${card_usb%:*}" == "${video_usb}" && -c "/dev/snd/pcmC${card##*/card}D0c" ]]; then
            echo "plughw:CARD=$(cat "${card}/id"),DEV=0"
            return 0
        fi
    done
    return 1
}

if [[ "${AUDIO_DEVICE}" == "auto" ]]; then
    AUDIO_DEVICE="$(camera_audio_card || echo none)"
fi

# ----- Encoder -----

# One tiny test encode tells whether an encoder really works here (driver, device and container runtime included)
encoder_works() {
    ffmpeg -hide_banner -loglevel error -f lavfi -i color=black:s=320x240 -frames:v 1 "$@" -f null - >/dev/null 2>&1
}
vaapi_device="${VAAPI_DEVICE:-$(ls /dev/dri/renderD* 2>/dev/null | head -n1)}"

if [[ "${INPUT_FORMAT}" == "h264" ]]; then
    ENCODER=copy
elif [[ "${ENCODER}" == "auto" ]]; then
    if encoder_works -init_hw_device "cuda=gpu:${NVIDIA_GPU}" -filter_hw_device gpu \
        -vf format=nv12,hwupload_cuda,colorspace_cuda=range=tv -c:v h264_nvenc -gpu "${NVIDIA_GPU}"; then
        # This also compiled colorspace_cuda's CUDA kernel (cached in CUDA_CACHE_PATH). Compiling it later, with the
        # camera and microphone already open, stalls capture for seconds.
        ENCODER=nvenc
    elif [[ -n "${vaapi_device}" ]] && encoder_works -vaapi_device "${vaapi_device}" -vf format=nv12,hwupload -c:v h264_vaapi; then ENCODER=vaapi
    else ENCODER=x264
    fi
fi

bitrate="${BITRATE_KBPS}k"
gop=$(( FPS * KEYINT_SECONDS * 2 ))   # upper bound only; keyframes are forced every KEYINT_SECONDS below

# Webcams deliver full-range video (JPEG's 0-255). Streams, the detector and players all expect limited range (16-235),
# so it is converted here and flagged, or dark and bright areas come out crushed. The BT.601 matrix is kept as is.
color_args=(-color_range tv -colorspace smpte170m -color_primaries bt709 -color_trc bt709)

# CPU filters for rotation and mirroring
cpu_filters=()
case "${ROTATION}" in
    90) cpu_filters+=("transpose=clock") ;;
    180) cpu_filters+=("hflip" "vflip") ;;
    270) cpu_filters+=("transpose=cclock") ;;
esac
(( hflip )) && cpu_filters+=("hflip")
join() { local IFS=,; echo "$*"; }

input_args=()
video_args=()
case "${ENCODER}" in
    copy)
        video_args=(-c:v copy)
        ;;
    nvenc)
        if [[ "${INPUT_FORMAT}" == "mjpeg" ]] && (( ! transform )); then
            # Fully on the GPU: NVDEC decodes the JPEGs, colorspace_cuda converts the range, NVENC encodes
            input_args=(-hwaccel cuda -hwaccel_device "${NVIDIA_GPU}" -hwaccel_output_format cuda)
            video_args=(-vf colorspace_cuda=range=tv)
        else
            video_args=(-vf "$(join "${cpu_filters[@]}" scale=out_range=tv format=nv12)")
        fi
        video_args+=(
            -c:v h264_nvenc -gpu "${NVIDIA_GPU}" -preset p4 -tune ll -zerolatency 1 -rc cbr -b:v "${bitrate}" -maxrate "${bitrate}"
            -bufsize "${bitrate}" -bf 0 -g "${gop}" -forced-idr 1 -no-scenecut 1 -spatial-aq 1 -profile:v high
        )
        ;;
    vaapi)
        [[ -n "${vaapi_device}" ]] || fail "ENCODER=vaapi, but there is no /dev/dri/renderD* device"
        input_args=(-vaapi_device "${vaapi_device}")
        video_args=(
            -vf "$(join "${cpu_filters[@]}" scale=out_range=tv format=nv12 hwupload)"
            -c:v h264_vaapi -rc_mode CBR -b:v "${bitrate}" -maxrate "${bitrate}" -bufsize "${bitrate}" -bf 0
            -g "${gop}" -profile:v high
        )
        ;;
    x264)
        video_args=(
            -vf "$(join "${cpu_filters[@]}" scale=out_range=tv format=yuv420p)"
            -c:v libx264 -preset veryfast -tune zerolatency -profile:v high -b:v "${bitrate}" -maxrate "${bitrate}"
            -bufsize "${bitrate}" -bf 0 -g "${gop}" -sc_threshold 0
        )
        ;;
    *)
        fail "ENCODER must be auto, nvenc, vaapi or x264 (got '${ENCODER}')"
        ;;
esac
if [[ "${ENCODER}" != "copy" ]]; then
    # Keyframes on a fixed clock line up with 2 second HLS segments downstream, even if the camera's frame rate dips
    video_args+=(-force_key_frames "expr:gte(t,n_forced*${KEYINT_SECONDS})" "${color_args[@]}")
fi

# ----- Pipeline -----

args=(
    -hide_banner -loglevel warning -nostdin -nostats
    -progress pipe:1 -stats_period 2
    # Video: kernel capture timestamps converted to wall clock, the same clock ALSA stamps audio with, so the
    # two stay in sync for as long as the stream runs
    "${input_args[@]}"
    -thread_queue_size 512
    -f v4l2 -input_format "${INPUT_FORMAT}" -video_size "${WIDTH}x${HEIGHT}" -framerate "${FPS}" -ts mono2abs
    -i "${CAMERA_DEVICE}"
)
if [[ "${AUDIO_DEVICE}" != "none" ]]; then
    args+=(-thread_queue_size 1024 -f alsa -channels "${AUDIO_CHANNELS}" -sample_rate 48000 -i "${AUDIO_DEVICE}")
fi
# Skip the first seconds, captured while the encoder was still starting and then sent in a burst. The ingest sets
# each track's RTSP clock from when it arrives, and a burst of backlog puts audio and video on different clocks, which
# RTSP readers such as the detector correct with a timestamp jump that stops them.
args+=(-ss 3)
args+=(-map 0:v "${video_args[@]}" -fps_mode passthrough)
if [[ "${AUDIO_DEVICE}" != "none" ]]; then
    # async resampling absorbs the small drift between the camera's clock and the microphone's
    args+=(-map 1:a -af aresample=async=1:min_hard_comp=0.1 -c:a aac -b:a "${AUDIO_BITRATE_KBPS}k" -ac "${AUDIO_CHANNELS}")
fi
# MediaMTX stream id: publish:<path>:<user>:<password>:<query>. The query tells the server's detector what to do.
# Rotation is already applied here, so the server never needs to rotate this camera.
args+=(
    # Send each audio frame as soon as it is encoded, rather than ~250 ms of them at a time
    -f mpegts -flush_packets 1 -pes_payload_size 0 -muxdelay 0 -muxpreload 0
    -srt_streamid "publish:${STREAM_NAME}:::detect=${detect}"
    "srt://${SERVER_HOST}:${SERVER_PORT}?mode=caller&transtype=live&pkt_size=1316&latency=$(( SRT_LATENCY_MS * 1000 ))&connect_timeout=5000"
)

log "Streaming ${camera_name:-camera} ${CAMERA_DEVICE} ${WIDTH}x${HEIGHT}@${FPS} ${INPUT_FORMAT}, encoder=${ENCODER} ${BITRATE_KBPS}k," \
    "rotation=${ROTATION} hflip=${hflip}, audio=${AUDIO_DEVICE}, detection=${detect}" \
    "-> srt://${SERVER_HOST}:${SERVER_PORT} as ${STREAM_NAME}"

# ffmpeg reports progress on stdout; keep only the latest frame count for the watchdog
ffmpeg "${args[@]}" > >(
    while IFS= read -r line; do
        [[ "${line}" == frame=* ]] && printf '%s\n' "${line#frame=}" > /tmp/frame
    done
) &
ffmpeg_pid=$!

stopping=0
trap 'stopping=1; kill -INT "${ffmpeg_pid}" 2>/dev/null' TERM INT

# Watchdog: ffmpeg can block forever on a camera that stops delivering frames or a network that stops accepting
# them. If the frame count does not move for WATCHDOG_SECONDS, exit non-zero so Docker restarts the pipeline.
last_frame=-1
last_change="${SECONDS}"
while kill -0 "${ffmpeg_pid}" 2>/dev/null; do
    sleep 1 & wait $! || true
    (( stopping )) && break
    frame="$(cat /tmp/frame 2>/dev/null || true)"
    if [[ -n "${frame}" && "${frame}" != "${last_frame}" ]]; then
        last_frame="${frame}"
        last_change="${SECONDS}"
    elif (( SECONDS - last_change > WATCHDOG_SECONDS )); then
        log "No frames sent for ${WATCHDOG_SECONDS} seconds; restarting"
        kill -KILL "${ffmpeg_pid}" 2>/dev/null
        exit 1
    fi
done

if (( stopping )); then
    # Give ffmpeg a moment to close the stream cleanly
    for _ in 1 2 3 4 5; do kill -0 "${ffmpeg_pid}" 2>/dev/null || break; sleep 1; done
    kill -KILL "${ffmpeg_pid}" 2>/dev/null || true
    log "Stopped"
    exit 0
fi

wait "${ffmpeg_pid}" && code=0 || code=$?
log "ffmpeg exited (code ${code})"
exit $(( code == 0 ? 1 : code ))
