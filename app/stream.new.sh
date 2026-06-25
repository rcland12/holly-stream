#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

STREAM_IP="${STREAM_IP:-127.0.0.1}"
STREAM_PORT="${STREAM_PORT:-1935}"
STREAM_APPLICATION="${STREAM_APPLICATION:-hollystream1}"
STREAM_KEY="${STREAM_KEY:-stream}"

CAMERA_INDEX="${CAMERA_INDEX:-0}"

CAMERA_WIDTH="${CAMERA_WIDTH:-3264}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-1848}"
CAMERA_FPS="${CAMERA_FPS:-15}"

AUDIO_ENABLED="${AUDIO_ENABLED:-False}"
AUDIO_DEVICE="${AUDIO_DEVICE:-hw:2,0}"

STREAM_QUALITY="${STREAM_QUALITY:-ultra}"

CAMERA_TNR_MODE="${CAMERA_TNR_MODE:-0}"
CAMERA_TNR_STRENGTH="${CAMERA_TNR_STRENGTH:-0}"
CAMERA_WBMODE="${CAMERA_WBMODE:-0}"

BOUNDED_QUEUES="${BOUNDED_QUEUES:-True}"

log_info "Jetson Nano Streaming Script Starting..."
if [ -f /etc/nv_tegra_release ]; then
  log_info "Jetson L4T Version:"
  cat /etc/nv_tegra_release
fi

if command -v jetson_clocks &> /dev/null; then
  log_info "Maximizing Jetson performance..."
  sudo jetson_clocks 2>/dev/null || log_warn "Could not set jetson_clocks (needs sudo on host)"
fi

if [ -z "${VIDEO_BITRATE:-}" ]; then
  if [ "$CAMERA_FPS" -ge 15 ]; then
    VIDEO_BITRATE=20000000
  else
    VIDEO_BITRATE=16000000
  fi
else
  VIDEO_BITRATE="${VIDEO_BITRATE}"
fi

CONTROL_RATE="${CONTROL_RATE:-1}"
H264_PRESET="${H264_PRESET:-1}"
PEAK_BITRATE="${PEAK_BITRATE:-0}"

KEYINT_SECONDS="${KEYINT_SECONDS:-2}"
GOP_SIZE=$((CAMERA_FPS * KEYINT_SECONDS))

RTMP_URI="rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"

log_info "Stream Configuration:"
echo -e "${BLUE}  Resolution:${NC} ${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps"
echo -e "${BLUE}  Bitrate:${NC} $(($VIDEO_BITRATE / 1000000))Mbps CBR"
echo -e "${BLUE}  Keyframes:${NC} every ${KEYINT_SECONDS}s (${GOP_SIZE} frames)"
echo -e "${BLUE}  Audio:${NC} ${AUDIO_ENABLED}"
echo -e "${BLUE}  Target:${NC} ${RTMP_URI}"
echo -e "${BLUE}  ISP:${NC} wbmode=${CAMERA_WBMODE}, tnr-mode=${CAMERA_TNR_MODE}, tnr-strength=${CAMERA_TNR_STRENGTH}"

if [ "${BOUNDED_QUEUES}" == "True" ]; then
  QUEUE_PROPS="max-size-time=4000000000 max-size-buffers=0 max-size-bytes=0 leaky=downstream"
else
  QUEUE_PROPS="max-size-buffers=0 max-size-time=0 max-size-bytes=0 leaky=downstream"
fi

build_camera_source() {
  echo "nvarguscamerasrc sensor-id=${CAMERA_INDEX} \
    wbmode=${CAMERA_WBMODE} \
    tnr-mode=${CAMERA_TNR_MODE} \
    tnr-strength=${CAMERA_TNR_STRENGTH} ! \
    video/x-raw(memory:NVMM),width=${CAMERA_WIDTH},height=${CAMERA_HEIGHT},framerate=${CAMERA_FPS}/1,format=NV12"
}

build_video_encoder() {
  echo "nvv4l2h264enc \
    control-rate=${CONTROL_RATE} \
    bitrate=${VIDEO_BITRATE} \
    preset-level=${H264_PRESET} \
    maxperf-enable=1 \
    iframeinterval=${GOP_SIZE} \
    insert-sps-pps=1 \
    insert-aud=1 \
    insert-vui=1 \
    num-B-Frames=0"
}

build_h264_parse() {
  echo "h264parse config-interval=1"
}

build_audio_pipeline() {
  AUDIO_BITRATE="${AUDIO_BITRATE:-128000}"
  AUDIO_RATE="${AUDIO_RATE:-48000}"
  AUDIO_CHANNELS="${AUDIO_CHANNELS:-1}"

  echo "alsasrc device=${AUDIO_DEVICE} ! \
    audio/x-raw,format=S16LE,channels=${AUDIO_CHANNELS},rate=${AUDIO_RATE} ! \
    queue ${QUEUE_PROPS} ! \
    voaacenc bitrate=${AUDIO_BITRATE} ! \
    queue ${QUEUE_PROPS} ! \
    mux."
}

build_rtmp_sink() {
  echo "rtmpsink location=\"${RTMP_URI} live=1\" sync=false"
}

if [ "${AUDIO_ENABLED}" == "True" ]; then
  if ! arecord -l 2>/dev/null | grep -q "card"; then
    log_warn "Cannot detect audio devices (may be in container). Will attempt ${AUDIO_DEVICE}"
  fi
fi

log_info "Starting stream..."

if [ "${AUDIO_ENABLED}" == "True" ]; then
  GST_DEBUG=2 gst-launch-1.0 -e -v \
    $(build_camera_source) ! \
    queue ${QUEUE_PROPS} ! \
    $(build_video_encoder) ! \
    $(build_h264_parse) ! \
    queue ${QUEUE_PROPS} ! \
    mux. \
    $(build_audio_pipeline) \
    flvmux name=mux streamable=true ! \
    queue ${QUEUE_PROPS} ! \
    $(build_rtmp_sink)
else
  GST_DEBUG=2 gst-launch-1.0 -e -v \
    $(build_camera_source) ! \
    queue ${QUEUE_PROPS} ! \
    $(build_video_encoder) ! \
    $(build_h264_parse) ! \
    queue ${QUEUE_PROPS} ! \
    flvmux streamable=true ! \
    queue ${QUEUE_PROPS} ! \
    $(build_rtmp_sink)
fi
