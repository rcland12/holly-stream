#!/bin/bash

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [ -z "$STREAM_IP" ]; then
    log_warn "STREAM_IP not defined. Defaulting to 127.0.0.1"
    STREAM_IP="127.0.0.1"
fi
if [ -z "$STREAM_PORT" ]; then
    log_warn "STREAM_PORT not defined. Defaulting to 1935"
    STREAM_PORT=1935
fi
if [ -z "$STREAM_APPLICATION" ]; then
    log_warn "STREAM_APPLICATION not defined. Defaulting to 'live'"
    STREAM_APPLICATION="live"
fi
if [ -z "$STREAM_KEY" ]; then
    log_warn "STREAM_KEY not defined. Defaulting to 'stream'"
    STREAM_KEY="stream"
fi
if [ -z "$CAMERA_INDEX" ]; then
    log_warn "CAMERA_INDEX not defined. Defaulting to 0"
    CAMERA_INDEX=0
fi
if [ -z "$CAMERA_WIDTH" ]; then
    log_warn "CAMERA_WIDTH not defined. Defaulting to 1280"
    CAMERA_WIDTH=1280
fi
if [ -z "$CAMERA_HEIGHT" ]; then
    log_warn "CAMERA_HEIGHT not defined. Defaulting to 720"
    CAMERA_HEIGHT=720
fi
if [ -z "$CAMERA_FPS" ]; then
    log_warn "CAMERA_FPS not defined. Defaulting to 30"
    CAMERA_FPS=30
fi
if [ -z "$AUDIO_ENABLED" ]; then
    log_warn "AUDIO_ENABLED not defined. Defaulting to False"
    AUDIO_ENABLED="False"
fi
if [ -z "$STREAM_QUALITY" ]; then
    log_warn "STREAM_QUALITY not defined. Options: ultra, high, smooth, balanced, fast. Defaulting to smooth"
    STREAM_QUALITY="smooth"
fi
if [ -z "$AUDIO_DEVICE" ]; then
    log_warn "AUDIO_DEVICE not defined. Defaulting to hw:2,0"
    AUDIO_DEVICE="hw:2,0"
fi

log_info "Jetson Nano Streaming Script Starting..."
if [ -f /etc/nv_tegra_release ]; then
    log_info "Jetson L4T Version:"
    cat /etc/nv_tegra_release
fi

if command -v jetson_clocks &> /dev/null; then
    log_info "Maximizing Jetson performance..."
    sudo jetson_clocks 2>/dev/null || log_warn "Could not set jetson_clocks (needs sudo)"
fi

case "$STREAM_QUALITY" in
    "ultra")
        VIDEO_BITRATE=12000000
        PEAK_BITRATE=15000000
        H264_PRESET=1
        CONTROL_RATE=0
        AUDIO_BITRATE=128000
        AUDIO_RATE=48000
        AUDIO_CHANNELS=1
        log_info "Using ULTRA quality preset (12 Mbps, maximum quality)"
        ;;
    "high")
        VIDEO_BITRATE=8000000
        PEAK_BITRATE=10000000
        H264_PRESET=1
        CONTROL_RATE=0
        AUDIO_BITRATE=128000
        AUDIO_RATE=48000
        AUDIO_CHANNELS=1
        log_info "Using HIGH quality preset (8 Mbps, excellent quality)"
        ;;
    "smooth")
        VIDEO_BITRATE=6000000
        PEAK_BITRATE=6500000
        H264_PRESET=2
        CONTROL_RATE=1
        AUDIO_BITRATE=128000
        AUDIO_RATE=48000
        AUDIO_CHANNELS=1
        log_info "Using SMOOTH quality preset (6 Mbps CBR, optimized for stable streaming)"
        ;;
    "balanced")
        VIDEO_BITRATE=5000000
        PEAK_BITRATE=6000000
        H264_PRESET=2
        CONTROL_RATE=1
        AUDIO_BITRATE=96000
        AUDIO_RATE=44100
        AUDIO_CHANNELS=1
        log_info "Using BALANCED quality preset (5 Mbps, good quality/performance)"
        ;;
    "fast")
        VIDEO_BITRATE=4000000
        PEAK_BITRATE=4500000
        H264_PRESET=2
        CONTROL_RATE=1
        AUDIO_BITRATE=64000
        AUDIO_RATE=44100
        AUDIO_CHANNELS=1
        log_info "Using FAST preset (4 Mbps, optimized for performance)"
        ;;
    *)
        VIDEO_BITRATE=6000000
        PEAK_BITRATE=6500000
        H264_PRESET=2
        CONTROL_RATE=1
        AUDIO_BITRATE=128000
        AUDIO_RATE=44100
        AUDIO_CHANNELS=1
        log_info "Using default SMOOTH quality preset"
        ;;
esac

GOP_SIZE=$((CAMERA_FPS * 2))
log_info "Keyframe interval: ${GOP_SIZE} frames ($(echo "scale=1; ${GOP_SIZE}/${CAMERA_FPS}" | bc)s)"

RTMP_URI="rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"

log_info "Stream Configuration:"
echo -e "${BLUE}  Resolution:${NC} ${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps"
echo -e "${BLUE}  Bitrate:${NC} $(($VIDEO_BITRATE / 1000000))Mbps (peak: $(($PEAK_BITRATE / 1000000))Mbps)"
echo -e "${BLUE}  Audio:${NC} ${AUDIO_ENABLED} (${AUDIO_CHANNELS} ch @ ${AUDIO_RATE}Hz, ${AUDIO_BITRATE}bps)"
echo -e "${BLUE}  Target:${NC} ${RTMP_URI}"

QUEUE_PROPS="max-size-buffers=0 max-size-time=0 max-size-bytes=0 leaky=downstream"

build_camera_source() {
    echo "nvarguscamerasrc sensor-id=${CAMERA_INDEX} \
        wbmode=1 \
        tnr-mode=2 \
        tnr-strength=1 ! \
        video/x-raw(memory:NVMM), \
            width=${CAMERA_WIDTH}, \
            height=${CAMERA_HEIGHT}, \
            framerate=${CAMERA_FPS}/1, \
            format=NV12"
}

build_video_encoder() {
    echo "nvv4l2h264enc \
        bitrate=${VIDEO_BITRATE} \
        peak-bitrate=${PEAK_BITRATE} \
        preset-level=${H264_PRESET} \
        control-rate=${CONTROL_RATE} \
        maxperf-enable=1 \
        insert-sps-pps=1 \
        insert-vui=1 \
        iframeinterval=${GOP_SIZE} \
        num-B-Frames=0"
}

build_audio_pipeline() {
    echo "alsasrc device=${AUDIO_DEVICE} ! \
        audio/x-raw,format=S16LE,channels=${AUDIO_CHANNELS},rate=${AUDIO_RATE} ! \
        queue ${QUEUE_PROPS} ! \
        voaacenc bitrate=${AUDIO_BITRATE} ! \
        queue ${QUEUE_PROPS} ! \
        mux."
}

build_rtmp_sink() {
    echo "rtmpsink location=\"${RTMP_URI} live=1\" \
        sync=false \
        max-lateness=1000000000"
}

if [ "${AUDIO_ENABLED}" == "True" ]; then
    if [ -n "$AUDIO_DEVICE" ]; then
        log_info "Using configured audio device: ${AUDIO_DEVICE}"
    else
        if arecord -l 2>/dev/null | grep -q "card"; then
            CARD_NUM=$(echo "${AUDIO_DEVICE}" | sed 's/.*:\([0-9]\+\).*/\1/')
            if ! arecord -l 2>/dev/null | grep -q "card ${CARD_NUM}"; then
                log_warn "Audio device ${AUDIO_DEVICE} not found. Attempting to find available device..."
                FIRST_CARD=$(arecord -l 2>/dev/null | grep "^card" | head -1 | sed 's/card \([0-9]\+\).*/\1/')
                if [ -n "$FIRST_CARD" ]; then
                    AUDIO_DEVICE="hw:${FIRST_CARD},0"
                    log_info "Using audio device: ${AUDIO_DEVICE}"
                else
                    log_error "Could not find any audio device. Disabling audio."
                    AUDIO_ENABLED="False"
                fi
            fi
        else
            log_warn "Cannot detect audio devices (may be in container). Will attempt to use ${AUDIO_DEVICE}"
        fi
    fi
fi

if [ "${AUDIO_ENABLED}" == "True" ]; then
    log_info "Starting stream with audio..."

    GST_DEBUG=2 gst-launch-1.0 -v \
        $(build_camera_source) ! \
        $(build_video_encoder) ! \
        h264parse ! \
        queue ${QUEUE_PROPS} ! \
        mux. \
        $(build_audio_pipeline) \
        flvmux name=mux \
            streamable=true \
            latency=100000000 ! \
        queue ${QUEUE_PROPS} ! \
        $(build_rtmp_sink)
else
    log_info "Starting stream without audio..."

    GST_DEBUG=2 gst-launch-1.0 -v \
        $(build_camera_source) ! \
        $(build_video_encoder) ! \
        h264parse ! \
        queue ${QUEUE_PROPS} ! \
        flvmux \
            streamable=true \
            latency=100000000 ! \
        queue ${QUEUE_PROPS} ! \
        $(build_rtmp_sink)
fi
