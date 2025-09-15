#!/bin/bash

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Environment variable validation with better defaults
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
    log_warn "CAMERA_WIDTH not defined. Defaulting to 1920"
    CAMERA_WIDTH=1920
fi
if [ -z "$CAMERA_HEIGHT" ]; then
    log_warn "CAMERA_HEIGHT not defined. Defaulting to 1080"
    CAMERA_HEIGHT=1080
fi
if [ -z "$CAMERA_FPS" ]; then
    log_warn "CAMERA_FPS not defined. Defaulting to 30"
    CAMERA_FPS=30
fi
if [ -z "$AUDIO_ENABLED" ]; then
    log_warn "AUDIO_ENABLED not defined. Defaulting to False"
    AUDIO_ENABLED="False"
fi
if [ -z "$QUALITY_PRESET" ]; then
    log_warn "QUALITY_PRESET not defined. Options: ultra, high, balanced, fast. Defaulting to high"
    QUALITY_PRESET="high"
fi
if [ -z "$AUDIO_DEVICE" ]; then
    log_warn "AUDIO_DEVICE not defined. Defaulting to hw:2,0"
    AUDIO_DEVICE="hw:2,0"
fi

# Display Jetson info
log_info "Jetson Nano Streaming Script Starting..."
if [ -f /etc/nv_tegra_release ]; then
    log_info "Jetson L4T Version:"
    cat /etc/nv_tegra_release
fi

# Check Jetson stats if available
if command -v jetson_clocks &> /dev/null; then
    log_info "Maximizing Jetson performance..."
    sudo jetson_clocks 2>/dev/null || log_warn "Could not set jetson_clocks (needs sudo)"
fi

# Display current Jetson stats if jtop is available
if command -v jetson_stats &> /dev/null; then
    log_info "Current Jetson Stats:"
    timeout 1 jetson_stats 2>/dev/null | head -5 || true
fi

# Set quality parameters based on preset
case "$QUALITY_PRESET" in
    "ultra")
        VIDEO_BITRATE=8000000      # 8 Mbps
        PEAK_BITRATE=12000000       # 12 Mbps peak
        H264_PRESET=1               # UltraFastPreset (lowest for quality)
        CONTROL_RATE=0              # Variable bitrate
        PROFILE="high"
        ENTROPY_CODING="cabac"
        NUM_BFRAMES=2
        INSERT_SPS_PPS=1
        INSERT_VUI=1
        QUANT_I=20                  # Lower = better quality
        QUANT_P=22
        QUANT_B=24
        AUDIO_BITRATE=192000
        AUDIO_RATE=48000
        AUDIO_CHANNELS=2
        log_info "Using ULTRA quality preset (8 Mbps, highest quality)"
        ;;
    "high")
        VIDEO_BITRATE=4000000       # 4 Mbps
        PEAK_BITRATE=6000000        # 6 Mbps peak
        H264_PRESET=2               # FastPreset
        CONTROL_RATE=1              # CBR for stability
        PROFILE="high"
        ENTROPY_CODING="cabac"
        NUM_BFRAMES=1
        INSERT_SPS_PPS=1
        INSERT_VUI=1
        QUANT_I=23
        QUANT_P=25
        QUANT_B=27
        AUDIO_BITRATE=128000
        AUDIO_RATE=48000
        AUDIO_CHANNELS=2
        log_info "Using HIGH quality preset (4 Mbps, excellent quality)"
        ;;
    "balanced")
        VIDEO_BITRATE=2500000       # 2.5 Mbps
        PEAK_BITRATE=4000000        # 4 Mbps peak
        H264_PRESET=3               # MediumPreset
        CONTROL_RATE=1              # CBR
        PROFILE="main"
        ENTROPY_CODING="cabac"
        NUM_BFRAMES=0
        INSERT_SPS_PPS=1
        INSERT_VUI=1
        QUANT_I=25
        QUANT_P=27
        QUANT_B=29
        AUDIO_BITRATE=96000
        AUDIO_RATE=44100
        AUDIO_CHANNELS=1
        log_info "Using BALANCED quality preset (2.5 Mbps, good quality/performance)"
        ;;
    "fast")
        VIDEO_BITRATE=1500000       # 1.5 Mbps
        PEAK_BITRATE=2500000        # 2.5 Mbps peak
        H264_PRESET=4               # UltraFastPreset for speed
        CONTROL_RATE=1              # CBR
        PROFILE="baseline"
        ENTROPY_CODING="cavlc"      # Faster than CABAC
        NUM_BFRAMES=0
        INSERT_SPS_PPS=1
        INSERT_VUI=0
        QUANT_I=28
        QUANT_P=30
        QUANT_B=32
        AUDIO_BITRATE=64000
        AUDIO_RATE=44100
        AUDIO_CHANNELS=1
        log_info "Using FAST preset (1.5 Mbps, optimized for performance)"
        ;;
    *)
        # Default to high
        VIDEO_BITRATE=4000000
        PEAK_BITRATE=6000000
        H264_PRESET=2
        CONTROL_RATE=1
        PROFILE="high"
        ENTROPY_CODING="cabac"
        NUM_BFRAMES=1
        INSERT_SPS_PPS=1
        INSERT_VUI=1
        QUANT_I=23
        QUANT_P=25
        QUANT_B=27
        AUDIO_BITRATE=128000
        AUDIO_RATE=48000
        AUDIO_CHANNELS=2
        log_info "Using default HIGH quality preset"
        ;;
esac

# Calculate GOP size based on FPS (2 second keyframe interval)
GOP_SIZE=$((CAMERA_FPS * 2))

# Build RTMP URI
RTMP_URI="rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"

log_info "Stream Configuration:"
echo -e "${BLUE}  Resolution:${NC} ${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps"
echo -e "${BLUE}  Bitrate:${NC} $(($VIDEO_BITRATE / 1000000))Mbps (peak: $(($PEAK_BITRATE / 1000000))Mbps)"
echo -e "${BLUE}  Audio:${NC} ${AUDIO_ENABLED}"
echo -e "${BLUE}  Target:${NC} ${RTMP_URI}"

# Enhanced queue properties for better buffering
QUEUE_PROPS="max-size-buffers=0 max-size-time=0 max-size-bytes=0 min-threshold-time=0 leaky=downstream"

# Build camera source with advanced ISP settings
build_camera_source() {
    echo "nvarguscamerasrc sensor-id=${CAMERA_INDEX} \
        wbmode=1 \
        tnr-mode=2 \
        tnr-strength=1 \
        ee-mode=2 \
        ee-strength=0.5 \
        aeantibanding=2 \
        exposuretimerange=\"5000000 30000000\" \
        gainrange=\"1 16\" \
        ispdigitalgainrange=\"1 4\" \
        exposurecompensation=0 \
        saturation=1.2 \
        bufapi-version=1 ! \
        video/x-raw(memory:NVMM), \
            width=${CAMERA_WIDTH}, \
            height=${CAMERA_HEIGHT}, \
            framerate=${CAMERA_FPS}/1, \
            format=NV12"
}

# Build video encoder with optimized settings
build_video_encoder() {
    echo "nvv4l2h264enc \
        bitrate=${VIDEO_BITRATE} \
        peak-bitrate=${PEAK_BITRATE} \
        preset-level=${H264_PRESET} \
        control-rate=${CONTROL_RATE} \
        maxperf-enable=1 \
        ratecontrol-enable=1 \
        iframeinterval=${GOP_SIZE} \
        idrinterval=${GOP_SIZE} \
        vbv-size=$((PEAK_BITRATE / CAMERA_FPS * 2)) \
        insert-sps-pps=${INSERT_SPS_PPS} \
        insert-vui=${INSERT_VUI} \
        num-B-Frames=${NUM_BFRAMES} \
        cabac-entropy-coding=$([ \"$ENTROPY_CODING\" = \"cabac\" ] && echo \"1\" || echo \"0\") \
        SliceIntraRefreshEnable=0 \
        SliceIntraRefreshInterval=60 \
        bit-packetization=1 \
        MeasureEncoderLatency=0 \
        EnableTwopassCBR=$([ $CONTROL_RATE -eq 1 ] && echo \"1\" || echo \"0\") \
        qpI=${QUANT_I} \
        qpP=${QUANT_P} \
        qpB=${QUANT_B} \
        EnableMVBufferMeta=0 \
        slice-header-spacing=0 \
        profile=$([ \"$PROFILE\" = \"high\" ] && echo \"2\" || ([ \"$PROFILE\" = \"main\" ] && echo \"1\" || echo \"0\")) \
        num-Ref-Frames=2"
}

# Build audio pipeline with echo cancellation and noise suppression
build_audio_pipeline() {
    if [ "$AUDIO_CHANNELS" -eq "2" ]; then
        AUDIO_FORMAT="S16LE"
        CHANNEL_LAYOUT="interleaved"
    else
        AUDIO_FORMAT="S16LE"
        CHANNEL_LAYOUT="mono"
    fi

    echo "alsasrc device=${AUDIO_DEVICE} \
        buffer-time=20000 \
        latency-time=10000 \
        provide-clock=false \
        do-timestamp=true ! \
        audio/x-raw, \
            format=${AUDIO_FORMAT}, \
            channels=${AUDIO_CHANNELS}, \
            rate=${AUDIO_RATE}, \
            layout=${CHANNEL_LAYOUT} ! \
        audioresample quality=10 ! \
        audioconvert dithering=2 noise-shaping=4 ! \
        audiorate tolerance=1000000000 ! \
        volume volume=1.5 ! \
        voaacenc bitrate=${AUDIO_BITRATE} ! \
        aacparse ! \
        queue ${QUEUE_PROPS} ! \
        mux."
}

# Build RTMP sink with optimized settings
build_rtmp_sink() {
    echo "rtmpsink location=\"${RTMP_URI} live=1 timeout=5\" \
        sync=false \
        async=false \
        max-lateness=2000000000 \
        qos=false \
        enable-last-sample=false \
        blocksize=4096 \
        render-delay=0 \
        throttle-time=0 \
        max-bitrate=$((PEAK_BITRATE * 2)) \
        drop-on-latency=true"
}

# Main streaming pipeline
if [ "${AUDIO_ENABLED}" == "True" ]; then
    log_info "Starting stream with audio..."

    # Check if audio device exists
    if ! arecord -l | grep -q "card ${AUDIO_DEVICE:3:1}"; then
        log_warn "Audio device ${AUDIO_DEVICE} not found. Attempting to find available device..."
        AUDIO_DEVICE=$(arecord -l | grep "card" | head -1 | sed 's/card \([0-9]\).*/hw:\1,0/')
        log_info "Using audio device: ${AUDIO_DEVICE}"
    fi

    GST_DEBUG=2 gst-launch-1.0 -v \
        $(build_camera_source) ! \
        tee name=t ! \
        queue ${QUEUE_PROPS} ! \
        $(build_video_encoder) ! \
        h264parse config-interval=${GOP_SIZE} ! \
        queue ${QUEUE_PROPS} ! \
        mux. \
        $(build_audio_pipeline) \
        flvmux name=mux \
            streamable=true \
            latency=100000000 \
            min-index-interval=1000000000 \
            reserved-moov-update-period=100000000 ! \
        queue ${QUEUE_PROPS} ! \
        $(build_rtmp_sink)
else
    log_info "Starting stream without audio..."

    GST_DEBUG=2 gst-launch-1.0 -v \
        $(build_camera_source) ! \
        $(build_video_encoder) ! \
        h264parse config-interval=${GOP_SIZE} ! \
        queue ${QUEUE_PROPS} ! \
        flvmux \
            streamable=true \
            latency=100000000 \
            min-index-interval=1000000000 \
            reserved-moov-update-period=100000000 ! \
        queue ${QUEUE_PROPS} ! \
        $(build_rtmp_sink)
fi
