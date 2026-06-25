#!/bin/bash
#
# Jetson Nano CSI-camera RTMP streaming pipeline.
#
# Pipeline:  nvarguscamerasrc -> nvvidconv (scale) -> nvv4l2h264enc -> flvmux -> rtmpsink
#
# Capture is decoupled from output: the sensor always runs in a known-good
# native mode and the hardware scaler (nvvidconv) resizes to the target. This
# makes every resolution preset work, even sizes the sensor cannot produce
# directly (e.g. 854x480).
#
# =============================================================================
# PRESETS
#
# Two preset axes control the stream. Set them as environment variables (e.g.
# in .env). Both have sensible defaults, so neither is required.
# =============================================================================
#
# STREAM_RESOLUTION  -> output dimensions + aspect ratio + framerate
#   (default: 1080p)
#
#   Preset        Output        Aspect   FPS   Notes
#   -----------   -----------   ------   ---   ----------------------------------
#   1080p         1920x1080     16:9     30    alias: 1080p30
#   1080p60       1920x1080     16:9     60    needs IMX477 (IMX219 maxes 1080p30)
#   720p          1280x720      16:9     30    alias: 720p30
#   720p60        1280x720      16:9     60
#   480p          854x480       16:9     30
#   360p          640x360       16:9     30
#   4:3           1440x1080     4:3      30    alias: 1080p_4:3; stretches unless
#                                              CAPTURE_WIDTH/HEIGHT set to a 4:3
#                                              sensor mode (e.g. 1640x1232)
#   720p_4:3      960x720       4:3      30
#   square        1080x1080     1:1      30    alias: 1:1; stretches (see 4:3 note)
#   portrait      1080x1920     9:16     30    alias: 9:16; stretches
#   custom        CAMERA_WIDTH x CAMERA_HEIGHT @ CAMERA_FPS  (legacy variables)
#   <W>x<H>       explicit, e.g. STREAM_RESOLUTION=1600x900  (fps = CAMERA_FPS|30)
#
# STREAM_QUALITY     -> encoder rate-control + bitrate (bitrate auto-scales with
#   (default: smooth)     the chosen resolution; override with VIDEO_BITRATE)
#
#   Preset      Rate ctrl   Rel. bitrate   Notes
#   ---------   ---------   ------------   -------------------------------------
#   ultra       VBR         highest        best image; can buffer on weak uplinks
#   high        CBR         high           excellent quality, stable bitrate
#   smooth      CBR         medium         default; balanced quality + stability
#   balanced    CBR         lower          good quality/performance trade-off
#   fast        CBR         lowest         best for constrained/flaky uplinks
#
# Other tunables (all optional, with defaults):
#   VIDEO_BITRATE   force exact target bitrate in bps (skips auto-calc)
#   GOP_SECONDS     keyframe interval in seconds (default 2)
#   CAPTURE_WIDTH / CAPTURE_HEIGHT   force the sensor capture mode (e.g. for a
#                                    native non-16:9 mode to avoid stretching)
#   CAMERA_FPS      overrides the preset framerate when set
#   WB_MODE         nvarguscamerasrc white balance (default 1 = auto)
#   GST_DEBUG       GStreamer debug level (default 2)
#
# Examples:
#   STREAM_RESOLUTION=720p  STREAM_QUALITY=fast    ./stream.sh
#   STREAM_RESOLUTION=1080p STREAM_QUALITY=high    ./stream.sh
#   STREAM_RESOLUTION=480p  VIDEO_BITRATE=1500000  ./stream.sh
#   STREAM_RESOLUTION=1600x900 GOP_SECONDS=1       ./stream.sh
# =============================================================================

set -o pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------------------------------------------------------------------------
# Connection / device defaults
# ---------------------------------------------------------------------------
STREAM_IP="${STREAM_IP:-127.0.0.1}"
STREAM_PORT="${STREAM_PORT:-1935}"
STREAM_APPLICATION="${STREAM_APPLICATION:-live}"
STREAM_KEY="${STREAM_KEY:-stream}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
AUDIO_ENABLED="${AUDIO_ENABLED:-False}"
AUDIO_DEVICE="${AUDIO_DEVICE:-hw:2,0}"
WB_MODE="${WB_MODE:-1}"               # nvarguscamerasrc white balance (1 = auto)

# Audio encode settings (only used when AUDIO_ENABLED=True)
AUDIO_BITRATE="${AUDIO_BITRATE:-128000}"
AUDIO_RATE="${AUDIO_RATE:-48000}"
AUDIO_CHANNELS="${AUDIO_CHANNELS:-1}"

# ---------------------------------------------------------------------------
# RESOLUTION PRESET  (output dimensions + aspect ratio + framerate)
#
# Set STREAM_RESOLUTION to one of the named presets below to switch the stream
# size/aspect ratio in one place. Anything not listed falls back to the legacy
# CAMERA_WIDTH / CAMERA_HEIGHT / CAMERA_FPS variables (use STREAM_RESOLUTION=custom
# to force that path explicitly).
#
#   Preset        Output        Aspect   FPS
#   ----------    -----------   ------   ---
#   1080p         1920x1080     16:9     30
#   1080p60       1920x1080     16:9     60   (requires IMX477; IMX219 maxes at 1080p30)
#   720p          1280x720      16:9     30
#   720p60        1280x720      16:9     60
#   480p          854x480       16:9     30
#   360p          640x360       16:9     30
#   4:3           1440x1080     4:3      30
#   720p_4:3      960x720       4:3      30
#   square        1080x1080     1:1      30
#   portrait      1080x1920     9:16     30
# ---------------------------------------------------------------------------
STREAM_RESOLUTION="${STREAM_RESOLUTION:-1080p}"

case "$STREAM_RESOLUTION" in
    1080p|1080p30) OUT_W=1920; OUT_H=1080; OUT_FPS=30 ;;
    1080p60)       OUT_W=1920; OUT_H=1080; OUT_FPS=60 ;;
    720p|720p30)   OUT_W=1280; OUT_H=720;  OUT_FPS=30 ;;
    720p60)        OUT_W=1280; OUT_H=720;  OUT_FPS=60 ;;
    480p)          OUT_W=854;  OUT_H=480;  OUT_FPS=30 ;;
    360p)          OUT_W=640;  OUT_H=360;  OUT_FPS=30 ;;
    4:3|1080p_4:3) OUT_W=1440; OUT_H=1080; OUT_FPS=30 ;;
    720p_4:3)      OUT_W=960;  OUT_H=720;  OUT_FPS=30 ;;
    square|1:1)    OUT_W=1080; OUT_H=1080; OUT_FPS=30 ;;
    portrait|9:16) OUT_W=1080; OUT_H=1920; OUT_FPS=30 ;;
    custom|"")
        OUT_W="${CAMERA_WIDTH:-1920}"
        OUT_H="${CAMERA_HEIGHT:-1080}"
        OUT_FPS="${CAMERA_FPS:-30}"
        log_info "Using custom resolution from CAMERA_WIDTH/HEIGHT/FPS"
        ;;
    *)
        # Allow an explicit WIDTHxHEIGHT string, e.g. STREAM_RESOLUTION=1600x900
        if [[ "$STREAM_RESOLUTION" =~ ^([0-9]+)x([0-9]+)$ ]]; then
            OUT_W="${BASH_REMATCH[1]}"
            OUT_H="${BASH_REMATCH[2]}"
            OUT_FPS="${CAMERA_FPS:-30}"
            log_info "Using explicit resolution ${OUT_W}x${OUT_H}"
        else
            log_warn "Unknown STREAM_RESOLUTION '${STREAM_RESOLUTION}', defaulting to 1080p"
            OUT_W=1920; OUT_H=1080; OUT_FPS=30
        fi
        ;;
esac

# CAMERA_FPS (if explicitly set) always wins, so existing .env files keep working.
[ -n "$CAMERA_FPS" ] && OUT_FPS="$CAMERA_FPS"

# ---------------------------------------------------------------------------
# CAPTURE MODE  (native sensor mode we feed into the hardware scaler)
#
# We capture at a safe 16:9 sensor mode valid on essentially every Jetson CSI
# camera (1080p30 / 720p60) and let nvvidconv scale to the output. Override with
# CAPTURE_WIDTH / CAPTURE_HEIGHT for a sensor-native aspect ratio (e.g. an
# IMX219 4:3 mode 1640x1232) to avoid stretching on non-16:9 presets.
# ---------------------------------------------------------------------------
if [ "$OUT_FPS" -gt 30 ]; then
    CAP_W="${CAPTURE_WIDTH:-1280}"
    CAP_H="${CAPTURE_HEIGHT:-720}"
else
    CAP_W="${CAPTURE_WIDTH:-1920}"
    CAP_H="${CAPTURE_HEIGHT:-1080}"
fi
CAP_FPS="$OUT_FPS"

# Never upscale: if the requested output is larger than the default capture
# mode, capture at the output size and rely on the sensor supporting it.
if [ "$OUT_W" -gt "$CAP_W" ] || [ "$OUT_H" -gt "$CAP_H" ]; then
    log_warn "Output ${OUT_W}x${OUT_H} exceeds default capture ${CAP_W}x${CAP_H}; capturing at output size (sensor must support it)."
    CAP_W="$OUT_W"
    CAP_H="$OUT_H"
fi

# ---------------------------------------------------------------------------
# QUALITY PRESET  (rate control + bitrate, derived from resolution)
#
# Bitrate is computed from pixels x fps x bits-per-pixel so it scales with the
# resolution preset instead of being a fixed number. Set VIDEO_BITRATE (bps) to
# override the computed value entirely.
#
# control-rate on nvv4l2h264enc:  1 = variable (VBR), 2 = constant (CBR).
# CBR is preferred for streaming: a flat bitrate avoids the spikes that overflow
# downstream buffers and cause rebuffering.
# ---------------------------------------------------------------------------
STREAM_QUALITY="${STREAM_QUALITY:-smooth}"

case "$STREAM_QUALITY" in
    ultra)
        BPP_NUM=15; CONTROL_RATE=1; H264_PRESET=4; TWOPASS=0; PEAK_PCT=140
        log_info "Quality: ULTRA (VBR, highest image quality, may buffer on weak uplinks)"
        ;;
    high)
        BPP_NUM=12; CONTROL_RATE=2; H264_PRESET=3; TWOPASS=1; PEAK_PCT=130
        log_info "Quality: HIGH (CBR, excellent quality)"
        ;;
    smooth)
        BPP_NUM=10; CONTROL_RATE=2; H264_PRESET=3; TWOPASS=1; PEAK_PCT=120
        log_info "Quality: SMOOTH (CBR, balanced quality, stable bitrate)"
        ;;
    balanced)
        BPP_NUM=8;  CONTROL_RATE=2; H264_PRESET=2; TWOPASS=1; PEAK_PCT=115
        log_info "Quality: BALANCED (CBR, good quality/performance)"
        ;;
    fast)
        BPP_NUM=6;  CONTROL_RATE=2; H264_PRESET=2; TWOPASS=1; PEAK_PCT=110
        log_info "Quality: FAST (CBR, lowest bitrate, best for constrained uplinks)"
        ;;
    *)
        BPP_NUM=10; CONTROL_RATE=2; H264_PRESET=3; TWOPASS=1; PEAK_PCT=120
        log_warn "Unknown STREAM_QUALITY '${STREAM_QUALITY}', defaulting to SMOOTH"
        ;;
esac
BPP_DEN=100

# Compute (or honor an explicit) target bitrate.
if [ -n "$VIDEO_BITRATE" ]; then
    log_info "Using explicit VIDEO_BITRATE=${VIDEO_BITRATE}"
else
    VIDEO_BITRATE=$(( OUT_W * OUT_H * OUT_FPS * BPP_NUM / BPP_DEN ))
    # Clamp to a sane floor so tiny resolutions still look acceptable.
    [ "$VIDEO_BITRATE" -lt 500000 ] && VIDEO_BITRATE=500000
fi
PEAK_BITRATE="${PEAK_BITRATE:-$(( VIDEO_BITRATE * PEAK_PCT / 100 ))}"

if [ "$CONTROL_RATE" = "2" ] && [ "$TWOPASS" = "1" ]; then
    TWOPASS_PROP="EnableTwopassCBR=1"
else
    TWOPASS_PROP=""
fi
[ "$CONTROL_RATE" = "2" ] && RC_LABEL="CBR" || RC_LABEL="VBR"

# Keyframe interval. Keep it short (default 2s) so the HLS packager can cut
# clean, evenly sized fragments and new viewers join quickly.
GOP_SECONDS="${GOP_SECONDS:-2}"
GOP_SIZE=$(( OUT_FPS * GOP_SECONDS ))

RTMP_URI="rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"

# ---------------------------------------------------------------------------
# Platform info / performance
# ---------------------------------------------------------------------------
log_info "Jetson Nano Streaming Script Starting..."
if [ -f /etc/nv_tegra_release ]; then
    log_info "Jetson L4T Version:"
    cat /etc/nv_tegra_release
fi
if command -v jetson_clocks &> /dev/null; then
    log_info "Maximizing Jetson performance..."
    sudo jetson_clocks 2>/dev/null || log_warn "Could not set jetson_clocks (needs sudo)"
fi

log_info "Stream Configuration:"
echo -e "${BLUE}  Resolution preset:${NC} ${STREAM_RESOLUTION}"
echo -e "${BLUE}  Capture:${NC} ${CAP_W}x${CAP_H} @ ${CAP_FPS}fps (sensor)"
echo -e "${BLUE}  Output:${NC} ${OUT_W}x${OUT_H} @ ${OUT_FPS}fps (scaled)"
echo -e "${BLUE}  Bitrate:${NC} $(( VIDEO_BITRATE / 1000 ))kbps ${RC_LABEL} (peak $(( PEAK_BITRATE / 1000 ))kbps)"
echo -e "${BLUE}  Keyframe interval:${NC} ${GOP_SIZE} frames (${GOP_SECONDS}s)"
echo -e "${BLUE}  Audio:${NC} ${AUDIO_ENABLED} (${AUDIO_CHANNELS}ch @ ${AUDIO_RATE}Hz, $(( AUDIO_BITRATE / 1000 ))kbps)"
echo -e "${BLUE}  Target:${NC} ${RTMP_URI}"

# ---------------------------------------------------------------------------
# Queue tuning
#
# Bounded, leaky-downstream queues keep latency from growing without limit when
# the network backs up: under congestion old frames are dropped (recovering at
# the next keyframe) instead of accumulating into an ever-growing buffer.
# ---------------------------------------------------------------------------
# General in-pipeline queue: small, bounded, no leak (keeps frames in order).
QUEUE_PROPS="max-size-buffers=4 max-size-bytes=0 max-size-time=0"
# Network-facing queue before the RTMP sink: bounded by time, leaks oldest data.
SINK_QUEUE="max-size-buffers=0 max-size-bytes=0 max-size-time=$(( GOP_SECONDS * 1000000000 )) leaky=downstream"

build_camera_source() {
    echo "nvarguscamerasrc sensor-id=${CAMERA_INDEX} \
        do-timestamp=true \
        wbmode=${WB_MODE} \
        tnr-mode=2 \
        tnr-strength=1 ! \
        video/x-raw(memory:NVMM), \
            width=${CAP_W}, \
            height=${CAP_H}, \
            framerate=${CAP_FPS}/1, \
            format=NV12 ! \
        nvvidconv ! \
        video/x-raw(memory:NVMM), \
            width=${OUT_W}, \
            height=${OUT_H}, \
            format=NV12"
}

build_video_encoder() {
    echo "nvv4l2h264enc \
        bitrate=${VIDEO_BITRATE} \
        peak-bitrate=${PEAK_BITRATE} \
        control-rate=${CONTROL_RATE} \
        preset-level=${H264_PRESET} \
        ${TWOPASS_PROP} \
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
        async=false \
        max-lateness=1000000000"
}

# ---------------------------------------------------------------------------
# Audio device resolution (best effort)
# ---------------------------------------------------------------------------
if [ "${AUDIO_ENABLED}" == "True" ]; then
    if [ -n "$AUDIO_DEVICE" ]; then
        log_info "Using configured audio device: ${AUDIO_DEVICE}"
    fi
    if arecord -l 2>/dev/null | grep -q "card"; then
        CARD_NUM=$(echo "${AUDIO_DEVICE}" | sed 's/.*:\([0-9]\+\).*/\1/')
        if ! arecord -l 2>/dev/null | grep -q "card ${CARD_NUM}"; then
            log_warn "Audio device ${AUDIO_DEVICE} not found. Searching for an available device..."
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
        log_warn "Cannot detect audio devices (may be in container). Will attempt ${AUDIO_DEVICE}"
    fi
fi

# ---------------------------------------------------------------------------
# Supervised run loop
#
# gst-launch exits on any camera glitch, network drop or RTMP disconnect. For a
# 24/7 camera we restart it automatically with a short backoff, and forward
# shutdown signals so `docker stop` ends the stream cleanly (-e flushes EOS).
# ---------------------------------------------------------------------------
GST_PID=""
RUNNING=1
GST_DEBUG_LEVEL="${GST_DEBUG:-2}"

shutdown() {
    log_info "Shutdown signal received, stopping stream..."
    RUNNING=0
    if [ -n "$GST_PID" ] && kill -0 "$GST_PID" 2>/dev/null; then
        kill -INT "$GST_PID" 2>/dev/null
        wait "$GST_PID" 2>/dev/null
    fi
    exit 0
}
trap shutdown INT TERM

launch_pipeline() {
    if [ "${AUDIO_ENABLED}" == "True" ]; then
        log_info "Starting stream with audio..."
        GST_DEBUG="$GST_DEBUG_LEVEL" gst-launch-1.0 -e \
            $(build_camera_source) ! \
            $(build_video_encoder) ! \
            h264parse ! \
            queue ${QUEUE_PROPS} ! \
            mux. \
            $(build_audio_pipeline) \
            flvmux name=mux \
                streamable=true \
                latency=100000000 ! \
            queue ${SINK_QUEUE} ! \
            $(build_rtmp_sink) &
    else
        log_info "Starting stream without audio..."
        GST_DEBUG="$GST_DEBUG_LEVEL" gst-launch-1.0 -e \
            $(build_camera_source) ! \
            $(build_video_encoder) ! \
            h264parse ! \
            queue ${QUEUE_PROPS} ! \
            flvmux \
                streamable=true \
                latency=100000000 ! \
            queue ${SINK_QUEUE} ! \
            $(build_rtmp_sink) &
    fi
    GST_PID=$!
}

BACKOFF=2
while [ "$RUNNING" -eq 1 ]; do
    launch_pipeline
    wait "$GST_PID"
    EXIT_CODE=$?
    GST_PID=""

    [ "$RUNNING" -eq 0 ] && break

    log_warn "Pipeline exited (code ${EXIT_CODE}). Restarting in ${BACKOFF}s..."
    sleep "$BACKOFF"
    # Exponential-ish backoff, capped at 10s.
    if [ "$BACKOFF" -lt 10 ]; then
        BACKOFF=$(( BACKOFF + 2 ))
    fi
done
