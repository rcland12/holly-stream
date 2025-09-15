#!/bin/bash

set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ -z "${AUDIO_ENABLED:-}" ]; then
  echo -e "${RED}AUDIO_ENABLED is required (True/False).${NC}"
  exit 1
fi

CAMERA_WIDTH="${CAMERA_WIDTH:-1920}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-1080}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_FPS="${CAMERA_FPS:-30}"
STREAM_IP="${STREAM_IP:-127.0.0.1}"
STREAM_PORT="${STREAM_PORT:-1935}"
STREAM_APPLICATION="${STREAM_APPLICATION:-live}"
STREAM_KEY="${STREAM_KEY:-stream}"
VIDEO_BITRATE="${VIDEO_BITRATE:-6000k}"
STREAM_QUALITY="${STREAM_QUALITY:-high}"
AUDIO_DEVICE="${AUDIO_DEVICE:-}"
USE_ENCODER="${USE_ENCODER:-jetson}"
RESTART_DELAY="${RESTART_DELAY:-3}"
CAMERA_BACKEND="${CAMERA_BACKEND:-auto}"

DIMS="${CAMERA_WIDTH}x${CAMERA_HEIGHT}"

echo -e "${YELLOW}CAMERA_WIDTH=${CAMERA_WIDTH}${NC}"
echo -e "${YELLOW}CAMERA_HEIGHT=${CAMERA_HEIGHT}${NC}"
echo -e "${YELLOW}CAMERA_INDEX=${CAMERA_INDEX}${NC}"
echo -e "${YELLOW}CAMERA_FPS=${CAMERA_FPS}${NC}"
echo -e "${YELLOW}STREAM=${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}${NC}"
echo -e "${YELLOW}STREAM_QUALITY=${STREAM_QUALITY}${NC}"
echo -e "${YELLOW}VIDEO_BITRATE=${VIDEO_BITRATE}${NC}"
echo -e "${YELLOW}AUDIO_DEVICE=${AUDIO_DEVICE}${NC}"
echo -e "${YELLOW}USE_ENCODER=${USE_ENCODER}${NC}"
echo -e "${YELLOW}CAMERA_BACKEND=${CAMERA_BACKEND}${NC}"

detect_jetson_model() {
  if [ -f /etc/nv_tegra_release ]; then
    echo -e "${GREEN}Detected Jetson (L4T):$(cat /etc/nv_tegra_release | tr '\n' ' ')${NC}"
  else
    echo -e "${YELLOW}Non-Jetson or L4T info not found.${NC}"
  fi
}

detect_jetson_model

check_camera_backend() {
  case "$CAMERA_BACKEND" in
    csi)
      if gst-inspect-1.0 nvarguscamerasrc >/dev/null 2>&1; then echo "csi"; else echo "usb"; fi
      return 0;;
    usb)
      if [ -e "/dev/video${CAMERA_INDEX}" ]; then echo "usb"; else echo "csi"; fi
      return 0;;
    auto|*)
      if gst-inspect-1.0 nvarguscamerasrc >/dev/null 2>&1; then
        echo "csi"; return 0
      fi
      if [ -e "/dev/video${CAMERA_INDEX}" ]; then
        echo "usb"; return 0
      fi
      echo "none"; return 1;;
  esac
}

set_quality_presets() {
  case "$STREAM_QUALITY" in
    ultra)
      VIDEO_BITRATE="10000k"
      MAX_BITRATE="12000k"
      BUFFER_SIZE="20000k"
      H264_PRESET=1
      PROFILE=2
      NUM_BFRAMES=2
      ;;
    high)
      VIDEO_BITRATE="6000k"
      MAX_BITRATE="8000k"
      BUFFER_SIZE="12000k"
      H264_PRESET=2
      PROFILE=2
      NUM_BFRAMES=1
      ;;
    medium)
      VIDEO_BITRATE="3000k"
      MAX_BITRATE="4000k"
      BUFFER_SIZE="6000k"
      H264_PRESET=3
      PROFILE=1
      NUM_BFRAMES=0
      ;;
    low)
      VIDEO_BITRATE="1500k"
      MAX_BITRATE="2000k"
      BUFFER_SIZE="3000k"
      H264_PRESET=4
      PROFILE=0
      NUM_BFRAMES=0
      ;;
    *)
      VIDEO_BITRATE="6000k"
      MAX_BITRATE="8000k"
      BUFFER_SIZE="12000k"
      H264_PRESET=2
      PROFILE=2
      NUM_BFRAMES=1
      ;;
  esac
}

set_quality_presets

echo -e "${GREEN}Querying camera capabilities...${NC}"
if command -v v4l2-ctl >/dev/null 2>&1; then
  if [ -e "/dev/video${CAMERA_INDEX}" ]; then
    v4l2-ctl --device=/dev/video"$CAMERA_INDEX" --list-formats-ext | head -40 || true
  else
    echo -e "${YELLOW}/dev/video${CAMERA_INDEX} not present (likely CSI camera).${NC}"
  fi
else
  echo -e "${YELLOW}v4l2-ctl not found; skipping.${NC}"
fi

build_audio_input() {
  if [ "${AUDIO_ENABLED}" != "True" ]; then
    echo ""; return 0
  fi

  if [ -n "${AUDIO_DEVICE:-}" ]; then
    if [[ "$AUDIO_DEVICE" == pulse:* ]]; then
      if command -v pactl >/dev/null 2>&1 && pactl info >/dev/null 2>&1; then
        echo "pulsesrc device=${AUDIO_DEVICE#pulse:} provide-clock=false do-timestamp=true ! audioconvert ! audioresample ! voaacenc bitrate=192000 ! aacparse ! queue ! mux."
        return 0
      else
        echo ""; return 0
      fi
    else
      echo "alsasrc device=${AUDIO_DEVICE} buffer-time=20000 latency-time=10000 provide-clock=false do-timestamp=true ! audioconvert ! audioresample ! voaacenc bitrate=192000 ! aacparse ! queue ! mux."
      return 0
    fi
  fi

  if command -v pactl >/dev/null 2>&1 && pactl info >/dev/null 2>&1; then
    echo "pulsesrc device=default provide-clock=false do-timestamp=true ! audioconvert ! audioresample ! voaacenc bitrate=192000 ! aacparse ! queue ! mux."
    return 0
  fi

  if command -v arecord >/dev/null 2>&1; then
    CARD="$(arecord -l 2>/dev/null | awk '/card [0-9]+:/ {print $2}' | sed 's/://;q')"
    if [ -n "$CARD" ]; then
      echo "alsasrc device=plughw:${CARD},0 buffer-time=20000 latency-time=10000 provide-clock=false do-timestamp=true ! audioconvert ! audioresample ! voaacenc bitrate=192000 ! aacparse ! queue ! mux."
      return 0
    fi
  fi

  echo ""
}

choose_encoder() {
  local req="$1"
  case "$req" in
    jetson|nvv4l2|nvv4l2h264enc|v4l2m2m)
      if gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then
        echo "nvv4l2h264enc"
      else
        echo "x264enc"
      fi
      ;;
    cpu|libx264|x264)
      echo "x264enc"
      ;;
    *)
      if gst-inspect-1.0 nvv4l2h264enc >/dev/null 2>&1; then
        echo "nvv4l2h264enc"
      else
        echo "x264enc"
      fi
      ;;
  esac
}

build_video_encoder() {
  local enc="$1"
  local vbps_k="${VIDEO_BITRATE%k}"
  local max_k="${MAX_BITRATE%k}"

  case "$enc" in
    nvv4l2h264enc)
      local vbv_bytes=$(( ${max_k:-8000} * 1000 / ${CAMERA_FPS} * 2 ))
      [ "$vbv_bytes" -le 0 ] && vbv_bytes=400000

      echo "nvv4l2h264enc \
        bitrate=$(( ${vbps_k:-6000} * 1000 )) \
        peak-bitrate=$(( ${max_k:-8000} * 1000 )) \
        preset-level=${H264_PRESET} \
        control-rate=1 \
        maxperf-enable=1 \
        iframeinterval=$((CAMERA_FPS * 2)) \
        idrinterval=$((CAMERA_FPS * 2)) \
        vbv-size=${vbv_bytes} \
        insert-sps-pps=1 \
        insert-vui=1 \
        num-B-Frames=${NUM_BFRAMES} \
        profile=${PROFILE} ! h264parse config-interval=$((CAMERA_FPS * 2))"
      ;;
    *)
      echo "videoconvert ! x264enc \
        tune=zerolatency \
        speed-preset=medium \
        byte-stream=true \
        key-int-max=$((CAMERA_FPS * 2)) \
        bframes=${NUM_BFRAMES} \
        bitrate=${vbps_k:-6000} \
        aud=true \
        threads=0 ! video/x-h264,profile=high ! h264parse config-interval=$((CAMERA_FPS * 2))"
      ;;
  esac
}

build_csi_source() {
  echo "nvarguscamerasrc sensor-id=${CAMERA_INDEX} \
        wbmode=1 tnr-mode=2 tnr-strength=1 ee-mode=2 ee-strength=0.5 \
        aeantibanding=2 exposuretimerange=\"5000000 30000000\" gainrange=\"1 16\" \
        ispdigitalgainrange=\"1 4\" exposurecompensation=0 saturation=1.0 ! \
        video/x-raw\(memory:NVMM\), width=${CAMERA_WIDTH}, height=${CAMERA_HEIGHT}, framerate=${CAMERA_FPS}/1, format=NV12"
}

build_usb_source() {
  echo "v4l2src device=/dev/video${CAMERA_INDEX} io-mode=2 do-timestamp=true ! \
        image/jpeg,framerate=${CAMERA_FPS}/1 ! jpegdec ! videoconvert ! \
        video/x-raw,format=NV12,width=${CAMERA_WIDTH},height=${CAMERA_HEIGHT},framerate=${CAMERA_FPS}/1 ! \
        nvvidconv ! video/x-raw\(memory:NVMM\),format=NV12,width=${CAMERA_WIDTH},height=${CAMERA_HEIGHT},framerate=${CAMERA_FPS}/1"
}

build_rtmp_sink() {
  echo "flvmux name=mux streamable=true latency=100000000 ! \
        queue max-size-buffers=0 max-size-time=0 max-size-bytes=0 leaky=downstream ! \
        rtmpsink location=\"rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}\" \
                 sync=false async=false qos=false"
}

CAMERA_TYPE="$(check_camera_backend || echo none)"
if [ "$CAMERA_TYPE" = "none" ]; then
  echo -e "${RED}No camera backend detected (nvarguscamerasrc or /dev/video*).${NC}"
  exit 1
fi

GOP_SIZE=$((CAMERA_FPS * 2))
KEYINT_MIN=$CAMERA_FPS
ENCODER="$(choose_encoder "$USE_ENCODER")"
[ "$ENCODER" = "nvv4l2h264enc" ] && echo -e "${GREEN}Using Jetson HW encoder (nvv4l2h264enc).${NC}" || echo -e "${YELLOW}Using CPU encoder (x264enc).${NC}"

AUDIO_INPUT="$(build_audio_input)"
if [ "${AUDIO_ENABLED}" == "True" ] && [ -z "${AUDIO_INPUT}" ]; then
  echo -e "${YELLOW}No usable audio input found; disabling audio to keep streaming.${NC}"
  AUDIO_ENABLED="False"
fi

run_once() {
  local enc="$1"
  local VIDEO_ENC SRC SINK PIPE AUDIO_PIPE

  VIDEO_ENC="$(build_video_encoder "$enc")"
  if [ "$CAMERA_TYPE" = "csi" ]; then
    SRC="$(build_csi_source)"
  else
    SRC="$(build_usb_source)"
  fi
  SINK="$(build_rtmp_sink)"

  if [ "${AUDIO_ENABLED}" == "True" ]; then
    AUDIO_PIPE="$(build_audio_input)"
    echo -e "${GREEN}Starting ${CAMERA_TYPE^^} stream with audio...${NC}"
    PIPE="GST_DEBUG=2 gst-launch-1.0 -e -v \
      ${SRC} ! queue max-size-buffers=0 max-size-time=0 max-size-bytes=0 leaky=downstream ! \
      ${VIDEO_ENC} ! ${SINK} \
      ${AUDIO_PIPE}"
  else
    echo -e "${GREEN}Starting ${CAMERA_TYPE^^} stream without audio...${NC}"
    PIPE="GST_DEBUG=2 gst-launch-1.0 -e -v \
      ${SRC} ! queue max-size-buffers=0 max-size-time=0 max-size-bytes=0 leaky=downstream ! \
      ${VIDEO_ENC} ! ${SINK}"
  fi

  # Show exactly what we will run (useful for debugging)
  echo -e "${YELLOW}Launching:${NC} ${PIPE}"

  # Run without eval; -l loads bashrc in some environments, -c runs the single command
  bash -lc "${PIPE}"
  return $?
}

while true; do
  run_once "$ENCODER"
  EC=$?
  if [ $EC -eq 0 ]; then echo -e "${GREEN}stream ended cleanly.${NC}"; break; fi
  echo -e "${YELLOW}stream exited with code ${EC}.${NC}"
  if [ "$ENCODER" = "nvv4l2h264enc" ]; then
    echo -e "${YELLOW}Falling back to CPU (x264enc).${NC}"
    ENCODER="x264enc"
    continue
  fi
  echo -e "${YELLOW}Retrying in ${RESTART_DELAY}s...${NC}"
  sleep "$RESTART_DELAY"
done




# #!/bin/bash

# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m'

# log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
# log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
# log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# if [ -z "$STREAM_IP" ]; then 
#     log_warn "STREAM_IP not defined. Defaulting to 127.0.0.1"
#     STREAM_IP="127.0.0.1"
# fi
# if [ -z "$STREAM_PORT" ]; then 
#     log_warn "STREAM_PORT not defined. Defaulting to 1935"
#     STREAM_PORT=1935
# fi
# if [ -z "$STREAM_APPLICATION" ]; then 
#     log_warn "STREAM_APPLICATION not defined. Defaulting to 'live'"
#     STREAM_APPLICATION="live"
# fi
# if [ -z "$STREAM_KEY" ]; then 
#     log_warn "STREAM_KEY not defined. Defaulting to 'stream'"
#     STREAM_KEY="stream"
# fi
# if [ -z "$CAMERA_INDEX" ]; then 
#     log_warn "CAMERA_INDEX not defined. Defaulting to 0"
#     CAMERA_INDEX=0
# fi
# if [ -z "$CAMERA_WIDTH" ]; then 
#     log_warn "CAMERA_WIDTH not defined. Defaulting to 1280"
#     CAMERA_WIDTH=1280
# fi
# if [ -z "$CAMERA_HEIGHT" ]; then 
#     log_warn "CAMERA_HEIGHT not defined. Defaulting to 720"
#     CAMERA_HEIGHT=720
# fi
# if [ -z "$CAMERA_FPS" ]; then 
#     log_warn "CAMERA_FPS not defined. Defaulting to 30"
#     CAMERA_FPS=30
# fi
# if [ -z "$AUDIO_ENABLED" ]; then 
#     log_warn "AUDIO_ENABLED not defined. Defaulting to False"
#     AUDIO_ENABLED="False"
# fi
# if [ -z "$QUALITY_PRESET" ]; then 
#     log_warn "QUALITY_PRESET not defined. Options: ultra, high, balanced, fast. Defaulting to high"
#     QUALITY_PRESET="high"
# fi
# if [ -z "$AUDIO_DEVICE" ]; then 
#     log_warn "AUDIO_DEVICE not defined. Defaulting to hw:2,0"
#     AUDIO_DEVICE="hw:2,0"
# fi

# log_info "Jetson Nano Streaming Script Starting..."
# if [ -f /etc/nv_tegra_release ]; then
#     log_info "Jetson L4T Version:"
#     cat /etc/nv_tegra_release
# fi

# if command -v jetson_clocks &> /dev/null; then
#     log_info "Maximizing Jetson performance..."
#     sudo jetson_clocks 2>/dev/null || log_warn "Could not set jetson_clocks (needs sudo)"
# fi

# case "$QUALITY_PRESET" in
#     "ultra")
#         VIDEO_BITRATE=6000000
#         PEAK_BITRATE=8000000
#         H264_PRESET=1
#         CONTROL_RATE=1
#         PROFILE=2
#         NUM_BFRAMES=2
#         INSERT_SPS_PPS=1
#         INSERT_VUI=1
#         AUDIO_BITRATE=192000
#         AUDIO_RATE=48000
#         AUDIO_CHANNELS=2
#         log_info "Using ULTRA quality preset (6 Mbps, highest quality)"
#         ;;
#     "high")
#         VIDEO_BITRATE=4000000
#         PEAK_BITRATE=6000000
#         H264_PRESET=2
#         CONTROL_RATE=1
#         PROFILE=2
#         NUM_BFRAMES=1
#         INSERT_SPS_PPS=1
#         INSERT_VUI=1
#         AUDIO_BITRATE=128000
#         AUDIO_RATE=48000
#         AUDIO_CHANNELS=2
#         log_info "Using HIGH quality preset (4 Mbps, excellent quality)"
#         ;;
#     "balanced")
#         VIDEO_BITRATE=2500000
#         PEAK_BITRATE=4000000
#         H264_PRESET=3
#         CONTROL_RATE=1
#         PROFILE=1
#         NUM_BFRAMES=0
#         INSERT_SPS_PPS=1
#         INSERT_VUI=1
#         AUDIO_BITRATE=96000
#         AUDIO_RATE=44100
#         AUDIO_CHANNELS=1
#         log_info "Using BALANCED quality preset (2.5 Mbps, good quality/performance)"
#         ;;
#     "fast")
#         VIDEO_BITRATE=1500000
#         PEAK_BITRATE=2500000
#         H264_PRESET=4
#         CONTROL_RATE=1
#         PROFILE=0
#         NUM_BFRAMES=0
#         INSERT_SPS_PPS=1
#         INSERT_VUI=0
#         AUDIO_BITRATE=64000
#         AUDIO_RATE=44100
#         AUDIO_CHANNELS=1
#         log_info "Using FAST preset (1.5 Mbps, optimized for performance)"
#         ;;
#     *)
#         VIDEO_BITRATE=4000000
#         PEAK_BITRATE=6000000
#         H264_PRESET=2
#         CONTROL_RATE=1
#         PROFILE=2
#         NUM_BFRAMES=1
#         INSERT_SPS_PPS=1
#         INSERT_VUI=1
#         AUDIO_BITRATE=128000
#         AUDIO_RATE=48000
#         AUDIO_CHANNELS=2
#         log_info "Using default HIGH quality preset"
#         ;;
# esac

# GOP_SIZE=$((CAMERA_FPS * 2))

# RTMP_URI="rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"

# log_info "Stream Configuration:"
# echo -e "${BLUE}  Resolution:${NC} ${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps"
# echo -e "${BLUE}  Bitrate:${NC} $(($VIDEO_BITRATE / 1000000))Mbps (peak: $(($PEAK_BITRATE / 1000000))Mbps)"
# echo -e "${BLUE}  Audio:${NC} ${AUDIO_ENABLED}"
# echo -e "${BLUE}  Target:${NC} ${RTMP_URI}"

# QUEUE_PROPS="max-size-buffers=0 max-size-time=0 max-size-bytes=0 leaky=downstream"

# build_camera_source() {
#     echo "nvarguscamerasrc sensor-id=${CAMERA_INDEX} \
#         wbmode=1 \
#         tnr-mode=2 \
#         tnr-strength=1 \
#         ee-mode=2 \
#         ee-strength=0.5 \
#         aeantibanding=2 \
#         exposuretimerange=\"5000000 30000000\" \
#         gainrange=\"1 16\" \
#         ispdigitalgainrange=\"1 4\" \
#         exposurecompensation=0 \
#         saturation=1.2 ! \
#         video/x-raw(memory:NVMM), \
#             width=${CAMERA_WIDTH}, \
#             height=${CAMERA_HEIGHT}, \
#             framerate=${CAMERA_FPS}/1, \
#             format=NV12"
# }

# build_video_encoder() {
#     echo "nvv4l2h264enc \
#         bitrate=${VIDEO_BITRATE} \
#         peak-bitrate=${PEAK_BITRATE} \
#         preset-level=${H264_PRESET} \
#         control-rate=${CONTROL_RATE} \
#         maxperf-enable=1 \
#         ratecontrol-enable=1 \
#         iframeinterval=${GOP_SIZE} \
#         idrinterval=${GOP_SIZE} \
#         vbv-size=$((PEAK_BITRATE / CAMERA_FPS * 2)) \
#         insert-sps-pps=${INSERT_SPS_PPS} \
#         insert-vui=${INSERT_VUI} \
#         num-B-Frames=${NUM_BFRAMES} \
#         profile=${PROFILE}"
# }

# build_audio_pipeline() {
#     if [ "$AUDIO_CHANNELS" -eq "2" ]; then
#         AUDIO_FORMAT="S16LE"
#     else
#         AUDIO_FORMAT="S16LE"
#     fi
    
#     echo "alsasrc device=${AUDIO_DEVICE} \
#         buffer-time=20000 \
#         latency-time=10000 \
#         provide-clock=false \
#         do-timestamp=true ! \
#         audio/x-raw, \
#             format=${AUDIO_FORMAT}, \
#             channels=${AUDIO_CHANNELS}, \
#             rate=${AUDIO_RATE} ! \
#         audioresample quality=10 ! \
#         audioconvert ! \
#         audiorate tolerance=1000000000 ! \
#         volume volume=1.5 ! \
#         voaacenc bitrate=${AUDIO_BITRATE} ! \
#         aacparse ! \
#         queue ${QUEUE_PROPS} ! \
#         mux."
# }

# build_rtmp_sink() {
#     echo "rtmpsink location=\"${RTMP_URI} live=1 timeout=5\" \
#         sync=false \
#         async=false \
#         max-lateness=2000000000 \
#         qos=false \
#         enable-last-sample=false \
#         blocksize=4096 \
#         render-delay=0 \
#         throttle-time=0 \
#         max-bitrate=$((PEAK_BITRATE * 2))"
# }

# check_audio_device() {
#     if ! command -v arecord &> /dev/null; then
#         log_warn "arecord command not found. Audio may not work properly."
#         log_warn "Install alsa-utils: apt-get install alsa-utils"
#         return 1
#     fi

#     if ! arecord -l 2>/dev/null | grep -q "card ${AUDIO_DEVICE:3:1}"; then
#         log_warn "Audio device ${AUDIO_DEVICE} not found. Attempting to find available device..."
#         local available_device=$(arecord -l 2>/dev/null | grep "card" | head -1 | sed 's/card \([0-9]\).*/hw:\1,0/')
#         if [ -n "$available_device" ]; then
#             AUDIO_DEVICE="$available_device"
#             log_info "Using audio device: ${AUDIO_DEVICE}"
#             return 0
#         else
#             log_error "No audio devices found. Disabling audio."
#             return 1
#         fi
#     fi
#     return 0
# }

# if [ "${AUDIO_ENABLED}" == "True" ]; then
#     log_info "Starting stream with audio..."

#     if check_audio_device; then
#         GST_DEBUG=2 gst-launch-1.0 -v \
#             $(build_camera_source) ! \
#             tee name=t ! \
#             queue ${QUEUE_PROPS} ! \
#             $(build_video_encoder) ! \
#             h264parse config-interval=${GOP_SIZE} ! \
#             queue ${QUEUE_PROPS} ! \
#             mux. \
#             $(build_audio_pipeline) \
#             flvmux name=mux \
#                 streamable=true \
#                 latency=100000000 ! \
#             queue ${QUEUE_PROPS} ! \
#             $(build_rtmp_sink)
#     else
#         log_warn "Falling back to video-only stream due to audio issues"
#         AUDIO_ENABLED="False"
#     fi
# fi

# if [ "${AUDIO_ENABLED}" != "True" ]; then
#     log_info "Starting stream without audio..."
    
#     GST_DEBUG=2 gst-launch-1.0 -v \
#         $(build_camera_source) ! \
#         $(build_video_encoder) ! \
#         h264parse config-interval=${GOP_SIZE} ! \
#         queue ${QUEUE_PROPS} ! \
#         flvmux \
#             streamable=true \
#             latency=100000000 ! \
#         queue ${QUEUE_PROPS} ! \
#         $(build_rtmp_sink)
# fi
