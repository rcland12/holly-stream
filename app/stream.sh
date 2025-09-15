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
USE_ENCODER="${USE_ENCODER:-cpu}"
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

detect_pi_model() {
  if [ -f /proc/device-tree/model ]; then
    local MODEL; MODEL=$(tr -d '\0' < /proc/device-tree/model)
    echo -e "${GREEN}Detected: ${MODEL}${NC}"
  fi
}

detect_pi_model

check_camera_backend() {
  case "$CAMERA_BACKEND" in
    libcamera)
      if command -v rpicam-vid >/dev/null 2>&1; then echo "libcamera"; else echo "usb"; fi
      return 0;;
    legacy)
      if command -v raspivid >/dev/null 2>&1; then echo "legacy"; else echo "usb"; fi
      return 0;;
    usb)
      if [ -e "/dev/video${CAMERA_INDEX}" ]; then echo "usb"; else echo "libcamera"; fi
      return 0;;
    *)
      if command -v rpicam-vid >/dev/null 2>&1; then
        echo "libcamera"; return 0
      fi
      if command -v raspivid >/dev/null 2>&1; then
        echo "legacy"; return 0
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
      CPU_PRESET="slow"
      V4L2M2M_PROFILE="high"
      V4L2M2M_LEVEL="4.2"
      VIDEO_BITRATE="10000k"
      MAX_BITRATE="12000k"
      BUFFER_SIZE="20000k"
      ;;
    high)
      CPU_PRESET="medium"
      V4L2M2M_PROFILE="high"
      V4L2M2M_LEVEL="4.1"
      VIDEO_BITRATE="6000k"
      MAX_BITRATE="8000k"
      BUFFER_SIZE="12000k"
      ;;
    medium)
      CPU_PRESET="veryfast"
      V4L2M2M_PROFILE="main"
      V4L2M2M_LEVEL="4.0"
      VIDEO_BITRATE="3000k"
      MAX_BITRATE="4000k"
      BUFFER_SIZE="6000k"
      ;;
    low)
      CPU_PRESET="ultrafast"
      V4L2M2M_PROFILE="baseline"
      V4L2M2M_LEVEL="3.1"
      VIDEO_BITRATE="1500k"
      MAX_BITRATE="2000k"
      BUFFER_SIZE="3000k"
      ;;
    *)
      CPU_PRESET="medium"
      V4L2M2M_PROFILE="high"
      V4L2M2M_LEVEL="4.1"
      VIDEO_BITRATE="6000k"
      MAX_BITRATE="8000k"
      BUFFER_SIZE="12000k"
      ;;
  esac
}

set_quality_presets

echo -e "${GREEN}Querying camera capabilities...${NC}"
if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl --device=/dev/video"$CAMERA_INDEX" --list-formats-ext | head -40 || true
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
        echo "-f pulse -thread_queue_size 8192 -i ${AUDIO_DEVICE#pulse:}"; return 0
      else
        echo ""
        return 0
      fi
    else
      echo "-f alsa -thread_queue_size 8192 -ar 48000 -ac 2 -i ${AUDIO_DEVICE}"; return 0
    fi
  fi

  if command -v pactl >/dev/null 2>&1 && pactl info >/dev/null 2>&1; then
    echo "-f pulse -thread_queue_size 8192 -i default"; return 0
  fi

  if command -v arecord >/dev/null 2>&1; then
    CARD_LINE="$(arecord -l 2>/dev/null | awk '/card [0-9]+:/ {print $2}' | sed 's/://;q')"
    if [ -n "$CARD_LINE" ]; then
      echo "-f alsa -thread_queue_size 8192 -ar 48000 -ac 2 -i plughw:${CARD_LINE},0"; return 0
    fi
  fi

  echo ""
}

choose_encoder() {
  local req="$1"
  if [[ "$req" == "nvenc" ]]; then req="v4l2m2m"; fi
  if [[ "$req" == "v4l2m2m" ]] && ffmpeg -hide_banner -encoders 2>/dev/null | grep -qE '(^| )h264_v4l2m2m'; then
    echo "v4l2m2m"
  else
    echo "cpu"
  fi
}

build_encoding_params() {
  case "$1" in
    v4l2m2m)
      echo "-c:v h264_v4l2m2m \
        -profile:v ${V4L2M2M_PROFILE} \
        -level:v ${V4L2M2M_LEVEL} \
        -b:v ${VIDEO_BITRATE} \
        -maxrate ${MAX_BITRATE} \
        -bufsize ${BUFFER_SIZE} \
        -g $((CAMERA_FPS * 2)) \
        -keyint_min ${CAMERA_FPS} \
        -sc_threshold 0 \
        -pix_fmt yuv420p"
      ;;
    *)
      echo "-c:v libx264 \
        -preset:v ${CPU_PRESET} \
        -tune:v zerolatency \
        -profile:v high \
        -level:v 4.2 \
        -crf 23 \
        -b:v ${VIDEO_BITRATE} \
        -maxrate ${MAX_BITRATE} \
        -bufsize ${BUFFER_SIZE} \
        -g $((CAMERA_FPS * 2)) \
        -keyint_min ${CAMERA_FPS} \
        -sc_threshold 0 \
        -pix_fmt yuv420p"
      ;;
  esac
}

CAMERA_TYPE="$(check_camera_backend || echo none)"
if [ "$CAMERA_TYPE" = "none" ]; then
  echo -e "${RED}No camera backend detected (rpicam-vid/raspivid or /dev/video*).${NC}"
  exit 1
fi

GOP_SIZE=$((CAMERA_FPS * 2))
KEYINT_MIN=$CAMERA_FPS
ENCODER="$(choose_encoder "$USE_ENCODER")"
[ "$ENCODER" = "v4l2m2m" ] && echo -e "${GREEN}Using V4L2 M2M (if it fails, auto-fallback to CPU).${NC}" || echo -e "${YELLOW}Using CPU (libx264).${NC}"

AUDIO_INPUT="$(build_audio_input)"
if [ "${AUDIO_ENABLED}" == "True" ] && [ -z "${AUDIO_INPUT}" ]; then
  echo -e "${YELLOW}No usable audio input found; disabling audio to keep streaming.${NC}"
  AUDIO_ENABLED="False"
fi

run_once() {
  local enc="$1"
  local ENCODING_PARAMS; ENCODING_PARAMS="$(build_encoding_params "$enc")"

  local MUX_TAIL_AUDIO="-c:a aac -b:a 192k -ar 48000 -ac 2 -af \"aresample=async=1:min_hard_comp=0.100000:first_pts=0\""
  local MUX_TAIL_NOAUDIO=""
  local COMMON_TAIL="-use_wallclock_as_timestamps 1 -fflags +genpts -fps_mode cfr -r ${CAMERA_FPS} \
                     -max_muxing_queue_size 2048 -rtmp_buffer 100 -flvflags no_duration_filesize \
                     -f flv \"rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}\""

  if [ "$CAMERA_TYPE" = "usb" ]; then
    if [ "${AUDIO_ENABLED}" == "True" ]; then
      echo -e "${GREEN}Starting USB stream with audio...${NC}"
      eval ffmpeg \
        -f v4l2 -thread_queue_size 4096 -framerate "$CAMERA_FPS" -video_size "$DIMS" -input_format mjpeg \
        -i "/dev/video${CAMERA_INDEX}" \
        ${AUDIO_INPUT} \
        -vf "format=yuv420p" \
        ${ENCODING_PARAMS} \
        -pix_fmt yuv420p -g ${GOP_SIZE} -keyint_min ${KEYINT_MIN} \
        ${MUX_TAIL_AUDIO} \
        ${COMMON_TAIL}
    else
      echo -e "${GREEN}Starting USB stream without audio...${NC}"
      eval ffmpeg \
        -f v4l2 -thread_queue_size 4096 -framerate "$CAMERA_FPS" -video_size "$DIMS" -input_format mjpeg \
        -i "/dev/video${CAMERA_INDEX}" \
        -vf "format=yuv420p" \
        ${ENCODING_PARAMS} \
        -pix_fmt yuv420p -g ${GOP_SIZE} -keyint_min ${KEYINT_MIN} \
        ${MUX_TAIL_NOAUDIO} \
        ${COMMON_TAIL}
    fi
    return $?
  fi

  if [ "$CAMERA_TYPE" = "libcamera" ] && command -v rpicam-vid >/dev/null 2>&1; then
    local LIBCAM_BITRATE; LIBCAM_BITRATE="$(echo ${VIDEO_BITRATE} | sed 's/k/000/')"
    if [ "${AUDIO_ENABLED}" == "True" ]; then
      echo -e "${GREEN}Starting rpicam stream with audio (copy video)...${NC}"
      eval rpicam-vid \
        --nopreview --inline --timeout 0 \
        --width "$CAMERA_WIDTH" --height "$CAMERA_HEIGHT" \
        --framerate "$CAMERA_FPS" --rotation 180 \
        --codec h264 --profile "${V4L2M2M_PROFILE}" --level "${V4L2M2M_LEVEL}" \
        --bitrate "${LIBCAM_BITRATE}" -o - \| \
      ffmpeg -thread_queue_size 2048 -f h264 -r "$CAMERA_FPS" -probesize 50M -analyzeduration 2M -i - \
        ${AUDIO_INPUT} \
        -c:v copy \
        -fps_mode cfr \
        ${MUX_TAIL_AUDIO} \
        ${COMMON_TAIL}
    else
      echo -e "${GREEN}Starting rpicam stream without audio (copy video)...${NC}"
      eval rpicam-vid \
        --nopreview --inline --timeout 0 \
        --width "$CAMERA_WIDTH" --height "$CAMERA_HEIGHT" \
        --framerate "$CAMERA_FPS" --rotation 180 \
        --codec h264 --profile "${V4L2M2M_PROFILE}" --level "${V4L2M2M_LEVEL}" \
        --bitrate "${LIBCAM_BITRATE}" -o - \| \
      ffmpeg -thread_queue_size 2048 -f h264 -r "$CAMERA_FPS" -probesize 50M -analyzeduration 2M -i - \
        -c:v copy \
        -fps_mode cfr \
        ${MUX_TAIL_NOAUDIO} \
        ${COMMON_TAIL}
    fi
    return $?
  fi

  if [ "$CAMERA_TYPE" = "legacy" ] && command -v raspivid >/dev/null 2>&1; then
    local RPI_BITRATE; RPI_BITRATE="$(echo ${VIDEO_BITRATE} | sed 's/k/000/')"
    if [ "${AUDIO_ENABLED}" == "True" ]; then
      echo -e "${GREEN}Starting raspivid stream with audio (copy video)...${NC}"
      eval raspivid --nopreview --timeout 0 \
        --width "$CAMERA_WIDTH" --height "$CAMERA_HEIGHT" \
        --framerate "$CAMERA_FPS" --rotation 180 \
        --bitrate "${RPI_BITRATE}" --profile high --inline -o - \| \
      ffmpeg -thread_queue_size 2048 -f h264 -r "$CAMERA_FPS" -probesize 50M -analyzeduration 2M -i - \
        ${AUDIO_INPUT} \
        -c:v copy \
        -fps_mode cfr \
        ${MUX_TAIL_AUDIO} \
        ${COMMON_TAIL}
    else
      echo -e "${GREEN}Starting raspivid stream without audio (copy video)...${NC}"
      eval raspivid --nopreview --timeout 0 \
        --width "$CAMERA_WIDTH" --height "$CAMERA_HEIGHT" \
        --framerate "$CAMERA_FPS" --rotation 180 \
        --bitrate "${RPI_BITRATE}" --profile high --inline -o - \| \
      ffmpeg -thread_queue_size 2048 -f h264 -r "$CAMERA_FPS" -probesize 50M -analyzeduration 2M -i - \
        -c:v copy \
        -fps_mode cfr \
        ${MUX_TAIL_NOAUDIO} \
        ${COMMON_TAIL}
    fi
    return $?
  fi

  echo -e "${RED}Unknown or unavailable CAMERA_TYPE=${CAMERA_TYPE}${NC}"
  return 2
}

while true; do
  run_once "$ENCODER"
  EC=$?
  if [ $EC -eq 0 ]; then echo -e "${GREEN}stream ended cleanly.${NC}"; break; fi
  echo -e "${YELLOW}stream exited with code ${EC}.${NC}"
  if [ "$CAMERA_TYPE" = "usb" ] && [ "$ENCODER" = "v4l2m2m" ]; then
    echo -e "${YELLOW}Falling back to CPU (libx264).${NC}"
    ENCODER="cpu"
    continue
  fi
  echo -e "${YELLOW}Retrying in ${RESTART_DELAY}s...${NC}"
  sleep "$RESTART_DELAY"
done
