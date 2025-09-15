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
VIDEO_QUALITY="${VIDEO_QUALITY:-high}"
AUDIO_DEVICE="${AUDIO_DEVICE:-}"
USE_ENCODER="${USE_ENCODER:-cpu}"
RESTART_DELAY="${RESTART_DELAY:-3}"

DIMS="${CAMERA_WIDTH}x${CAMERA_HEIGHT}"

echo -e "${YELLOW}CAMERA_WIDTH=${CAMERA_WIDTH}${NC}"
echo -e "${YELLOW}CAMERA_HEIGHT=${CAMERA_HEIGHT}${NC}"
echo -e "${YELLOW}CAMERA_INDEX=${CAMERA_INDEX}${NC}"
echo -e "${YELLOW}CAMERA_FPS=${CAMERA_FPS}${NC}"
echo -e "${YELLOW}STREAM=${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}${NC}"
echo -e "${YELLOW}VIDEO_QUALITY=${VIDEO_QUALITY}${NC}"
echo -e "${YELLOW}VIDEO_BITRATE=${VIDEO_BITRATE}${NC}"
echo -e "${YELLOW}AUDIO_DEVICE=${AUDIO_DEVICE}${NC}"
echo -e "${YELLOW}USE_ENCODER=${USE_ENCODER}${NC}"

set_quality_presets() {
  case "$VIDEO_QUALITY" in
    ultra)
        CPU_PRESET="slow"
        NVENC_PRESET="p7"
        NVENC_RC="vbr"
        NVENC_CQ="19"
        VIDEO_BITRATE="10000k"
        MAX_BITRATE="12000k"
        BUFFER_SIZE="20000k"
        ;;
    high)
        CPU_PRESET="medium"
        NVENC_PRESET="p5"
        NVENC_RC="vbr"
        NVENC_CQ="23"
        VIDEO_BITRATE="6000k"
        MAX_BITRATE="8000k"
        BUFFER_SIZE="12000k"
        ;;
    medium)
        CPU_PRESET="veryfast"
        NVENC_PRESET="p4"
        NVENC_RC="cbr"
        NVENC_CQ="27"
        VIDEO_BITRATE="3000k"
        MAX_BITRATE="4000k"
        BUFFER_SIZE="6000k"
        ;;
    low)
        CPU_PRESET="ultrafast"
        NVENC_PRESET="p2"
        NVENC_RC="cbr"
        NVENC_CQ="30"
        VIDEO_BITRATE="1500k"
        MAX_BITRATE="2000k"
        BUFFER_SIZE="3000k"
        ;;
    *)
        CPU_PRESET="medium"
        NVENC_PRESET="p5"
        NVENC_RC="vbr"
        NVENC_CQ="23"
        VIDEO_BITRATE="6000k"
        MAX_BITRATE="8000k"
        BUFFER_SIZE="12000k"
        ;;
  esac
}

set_quality_presets

echo -e "${GREEN}Querying camera capabilities...${NC}"
if command -v v4l2-ctl >/dev/null 2>&1; then
  v4l2-ctl --device=/dev/video"$CAMERA_INDEX" --list-formats-ext | head -20 || true
else
  echo -e "${YELLOW}v4l2-ctl not found; skipping.${NC}"
fi

build_encoding_params() {
  case "$1" in
    nvenc)
        echo "-c:v h264_nvenc \
        -preset:v $NVENC_PRESET \
        -tune:v hq \
        -rc:v $NVENC_RC \
        -rc-lookahead:v 32 \
        -spatial-aq:v 1 \
        -temporal-aq:v 1 \
        -cq:v $NVENC_CQ \
        -b:v $VIDEO_BITRATE \
        -maxrate:v $MAX_BITRATE \
        -bufsize:v $BUFFER_SIZE \
        -profile:v high \
        -level:v 4.2 \
        -coder:v cabac \
        -b_ref_mode:v middle \
        -dpb_size:v 4"
        ;;
    *)
        echo "-c:v libx264 \
        -preset:v $CPU_PRESET \
        -tune:v zerolatency \
        -profile:v high \
        -level:v 4.2 \
        -crf 23 \
        -b:v $VIDEO_BITRATE \
        -maxrate:v $MAX_BITRATE \
        -bufsize:v $BUFFER_SIZE"
        ;;
  esac
}

build_audio_input() {
  if [ "${AUDIO_ENABLED}" == "True" ]; then
    if [ -n "$AUDIO_DEVICE" ]; then
      if [[ "$AUDIO_DEVICE" == pulse:* ]]; then
        if command -v pactl >/dev/null 2>&1 && pactl info >/dev/null 2>&1; then
          echo "-f pulse -thread_queue_size 8192 -i ${AUDIO_DEVICE#pulse:}"
        else
          echo "-f alsa -thread_queue_size 8192 -ar 48000 -ac 2 -i default"
        fi
      else
        echo "-f alsa -thread_queue_size 8192 -ar 48000 -ac 2 -i ${AUDIO_DEVICE}"
      fi
    else
      if command -v pactl >/dev/null 2>&1 && pactl info >/dev/null 2>&1; then
        echo "-f pulse -thread_queue_size 8192 -i default"
      else
        echo "-f alsa -thread_queue_size 8192 -ar 48000 -ac 2 -i default"
      fi
    fi
  else
    echo ""
  fi
}

choose_encoder() {
  if [ "$USE_ENCODER" = "nvenc" ] && ffmpeg -hide_banner -encoders 2>/dev/null | grep -qE '(^| )h264_nvenc'; then
    echo "nvenc"
  else
    echo "cpu"
  fi
}

run_ffmpeg_once() {
  local enc="$1"
  local ENCODING_PARAMS; ENCODING_PARAMS="$(build_encoding_params "$enc")"
  local AUDIO_INPUT; AUDIO_INPUT="$(build_audio_input)"

  if [ "${AUDIO_ENABLED}" == "True" ]; then
    echo -e "${GREEN}Starting stream with audio...${NC}"
    ffmpeg \
      -f v4l2 \
      -thread_queue_size 4096 \
      -framerate "$CAMERA_FPS" \
      -video_size "$DIMS" \
      -input_format mjpeg \
      -i "/dev/video${CAMERA_INDEX}" \
      $AUDIO_INPUT \
      -use_wallclock_as_timestamps 1 \
      -fflags +genpts \
      -vf "format=yuv420p" \
      $ENCODING_PARAMS \
      -pix_fmt yuv420p \
      -r "$CAMERA_FPS" \
      -g $((CAMERA_FPS * 2)) \
      -keyint_min "$CAMERA_FPS" \
      -sc_threshold 0 \
      -c:a aac \
      -b:a 192k \
      -ar 48000 \
      -ac 2 \
      -af "aresample=async=1:min_hard_comp=0.100000:first_pts=0" \
      -vsync cfr \
      -max_muxing_queue_size 2048 \
      -rtmp_buffer 100 \
      -flvflags no_duration_filesize \
      -f flv "rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"
  else
    echo -e "${GREEN}Starting stream without audio...${NC}"
    ffmpeg \
      -f v4l2 \
      -thread_queue_size 4096 \
      -framerate "$CAMERA_FPS" \
      -video_size "$DIMS" \
      -input_format mjpeg \
      -i "/dev/video${CAMERA_INDEX}" \
      -use_wallclock_as_timestamps 1 \
      -fflags +genpts \
      -vf "format=yuv420p" \
      $ENCODING_PARAMS \
      -pix_fmt yuv420p \
      -r "$CAMERA_FPS" \
      -g $((CAMERA_FPS * 2)) \
      -keyint_min "$CAMERA_FPS" \
      -sc_threshold 0 \
      -vsync cfr \
      -max_muxing_queue_size 2048 \
      -rtmp_buffer 100 \
      -flvflags no_duration_filesize \
      -f flv "rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"
  fi
}

ENCODER="$(choose_encoder)"
[ "$ENCODER" = "nvenc" ] && echo -e "${GREEN}Using NVENC (if it fails, auto-fallback to CPU).${NC}" || echo -e "${YELLOW}Using CPU (libx264).${NC}"

while true; do
  run_ffmpeg_once "$ENCODER"
  EC=$?
  if [ $EC -eq 0 ]; then echo -e "${GREEN}ffmpeg exited cleanly.${NC}"; break; fi
  echo -e "${YELLOW}ffmpeg exited with code ${EC}.${NC}"
  if [ "$ENCODER" = "nvenc" ]; then
    echo -e "${YELLOW}Falling back to CPU (libx264).${NC}"
    ENCODER="cpu"
    continue
  fi
  echo -e "${YELLOW}Retrying in ${RESTART_DELAY}s...${NC}"
  sleep "$RESTART_DELAY"
done
