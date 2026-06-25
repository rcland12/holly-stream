#!/bin/bash

set -e

STREAM_IP="${STREAM_IP:-192.168.1.120}"
STREAM_PORT="${STREAM_PORT:-1935}"
STREAM_APPLICATION="${STREAM_APPLICATION:-hollystream1}"
STREAM_KEY="${STREAM_KEY:-hollyvideostream1}"

CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-3264}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-1848}"
CAMERA_FPS="${CAMERA_FPS:-10}"

VIDEO_BITRATE="${VIDEO_BITRATE:-18000000}"
CONTROL_RATE="${CONTROL_RATE:-1}"
PRESET_LEVEL="${PRESET_LEVEL:-1}"
MAXPERF="${MAXPERF:-1}"

KEYINT_SECONDS="${KEYINT_SECONDS:-2}"
GOP_SIZE=$(( CAMERA_FPS * KEYINT_SECONDS ))

AUDIO_ENABLED="${AUDIO_ENABLED:-False}"
AUDIO_DEVICE="${AUDIO_DEVICE:-hw:2,0}"
AUDIO_BITRATE="${AUDIO_BITRATE:-128000}"
AUDIO_RATE="${AUDIO_RATE:-48000}"
AUDIO_CHANNELS="${AUDIO_CHANNELS:-1}"

RTMP_URI="rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"

if [ "${AUDIO_ENABLED}" = "True" ]; then
  GST_DEBUG=2 exec gst-launch-1.0 -e -v \
    nvarguscamerasrc sensor-id="${CAMERA_INDEX}" ! \
      "video/x-raw(memory:NVMM),width=${CAMERA_WIDTH},height=${CAMERA_HEIGHT},framerate=${CAMERA_FPS}/1,format=NV12" ! \
    nvv4l2h264enc \
      control-rate="${CONTROL_RATE}" \
      bitrate="${VIDEO_BITRATE}" \
      preset-level="${PRESET_LEVEL}" \
      maxperf-enable="${MAXPERF}" \
      iframeinterval="${GOP_SIZE}" \
      insert-sps-pps=1 \
      insert-aud=1 \
      num-B-Frames=0 ! \
    h264parse config-interval=1 ! \
    flvmux name=mux streamable=true ! \
    rtmpsink location="${RTMP_URI} live=1" sync=false \
    alsasrc device="${AUDIO_DEVICE}" ! \
      "audio/x-raw,format=S16LE,channels=${AUDIO_CHANNELS},rate=${AUDIO_RATE}" ! \
    voaacenc bitrate="${AUDIO_BITRATE}" ! \
    mux.
else
  GST_DEBUG=2 exec gst-launch-1.0 -e -v \
    nvarguscamerasrc sensor-id="${CAMERA_INDEX}" ! \
      "video/x-raw(memory:NVMM),width=${CAMERA_WIDTH},height=${CAMERA_HEIGHT},framerate=${CAMERA_FPS}/1,format=NV12" ! \
    nvv4l2h264enc \
      control-rate="${CONTROL_RATE}" \
      bitrate="${VIDEO_BITRATE}" \
      preset-level="${PRESET_LEVEL}" \
      maxperf-enable="${MAXPERF}" \
      iframeinterval="${GOP_SIZE}" \
      insert-sps-pps=1 \
      insert-aud=1 \
      num-B-Frames=0 ! \
    h264parse config-interval=1 ! \
    flvmux streamable=true ! \
    rtmpsink location="${RTMP_URI} live=1" sync=false
fi
