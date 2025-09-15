# #!/bin/bash

# RTMP_URI="rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"

# if [ "${AUDIO_ENABLED}" == "True" ]; then
#     QUEUE_PROPS="max-size-buffers=0 max-size-time=0 max-size-bytes=0 leaky=downstream"

#     gst-launch-1.0 -v \
#         nvarguscamerasrc sensor-id=${CAMERA_INDEX} ! \
#         "video/x-raw(memory:NVMM), width=${CAMERA_WIDTH}, height=${CAMERA_HEIGHT}, framerate=${CAMERA_FPS}/1, format=NV12" ! \
#         nvv4l2h264enc \
#             bitrate=2000000 \
#             peak-bitrate=4000000 \
#             preset-level=4 \
#             control-rate=1 \
#             maxperf-enable=1 ! \
#         h264parse ! \
#         queue $QUEUE_PROPS ! \
#         mux. \
#         alsasrc device=hw:2,0 provide-clock=false ! \
#         "audio/x-raw, format=S16LE, channels=1, rate=44100" ! \
#         audioresample ! \
#         audioconvert ! \
#         audiorate ! \
#         voaacenc \
#             bitrate=64000 ! \
#         aacparse ! \
#         queue $QUEUE_PROPS ! \
#         mux. \
#         flvmux name=mux streamable=true latency=100000000 ! \
#         queue $QUEUE_PROPS ! \
#         rtmpsink location="${RTMP_URI} live=1" \
#             sync=false \
#             max-lateness=1000000000

# else
#     gst-launch-1.0 -v \
#         nvarguscamerasrc sensor-id=${CAMERA_INDEX} \
#             wbmode=1 \
#             tnr-mode=2 \
#             tnr-strength=1 ! \
#         "video/x-raw(memory:NVMM), width=${CAMERA_WIDTH}, height=${CAMERA_HEIGHT}, framerate=${CAMERA_FPS}/1, format=NV12" ! \
#         nvv4l2h264enc \
#             bitrate=2000000 \
#             peak-bitrate=4000000 \
#             preset-level=4 \
#             control-rate=1 \
#             maxperf-enable=1 \
#             iframeinterval=30 ! \
#         h264parse ! \
#         queue max-size-buffers=0 max-size-time=0 max-size-bytes=0 leaky=downstream ! \
#         flvmux streamable=true latency=100000000 ! \
#         rtmpsink location="${RTMP_URI} live=1" \
#             sync=false \
#             max-lateness=1000000000
# fi