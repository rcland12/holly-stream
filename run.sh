#!/bin/bash

source .env

[[ -z $OBJECT_DETECTION ]] && echo "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False." && exit 1

if [[ ! "${OBJECT_DETECTION}" =~ ^(True|False)$ ]]; then
    echo "Invalid input for OBJECT_DETECTION. Expecting True or False; received ${OBJECT_DETECTION}."
    exit 120
fi

if [[ "${OBJECT_DETECTION}" == "True" ]]; then
    docker compose up -d triton
    
    echo "Waiting to start Holly Stream until Triton is healthy."
    for ((attempt=1; attempt<=60; attempt++)); do
        if docker compose exec -it triton curl -s -f "http://localhost:8000/v2/health/ready" > /dev/null; then
            break
        fi
        sleep 1
        [[ $attempt -eq 60 ]] && echo "Triton failed all health checks after 60 seconds. Stopping all services." && exit 120
    done
fi

docker compose up -d app
echo "Holly Stream has started. Performing health check..."
sleep 10

for i in {1..12}; do
    if [ "$( docker container inspect -f '{{.State.Running}}' holly-stream-app )" = "true" ]; then
        echo "Holly Stream STATUS: HEALTHY"
        break
    elif [ $i -eq 12 ]; then
        echo "Holly STREAM STATUS: UNHEALTHY"
        echo "Shutting down."
        docker compose down
        exit 1
    else
        echo "Health check attempt: $i/12"
        sleep 5
    fi
done

echo "System running."



# RED='\033[0;31m'
# GREEN='\033[0;32m'
# YELLOW='\033[1;33m'
# BLUE='\033[0;34m'
# NC='\033[0m'

# log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
# log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
# log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# source .env

# detect_pi_model() {
#     if [ -f /proc/device-tree/model ]; then
#         MODEL=$(tr -d '\0' < /proc/device-tree/model)
#         log_info "Detected: $MODEL" >&2

#         if [[ $MODEL == *"Pi 4"* ]] || [[ $MODEL == *"Pi 5"* ]]; then
#             echo "pi4"
#         elif [[ $MODEL == *"Pi 3"* ]]; then
#             echo "pi3"
#         elif [[ $MODEL == *"Pi Zero 2"* ]]; then
#             echo "pizero2"
#         else
#             echo "unknown"
#         fi
#     else
#         echo "unknown"
#     fi
# }

# optimize_system() {
#     log_info "Optimizing system for streaming..."
    
#     # requires reboot to take effect
#     if [ -w /boot/config.txt ]; then
#         if ! grep -q "gpu_mem=" /boot/config.txt; then
#             log_warn "Consider adding 'gpu_mem=256' to /boot/config.txt for better performance"
#         fi
#     fi

#     if [ -w /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor ]; then
#         echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1
#         log_info "CPU governor set to performance mode"
#     fi

#     sudo sysctl -w net.core.rmem_max=26214400 > /dev/null 2>&1
#     sudo sysctl -w net.core.wmem_max=26214400 > /dev/null 2>&1
#     sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 26214400" > /dev/null 2>&1
#     sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 26214400" > /dev/null 2>&1
#     log_info "Network buffers optimized"
# }

# if [ -z "$OBJECT_DETECTION" ]; then 
#     log_error "The environment variable OBJECT_DETECTION is required. This is a boolean value True/False."
#     exit 1
# fi

# if [ -z "$MOTION_DETECTION" ]; then 
#     log_error "The environment variable MOTION_DETECTION is required. This is a boolean value True/False."
#     exit 1
# fi

# if [ "${OBJECT_DETECTION}" == "True" ] && [ "${MOTION_DETECTION}" == "True" ]; then 
#     log_error "Both OBJECT_DETECTION and MOTION_DETECTION cannot be True, choose only one"
#     exit 1
# fi

# PI_MODEL=$(detect_pi_model)

# if [ -f /proc/device-tree/model ]; then
#     FULL_MODEL=$(tr -d '\0' < /proc/device-tree/model)
#     log_info "Detected: $FULL_MODEL"
# fi

# if [ -z "$QUALITY_PRESET" ]; then 
#     log_warn "QUALITY_PRESET not defined. Options: ultra, high, balanced, fast. Defaulting to high"
#     QUALITY_PRESET="high"
# fi

# log_info "PI_MODEL detected as: '$PI_MODEL'"
# log_info "QUALITY_PRESET requested: '$QUALITY_PRESET'"

# case "$QUALITY_PRESET" in
#     "ultra")
#         # Ultra quality - for Pi 4/5 only
#         if [[ "$PI_MODEL" == "pi4" ]]; then
#             DEFAULT_WIDTH=1920
#             DEFAULT_HEIGHT=1080
#             DEFAULT_FPS=30
#             VIDEO_BITRATE=8000
#             AUDIO_BITRATE=192
#             H264_PROFILE="high"
#             H264_LEVEL="4.2"
#             FFMPEG_PRESET="medium"
#             DENOISE_MODE="cdn_hq"
#             SHARPNESS=1.2
#             CONTRAST=1.1
#             log_info "Using ULTRA quality preset (8 Mbps, 1080p)"
#         else
#             log_warn "Ultra preset only supported on Pi 4/5, falling back to high"
#             DEFAULT_WIDTH=1920
#             DEFAULT_HEIGHT=1080
#             DEFAULT_FPS=30
#             VIDEO_BITRATE=4500
#             AUDIO_BITRATE=128
#             H264_PROFILE="high"
#             H264_LEVEL="4.1"
#             FFMPEG_PRESET="veryfast"
#             DENOISE_MODE="cdn_fast"
#             SHARPNESS=1.1
#             CONTRAST=1.05
#             log_info "Using HIGH quality preset (4.5 Mbps, 1080p)"
#         fi
#         ;;
#     "high")
#         # High quality - good for Pi 3/4/5
#         DEFAULT_WIDTH=1920
#         DEFAULT_HEIGHT=1080
#         DEFAULT_FPS=30
#         VIDEO_BITRATE=4500
#         AUDIO_BITRATE=128
#         H264_PROFILE="high"
#         H264_LEVEL="4.1"
#         FFMPEG_PRESET="veryfast"
#         DENOISE_MODE="cdn_fast"
#         SHARPNESS=1.1
#         CONTRAST=1.05
#         log_info "Using HIGH quality preset (4.5 Mbps, 1080p)"
#         ;;
#     "balanced")
#         # Balanced - good for most use cases
#         DEFAULT_WIDTH=1280
#         DEFAULT_HEIGHT=720
#         DEFAULT_FPS=30
#         VIDEO_BITRATE=2500
#         AUDIO_BITRATE=96
#         H264_PROFILE="main"
#         H264_LEVEL="4.0"
#         FFMPEG_PRESET="ultrafast"
#         DENOISE_MODE="cdn_fast"
#         SHARPNESS=1.0
#         CONTRAST=1.0
#         log_info "Using BALANCED quality preset (2.5 Mbps, 720p)"
#         ;;
#     "fast")
#         # Fast - for low latency or older Pi models
#         DEFAULT_WIDTH=1280
#         DEFAULT_HEIGHT=720
#         DEFAULT_FPS=25
#         VIDEO_BITRATE=1500
#         AUDIO_BITRATE=64
#         H264_PROFILE="baseline"
#         H264_LEVEL="3.1"
#         FFMPEG_PRESET="ultrafast"
#         DENOISE_MODE="cdn_off"
#         SHARPNESS=1.0
#         CONTRAST=1.0
#         log_info "Using FAST preset (1.5 Mbps, 720p, optimized for latency)"
#         ;;
#     *)
#         # Default to balanced
#         log_warn "Unknown quality preset '$QUALITY_PRESET', using balanced"
#         QUALITY_PRESET="balanced"
#         DEFAULT_WIDTH=1280
#         DEFAULT_HEIGHT=720
#         DEFAULT_FPS=30
#         VIDEO_BITRATE=2500
#         AUDIO_BITRATE=96
#         H264_PROFILE="main"
#         H264_LEVEL="4.0"
#         FFMPEG_PRESET="ultrafast"
#         DENOISE_MODE="cdn_fast"
#         SHARPNESS=1.0
#         CONTRAST=1.0
#         ;;
# esac

# CAMERA_AUDIO=${CAMERA_AUDIO:-"False"}
# CAMERA_FPS=${CAMERA_FPS:-$DEFAULT_FPS}
# CAMERA_WIDTH=${CAMERA_WIDTH:-$DEFAULT_WIDTH}
# CAMERA_HEIGHT=${CAMERA_HEIGHT:-$DEFAULT_HEIGHT}
# CAMERA_ROTATION=${CAMERA_ROTATION:-180}
# STREAM_IP=${STREAM_IP:-"127.0.0.1"}
# STREAM_PORT=${STREAM_PORT:-1935}
# STREAM_APPLICATION=${STREAM_APPLICATION:-"live"}
# STREAM_KEY=${STREAM_KEY:-"stream"}
# MODEL_DIMS="${CAMERA_WIDTH}x${CAMERA_HEIGHT}"
# GOP_SIZE=$((CAMERA_FPS * 2))

# optimize_system

# log_info "Stream Configuration:"
# echo -e "${BLUE}  Pi Model:${NC} $PI_MODEL"
# echo -e "${BLUE}  Quality:${NC} $QUALITY_PRESET"
# echo -e "${BLUE}  Resolution:${NC} ${CAMERA_WIDTH}x${CAMERA_HEIGHT} @ ${CAMERA_FPS}fps"
# echo -e "${BLUE}  Video Bitrate:${NC} ${VIDEO_BITRATE}k"
# echo -e "${BLUE}  Audio:${NC} ${CAMERA_AUDIO} (${AUDIO_BITRATE}k)"
# echo -e "${BLUE}  Target:${NC} rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"

# if [ "${OBJECT_DETECTION}" == "True" ]; then
#     log_info "Starting Object Detection mode..."
#     docker compose up -d triton

#     ATTEMPT=1
#     RETRIES=60
#     INTERVAL=1
#     TOTAL_TIME=$((RETRIES * INTERVAL))

#     log_info "Waiting for Triton to become healthy..."
#     while [ $ATTEMPT -le $RETRIES ]; do
#         url="http://localhost:8000/v2/health/ready"
#         response=$(curl --write-out "%{http_code}" --silent --output /dev/null "$url")

#         if [ $response -eq 200 ]; then
#             log_info "Triton STATUS: HEALTHY"
#             break
#         else
#             echo -ne "\rWaiting... ($ATTEMPT/$RETRIES)"
#             ATTEMPT=$((ATTEMPT + 1))
#             sleep $INTERVAL
#         fi
#     done

#     if [ $ATTEMPT -gt $RETRIES ]; then
#         log_error "Triton failed all health checks after $TOTAL_TIME seconds. Stopping all services."
#         exit 120
#     fi

#     nohup $PWD/.stream_env/bin/python3 app/main.py > .log.out 2>&1 &
#     echo $! > .process.pid
#     log_info "Holly-stream with object detection started. PID: $(cat .process.pid)"

# elif [ "${MOTION_DETECTION}" == "True" ]; then
#     log_info "Starting Motion Detection mode..."
#     nohup $PWD/.stream_env/bin/python3 app/motion.py > .log.out 2>&1 &
#     echo $! > .process.pid
#     log_info "Motion detection started. PID: $(cat .process.pid)"

# elif [ "${OBJECT_DETECTION}" == "False" ]; then
#     log_info "Starting direct streaming mode..."

#     if ! rpicam-hello --list-cameras > /dev/null 2>&1; then
#         log_error "No camera detected or rpicam-apps not installed"
#         exit 1
#     fi

#     RPICAM_CMD="rpicam-vid \
#         --nopreview \
#         --inline \
#         --timeout 0 \
#         --framerate $CAMERA_FPS \
#         --width $CAMERA_WIDTH \
#         --height $CAMERA_HEIGHT \
#         --rotation $CAMERA_ROTATION \
#         --brightness 0.1 \
#         --contrast $CONTRAST \
#         --sharpness $SHARPNESS \
#         --denoise $DENOISE_MODE \
#         --autofocus-mode continuous"

#     if [[ "$PI_MODEL" == "pi4" ]] && [[ "$QUALITY_PRESET" == "high" || "$QUALITY_PRESET" == "ultra" ]]; then
#         RPICAM_CMD="$RPICAM_CMD --hdr"
#         log_info "HDR mode enabled"
#     fi

#     RPICAM_CMD="$RPICAM_CMD --codec h264"

#     if [ ! -z "$CAMERA_TUNING" ]; then
#         RPICAM_CMD="$RPICAM_CMD --tuning $CAMERA_TUNING"
#     fi

#     RPICAM_CMD="$RPICAM_CMD --listen -o -"
    
#     if [ "${CAMERA_AUDIO}" == "True" ]; then
#         log_info "Configuring stream with audio..."

#         AUDIO_DEVICE="default"
#         if arecord -l 2>/dev/null | grep -q "USB"; then
#             log_info "USB audio device detected"
#             AUDIO_DEVICE="hw:1,0"
#         fi

#         if ! arecord -D $AUDIO_DEVICE -d 1 -f cd -t raw > /dev/null 2>&1; then
#             log_warn "Audio device $AUDIO_DEVICE not working, falling back to default"
#             AUDIO_DEVICE="default"
#         fi

#         stream_command="$RPICAM_CMD | \
# ffmpeg \
#     -thread_queue_size 2048 \
#     -f h264 \
#     -i - \
#     -f alsa \
#     -thread_queue_size 2048 \
#     -ar 48000 \
#     -ac 2 \
#     -i $AUDIO_DEVICE \
#     -nostdin \
#     -c:v copy \
#     -c:a aac \
#     -b:a ${AUDIO_BITRATE}k \
#     -ar 48000 \
#     -af \"aresample=async=1:min_hard_comp=0.100000:first_pts=0\" \
#     -vsync cfr \
#     -g $GOP_SIZE \
#     -max_muxing_queue_size 2048 \
#     -rtmp_buffer 100 \
#     -flvflags no_duration_filesize \
#     -f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY"
#     else
#         log_info "Configuring stream without audio..."

#         if [[ "$PI_MODEL" == "pi4" ]]; then
#             stream_command="$RPICAM_CMD | \
# ffmpeg \
#     -f h264 \
#     -i - \
#     -nostdin \
#     -c:v copy \
#     -g $GOP_SIZE \
#     -flvflags no_duration_filesize \
#     -rtmp_buffer 100 \
#     -f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY"
#         else
#             stream_command="$RPICAM_CMD | \
# ffmpeg \
#     -f h264 \
#     -i - \
#     -nostdin \
#     -c:v libx264 \
#     -profile:v $H264_PROFILE \
#     -level:v $H264_LEVEL \
#     -preset $FFMPEG_PRESET \
#     -tune zerolatency \
#     -b:v ${VIDEO_BITRATE}k \
#     -maxrate ${VIDEO_BITRATE}k \
#     -bufsize $((VIDEO_BITRATE * 2))k \
#     -g $GOP_SIZE \
#     -keyint_min $((GOP_SIZE / 2)) \
#     -sc_threshold 0 \
#     -pix_fmt yuv420p \
#     -r $CAMERA_FPS \
#     -vsync cfr \
#     -flvflags no_duration_filesize \
#     -rtmp_buffer 100 \
#     -f flv rtmp://$STREAM_IP:$STREAM_PORT/$STREAM_APPLICATION/$STREAM_KEY"
#         fi
#     fi

#     log_info "Starting stream..."
#     echo $stream_command
#     log_info "Command: $stream_command"

#     nohup bash -c "$stream_command" > .log.out 2>&1 &
#     STREAM_PID=$!
#     echo $STREAM_PID > .process.pid

#     log_info "Holly-stream started. PID: $STREAM_PID"
#     log_info "Monitor logs with: tail -f .log.out"

#     sleep 3
#     if kill -0 $STREAM_PID 2>/dev/null; then
#         log_info "Stream process is running"
#         if pgrep -f "ffmpeg.*rtmp://$STREAM_IP" > /dev/null; then
#             log_info "FFmpeg streaming process detected - stream should be active"
#         else
#             log_warn "FFmpeg process not detected - check .log.out for errors"
#         fi
#     else
#         log_error "Stream failed to start. Check .log.out for details"
#         if [ -f .log.out ]; then
#             log_error "Last few lines from log:"
#             tail -5 .log.out
#         fi
#         exit 1
#     fi

# else
#     log_error "Invalid input for OBJECT_DETECTION or MOTION_DETECTION. At least one value must be True."
#     exit 120
# fi

# log_info "Stream started successfully. Use your separate stop script to terminate."
