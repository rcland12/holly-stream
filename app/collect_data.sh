#!/bin/bash
#
# Save camera frames at an interval for building a custom training dataset.
#
# Frames use the same sensor mode, ISP tuning (white balance, noise reduction)
# and output size as the stream, so training images look like what the model
# sees. The camera can only be opened by one process, so stop the stream first:
#
#   docker-compose stop app
#   mkdir -p data
#   docker-compose run --rm -v "$PWD/data:/data" --entrypoint /opt/holly-stream/collect_data.sh app \
#       --count 200 --period 5
#   docker-compose start app
#
# Images are saved as data/images/<prefix>_00000.jpg, ... Repeated runs continue
# the numbering, so a dataset can be built up over several sessions.

set -euo pipefail

OUTPUT=/data/images
COUNT=200
PERIOD=5
PREFIX=$(hostname)
WIDTH=1280
HEIGHT=720

usage() {
    echo "Usage: collect_data.sh [--output DIR] [--count N] [--period SECONDS] [--prefix NAME] [--size WxH]"
    echo "  --output   directory to save images to (default ${OUTPUT})"
    echo "  --count    number of images to save (default ${COUNT})"
    echo "  --period   whole seconds between images (default ${PERIOD})"
    echo "  --prefix   filename prefix (default: hostname)"
    echo "  --size     saved image size (default ${WIDTH}x${HEIGHT}, the 720p stream size)"
}

while [ $# -gt 0 ]; do
    case "$1" in
        --output) OUTPUT="$2"; shift 2 ;;
        --count) COUNT="$2"; shift 2 ;;
        --period) PERIOD="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --size) WIDTH="${2%x*}"; HEIGHT="${2#*x}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
done

FPS="${CAMERA_FPS:-30}"
CAPTURE_WIDTH="${CAPTURE_WIDTH:-1920}"
CAPTURE_HEIGHT="${CAPTURE_HEIGHT:-1080}"

mkdir -p "$OUTPUT"
# Continue numbering after the highest existing <prefix>_NNNNN.jpg.
START=$(find "$OUTPUT" -maxdepth 1 -name "${PREFIX}_*.jpg" -printf '%f\n' |
        sed -nE "s/^${PREFIX}_([0-9]+)\.jpg$/\1/p" | sort -n | tail -1)
START=$(( ${START:+10#$START + 1} + 0 ))

# Capture one extra image and discard it: the first frame arrives before auto
# exposure and white balance have settled.
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
BUFFERS=$(( (COUNT + 1) * PERIOD * FPS + FPS ))

echo "Saving ${COUNT} images (${WIDTH}x${HEIGHT}, every ${PERIOD}s) to ${OUTPUT}/${PREFIX}_$(printf %05d "$START").jpg ..."

( while sleep "$PERIOD"; do
    n=$(( $(find "$TMP" -name '*.jpg' | wc -l) - 1 ))
    [ "$n" -gt 0 ] && [ "$n" -le "$COUNT" ] && echo "  ${n}/${COUNT}"
  done ) &
PROGRESS=$!

GST_DEBUG=1 gst-launch-1.0 -q -e \
    nvarguscamerasrc sensor-id="${CAMERA_INDEX:-0}" num-buffers="$BUFFERS" \
        wbmode="${CAMERA_WBMODE:-1}" tnr-mode="${CAMERA_TNR_MODE:-1}" tnr-strength="${CAMERA_TNR_STRENGTH:-0.5}" ! \
    "video/x-raw(memory:NVMM),width=${CAPTURE_WIDTH},height=${CAPTURE_HEIGHT},framerate=${FPS}/1,format=NV12" ! \
    nvvidconv ! "video/x-raw(memory:NVMM),width=${WIDTH},height=${HEIGHT},format=I420" ! \
    videorate drop-only=true ! "video/x-raw(memory:NVMM),framerate=1/${PERIOD}" ! \
    nvjpegenc quality=95 ! \
    multifilesink location="${TMP}/frame_%05d.jpg" > /dev/null

{ kill "$PROGRESS" && wait "$PROGRESS"; } 2>/dev/null || true

index=$START
for frame in $(ls "$TMP"/frame_*.jpg | sort | tail -n +2 | head -n "$COUNT"); do
    mv "$frame" "${OUTPUT}/${PREFIX}_$(printf %05d "$index").jpg"
    index=$(( index + 1 ))
done
# The container runs as root; hand the images to whoever owns the mounted folder.
chown -R --reference="$(dirname "$OUTPUT")" "$OUTPUT" 2>/dev/null || true
echo "Saved $(( index - START )) images to ${OUTPUT} (${PREFIX}_$(printf %05d "$START").jpg - ${PREFIX}_$(printf %05d $(( index - 1 ))).jpg)"
