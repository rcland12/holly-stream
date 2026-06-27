#!/bin/bash
#
# Raspberry Pi CSI/USB-camera RTMP streaming pipeline.
#
# Backends (auto-detected, override with CAMERA_BACKEND):
#   libcamera  -> rpicam-vid (HW H.264 encode) | ffmpeg (mux)   [Pi camera stack]
#   legacy     -> raspivid   (HW H.264 encode) | ffmpeg (mux)   [old firmware stack]
#   usb        -> ffmpeg (v4l2 capture + h264_v4l2m2m or libx264 encode)
#
# Capture is decoupled from output: on the libcamera path the sensor runs a
# full-FOV mode and the ISP downscales to the target. This keeps a wide lens's
# full field of view AND a stream size the Pi HW encoder can handle (<=1080p).
#
# =============================================================================
# PRESETS  (set as environment variables, e.g. in .env -- all have defaults)
# =============================================================================
#
# STREAM_RESOLUTION  -> output dimensions + aspect ratio + framerate
#   (default: 4:3 -- matches the IMX519's 4:3 full-FOV sensor, no stretching)
#
#   Preset        Output        Aspect   FPS   Notes
#   -----------   -----------   ------   ---   ----------------------------------
#   4:3           1440x1080     4:3      30    full-FOV match for IMX519/IMX477
#   720p_4:3      960x720       4:3      30    lower-bandwidth 4:3
#   1080p         1920x1080     16:9     30    crops/stretches a 4:3 sensor mode
#   720p          1280x720      16:9     30
#   720p60        1280x720      16:9     60
#   480p          854x480       16:9     30
#   square        1080x1080     1:1      30
#   custom        CAMERA_WIDTH x CAMERA_HEIGHT @ CAMERA_FPS  (legacy variables)
#   <W>x<H>       explicit, e.g. STREAM_RESOLUTION=1600x1200
#
#   NOTE: the Pi 4 hardware H.264 encoder is limited to ~1080 lines, so keep the
#   output height <= 1080. Capture can be much larger; it is downscaled.
#
# STREAM_QUALITY     -> rate control + bitrate (bitrate auto-scales with the
#   (default: high)     resolution; override with VIDEO_BITRATE, e.g. 6000k)
#
#   Preset      Rel. bitrate   Notes
#   ---------   ------------   --------------------------------------------------
#   ultra       highest        best image; needs strong uplink
#   high        high           excellent quality (default)
#   smooth      medium         balanced quality + stability
#   balanced    lower          good quality/performance        (alias: medium)
#   fast        lowest         best for constrained uplinks     (alias: low)
#
# FULL FIELD OF VIEW (libcamera path):
#   CAPTURE_WIDTH / CAPTURE_HEIGHT select the sensor mode fed to the ISP. Default
#   is 2328x1748 -- the IMX519's full-sensor binned mode (full 4:3 FOV @ 30fps).
#   The mode is verified against `rpicam-vid --list-cameras`; if your camera does
#   not have it, the script falls back to auto mode (so it stays camera-agnostic).
#   Discover your camera's full-FOV mode (look for crop "(0, 0)/<full>"):
#       rpicam-vid --list-cameras
#
# Other tunables (all optional, with defaults):
#   VIDEO_BITRATE     force bitrate, e.g. 6000k (skips auto-calc)
#   GOP_SECONDS       keyframe interval in seconds (default 2; alias KEYINT_SECONDS)
#   CAMERA_FPS        overrides the preset framerate when set
#   CAMERA_ROTATION   0 or 180 (default 180)
#   CAMERA_BACKEND    auto | libcamera | legacy | usb   (default auto)
#   USE_ENCODER       cpu | v4l2m2m   (USB path only; default cpu)
#   AUDIO_ENABLED     True | False    (REQUIRED)
#   AUDIO_DEVICE      e.g. hw:1,0 or pulse:<name> (auto-detected if empty)
#   RESTART_DELAY     seconds between restart attempts (default 3)
#
# Examples:
#   STREAM_RESOLUTION=4:3   STREAM_QUALITY=high   AUDIO_ENABLED=True  ./stream.sh
#   STREAM_RESOLUTION=720p_4:3 STREAM_QUALITY=smooth AUDIO_ENABLED=False ./stream.sh
# =============================================================================

set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

if [ -z "${AUDIO_ENABLED:-}" ]; then
  log_error "AUDIO_ENABLED is required (True/False)."
  exit 1
fi

# ---------------------------------------------------------------------------
# Connection / device defaults
# ---------------------------------------------------------------------------
STREAM_IP="${STREAM_IP:-127.0.0.1}"
STREAM_PORT="${STREAM_PORT:-1935}"
STREAM_APPLICATION="${STREAM_APPLICATION:-live}"
STREAM_KEY="${STREAM_KEY:-stream}"
CAMERA_INDEX="${CAMERA_INDEX:-0}"
CAMERA_ROTATION="${CAMERA_ROTATION:-180}"
AUDIO_DEVICE="${AUDIO_DEVICE:-}"

# ----- Image tuning (libcamera/rpicam-vid). Empty = camera auto. -----
# To BRIGHTEN a dark image, prefer CAMERA_EV (exposure compensation in stops,
# e.g. 0.5 or 1.0) -- it lets auto-exposure target brighter while balancing
# shutter/gain. CAMERA_BRIGHTNESS (-1.0..1.0) is a simpler post-lift. Raising
# gain brightens but adds noise (and noise costs bitrate), so pair brightening
# with CAMERA_DENOISE=cdn_hq to keep the stream clean.
CAMERA_EV="${CAMERA_EV:-}"               # exposure compensation, stops (brightens)
CAMERA_BRIGHTNESS="${CAMERA_BRIGHTNESS:-}"   # -1.0..1.0 additive brightness
CAMERA_GAIN="${CAMERA_GAIN:-}"           # analogue gain (ISO-like); higher = brighter + noisier
CAMERA_SHUTTER="${CAMERA_SHUTTER:-}"     # fixed shutter in us; longer = brighter (caps fps)
CAMERA_CONTRAST="${CAMERA_CONTRAST:-}"   # 0.0..~2.0 (1.0 = normal)
CAMERA_SATURATION="${CAMERA_SATURATION:-}"
CAMERA_SHARPNESS="${CAMERA_SHARPNESS:-}"
CAMERA_AWB="${CAMERA_AWB:-}"             # auto|incandescent|tungsten|fluorescent|indoor|daylight|cloudy
CAMERA_DENOISE="${CAMERA_DENOISE:-}"     # off|cdn_off|cdn_fast|cdn_hq
CAMERA_METERING="${CAMERA_METERING:-}"   # centre|spot|average
CAMERA_EXPOSURE="${CAMERA_EXPOSURE:-}"   # normal|sport|long
USE_ENCODER="${USE_ENCODER:-cpu}"
RESTART_DELAY="${RESTART_DELAY:-3}"
CAMERA_BACKEND="${CAMERA_BACKEND:-auto}"

# ---------------------------------------------------------------------------
# RESOLUTION PRESET  -> OUT_W / OUT_H / OUT_FPS (the encoded/streamed size)
# ---------------------------------------------------------------------------
STREAM_RESOLUTION="${STREAM_RESOLUTION:-4:3}"

case "$STREAM_RESOLUTION" in
    4:3|1080p_4:3) OUT_W=1440; OUT_H=1080; OUT_FPS=30 ;;
    720p_4:3)      OUT_W=960;  OUT_H=720;  OUT_FPS=30 ;;
    1080p|1080p30) OUT_W=1920; OUT_H=1080; OUT_FPS=30 ;;
    720p|720p30)   OUT_W=1280; OUT_H=720;  OUT_FPS=30 ;;
    720p60)        OUT_W=1280; OUT_H=720;  OUT_FPS=60 ;;
    480p)          OUT_W=854;  OUT_H=480;  OUT_FPS=30 ;;
    square|1:1)    OUT_W=1080; OUT_H=1080; OUT_FPS=30 ;;
    custom|"")
        OUT_W="${CAMERA_WIDTH:-1920}"
        OUT_H="${CAMERA_HEIGHT:-1080}"
        OUT_FPS="${CAMERA_FPS:-30}"
        log_info "Using custom resolution from CAMERA_WIDTH/HEIGHT/FPS"
        ;;
    *)
        if [[ "$STREAM_RESOLUTION" =~ ^([0-9]+)x([0-9]+)$ ]]; then
            OUT_W="${BASH_REMATCH[1]}"
            OUT_H="${BASH_REMATCH[2]}"
            OUT_FPS="${CAMERA_FPS:-30}"
            log_info "Using explicit resolution ${OUT_W}x${OUT_H}"
        else
            log_warn "Unknown STREAM_RESOLUTION '${STREAM_RESOLUTION}', defaulting to 4:3 (1440x1080)"
            OUT_W=1440; OUT_H=1080; OUT_FPS=30
        fi
        ;;
esac

# CAMERA_FPS (if explicitly set) always wins, so existing configs keep working.
[ -n "${CAMERA_FPS:-}" ] && OUT_FPS="$CAMERA_FPS"

# Keep these populated for the 'custom' path / display.
CAMERA_WIDTH="${CAMERA_WIDTH:-$OUT_W}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-$OUT_H}"
CAMERA_FPS="$OUT_FPS"
DIMS="${OUT_W}x${OUT_H}"

# ---------------------------------------------------------------------------
# CAPTURE MODE (libcamera) -> full-FOV sensor mode fed to the ISP downscaler.
# Default = IMX519 full-sensor binned mode. Verified at runtime; falls back to
# auto if unavailable, so the script works on any Pi camera.
# ---------------------------------------------------------------------------
CAPTURE_WIDTH="${CAPTURE_WIDTH:-2328}"
CAPTURE_HEIGHT="${CAPTURE_HEIGHT:-1748}"

# ---------------------------------------------------------------------------
# KEYFRAME interval. Keep ~2s so keyframes align with the downstream HLS
# fragments (cleaner fragments, faster joins).
# ---------------------------------------------------------------------------
GOP_SECONDS="${GOP_SECONDS:-${KEYINT_SECONDS:-2}}"
GOP_SIZE=$(( OUT_FPS * GOP_SECONDS ))
KEYINT_MIN="$OUT_FPS"

# ---------------------------------------------------------------------------
# QUALITY PRESET  -> bits-per-pixel + encoder params. Bitrate is derived from
# resolution x fps so it scales with the chosen size. control: maxrate ~= bitrate
# and a 1s buffer give a stable, CBR-like rate (minimal buffering downstream).
# ---------------------------------------------------------------------------
STREAM_QUALITY="${STREAM_QUALITY:-high}"

case "$STREAM_QUALITY" in
    ultra)
        BPP_NUM=15; CPU_PRESET="slow";     V4L2M2M_PROFILE="high";     V4L2M2M_LEVEL="4.2"
        log_info "Quality: ULTRA (highest image quality, needs strong uplink)" ;;
    high)
        BPP_NUM=12; CPU_PRESET="medium";   V4L2M2M_PROFILE="high";     V4L2M2M_LEVEL="4.1"
        log_info "Quality: HIGH (excellent quality)" ;;
    smooth)
        BPP_NUM=10; CPU_PRESET="veryfast"; V4L2M2M_PROFILE="high";     V4L2M2M_LEVEL="4.1"
        log_info "Quality: SMOOTH (balanced quality, stable bitrate)" ;;
    balanced|medium)
        BPP_NUM=8;  CPU_PRESET="veryfast"; V4L2M2M_PROFILE="main";     V4L2M2M_LEVEL="4.0"
        log_info "Quality: BALANCED (good quality/performance)" ;;
    fast|low)
        BPP_NUM=6;  CPU_PRESET="ultrafast"; V4L2M2M_PROFILE="baseline"; V4L2M2M_LEVEL="3.1"
        log_info "Quality: FAST (lowest bitrate, best for constrained uplinks)" ;;
    *)
        BPP_NUM=12; CPU_PRESET="medium";   V4L2M2M_PROFILE="high";     V4L2M2M_LEVEL="4.1"
        log_warn "Unknown STREAM_QUALITY '${STREAM_QUALITY}', defaulting to HIGH" ;;
esac
BPP_DEN=100

# Compute (or honor explicit) bitrate, in integer kbps.
if [ -n "${VIDEO_BITRATE:-}" ]; then
    BR_KBPS="${VIDEO_BITRATE%k}"
    log_info "Using explicit VIDEO_BITRATE=${VIDEO_BITRATE}"
else
    BR_KBPS=$(( OUT_W * OUT_H * OUT_FPS * BPP_NUM / (BPP_DEN * 1000) ))
fi
# Clamp to a sane range (Pi HW encoder practical ceiling ~10 Mbps).
[ "$BR_KBPS" -lt 800 ]   && BR_KBPS=800
[ "$BR_KBPS" -gt 10000 ] && BR_KBPS=10000

VIDEO_BITRATE="${BR_KBPS}k"
MAX_BITRATE="$(( BR_KBPS * 115 / 100 ))k"   # small headroom over target
BUFFER_SIZE="${BR_KBPS}k"                    # 1s VBV -> tight, CBR-like
RPICAM_BPS=$(( BR_KBPS * 1000 ))

# ---------------------------------------------------------------------------
# Show config
# ---------------------------------------------------------------------------
detect_pi_model() {
  if [ -f /proc/device-tree/model ]; then
    local MODEL; MODEL=$(tr -d '\0' < /proc/device-tree/model)
    log_info "Detected: ${MODEL}"
  fi
}
detect_pi_model

log_info "Stream Configuration:"
echo -e "${BLUE}  Resolution preset:${NC} ${STREAM_RESOLUTION}"
echo -e "${BLUE}  Output:${NC} ${OUT_W}x${OUT_H} @ ${OUT_FPS}fps"
echo -e "${BLUE}  Capture (libcamera):${NC} ${CAPTURE_WIDTH}x${CAPTURE_HEIGHT} (full-FOV, downscaled)"
echo -e "${BLUE}  Quality:${NC} ${STREAM_QUALITY} -> ${VIDEO_BITRATE} (max ${MAX_BITRATE}, buf ${BUFFER_SIZE})"
echo -e "${BLUE}  Keyframe interval:${NC} ${GOP_SIZE} frames (${GOP_SECONDS}s)"
echo -e "${BLUE}  Rotation:${NC} ${CAMERA_ROTATION}"
echo -e "${BLUE}  Audio:${NC} ${AUDIO_ENABLED} (device='${AUDIO_DEVICE}')"
echo -e "${BLUE}  Backend/Encoder:${NC} ${CAMERA_BACKEND} / ${USE_ENCODER}"
echo -e "${BLUE}  Target:${NC} rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}"

# ---------------------------------------------------------------------------
# Backend detection
# ---------------------------------------------------------------------------
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
      if command -v rpicam-vid >/dev/null 2>&1; then echo "libcamera"; return 0; fi
      if command -v raspivid >/dev/null 2>&1; then echo "legacy"; return 0; fi
      if [ -e "/dev/video${CAMERA_INDEX}" ]; then echo "usb"; return 0; fi
      echo "none"; return 1;;
  esac
}

CAMERA_TYPE="$(check_camera_backend || echo none)"
if [ "$CAMERA_TYPE" = "none" ]; then
  log_error "No camera backend detected (rpicam-vid/raspivid or /dev/video*)."
  exit 1
fi
log_info "Camera backend: ${CAMERA_TYPE}"

# The Pi hardware H.264 encoder (libcamera/legacy paths) tops out near 1080 lines.
if [ "$CAMERA_TYPE" != "usb" ] && [ "$OUT_H" -gt 1080 ]; then
  log_warn "Output height ${OUT_H} exceeds the Pi HW encoder's ~1080-line limit; expect failure or fallback. Use an output <=1080 tall."
fi

# ---------------------------------------------------------------------------
# Verify the full-FOV capture mode exists (libcamera). Keeps us camera-agnostic:
# unknown mode -> auto (rpicam picks a mode, FOV may be cropped).
# ---------------------------------------------------------------------------
LIBCAMERA_MODE_ARG=""
if [ "$CAMERA_TYPE" = "libcamera" ] && [ -n "$CAPTURE_WIDTH" ] && [ -n "$CAPTURE_HEIGHT" ]; then
  if rpicam-vid --list-cameras 2>/dev/null | grep -q "${CAPTURE_WIDTH}x${CAPTURE_HEIGHT}"; then
    LIBCAMERA_MODE_ARG="--mode ${CAPTURE_WIDTH}:${CAPTURE_HEIGHT}"
    log_info "Full-FOV sensor mode ${CAPTURE_WIDTH}x${CAPTURE_HEIGHT} available."
  else
    log_warn "Sensor mode ${CAPTURE_WIDTH}x${CAPTURE_HEIGHT} not found on this camera; using auto mode (FOV may be cropped)."
  fi
fi

# ---------------------------------------------------------------------------
# Audio input (ffmpeg). Auto-detect ALSA/Pulse if AUDIO_DEVICE is empty.
# ---------------------------------------------------------------------------
build_audio_input() {
  if [ "${AUDIO_ENABLED}" != "True" ]; then echo ""; return 0; fi

  if [ -n "${AUDIO_DEVICE:-}" ]; then
    if [[ "$AUDIO_DEVICE" == pulse:* ]]; then
      if command -v pactl >/dev/null 2>&1 && pactl info >/dev/null 2>&1; then
        echo "-f pulse -thread_queue_size 8192 -i ${AUDIO_DEVICE#pulse:}"; return 0
      else
        echo ""; return 0
      fi
    else
      echo "-f alsa -thread_queue_size 8192 -ar 48000 -ac 2 -i ${AUDIO_DEVICE}"; return 0
    fi
  fi

  if command -v pactl >/dev/null 2>&1 && pactl info >/dev/null 2>&1; then
    echo "-f pulse -thread_queue_size 8192 -i default"; return 0
  fi

  if command -v arecord >/dev/null 2>&1; then
    local CARD_LINE
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

# CBR-like ffmpeg encoder params (USB path). maxrate~=bitrate + 1s bufsize.
build_encoding_params() {
  case "$1" in
    v4l2m2m)
      echo "-c:v h264_v4l2m2m \
        -profile:v ${V4L2M2M_PROFILE} \
        -level:v ${V4L2M2M_LEVEL} \
        -b:v ${VIDEO_BITRATE} \
        -maxrate ${MAX_BITRATE} \
        -bufsize ${BUFFER_SIZE} \
        -g ${GOP_SIZE} \
        -keyint_min ${KEYINT_MIN} \
        -sc_threshold 0 \
        -pix_fmt yuv420p"
      ;;
    *)
      echo "-c:v libx264 \
        -preset:v ${CPU_PRESET} \
        -tune:v zerolatency \
        -profile:v high \
        -level:v 4.2 \
        -b:v ${VIDEO_BITRATE} \
        -maxrate ${MAX_BITRATE} \
        -bufsize ${BUFFER_SIZE} \
        -g ${GOP_SIZE} \
        -keyint_min ${KEYINT_MIN} \
        -sc_threshold 0 \
        -pix_fmt yuv420p"
      ;;
  esac
}

# Assemble optional rpicam-vid image-tuning flags (only those that are set).
build_libcamera_tuning() {
  local t=""
  [ -n "$CAMERA_EV" ]         && t="$t --ev $CAMERA_EV"
  [ -n "$CAMERA_BRIGHTNESS" ] && t="$t --brightness $CAMERA_BRIGHTNESS"
  [ -n "$CAMERA_GAIN" ]       && t="$t --gain $CAMERA_GAIN"
  [ -n "$CAMERA_SHUTTER" ]    && t="$t --shutter $CAMERA_SHUTTER"
  [ -n "$CAMERA_CONTRAST" ]   && t="$t --contrast $CAMERA_CONTRAST"
  [ -n "$CAMERA_SATURATION" ] && t="$t --saturation $CAMERA_SATURATION"
  [ -n "$CAMERA_SHARPNESS" ]  && t="$t --sharpness $CAMERA_SHARPNESS"
  [ -n "$CAMERA_AWB" ]        && t="$t --awb $CAMERA_AWB"
  [ -n "$CAMERA_DENOISE" ]    && t="$t --denoise $CAMERA_DENOISE"
  [ -n "$CAMERA_METERING" ]   && t="$t --metering $CAMERA_METERING"
  [ -n "$CAMERA_EXPOSURE" ]   && t="$t --exposure $CAMERA_EXPOSURE"
  echo "$t"
}
LIBCAMERA_TUNING="$(build_libcamera_tuning)"
[ -n "$LIBCAMERA_TUNING" ] && log_info "Image tuning:${LIBCAMERA_TUNING}"

ENCODER="$(choose_encoder "$USE_ENCODER")"
[ "$ENCODER" = "v4l2m2m" ] && log_info "USB encoder: V4L2 M2M (auto-fallback to CPU on failure)." || log_info "USB encoder: CPU (libx264)."

AUDIO_INPUT="$(build_audio_input)"
if [ "${AUDIO_ENABLED}" == "True" ] && [ -z "${AUDIO_INPUT}" ]; then
  log_warn "No usable audio input found; disabling audio to keep streaming."
  AUDIO_ENABLED="False"
fi

# ---------------------------------------------------------------------------
# One streaming attempt for the active backend.
# ---------------------------------------------------------------------------
run_once() {
  local enc="$1"
  local ENCODING_PARAMS; ENCODING_PARAMS="$(build_encoding_params "$enc")"

  # async audio resampling keeps A/V in sync over a live stream.
  local MUX_TAIL_AUDIO="-c:a aac -b:a 128k -ar 48000 -ac 2 -af \"aresample=async=1:min_hard_comp=0.100000:first_pts=0\""
  local MUX_TAIL_NOAUDIO=""
  local COMMON_TAIL="-use_wallclock_as_timestamps 1 -fflags +genpts -fps_mode cfr -r ${OUT_FPS} \
                     -max_muxing_queue_size 2048 -rtmp_buffer 100 -flvflags no_duration_filesize \
                     -f flv \"rtmp://${STREAM_IP}:${STREAM_PORT}/${STREAM_APPLICATION}/${STREAM_KEY}\""

  # ---- USB (v4l2 capture + ffmpeg encode) ----
  if [ "$CAMERA_TYPE" = "usb" ]; then
    if [ "${AUDIO_ENABLED}" == "True" ]; then
      log_info "Starting USB stream with audio..."
      eval ffmpeg \
        -f v4l2 -thread_queue_size 4096 -framerate "$OUT_FPS" -video_size "$DIMS" -input_format mjpeg \
        -i "/dev/video${CAMERA_INDEX}" \
        ${AUDIO_INPUT} \
        -vf "format=yuv420p" \
        ${ENCODING_PARAMS} \
        ${MUX_TAIL_AUDIO} \
        ${COMMON_TAIL}
    else
      log_info "Starting USB stream without audio..."
      eval ffmpeg \
        -f v4l2 -thread_queue_size 4096 -framerate "$OUT_FPS" -video_size "$DIMS" -input_format mjpeg \
        -i "/dev/video${CAMERA_INDEX}" \
        -vf "format=yuv420p" \
        ${ENCODING_PARAMS} \
        ${MUX_TAIL_NOAUDIO} \
        ${COMMON_TAIL}
    fi
    return $?
  fi

  # ---- libcamera (rpicam-vid HW encode, full-FOV capture, ffmpeg mux) ----
  if [ "$CAMERA_TYPE" = "libcamera" ] && command -v rpicam-vid >/dev/null 2>&1; then
    if [ "${AUDIO_ENABLED}" == "True" ]; then
      log_info "Starting rpicam stream with audio (HW encode, copy video)..."
      eval rpicam-vid \
        --nopreview --inline --timeout 0 \
        ${LIBCAMERA_MODE_ARG} \
        --width "$OUT_W" --height "$OUT_H" \
        --framerate "$OUT_FPS" --rotation "$CAMERA_ROTATION" \
        --codec h264 --profile "${V4L2M2M_PROFILE}" --level "${V4L2M2M_LEVEL}" \
        --intra "${GOP_SIZE}" \
        ${LIBCAMERA_TUNING} \
        --bitrate "${RPICAM_BPS}" -o - \| \
      ffmpeg -thread_queue_size 2048 -f h264 -r "$OUT_FPS" -probesize 50M -analyzeduration 2M -i - \
        ${AUDIO_INPUT} \
        -c:v copy \
        -fps_mode cfr \
        ${MUX_TAIL_AUDIO} \
        ${COMMON_TAIL}
    else
      log_info "Starting rpicam stream without audio (HW encode, copy video)..."
      eval rpicam-vid \
        --nopreview --inline --timeout 0 \
        ${LIBCAMERA_MODE_ARG} \
        --width "$OUT_W" --height "$OUT_H" \
        --framerate "$OUT_FPS" --rotation "$CAMERA_ROTATION" \
        --codec h264 --profile "${V4L2M2M_PROFILE}" --level "${V4L2M2M_LEVEL}" \
        --intra "${GOP_SIZE}" \
        ${LIBCAMERA_TUNING} \
        --bitrate "${RPICAM_BPS}" -o - \| \
      ffmpeg -thread_queue_size 2048 -f h264 -r "$OUT_FPS" -probesize 50M -analyzeduration 2M -i - \
        -c:v copy \
        -fps_mode cfr \
        ${MUX_TAIL_NOAUDIO} \
        ${COMMON_TAIL}
    fi
    return $?
  fi

  # ---- legacy (raspivid HW encode, ffmpeg mux) ----
  if [ "$CAMERA_TYPE" = "legacy" ] && command -v raspivid >/dev/null 2>&1; then
    if [ "${AUDIO_ENABLED}" == "True" ]; then
      log_info "Starting raspivid stream with audio (HW encode, copy video)..."
      eval raspivid --nopreview --timeout 0 \
        --width "$OUT_W" --height "$OUT_H" \
        --framerate "$OUT_FPS" --rotation "$CAMERA_ROTATION" \
        --bitrate "${RPICAM_BPS}" --profile high --inline --intra "${GOP_SIZE}" -o - \| \
      ffmpeg -thread_queue_size 2048 -f h264 -r "$OUT_FPS" -probesize 50M -analyzeduration 2M -i - \
        ${AUDIO_INPUT} \
        -c:v copy \
        -fps_mode cfr \
        ${MUX_TAIL_AUDIO} \
        ${COMMON_TAIL}
    else
      log_info "Starting raspivid stream without audio (HW encode, copy video)..."
      eval raspivid --nopreview --timeout 0 \
        --width "$OUT_W" --height "$OUT_H" \
        --framerate "$OUT_FPS" --rotation "$CAMERA_ROTATION" \
        --bitrate "${RPICAM_BPS}" --profile high --inline --intra "${GOP_SIZE}" -o - \| \
      ffmpeg -thread_queue_size 2048 -f h264 -r "$OUT_FPS" -probesize 50M -analyzeduration 2M -i - \
        -c:v copy \
        -fps_mode cfr \
        ${MUX_TAIL_NOAUDIO} \
        ${COMMON_TAIL}
    fi
    return $?
  fi

  log_error "Unknown or unavailable CAMERA_TYPE=${CAMERA_TYPE}"
  return 2
}

# ---------------------------------------------------------------------------
# Supervised run loop: restart on failure; USB v4l2m2m -> CPU fallback once.
# ---------------------------------------------------------------------------
while true; do
  run_once "$ENCODER"
  EC=$?
  if [ "$EC" -eq 0 ]; then log_info "Stream ended cleanly."; break; fi
  log_warn "Stream exited with code ${EC}."
  if [ "$CAMERA_TYPE" = "usb" ] && [ "$ENCODER" = "v4l2m2m" ]; then
    log_warn "Falling back to CPU (libx264)."
    ENCODER="cpu"
    continue
  fi
  log_warn "Retrying in ${RESTART_DELAY}s..."
  sleep "$RESTART_DELAY"
done
