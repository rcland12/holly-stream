# Holly detector: SRT in -> NVDEC -> TensorRT YOLO -> boxes drawn on NV12 -> NVENC -> RTMP out.
#
# Ubuntu's FFmpeg is built with NVENC/NVDEC (cuvid) and SRT, and the NVIDIA container runtime mounts the
# driver's codec libraries when NVIDIA_DRIVER_CAPABILITIES includes "video". PyAV is compiled against that
# FFmpeg, because PyAV's binary wheels bundle their own FFmpeg without NVIDIA codecs.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        ffmpeg \
        fonts-dejavu-core \
        libavcodec-dev \
        libavdevice-dev \
        libavfilter-dev \
        libavformat-dev \
        libavutil-dev \
        libswresample-dev \
        libswscale-dev \
        pkg-config \
        python3-dev \
        python3-venv && \
    rm -rf /var/lib/apt/lists/*

# torch is its own layer: it is ~3 GB and changes far less often than the other requirements.
# The cu126 build runs on NVIDIA drivers 560 and newer.
RUN python3 -m venv ${VIRTUAL_ENV} && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cu126

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --no-binary av -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

WORKDIR /app
COPY holly ./holly

CMD ["python", "-m", "holly.supervisor"]
