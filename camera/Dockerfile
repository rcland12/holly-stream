# Holly camera for Linux: USB webcam -> H.264 (NVENC, VAAPI or x264) + AAC -> MPEG-TS -> SRT to the Holly server.
#
# Ubuntu's FFmpeg is built with NVENC/NVDEC, CUDA filters, VAAPI and SRT. The NVIDIA container runtime mounts the
# driver's codec libraries when the container runs with a GPU (the nvidia profile in compose.yml); the VA drivers
# below cover Intel and AMD GPUs.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_CACHE_PATH=/cache/cuda

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        alsa-utils \
        ffmpeg \
        intel-media-va-driver \
        mesa-va-drivers \
        v4l-utils && \
    rm -rf /var/lib/apt/lists/*

COPY holly-camera.sh /usr/local/bin/holly-camera.sh

ENTRYPOINT ["/usr/local/bin/holly-camera.sh"]
