# syntax=docker/dockerfile:1

# Pin the base image by digest once you have a build you trust:
#   docker buildx imagetools inspect debian:bookworm-slim
#   docker build --build-arg BASE_IMAGE=debian:bookworm-slim@sha256:<digest> ...
ARG BASE_IMAGE=debian:bookworm-slim

# ---------------------------------------------------------------------------
# Builder stage: libcamera + rpicam-apps from pinned commits
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

# Required. Get these from a container that currently works:
#   docker run --rm <your-working-image> sh -c 'true'   # (image no longer has the repos)
# or, if the build tree is gone, from the upstream repos:
#   git ls-remote https://github.com/raspberrypi/libcamera.git HEAD
#   git ls-remote https://github.com/raspberrypi/rpicam-apps.git HEAD
# Record them in .env / CI so rebuilds are reproducible.
ARG LIBCAMERA_REF
ARG RPICAM_APPS_REF

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libavcodec-dev \
        libavdevice-dev \
        libavformat-dev \
        libavutil-dev \
        libboost-dev \
        libboost-program-options-dev \
        libdrm-dev \
        libexif-dev \
        libglib2.0-dev \
        libgnutls28-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libjpeg-dev \
        libpng-dev \
        libqt5core5a \
        libqt5gui5 \
        libqt5widgets5 \
        libssl-dev \
        libswresample-dev \
        libswscale-dev \
        libtiff5-dev \
        libudev-dev \
        meson \
        ninja-build \
        openssl \
        pkg-config \
        pybind11-dev \
        python3-dev \
        python3-jinja2 \
        python3-pip \
        python3-ply \
        python3-yaml \
        qtbase5-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Shallow fetch of one exact commit. GitHub permits fetching by SHA; if this
# ever fails, fall back to a full clone followed by `git checkout <ref>`.
#
# Note the double install: the plain `ninja install` puts libcamera into the
# builder so rpicam-apps can link against it, and the DESTDIR install stages a
# clean copy for the runtime stage. Both are needed.
RUN set -eux; \
    test -n "${LIBCAMERA_REF}"; \
    mkdir libcamera && cd libcamera; \
    git init -q; \
    git remote add origin https://github.com/raspberrypi/libcamera.git; \
    git fetch --depth=1 origin "${LIBCAMERA_REF}"; \
    git checkout -q FETCH_HEAD; \
    meson setup build \
        --buildtype=release \
        -Dpipelines=rpi/vc4,rpi/pisp \
        -Dipas=rpi/vc4,rpi/pisp \
        -Dv4l2=enabled \
        -Dgstreamer=enabled \
        -Dtest=false \
        -Dlc-compliance=disabled \
        -Dcam=disabled \
        -Dqcam=disabled \
        -Ddocumentation=disabled \
        -Dpycamera=enabled; \
    ninja -C build; \
    ninja -C build install; \
    DESTDIR=/build/libcamera-install ninja -C build install

RUN set -eux; \
    test -n "${RPICAM_APPS_REF}"; \
    mkdir rpicam-apps && cd rpicam-apps; \
    git init -q; \
    git remote add origin https://github.com/raspberrypi/rpicam-apps.git; \
    git fetch --depth=1 origin "${RPICAM_APPS_REF}"; \
    git checkout -q FETCH_HEAD; \
    meson setup build \
        -Denable_libav=disabled \
        -Denable_drm=enabled \
        -Denable_egl=disabled \
        -Denable_qt=disabled \
        -Denable_opencv=disabled \
        -Denable_tflite=disabled; \
    meson compile -C build; \
    DESTDIR=/build/rpicam-install meson install -C build

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM ${BASE_IMAGE}

WORKDIR /root/app

# python3-numpy, python3-yaml and python3-prctl are deliberately NOT installed
# here. All three are pulled in by pip (opencv-python, picamera2), and
# having an apt copy in /usr/lib alongside a pip copy in /usr/local means two
# of each on sys.path, with different BLAS linkage in numpy's case. Let pip own
# them. libcap-dev is here so pip can compile python-prctl, which ships no
# wheels; build-essential and python3-dev cover the rest of that build.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        alsa-utils \
        build-essential \
        ffmpeg \
        gstreamer1.0-plugins-base \
        gstreamer1.0-plugins-good \
        gstreamer1.0-alsa \
        kmod \
        libatlas-base-dev \
        libblas3 \
        libboost-program-options1.74.0 \
        libcap-dev \
        libdrm2 \
        libexif12 \
        libgfortran5 \
        libglib2.0-0 \
        libgnutls30 \
        libgstreamer1.0-0 \
        libgstreamer-plugins-base1.0-0 \
        libjpeg62-turbo \
        liblapack3 \
        libpng16-16 \
        libssl3 \
        libtiff6 \
        pkg-config \
        pulseaudio \
        pulseaudio-utils \
        python3 \
        python3-dev \
        python3-pip \
        udev \
        v4l-utils && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/libcamera-install/usr/local /usr/local
COPY --from=builder /build/rpicam-install/usr/local /usr/local

# Upgrading pip is the actual fix for the typing-extensions failure: bookworm
# ships pip 23.0.1, which does not canonicalize wheel metadata names and so
# rejects typing_extensions>=4.16 and falls back to an sdist it cannot build.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --break-system-packages --no-cache-dir --upgrade pip && \
    pip3 install --break-system-packages --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

ENV LD_LIBRARY_PATH=/usr/local/lib/aarch64-linux-gnu:/usr/local/lib:/usr/lib/aarch64-linux-gnu/blas:/usr/lib/aarch64-linux-gnu/lapack:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:$PATH
# meson installs the libcamera Python bindings here, which is not on Debian's default sys.path
ENV PYTHONPATH=/usr/local/lib/python3/dist-packages

COPY main.py collect_data.py stream.sh entrypoint.sh ./
COPY ./images/ ./images/

RUN echo 'SUBSYSTEM=="dma_heap", GROUP="video", MODE="0660"' > /etc/udev/rules.d/10-dma-heap.rules && \
    echo 'SUBSYSTEM=="media", GROUP="video", MODE="0660"' >> /etc/udev/rules.d/10-dma-heap.rules && \
    echo 'SUBSYSTEM=="video4linux", GROUP="video", MODE="0660"' >> /etc/udev/rules.d/10-dma-heap.rules && \
    echo "/usr/lib/aarch64-linux-gnu/blas" >> /etc/ld.so.conf.d/blas.conf && \
    ldconfig && \
    mkdir -p /dev/dma_heap && \
    mkdir -p /run/udev && \
    useradd -m -G video,audio camera-user && \
    chmod +x stream.sh entrypoint.sh

CMD ["/bin/bash", "entrypoint.sh"]
