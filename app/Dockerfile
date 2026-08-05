FROM debian:bookworm-slim AS builder

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
RUN git clone --depth=1 https://github.com/raspberrypi/libcamera.git && \
    cd libcamera && \
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
        -Dpycamera=enabled && \
    ninja -C build && \
    ninja -C build install && \
    DESTDIR=/build/libcamera-install \
        ninja -C build install

RUN git clone --depth=1 https://github.com/raspberrypi/rpicam-apps.git && \
    cd rpicam-apps && \
    meson setup build \
        -Denable_libav=disabled \
        -Denable_drm=enabled \
        -Denable_egl=disabled \
        -Denable_qt=disabled \
        -Denable_opencv=disabled \
        -Denable_tflite=disabled && \
    meson compile -C build && \
    DESTDIR=/build/rpicam-install meson install -C build

FROM debian:bookworm-slim

WORKDIR /root/app
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
        python3-numpy \
        python3-pip \
        python3-prctl \
        python3-yaml \
        udev \
        v4l-utils && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/libcamera-install/usr/local /usr/local
COPY --from=builder /build/rpicam-install/usr/local /usr/local

RUN pip3 install --break-system-packages --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --break-system-packages --no-cache-dir \
        imutils \
        opencv-python \
        picamera2 \
        python-dotenv \
        tritonclient[all]

ENV LD_LIBRARY_PATH=/usr/local/lib/aarch64-linux-gnu:/usr/local/lib:/usr/lib/aarch64-linux-gnu/blas:/usr/lib/aarch64-linux-gnu/lapack:$LD_LIBRARY_PATH
ENV PATH=/usr/local/bin:$PATH

COPY main.py stream.sh entrypoint.sh ./
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
