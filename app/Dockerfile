FROM debian:bookworm-slim
WORKDIR /root/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git wget cmake meson python3.10 python3-pip \
    python3-dev python3-yaml python3-ply python3-jinja2 \
    ffmpeg build-essential libpng-dev libboost-dev \
    libgnutls28-dev openssl libtiff5-dev pybind11-dev \
    qtbase5-dev libqt5core5a libqt5gui5 libqt5widgets5 \
    libglib2.0-dev libgstreamer-plugins-base1.0-dev ninja-build \
    libboost-program-options-dev libdrm-dev libexif-dev

RUN pip3 install --upgrade --no-cache-dir --break-system-packages pip && \
    python3 -m pip install --no-cache-dir --break-system-packages \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir --break-system-packages \
    opencv-python imutils python-dotenv tritonclient[all]

RUN git clone https://github.com/raspberrypi/libcamera.git && \
    cd libcamera && \
    meson setup build \
    --buildtype=release \
    -D pipelines=rpi/vc4,rpi/pisp \
    -D ipas=rpi/vc4,rpi/pisp \
    -D v4l2=true \
    -D gstreamer=enabled \
    -D test=false \
    -D lc-compliance=disabled \
    -D cam=disabled \
    -D qcam=disabled \
    -D documentation=disabled \
    -D pycamera=enabled && \
    ninja -C build && \
    ninja -C build install

RUN git clone https://github.com/raspberrypi/rpicam-apps.git && \
    cd rpicam-apps && \
    meson setup build \
    -D enable_libav=false \
    -D enable_drm=true \
    -D enable_egl=false \
    -D enable_qt=false \
    -D enable_opencv=false \
    -D enable_tflite=false && \
    meson compile -C build && \
    meson install -C build

ENV LD_LIBRARY_PATH=/usr/local/lib/aarch64-linux-gnu
COPY main.py stream.sh entrypoint.sh ./
CMD ["/bin/bash", "entrypoint.sh"]