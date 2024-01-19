FROM nvcr.io/nvidia/l4t-base:r32.7.1
ARG APPDIR="/root/app"
ARG BUILDDIR="/root/build"

# OpenCV (4.5.1) installation from source
# The default OpenCV version on the Jetson Nano (Jetpack 4.6.1) is 4.1.1
# The default version 4.1.1 causes dependency issues
WORKDIR ${BUILDDIR}
ARG OPENCV_VERSION="4.5.1"

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    build-essential cmake git wget curl unzip ca-certificates ffmpeg \
    libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    python2.7-dev python3.6-dev python-dev python-numpy python3-numpy python3-pip \
    libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev \
    libv4l-dev v4l-utils qv4l2 v4l2ucp libopenblas-base libopenmpi-dev \
    software-properties-common autoconf automake rapidjson-dev patchelf zlib1g-dev \
    libb64-dev libre2-dev libssl-dev libtool libboost-dev libcurl4-openssl-dev

RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip \
    && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip \
    && unzip opencv.zip && unzip opencv_contrib.zip \
    && mv opencv-${OPENCV_VERSION} opencv && mv opencv_contrib-${OPENCV_VERSION} opencv_contrib \
    && rm opencv.zip && rm opencv_contrib.zip \
    && mkdir ./opencv/build && cd ./opencv/build \
    && cmake \
    -D WITH_CUDA=ON \
    -D WITH_CUDNN=ON \
    -D CUDA_ARCH_BIN=5.3 \
    -D CUDA_ARCH_PTX="" \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -D WITH_GSTREAMER=ON \
    -D WITH_LIBV4L=ON \
    -D BUILD_opencv_python2=ON \
    -D BUILD_opencv_python3=ON \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local .. \
    && make -j$(nproc) && make install && make clean \
    && apt-get update && cd ../.. && rm -rf opencv opencv_contrib

# PyTorch (1.8.0), Torchvision (0.9.0), and Triton Server (2.19.0) installation via pip wheel
ARG PYTORCH_VERSION="1.8.0"
ARG TORCHVISION_VERSION="0.9.0"
ARG TORCHVISION_BRANCH="release/0.9"
ARG TRITON_VERSION="2.19.0"
ARG JETPACK_VERSION="4.6.1"

RUN python3 -m pip install --upgrade --no-cache-dir pip \
    && python3 -m pip install --no-cache-dir \
    Cython setuptools==58.3.0 grpcio-tools numpy==1.19.4 future attrdict \
    pillow==8.4.0 nanocamera python-dotenv \
    && python3 -m pip install imutils \
    && wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl \
    -O torch-${PYTORCH_VERSION}-cp36-cp36m-linux_aarch64.whl \
    && python3 -m pip install --no-cache-dir torch-${PYTORCH_VERSION}-cp36-cp36m-linux_aarch64.whl \
    && rm torch-${PYTORCH_VERSION}-cp36-cp36m-linux_aarch64.whl \
    && git clone --branch ${TORCHVISION_BRANCH} https://github.com/pytorch/vision torchvision \
    && cd torchvision \
    && export BUILD_VERSION=${TORCHVISION_VERSION} \
    && python3 setup.py install --user \
    && cd .. && rm -rf torchvision \
    && mkdir /tritonserver && cd /tritonserver \
    && wget https://github.com/triton-inference-server/server/releases/download/v${TRITON_VERSION}/tritonserver${TRITON_VERSION}-jetpack${JETPACK_VERSION}.tgz \
    && tar -xzf tritonserver${TRITON_VERSION}-jetpack${JETPACK_VERSION}.tgz -C /tritonserver/ \
    && rm tritonserver${TRITON_VERSION}-jetpack${JETPACK_VERSION}.tgz \
    && rm -rf /tritonserver/backends/identity/ \
    && rm -rf /tritonserver/backends/onnxruntime/ \
    && rm -rf /tritonserver/backends/pytorch/ \
    && rm -rf /tritonserver/backends/tensorflow1/ \
    && rm -rf /tritonserver/backends/tensorflow2/

ENV LD_LIBRARY_PATH=/tritonserver/backends/python:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/tritonserver/backends/tensorrt:$LD_LIBRARY_PATH
ENV PATH=$PATH:/tritonserver/bin

CMD ["tritonserver", "--backend-directory", "/tritonserver/backends", "--model-repository", "/root/app/triton"]