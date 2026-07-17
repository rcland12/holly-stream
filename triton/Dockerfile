FROM nvcr.io/nvidia/l4t-base:r32.7.1

ARG OPENCV_VERSION="4.5.1"
ARG PYTORCH_VERSION="1.8.0"
ARG TORCHVISION_VERSION="0.9.0"
ARG TORCHVISION_BRANCH="release/0.9"
ARG TRITON_VERSION="2.19.0"
ARG JETPACK_VERSION="4.6.1"
ARG BUILD_VERSION=${TORCHVISION_VERSION}

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        autoconf \
        automake \
        build-essential \
        ca-certificates \
        cmake \
        curl \
        ffmpeg \
        git \
        libavcodec-dev \
        libavformat-dev \
        libb64-dev \
        libboost-dev \
        libcurl4-openssl-dev \
        libdc1394-22-dev \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libgtk2.0-dev \
        libjpeg-dev \
        libopenblas-base \
        libopenmpi-dev \
        libpng-dev \
        libre2-dev \
        libssl-dev \
        libswscale-dev \
        libtbb-dev \
        libtbb2 \
        libtiff-dev \
        libtool \
        libv4l-dev \
        patchelf \
        pkg-config \
        python-dev \
        python-numpy \
        python2.7-dev \
        python3-numpy \
        python3-pip \
        python3.6-dev \
        qv4l2 \
        rapidjson-dev \
        software-properties-common \
        unzip \
        v4l-utils \
        v4l2ucp \
        wget \
        zlib1g-dev

WORKDIR /app/build
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mv opencv-${OPENCV_VERSION} opencv && \
    mv opencv_contrib-${OPENCV_VERSION} opencv_contrib && \
    rm opencv.zip && \
    rm opencv_contrib.zip && \
    mkdir ./opencv/build && \
    cd ./opencv/build && \
    cmake \
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
        -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && \
    make install && \
    make clean && \
    apt-get update && \
    cd ../.. && \
    rm -rf opencv opencv_contrib

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install setuptools && \
    python3 -m pip install --no-cache-dir \
        attrdict==2.0.1 \
        Cython==3.0.12 \
        future==1.0.0 \
        grpcio-tools==1.48.2 \
        imutils==0.5.4 \
        nanocamera==0.1.5 \
        numpy==1.19.4 \
        pillow==8.4.0 \
        python-dotenv==0.20.0 && \
    wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl \
        -O torch-${PYTORCH_VERSION}-cp36-cp36m-linux_aarch64.whl && \
    python3 -m pip install --no-cache-dir torch-${PYTORCH_VERSION}-cp36-cp36m-linux_aarch64.whl && \
    rm torch-${PYTORCH_VERSION}-cp36-cp36m-linux_aarch64.whl && \
    git clone --branch ${TORCHVISION_BRANCH} https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    python3 setup.py install --user && \
    cd .. && \
    rm -rf torchvision

WORKDIR /opt/tritonserver
RUN wget https://github.com/triton-inference-server/server/releases/download/v${TRITON_VERSION}/tritonserver${TRITON_VERSION}-jetpack${JETPACK_VERSION}.tgz && \
    tar -xzf tritonserver${TRITON_VERSION}-jetpack${JETPACK_VERSION}.tgz -C /opt/tritonserver && \
    rm tritonserver${TRITON_VERSION}-jetpack${JETPACK_VERSION}.tgz && \
    rm -rf /opt/tritonserver/backends/identity/ \
        /opt/tritonserver/backends/onnxruntime/ \
        /opt/tritonserver/backends/pytorch/ \
        /opt/tritonserver/backends/tensorflow1/ \
        /opt/tritonserver/backends/tensorflow2/

ENV LD_LIBRARY_PATH=/opt/tritonserver/backends/python:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/opt/tritonserver/backends/tensorrt:$LD_LIBRARY_PATH
ENV PATH=$PATH:/opt/tritonserver/bin

CMD ["/bin/sh", "-c", "tritonserver --model-repository=/models --cuda-memory-pool-byte-size=0:268435456 --pinned-memory-pool-byte-size=134217728 --response-cache-byte-size=0 --allow-metrics=false"]
