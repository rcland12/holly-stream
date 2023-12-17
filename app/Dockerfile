FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3
ARG APPDIR="/root/app"
ARG BUILDDIR="/root/build"

RUN sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf" && \
    ldconfig && \
    apt-get update && \
    apt-get install -y \
    git cmake curl unzip pkg-config libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk2.0-dev libcanberra-gtk* \
    python3-dev python3-numpy python3-pip \
    libxvidcore-dev libx264-dev libgtk-3-dev \
    libtbb2 libtbb-dev libdc1394-22-dev \
    libv4l-dev v4l-utils \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    libavresample-dev libvorbis-dev libxine2-dev \
    libfaac-dev libmp3lame-dev libtheora-dev \
    libopencore-amrnb-dev libopencore-amrwb-dev \
    libopenblas-dev libatlas-base-dev libblas-dev libopenblas-base \
    liblapack-dev libeigen3-dev gfortran \
    libhdf5-dev protobuf-compiler \
    libprotobuf-dev libgoogle-glog-dev libgflags-dev \
    libopenmpi-dev libomp-dev ffmpeg \
    libjpeg-dev zlib1g-dev libpython3-dev libfreetype6-dev


# PyTorch (1.8.0) and Torchvision (0.9.0) installation via pip wheel
WORKDIR ${BUILDDIR}
ARG PYTORCH_VERSION="1.8.0"
ARG TORCHVISION_VERSION="0.9.0"
ARG TORCHVISION_BRANCH="release/0.9"

# PyTorch
RUN wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl \
    -O torch-${PYTORCH_VERSION}-cp36-cp36m-linux_aarch64.whl && \
    pip3 install Cython torch-${PYTORCH_VERSION}-cp36-cp36m-linux_aarch64.whl && \
    rm torch-${PYTORCH_VERSION}-cp36-cp36m-linux_aarch64.whl

# Torchvision
RUN git clone --branch ${TORCHVISION_BRANCH} https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    export BUILD_VERSION=${TORCHVISION_VERSION} && \
    python3 setup.py install --user && \
    cd .. && rm -rf torchvision


# Nvidia Triton Inference Server Client installation
ARG TRITON_VERSION="2.19.0"
RUN wget https://github.com/triton-inference-server/server/releases/download/v${TRITON_VERSION}/tritonserver${TRITON_VERSION}-jetpack4.6.1.tgz && \
    mkdir tritonserver && \
    tar -xzf tritonserver${TRITON_VERSION}-jetpack4.6.1.tgz -C ${BUILDDIR}/tritonserver/

RUN pip3 install --upgrade pip && \
    pip3 install --upgrade grpcio-tools numpy==1.19.4 future attrdict nanocamera imutils python-dotenv && \
    pip3 install --upgrade ${BUILDDIR}/tritonserver/clients/python/tritonclient-${TRITON_VERSION}-py3-none-any.whl[all] && \
    rm tritonserver${TRITON_VERSION}-jetpack4.6.1.tgz

WORKDIR ${APPDIR}
COPY main.py utilities.py ./
COPY images/*.png ./images/
CMD ["python3", "main.py"]
