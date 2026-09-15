# Holly Stream for Jetson Nano (JetPack 4.6.x / L4T R32.7, DeepStream 6.0.1)
#
# Build on the Jetson itself (docker-compose build); the nvidia runtime must be
# the default runtime so CUDA/TensorRT headers are mounted during the build.

# The samples image ships the DeepStream headers needed to compile.
FROM nvcr.io/nvidia/deepstream-l4t:6.0.1-samples AS build

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY src/ .
RUN make

FROM nvcr.io/nvidia/deepstream-l4t:6.0.1-base

COPY --from=build /src/holly-stream /src/libnvdsinfer_custom_impl_yolo.so /opt/holly-stream/
COPY config/ /opt/holly-stream/config/
COPY entrypoint.sh collect_data.sh /opt/holly-stream/

ENV GST_DEBUG=1
ENTRYPOINT ["/opt/holly-stream/entrypoint.sh"]
