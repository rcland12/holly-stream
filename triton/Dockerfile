FROM nvcr.io/nvidia/tritonserver:23.10-py3

RUN apt-get update -y && \
    apt-get install -y \
    libgl1 && \
    pip3 install \
    opencv-python \
    torch

EXPOSE 8000-8002
CMD ["tritonserver", "--model-repository=/models"]