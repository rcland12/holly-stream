FROM nvcr.io/nvidia/tritonserver:24.01-py3

RUN apt-get update -y && \
    apt-get install -y python3-opencv && \
    pip3 install --no-cache-dir --upgrade pip && \
    && pip3 install --no-cache-dir --upgrade wheel setuptools && \
    && pip3 install --no-cache-dir torch torchvision python-dotenv opencv-python

CMD ["sh", "-c", "tritonserver --model-repository=${MODEL_REPOSITORY}"]