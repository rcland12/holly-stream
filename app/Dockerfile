FROM debian:bullseye-slim
WORKDIR /root/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.9 python3-pip python3-dev ffmpeg build-essential && \
    pip3 install --upgrade --no-cache-dir pip && \
    python3 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    python3 -m pip install --no-cache-dir opencv-python \
    imutils python-dotenv tritonclient[all]

COPY main.py utilities.py stream.sh entrypoint.sh ./
CMD ["/bin/bash", "entrypoint.sh"]