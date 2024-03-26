FROM debian:bullseye-slim
WORKDIR /root/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 python3-pip ffmpeg \
    && python3 -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && python3 -m pip install --no-cache-dir opencv-python \
    imutils python-dotenv tritonclient[all]

COPY main.py stream.sh entrypoint.sh ./
CMD ["/bin/bash", "entrypoint.sh"]
