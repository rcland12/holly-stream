FROM ultralytics/ultralytics:latest-cpu
WORKDIR /home/holly
COPY main.py predict.py ./

RUN apt update && \
    apt install -y ffmpeg && \
    pip install supervision py-cpuinfo

CMD ["python3", "main.py", "--ip", "127.0.0.1", "--port", "1935", "--application", "live", "--stream_key", "stream", "--capture_index", "0", "--model", "yolov8n.pt"]
