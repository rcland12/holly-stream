FROM nvcr.io/nvidia/tritonserver:24.01-py3

# The python backend in 24.01 is built against numpy 1.x; with numpy 2 the preprocess
# output tensor arrives empty ("onnx runtime error 2: not enough space: expected 4915200, got 0").
# NMS runs inside the ONNX model, so torch/torchvision are no longer needed.
RUN pip3 install --no-cache-dir \
        numpy==1.26.4 \
        opencv-python-headless==4.10.0.84

CMD ["sh", "-c", "tritonserver --model-repository=${MODEL_REPOSITORY:-/models} --allow-metrics=false"]
