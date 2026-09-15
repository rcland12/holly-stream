import ast
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import onnx
import tensorrt as trt
import torch

log = logging.getLogger(__name__)

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
}


class Engine:
    """
    A TensorRT engine for a YOLO detection model exported to ONNX by Ultralytics.

    The engine is built from the ONNX file on first use (FP16 when the GPU supports it) and cached next to
    it, keyed by the model contents, TensorRT version and GPU, so a rebuild only happens when one changes.
    Inputs and outputs live on the GPU as torch tensors, so inference never copies through the CPU.

    Expects one input [1, 3, H, W] and one output [1, N, 6] of [x1, y1, x2, y2, score, class] rows.
    """

    def __init__(self, onnx_path: Path, cache_dir: Path, fp16: bool = True):
        self.device = torch.device("cuda")
        self.names = self._read_names(onnx_path)

        engine_bytes = self._load_or_build(onnx_path, cache_dir, fp16)
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        self._context = self._engine.create_execution_context()
        # TensorRT adds extra synchronization when it runs on the default CUDA stream
        self._stream = torch.cuda.Stream()

        self._inputs: List[Tuple[str, torch.Tensor]] = []
        self._outputs: List[Tuple[str, torch.Tensor]] = []
        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            shape = tuple(self._engine.get_tensor_shape(name))
            tensor = torch.empty(shape, dtype=_TRT_TO_TORCH[self._engine.get_tensor_dtype(name)], device=self.device)
            self._context.set_tensor_address(name, tensor.data_ptr())
            mode = self._engine.get_tensor_mode(name)
            (self._inputs if mode == trt.TensorIOMode.INPUT else self._outputs).append((name, tensor))

        if len(self._inputs) != 1 or len(self._outputs) != 1:
            raise ValueError(f"Expected one input and one output, got {len(self._inputs)} and {len(self._outputs)}")
        self.input_shape: Tuple[int, ...] = tuple(self._inputs[0][1].shape)
        self.input_dtype = self._inputs[0][1].dtype
        output_shape = tuple(self._outputs[0][1].shape)
        if len(self.input_shape) != 4 or output_shape[-1] != 6:
            raise ValueError(f"Unsupported model shapes: input {self.input_shape}, output {output_shape}")

    @property
    def input_size(self) -> Tuple[int, int]:
        """(height, width) of the model input."""
        return self.input_shape[2], self.input_shape[3]

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Run inference.

        Args:
            images (torch.Tensor): [1, 3, H, W] float tensor on the GPU, RGB scaled to 0-1.

        Returns:
            torch.Tensor: [N, 6] detections on the GPU, valid until the next call.
        """
        # The input was prepared on the caller's stream, so wait for that work before reading it
        self._stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._stream):
            self._inputs[0][1].copy_(images)
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT inference failed")
        self._stream.synchronize()
        return self._outputs[0][1][0].float()

    @staticmethod
    def _read_names(onnx_path: Path) -> Dict[int, str]:
        # Ultralytics stores the class names in the ONNX metadata as a Python dict literal
        model = onnx.load(str(onnx_path), load_external_data=False)
        metadata = {prop.key: prop.value for prop in model.metadata_props}
        if "names" not in metadata:
            raise ValueError(f"{onnx_path} has no 'names' metadata; export it with Ultralytics")
        return {int(k): v for k, v in ast.literal_eval(metadata["names"]).items()}

    @staticmethod
    def _load_or_build(onnx_path: Path, cache_dir: Path, fp16: bool) -> bytes:
        onnx_bytes = onnx_path.read_bytes()
        key = json.dumps(
            {
                "onnx": hashlib.sha256(onnx_bytes).hexdigest(),
                "tensorrt": trt.__version__,
                "gpu": torch.cuda.get_device_name(),
                "fp16": fp16,
            },
            sort_keys=True,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        engine_path = cache_dir / f"{onnx_path.stem}-{hashlib.sha256(key.encode()).hexdigest()[:12]}.engine"
        if engine_path.exists():
            log.info("Loading cached TensorRT engine %s", engine_path)
            return engine_path.read_bytes()

        log.info("Building TensorRT engine for %s (fp16=%s), this takes a few minutes the first time", onnx_path, fp16)
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(0)
        parser = trt.OnnxParser(network, logger)
        if not parser.parse(onnx_bytes):
            errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
            raise RuntimeError(f"Could not parse {onnx_path}:\n{errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT could not build an engine for {onnx_path}")
        engine_bytes = bytes(serialized)
        tmp = engine_path.with_suffix(".tmp")
        tmp.write_bytes(engine_bytes)
        tmp.rename(engine_path)
        log.info("Saved TensorRT engine %s", engine_path)
        return engine_bytes
