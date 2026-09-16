import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


def _bool(value: str) -> bool:
    return value.lower() in ("1", "true", "yes", "on")


@dataclass
class Config:
    """Detector settings, read from environment variables. See server/.env.example."""

    # MediaMTX ingest: cameras publish to it over SRT, workers read from it over RTSP
    ingest_api: str = field(default_factory=lambda: _env("INGEST_API", "http://holly-ingest:9997"))
    ingest_rtsp: str = field(default_factory=lambda: _env("INGEST_RTSP", "rtsp://holly-ingest:8554"))
    # Paths under this prefix are the detector's own annotated copies, not cameras
    annotated_prefix: str = field(default_factory=lambda: _env("ANNOTATED_PREFIX", "annotated/"))

    # Where each annotated stream is published; {name} is the camera's stream name. The ingest serves it to LAN
    # viewers and relays it on to nginx-rtmp (see RELAY_URL in compose).
    output_url: str = field(default_factory=lambda: _env("OUTPUT_URL", "rtmp://holly-ingest:1935/annotated/{name}"))

    model: Path = field(default_factory=lambda: Path("/models") / _env("MODEL", "yolo26m_480x640.onnx"))
    engine_cache: Path = field(default_factory=lambda: Path(_env("ENGINE_CACHE", "/models/engines")))
    confidence: float = field(default_factory=lambda: float(_env("CONFIDENCE_THRESHOLD", "0.35")))
    classes: List[int] = field(default_factory=lambda: list(ast.literal_eval(_env("CLASSES", "[]"))))
    smoothing: float = field(default_factory=lambda: float(_env("BOX_SMOOTHING", "0.5")))

    bitrate_kbps: int = field(default_factory=lambda: int(_env("BITRATE_KBPS", "4500")))
    keyint_seconds: int = field(default_factory=lambda: int(_env("KEYINT_SECONDS", "2")))
    nvenc_preset: str = field(default_factory=lambda: _env("NVENC_PRESET", "p5"))

    # Saves an unannotated frame from every camera each N seconds for building a training dataset; 0 disables
    snapshot_interval: float = field(default_factory=lambda: float(_env("SNAPSHOT_INTERVAL", "0")))
    snapshot_dir: Path = field(default_factory=lambda: Path(_env("SNAPSHOT_DIR", "/data/snapshots")))

    status_file: Path = field(default_factory=lambda: Path(_env("STATUS_FILE", "/data/status.json")))
    stats_interval: float = field(default_factory=lambda: float(_env("STATS_INTERVAL", "10")))
    log_stats: bool = field(default_factory=lambda: _bool(_env("LOG_STATS", "false")))
