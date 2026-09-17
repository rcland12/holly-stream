"""
Holly detector supervisor.

Watches the MediaMTX ingest for cameras that ask for detection (DETECTION=true in the camera's settings, sent as
detect=1 in its SRT stream id). When such a camera connects, a worker process is started for it and publishes
annotated/<stream name>; when the camera goes away, its worker is stopped. Cameras without detection are ignored
here and relayed plain by the ingest. Nothing on the server changes when cameras are added, moved or removed.

Each camera runs in its own process so one misbehaving stream cannot stall or crash the others.
"""

import json
import logging
import multiprocessing as mp
import queue
import signal
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional
from urllib.parse import parse_qs

from holly.config import Config

log = logging.getLogger("holly.supervisor")

POLL_SECONDS = 2.0
RESTART_BACKOFF_SECONDS = (2, 5, 10, 30)


@dataclass
class Camera:
    name: str
    remote: str
    rotate_180: bool


@dataclass
class Worker:
    camera: Camera
    process: mp.Process
    started: float = field(default_factory=time.monotonic)
    stats: Dict = field(default_factory=dict)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(processName)s %(name)s: %(message)s")


def _api(config: Config, path: str) -> Dict:
    with urllib.request.urlopen(f"{config.ingest_api}{path}", timeout=5) as response:
        return json.load(response)


def fetch_cameras(config: Config) -> Dict[str, Camera]:
    """Connected cameras that asked for detection, keyed by stream name."""
    conns = {c["id"]: c for c in _api(config, "/v3/srtconns/list?itemsPerPage=1000")["items"]}
    cameras = {}
    for path in _api(config, "/v3/paths/list?itemsPerPage=1000")["items"]:
        name, source = path["name"], path.get("source") or {}
        if not path.get("ready") or name.startswith(config.annotated_prefix) or not source:
            continue
        conn = conns.get(source.get("id"), {})
        # Cameras pass options in their SRT stream id, e.g. publish:<name>:::rotate=180&detect=1
        options = parse_qs(conn.get("query", ""))
        if options.get("detect", ["0"])[0] != "1":
            continue
        cameras[name] = Camera(
            name=name,
            remote=conn.get("remoteAddr", "").rsplit(":", 1)[0],
            rotate_180=options.get("rotate", ["0"])[0] == "180",
        )
    return cameras


def _exit(signum: int, frame: object) -> None:
    raise SystemExit(0)


def worker_main(config: Config, camera: Camera, reports: mp.Queue) -> None:
    _setup_logging()
    signal.signal(signal.SIGTERM, _exit)
    logging.getLogger("libav").setLevel(logging.CRITICAL)

    import av

    from holly.engine import Engine
    from holly.pipeline import run_camera

    av.logging.set_level(av.logging.ERROR)
    engine = Engine(config.model, config.engine_cache)

    def report(stats: Dict) -> None:
        reports.put((camera.name, stats))
        if config.log_stats:
            log.info("%s: %s", camera.name, stats)

    try:
        run_camera(config, engine, camera.name, camera.rotate_180, report)
    except (av.error.FFmpegError, OSError) as e:
        # The camera or the ingest went away mid-stream; expected, and the supervisor restarts the worker
        log.warning("%s: stream interrupted: %s", camera.name, e)
        raise SystemExit(1)


def build_engine(config: Config) -> None:
    """Build (or load) the TensorRT engine once, so workers starting together don't all build it."""
    _setup_logging()
    from holly.engine import Engine

    engine = Engine(config.model, config.engine_cache)
    log.info(
        "Model %s: input %s, %d classes, drawing %s",
        config.model.name, engine.input_shape, len(engine.names),
        [engine.names[c] for c in config.classes] if config.classes else "all classes",
    )


class Supervisor:
    def __init__(self, config: Config):
        self.config = config
        self.ctx = mp.get_context("spawn")  # CUDA cannot be used in forked processes
        self.reports: mp.Queue = self.ctx.Queue()
        self.workers: Dict[str, Worker] = {}
        self.restart_at: Dict[str, float] = {}
        self.failures: Dict[str, int] = {}
        self.api_ok = True

    def run(self) -> None:
        builder = self.ctx.Process(target=build_engine, args=(self.config,), name="engine-build")
        builder.start()
        builder.join()
        if builder.exitcode != 0:
            raise SystemExit(f"Could not load model {self.config.model}")

        log.info("Watching %s for cameras; publishing to %s", self.config.ingest_api, self.config.output_url)
        while True:
            self._tick()
            time.sleep(POLL_SECONDS)

    def stop(self) -> None:
        for name in list(self.workers):
            self._stop_worker(name, "detector shutting down")

    def _tick(self) -> None:
        try:
            cameras = fetch_cameras(self.config)
            if not self.api_ok:
                log.info("Ingest API reachable again")
            self.api_ok = True
        except Exception as e:
            if self.api_ok:
                log.warning("Cannot reach the ingest API at %s: %s", self.config.ingest_api, e)
            self.api_ok = False
            cameras = None

        self._drain_reports()

        for name, worker in list(self.workers.items()):
            if not worker.process.is_alive():
                code = worker.process.exitcode
                del self.workers[name]
                ran_long = time.monotonic() - worker.started > 60
                self.failures[name] = 0 if ran_long else self.failures.get(name, 0) + 1
                delay = RESTART_BACKOFF_SECONDS[min(self.failures[name], len(RESTART_BACKOFF_SECONDS) - 1)]
                self.restart_at[name] = time.monotonic() + delay
                log.info("Worker for %s exited (code %s)", name, code)

        if cameras is not None:
            for name in list(self.workers):
                if name not in cameras:
                    self._stop_worker(name, "camera went offline")
            for name, camera in cameras.items():
                worker = self.workers.get(name)
                if worker is not None and worker.camera.rotate_180 != camera.rotate_180:
                    self._stop_worker(name, "camera settings changed")
                    worker = None
                if worker is None and time.monotonic() >= self.restart_at.get(name, 0):
                    self._start_worker(camera)
            for name in list(self.restart_at):
                if name not in cameras:
                    self.restart_at.pop(name, None)
                    self.failures.pop(name, None)

        self._write_status(cameras)

    def _start_worker(self, camera: Camera) -> None:
        process = self.ctx.Process(
            target=worker_main, args=(self.config, camera, self.reports), name=camera.name, daemon=True
        )
        process.start()
        self.workers[camera.name] = Worker(camera, process)
        log.info("Camera online: %s from %s (rotate_180=%s)", camera.name, camera.remote, camera.rotate_180)

    def _stop_worker(self, name: str, reason: str) -> None:
        worker = self.workers.pop(name, None)
        if worker is None:
            return
        log.info("Stopping worker for %s: %s", name, reason)
        worker.process.terminate()
        worker.process.join(timeout=10)
        if worker.process.is_alive():
            worker.process.kill()

    def _drain_reports(self) -> None:
        while True:
            try:
                name, stats = self.reports.get_nowait()
            except queue.Empty:
                return
            if name in self.workers:
                self.workers[name].stats = {**stats, "updated": time.time()}

    def _write_status(self, cameras: Optional[Dict[str, Camera]]) -> None:
        status = {
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ingest_reachable": self.api_ok,
            "model": self.config.model.name,
            "cameras": {},
        }
        for name in sorted(set(self.workers) | set(cameras or {})):
            worker = self.workers.get(name)
            camera = worker.camera if worker else (cameras or {}).get(name)
            stats = worker.stats if worker else {}
            fresh = bool(stats) and time.time() - stats.get("updated", 0) < self.config.stats_interval * 3
            status["cameras"][name] = {
                "remote": camera.remote,
                "rotate_180": camera.rotate_180,
                "state": "running" if fresh else ("starting" if worker else "restarting"),
                **{k: v for k, v in stats.items() if k != "updated"},
            }
        tmp = self.config.status_file.with_suffix(".tmp")
        try:
            self.config.status_file.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_text(json.dumps(status, indent=2))
            tmp.replace(self.config.status_file)
        except OSError as e:
            log.debug("Cannot write %s: %s", self.config.status_file, e)


def main() -> None:
    _setup_logging()
    supervisor = Supervisor(Config())

    def shutdown(*_):
        supervisor.stop()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    supervisor.run()


if __name__ == "__main__":
    main()
