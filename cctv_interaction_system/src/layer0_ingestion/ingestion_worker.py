"""Video ingestion worker.

Reads frames from an RTSP source (or mock), applies frame decimation
(detection stride), and pushes them onto the detection / tracking queues.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Optional

from config.settings import get_settings
from src.common.logger import get_logger
from src.common.metrics import FRAMES_DROPPED, FRAMES_INGESTED, INGESTION_LATENCY

from .ffmpeg_pipeline import FFmpegPipeline, MockSource

logger = get_logger()


class IngestionWorker(threading.Thread):
    """One thread per camera. Reads frames, applies decimation, enqueues."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        detection_queue: "queue.Queue",
        tracking_queue: "queue.Queue",
        detection_stride: int = 5,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        use_mock: bool = False,
        use_hwaccel: bool = True,
        max_queue_size: int = 256,
        reconnect_delay: float = 2.0,
        max_reconnect: int = 5,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or f"ingest-{camera_id}", daemon=True)
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.detection_queue = detection_queue
        self.tracking_queue = tracking_queue
        self.detection_stride = max(1, detection_stride)
        self.width = width
        self.height = height
        self.fps = fps
        self.use_mock = use_mock
        self.use_hwaccel = use_hwaccel
        self.max_queue_size = max_queue_size
        self.reconnect_delay = reconnect_delay
        self.max_reconnect = max_reconnect
        self._stop = threading.Event()
        self.frame_counter = 0

    def stop(self) -> None:
        self._stop.set()

    def _make_source(self):
        if self.use_mock:
            return MockSource(self.camera_id, self.width, self.height, self.fps,
                              seed=hash(self.camera_id) & 0xFFFFFFFF)
        return FFmpegPipeline(
            self.camera_id, self.rtsp_url, self.width, self.height, self.fps,
            use_hwaccel=self.use_hwaccel, reconnect_delay=self.reconnect_delay,
            max_reconnect=self.max_reconnect,
        )

    def _enqueue(self, q: "queue.Queue", packet: dict) -> bool:
        try:
            q.put_nowait(packet)
            return True
        except queue.Full:
            # Drop oldest by popping one then putting
            try:
                q.get_nowait()
                q.put_nowait(packet)
                FRAMES_DROPPED.labels(self.camera_id).inc()
                return True
            except Exception:
                FRAMES_DROPPED.labels(self.camera_id).inc()
                return False

    def run(self) -> None:
        reconnect_attempts = 0
        while not self._stop.is_set():
            source = self._make_source()
            try:
                source.start()
                logger.info(f"[{self.camera_id}] ingestion source started")
                while not self._stop.is_set():
                    t0 = time.time()
                    frame = source.read_frame()
                    if frame is None:
                        logger.warning(f"[{self.camera_id}] source returned None — reconnecting")
                        break
                    self.frame_counter += 1
                    ts = time.time()

                    # Always feed tracking
                    track_pkt = {
                        "camera_id": self.camera_id,
                        "frame_id": self.frame_counter,
                        "timestamp": ts,
                        "data": frame,
                        "width": self.width,
                        "height": self.height,
                        "frame_type": "tracking",
                    }
                    self._enqueue(self.tracking_queue, track_pkt)

                    # Only feed detection every Nth frame
                    if self.frame_counter % self.detection_stride == 0:
                        det_pkt = {
                            "camera_id": self.camera_id,
                            "frame_id": self.frame_counter,
                            "timestamp": ts,
                            "data": frame,
                            "width": self.width,
                            "height": self.height,
                            "frame_type": "detection",
                        }
                        self._enqueue(self.detection_queue, det_pkt)

                    FRAMES_INGESTED.labels(self.camera_id).inc()
                    INGESTION_LATENCY.observe(time.time() - t0)
            except Exception as e:
                logger.error(f"[{self.camera_id}] ingestion error: {e}")
            finally:
                try:
                    source.close()
                except Exception:
                    pass

            if self._stop.is_set():
                break
            reconnect_attempts += 1
            if reconnect_attempts > self.max_reconnect:
                logger.error(f"[{self.camera_id}] max reconnect attempts reached — exiting")
                break
            logger.info(f"[{self.camera_id}] reconnecting in {self.reconnect_delay}s "
                        f"(attempt {reconnect_attempts}/{self.max_reconnect})")
            self._stop.wait(self.reconnect_delay)

        logger.info(f"[{self.camera_id}] ingestion worker stopped")
