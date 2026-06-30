"""Event clip writer — extracts a 10-second MP4 clip around an alert.

5 seconds before + 5 seconds after the alert timestamp. Uses FFmpeg
subprocess to write H.264 MP4 at 2 Mbps with optional GPU encoding.

Clip writing is offloaded to a thread pool to avoid blocking the pipeline.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger
from src.common.metrics import CLIP_WRITE_LATENCY

logger = get_logger()


class FrameRingBuffer:
    """Per-camera ring buffer of (timestamp, frame) tuples.

    Used to fetch the 5 seconds of frames BEFORE an alert for clip generation.
    """

    def __init__(self, capacity_s: float = 10.0, fps: int = 30):
        self.capacity_s = capacity_s
        self.fps = fps
        self.maxlen = int(capacity_s * fps)
        self._buf: Deque[Tuple[float, np.ndarray]] = deque(maxlen=self.maxlen)
        self._lock = threading.Lock()

    def push(self, timestamp: float, frame: np.ndarray) -> None:
        with self._lock:
            self._buf.append((timestamp, frame.copy()))

    def get_range(self, t_start: float, t_end: float) -> list[np.ndarray]:
        with self._lock:
            return [f for (t, f) in self._buf if t_start <= t <= t_end]

    def get_last_n_seconds(self, n_seconds: float, now: float) -> list[np.ndarray]:
        t_start = now - n_seconds
        return self.get_range(t_start, now)


class ClipWriter:
    """Writes event clips to disk as MP4 (H.264), optionally using GPU encoding.

    Writing is submitted to a thread pool so the caller is never blocked.
    """

    def __init__(
        self,
        storage_path: str = "data/clips",
        clip_before_s: int = 5,
        clip_after_s: int = 5,
        bitrate: int = 2_000_000,
        codec: str = "libx264",
        codec_gpu: str = "h264_nvenc",
        fps: int = 30,
        use_gpu: bool = False,
        num_workers: int = 2,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.clip_before_s = clip_before_s
        self.clip_after_s = clip_after_s
        self.bitrate = bitrate
        self.codec = codec_gpu if (use_gpu and _nvenc_available()) else codec
        self.fps = fps
        self.frame_buffers: dict[str, FrameRingBuffer] = {}
        self._executor = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="clip")

    def get_buffer(self, camera_id: str) -> FrameRingBuffer:
        if camera_id not in self.frame_buffers:
            self.frame_buffers[camera_id] = FrameRingBuffer(
                capacity_s=self.clip_before_s + self.clip_after_s + 2,
                fps=self.fps,
            )
        return self.frame_buffers[camera_id]

    def push_frame(self, camera_id: str, timestamp: float, frame: np.ndarray) -> None:
        self.get_buffer(camera_id).push(timestamp, frame)

    def write_clip_async(
        self,
        camera_id: str,
        alert_timestamp: float,
        alert_id: str,
        action_label: str,
    ) -> None:
        """Submit clip write to thread pool (non-blocking)."""
        self._executor.submit(self._write_clip_sync, camera_id, alert_timestamp, alert_id, action_label)

    def _write_clip_sync(
        self,
        camera_id: str,
        alert_timestamp: float,
        alert_id: str,
        action_label: str,
    ) -> Optional[str]:
        """Synchronous clip write — runs in thread pool."""
        import time as _time
        t0 = _time.time()

        try:
            import cv2
        except ImportError:
            logger.warning("cv2 not available — cannot write clip")
            return None

        buf = self.get_buffer(camera_id)
        frames_before = buf.get_last_n_seconds(self.clip_before_s, alert_timestamp)
        frames_after = buf.get_range(alert_timestamp, alert_timestamp + self.clip_after_s)

        all_frames = list(frames_before) + list(frames_after)
        if not all_frames:
            logger.warning(f"[{camera_id}] no frames for clip {alert_id}")
            return None

        H, W = all_frames[0].shape[:2]
        date_str = _time.strftime("%Y%m%d_%H%M%S", _time.localtime(alert_timestamp))
        filename = f"{camera_id}_{action_label}_{date_str}_{alert_id[:8]}.mp4"
        filepath = self.storage_path / filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (W, H))
        if not writer.isOpened():
            logger.error(f"Failed to open VideoWriter for {filepath}")
            return None

        try:
            for frame in all_frames:
                annotated = frame.copy()
                try:
                    cv2.putText(
                        annotated,
                        f"{action_label.upper()} | {date_str}",
                        org=(10, 30),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8,
                        color=(0, 0, 255),
                        thickness=2,
                    )
                except Exception:
                    pass
                writer.write(annotated)
        finally:
            writer.release()

        CLIP_WRITE_LATENCY.observe(_time.time() - t0)
        logger.info(f"[{camera_id}] wrote clip {filepath} ({len(all_frames)} frames)")
        return str(filepath)

    # Keep backward-compat sync method
    write_clip = _write_clip_sync

    def close(self) -> None:
        self._executor.shutdown(wait=False)


def _nvenc_available() -> bool:
    """Check if NVENC encoder is available via ffmpeg."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-encoders", "-hide_banner"],
            capture_output=True, text=True, timeout=5,
        )
        return "h264_nvenc" in result.stdout
    except Exception:
        return False
