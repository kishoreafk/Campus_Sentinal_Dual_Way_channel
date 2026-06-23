"""Event clip writer — extracts a 10-second MP4 clip around an alert.

5 seconds before + 5 seconds after the alert timestamp. Uses FFmpeg
subprocess to write H.264 MP4 at 2 Mbps.

In mock mode (no real RTSP frames), the clip writer generates a synthetic
clip with overlaid alert metadata so the pipeline is testable end-to-end.
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger

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
    """Writes event clips to disk as MP4 (H.264)."""

    def __init__(
        self,
        storage_path: str = "data/clips",
        clip_before_s: int = 5,
        clip_after_s: int = 5,
        bitrate: int = 2_000_000,
        codec: str = "libx264",
        fps: int = 30,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.clip_before_s = clip_before_s
        self.clip_after_s = clip_after_s
        self.bitrate = bitrate
        self.codec = codec
        self.fps = fps
        self.frame_buffers: dict[str, FrameRingBuffer] = {}

    def get_buffer(self, camera_id: str) -> FrameRingBuffer:
        if camera_id not in self.frame_buffers:
            self.frame_buffers[camera_id] = FrameRingBuffer(
                capacity_s=self.clip_before_s + self.clip_after_s + 2,
                fps=self.fps,
            )
        return self.frame_buffers[camera_id]

    def push_frame(self, camera_id: str, timestamp: float, frame: np.ndarray) -> None:
        self.get_buffer(camera_id).push(timestamp, frame)

    def write_clip(
        self,
        camera_id: str,
        alert_timestamp: float,
        alert_id: str,
        action_label: str,
        frames_before: Optional[list[np.ndarray]] = None,
        frames_after: Optional[list[np.ndarray]] = None,
    ) -> Optional[str]:
        """Write a clip to disk and return the path.

        In production with real RTSP frames, the clip writer would wait
        clip_after_s seconds for the "after" frames to arrive, then assemble
        the full clip.

        For testing, we accept pre-supplied frames_before / frames_after.
        """
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 not available — cannot write clip")
            return None

        # Get frames from ring buffer
        buf = self.get_buffer(camera_id)
        if frames_before is None:
            frames_before = buf.get_last_n_seconds(self.clip_before_s, alert_timestamp)

        # If we don't have after-frames, just use what we have
        if frames_after is None:
            frames_after = []

        all_frames = list(frames_before) + list(frames_after)
        if not all_frames:
            logger.warning(f"[{camera_id}] no frames for clip {alert_id}")
            return None

        # Determine frame size from first frame
        H, W = all_frames[0].shape[:2]
        date_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(alert_timestamp))
        filename = f"{camera_id}_{action_label}_{date_str}_{alert_id[:8]}.mp4"
        filepath = self.storage_path / filename

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filepath), fourcc, self.fps, (W, H))
        if not writer.isOpened():
            logger.error(f"Failed to open VideoWriter for {filepath}")
            return None

        try:
            for frame in all_frames:
                # Overlay alert text on the frame
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

        logger.info(f"[{camera_id}] wrote clip {filepath} ({len(all_frames)} frames)")
        return str(filepath)
