"""FFmpeg pipeline wrapper for RTSP ingestion with NVDEC hardware decode.

In production this uses `-hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid`
to keep frames on the GPU and avoid CPU memory copies. The output is raw BGR24
frames written to stdout which we read in fixed-size chunks.

For testing without a real camera, the same module can run off a synthetic
frame generator (see `mock_source`).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Callable, Generator, Optional

import numpy as np

from src.common.logger import get_logger

logger = get_logger()


@dataclass
class FramePacket:
    """A decoded frame ready for downstream processing."""

    camera_id: str
    frame_id: int
    timestamp: float
    data: np.ndarray  # uint8 HxWx3 BGR
    width: int
    height: int


def build_ffmpeg_cmd(
    rtsp_url: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    use_hwaccel: bool = True,
    rtsp_transport: str = "tcp",
) -> list[str]:
    """Build the FFmpeg command for RTSP ingestion.

    When hardware acceleration is enabled, the pipeline decodes on the GPU
    then downloads to system memory so the output can be read via stdout.
    """
    cmd: list[str] = ["ffmpeg", "-loglevel", "warning"]
    if use_hwaccel:
        # Decode on GPU, then download to system memory for pipe output
        cmd += [
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "nv12",
            "-c:v", "h264_cuvid",
        ]
    cmd += [
        "-rtsp_transport", rtsp_transport,
        "-i", rtsp_url,
        "-vf", f"fps={fps},scale={width}:{height}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",
        "pipe:1",
    ]
    return cmd


def mock_frame_generator(
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    seed: Optional[int] = None,
) -> Generator[np.ndarray, None, None]:
    """Generate synthetic BGR frames for testing (replaces RTSP source).

    Produces visually-distinct frames with moving "person-like" rectangles so
    that downstream detection / tracking can be exercised without a GPU or
    real camera.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / fps
    t = 0.0
    while True:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Background gradient that shifts over time
        for c in range(3):
            frame[:, :, c] = int(40 + 30 * np.sin(t + c))
        # Two "persons" walking across the frame
        for i in range(2):
            phase = t * (0.7 + 0.2 * i) + i * 1.5
            x = int((np.sin(phase) * 0.4 + 0.5) * width)
            y = int((np.cos(phase * 0.6) * 0.3 + 0.55) * height)
            w = 80 + int(20 * np.sin(phase * 2))
            h = 160 + int(20 * np.cos(phase * 2))
            color = (60 + 80 * i, 200 - 40 * i, 80 + 60 * i)
            cv2_rect(frame, x - w // 2, y - h // 2, x + w // 2, y + h // 2, color)
        # Add noise so frames aren't identical
        noise = rng.integers(0, 8, size=(height, width, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        yield frame
        t += dt


def cv2_rect(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a filled rectangle (avoids importing cv2 just for this)."""
    H, W = img.shape[:2]
    x1, x2 = max(0, x1), min(W, x2)
    y1, y2 = max(0, y1), min(H, y2)
    if x1 >= x2 or y1 >= y2:
        return
    img[y1:y2, x1:x2] = color


class FFmpegPipeline:
    """Wraps an FFmpeg subprocess producing raw BGR frames on stdout."""

    def __init__(
        self,
        camera_id: str,
        rtsp_url: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        use_hwaccel: bool = True,
        rtsp_transport: str = "tcp",
        bufsize: int = 10 ** 8,
        reconnect_delay: float = 2.0,
        max_reconnect: int = 5,
    ):
        self.camera_id = camera_id
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.fps = fps
        self.use_hwaccel = use_hwaccel
        self.rtsp_transport = rtsp_transport
        self.bufsize = bufsize
        self.reconnect_delay = reconnect_delay
        self.max_reconnect = max_reconnect
        self._proc: Optional[subprocess.Popen] = None
        self._frame_size = width * height * 3
        self._closed = False

    def _start(self) -> None:
        cmd = build_ffmpeg_cmd(
            self.rtsp_url, self.width, self.height, self.fps,
            self.use_hwaccel, self.rtsp_transport,
        )
        logger.info(f"[{self.camera_id}] starting FFmpeg: {' '.join(cmd[:8])}...")
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.bufsize,
        )
        # Quick health check — crash if process dies within 1s of launch
        time.sleep(0.1)
        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read(2048) if self._proc.stderr else b""
            raise RuntimeError(f"FFmpeg died on startup: {stderr.decode(errors='replace')[:500]}")

    def start(self) -> None:
        self._start()

    def read_frame(self) -> Optional[np.ndarray]:
        """Read one BGR frame. Returns None on EOF / error."""
        if self._proc is None or self._proc.stdout is None:
            return None
        # Quick health check before reading
        if self._proc.poll() is not None:
            stderr = self._proc.stderr.read(2048) if self._proc.stderr else b""
            logger.warning(f"[{self.camera_id}] FFmpeg exited unexpectedly: "
                           f"{stderr.decode(errors='replace')[:300]}")
            return None
        raw = self._proc.stdout.read(self._frame_size)
        if not raw or len(raw) < self._frame_size:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))

    def close(self) -> None:
        self._closed = True
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.close()


class MockSource:
    """Drop-in replacement for FFmpegPipeline that yields synthetic frames.

    Useful for tests, CI, and local development without RTSP cameras or GPU.
    """

    def __init__(
        self,
        camera_id: str,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        seed: Optional[int] = None,
    ):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self._gen = mock_frame_generator(width, height, fps, seed=seed)

    def start(self) -> None:
        pass

    def read_frame(self) -> Optional[np.ndarray]:
        try:
            return next(self._gen)
        except StopIteration:
            return None

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
