"""TensorRT engine wrapper for YOLOv8-Pose.

This module provides two implementations:

  1. `TensorRTDetector`   — production, requires `tensorrt`, `pycuda`, and an
                            exported `.engine` file. Loaded lazily so the
                            package imports cleanly without GPU deps.
  2. `MockDetector`       — CPU-only stub that returns plausible-looking
                            detections for testing. Deterministic per-frame
                            via frame_id seed.

The `Detector` factory picks the right implementation based on settings.
"""

from __future__ import annotations

import abc
import math
import os
import time
from typing import List, Optional

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger
from src.common.schemas import Detection, FrameDetections

logger = get_logger()


# COCO keypoint names (17)
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class BaseDetector(abc.ABC):
    """Abstract detector interface."""

    @abc.abstractmethod
    def detect_batch(self, frames: List[np.ndarray], metas: List[dict]) -> List[FrameDetections]:
        ...

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------
# Mock detector (CPU)
# ---------------------------------------------------------------------
class MockDetector(BaseDetector):
    """Deterministic mock detector.

    Generates 1-3 synthetic person detections per frame using a deterministic
    PRNG seeded by (camera_id, frame_id). Bounding boxes follow smooth
    sinusoidal trajectories so downstream tracking / pair analysis can be
    exercised end-to-end.
    """

    def __init__(self, conf_threshold: float = 0.5, iou_threshold: float = 0.65):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def detect_batch(self, frames: List[np.ndarray], metas: List[dict]) -> List[FrameDetections]:
        results: List[FrameDetections] = []
        for frame, meta in zip(frames, metas):
            H, W = frame.shape[:2]
            camera_id = meta["camera_id"]
            frame_id = meta["frame_id"]
            ts = meta["timestamp"]
            rng = np.random.default_rng(hash((camera_id, frame_id)) & 0xFFFFFFFF)

            n_persons = int(rng.integers(1, 4))
            detections: List[Detection] = []
            for i in range(n_persons):
                # Smooth trajectory
                phase = frame_id * 0.05 + i * 1.5
                cx = W * (0.5 + 0.3 * math.sin(phase))
                cy = H * (0.55 + 0.2 * math.cos(phase * 0.7))
                w = 80 + int(20 * math.sin(phase * 2))
                h = 160 + int(20 * math.cos(phase * 2))
                x1 = max(0.0, cx - w / 2)
                y1 = max(0.0, cy - h / 2)
                x2 = min(W, cx + w / 2)
                y2 = min(H, cy + h / 2)
                conf = float(rng.uniform(0.7, 0.97))
                # Keypoints: simple layout relative to bbox
                kps = self._synthetic_keypoints(x1, y1, x2, y2, rng)
                kp_scores = [float(rng.uniform(0.5, 0.95)) for _ in range(17)]
                detections.append(Detection(
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    confidence=conf,
                    keypoints=kps,
                    keypoint_scores=kp_scores,
                ))
            results.append(FrameDetections(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=ts,
                detections=detections,
            ))
        return results

    @staticmethod
    def _synthetic_keypoints(x1, y1, x2, y2, rng) -> List[List[float]]:
        """Place 17 COCO keypoints within bbox (rough anthropomorphic layout)."""
        cx = (x1 + x2) / 2
        w = x2 - x1
        h = y2 - y1
        # Relative offsets (x, y) within normalised bbox, anchored top-left
        layout = [
            (0.50, 0.10),  # nose
            (0.45, 0.08),  # left_eye
            (0.55, 0.08),  # right_eye
            (0.40, 0.10),  # left_ear
            (0.60, 0.10),  # right_ear
            (0.35, 0.25),  # left_shoulder
            (0.65, 0.25),  # right_shoulder
            (0.30, 0.45),  # left_elbow
            (0.70, 0.45),  # right_elbow
            (0.28, 0.65),  # left_wrist
            (0.72, 0.65),  # right_wrist
            (0.42, 0.55),  # left_hip
            (0.58, 0.55),  # right_hip
            (0.40, 0.78),  # left_knee
            (0.60, 0.78),  # right_knee
            (0.40, 0.95),  # left_ankle
            (0.60, 0.95),  # right_ankle
        ]
        kps: List[List[float]] = []
        for rx, ry in layout:
            kp_x = x1 + rx * w + float(rng.normal(0, 2))
            kp_y = y1 + ry * h + float(rng.normal(0, 2))
            conf = float(rng.uniform(0.55, 0.95))
            kps.append([kp_x, kp_y, conf])
        return kps


# ---------------------------------------------------------------------
# TensorRT detector (production)
# ---------------------------------------------------------------------
class TensorRTDetector(BaseDetector):
    """Real TensorRT detector — requires tensorrt + pycuda + .engine file.

    Lazy import so the package loads cleanly in CPU-only environments.
    """

    def __init__(
        self,
        engine_path: str,
        imgsz: int = 640,
        max_batch: int = 32,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.65,
        device_id: int = 0,
    ):
        import tensorrt as trt  # noqa
        import pycuda.driver as cuda  # noqa
        import pycuda.autoinit  # noqa

        self.trt = trt
        self.cuda = cuda
        self.imgsz = imgsz
        self.max_batch = max_batch
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device_id = device_id

        logger.info(f"Loading TensorRT engine: {engine_path}")
        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(trt_logger).deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Allocate persistent buffers for max_batch
        self.d_input = cuda.mem_alloc(max_batch * 3 * imgsz * imgsz * 2)  # FP16
        # YOLOv8-Pose output: [B, 56, 8400] (4 bbox + 1 conf + 17*3 kp = 56)
        self.d_output = cuda.mem_alloc(max_batch * 56 * 8400 * 2)
        self.h_output = np.empty((max_batch, 56, 8400), dtype=np.float16)

    def detect_batch(self, frames: List[np.ndarray], metas: List[dict]) -> List[FrameDetections]:
        # NOTE: full implementation requires preprocessing (letterbox),
        #       inference, NMS, keypoint extraction. This is a stub that
        #       shows the API; production code lives in `tensorrt_engine.py`.
        raise NotImplementedError(
            "TensorRTDetector is a stub — use MockDetector for CPU testing, "
            "or implement full inference in tensorrt_engine.py once an "
            "engine file is available."
        )

    def close(self) -> None:
        if hasattr(self, "d_input"):
            self.d_input.free()
            self.d_output.free()


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def Detector(use_mock: Optional[bool] = None) -> BaseDetector:
    """Factory — picks the right detector implementation."""
    if use_mock is None:
        cfg = get_settings().layer1
        use_mock = cfg.use_mock or get_settings().mock_mode

    if use_mock:
        cfg = get_settings().layer1
        return MockDetector(
            conf_threshold=cfg.conf_threshold,
            iou_threshold=cfg.iou_threshold,
        )

    cfg = get_settings().layer1
    engine_path = cfg.model_path
    if not os.path.exists(engine_path):
        logger.warning(
            f"TensorRT engine not found at {engine_path}, falling back to MockDetector"
        )
        return MockDetector(cfg.conf_threshold, cfg.iou_threshold)
    return TensorRTDetector(
        engine_path=engine_path,
        imgsz=cfg.imgsz,
        max_batch=cfg.max_batch,
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
        device_id=cfg.device_id,
    )
