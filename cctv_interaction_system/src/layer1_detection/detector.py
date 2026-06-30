"""TensorRT engine wrapper for YOLOv8-Pose.

This module provides three implementations:

  1. `TensorRTDetector`   — production, requires `tensorrt`, `pycuda`, and an
                            exported `.engine` file. Loaded lazily so the
                            package imports cleanly without GPU deps.
  2. `ONNXDetector`       — ONNX Runtime fallback (no TensorRT needed).
  3. `MockDetector`       — CPU-only stub that returns plausible-looking
                            detections for testing. Deterministic per-frame
                            via frame_id seed.

The `Detector` factory picks the right implementation based on settings.
"""

from __future__ import annotations

import abc
import hashlib
import math
import os
import time
from typing import List, Optional

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger
from src.common.metrics import DETECTION_BATCH_SIZE, DETECTION_LATENCY
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
# TensorRT detector (production)
# ---------------------------------------------------------------------
class TensorRTDetector(BaseDetector):
    """YOLOv8-Pose via TensorRT engine file.

    Requires: tensorrt, pycuda, and a valid .engine file.
    """

    def __init__(
        self,
        engine_path: str,
        imgsz: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.65,
        half: bool = True,
        device_id: int = 0,
        warmup_iters: int = 10,
    ):
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.half = half
        self.device_id = device_id

        import tensorrt as trt
        import pycuda.driver as cuda

        cuda.init()
        self._cuda_ctx = cuda.Device(device_id).make_context()
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)

        with open(engine_path, "rb") as f:
            self._engine = self._runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

        self._allocate_buffers()

        # Warmup
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(warmup_iters):
            self._infer([dummy])

    def _allocate_buffers(self):
        import pycuda.driver as cuda
        import tensorrt as trt

        self._inputs = []
        self._outputs = []
        self._bindings = []
        for i in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(i)
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            shape = self._engine.get_binding_shape(i)
            size = int(np.prod(shape))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))
            if self._engine.binding_is_input(i):
                self._inputs.append({"name": name, "host": host_mem, "device": device_mem, "shape": shape, "dtype": dtype})
            else:
                self._outputs.append({"name": name, "host": host_mem, "device": device_mem, "shape": shape, "dtype": dtype})

    def _preprocess(self, frames: List[np.ndarray]) -> np.ndarray:
        import cv2
        batch = []
        for f in frames:
            h, w = f.shape[:2]
            r = self.imgsz / max(h, w)
            nh, nw = int(h * r), int(w * r)
            resized = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            dx = (self.imgsz - nw) // 2
            dy = (self.imgsz - nh) // 2
            canvas[dy:dy+nh, dx:dx+nw] = resized
            normed = canvas.astype(np.float32) / 255.0
            if self.half:
                normed = normed.astype(np.float16)
            batch.append(normed.transpose(2, 0, 1))
        return np.stack(batch, axis=0) if len(batch) > 1 else batch[0][None, ...]

    def _infer(self, frames: List[np.ndarray]) -> np.ndarray:
        import pycuda.driver as cuda
        self._cuda_ctx.push()
        try:
            inp = self._preprocess(frames)
            B = inp.shape[0]
            self._context.set_binding_shape(0, inp.shape)
            np.copyto(self._inputs[0]["host"], inp.ravel())
            cuda.memcpy_htod_async(self._inputs[0]["device"], self._inputs[0]["host"])
            self._context.execute_async_v2(self._bindings)
            cuda.memcpy_dtoh_async(self._outputs[0]["host"], self._outputs[0]["device"])
            out = np.copy(self._outputs[0]["host"])
            out = out.reshape(B, -1, out.shape[0] // B // out.shape[1])
            return out
        finally:
            self._cuda_ctx.pop()

    def detect_batch(self, frames: List[np.ndarray], metas: List[dict]) -> List[FrameDetections]:
        t0 = time.time()
        from .postprocess import parse_yolov8_pose_output
        raw = self._infer(frames)
        dets_list = parse_yolov8_pose_output(
            raw,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            orig_shape=(frames[0].shape[0], frames[0].shape[1]) if frames else (720, 1280),
            input_size=self.imgsz,
        )
        DETECTION_LATENCY.observe(time.time() - t0)
        DETECTION_BATCH_SIZE.observe(len(frames))
        results = []
        for dets, meta in zip(dets_list, metas):
            detections = [
                Detection(
                    bbox=d["bbox"],
                    confidence=d["confidence"],
                    keypoints=d["keypoints"],
                    keypoint_scores=d["keypoint_scores"],
                ) for d in dets
            ]
            results.append(FrameDetections(
                camera_id=meta["camera_id"],
                frame_id=meta["frame_id"],
                timestamp=meta["timestamp"],
                detections=detections,
            ))
        return results

    def close(self) -> None:
        self._cuda_ctx.pop()
        self._context = None
        self._engine = None


# ---------------------------------------------------------------------
# ONNX Runtime fallback detector (no TensorRT needed)
# ---------------------------------------------------------------------
class ONNXDetector(BaseDetector):
    """YOLOv8-Pose via ONNX Runtime — works without TensorRT.

    Requires: onnxruntime-gpu (or onnxruntime for CPU).
    """

    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 640,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.65,
        device_id: int = 0,
    ):
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        import onnxruntime as ort
        providers = [f"CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def _preprocess(self, frames: List[np.ndarray]) -> np.ndarray:
        import cv2
        batch = []
        for f in frames:
            h, w = f.shape[:2]
            r = self.imgsz / max(h, w)
            nh, nw = int(h * r), int(w * r)
            resized = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            dx = (self.imgsz - nw) // 2
            dy = (self.imgsz - nh) // 2
            canvas[dy:dy+nh, dx:dx+nw] = resized
            normed = canvas.astype(np.float32) / 255.0
            batch.append(normed.transpose(2, 0, 1))
        return np.stack(batch, axis=0) if len(batch) > 1 else batch[0][None, ...]

    def detect_batch(self, frames: List[np.ndarray], metas: List[dict]) -> List[FrameDetections]:
        t0 = time.time()
        from .postprocess import parse_yolov8_pose_output
        inp = self._preprocess(frames)
        raw = self._session.run(None, {self._input_name: inp})[0]
        dets_list = parse_yolov8_pose_output(
            raw,
            conf_threshold=self.conf_threshold,
            iou_threshold=self.iou_threshold,
            orig_shape=(frames[0].shape[0], frames[0].shape[1]) if frames else (720, 1280),
            input_size=self.imgsz,
        )
        DETECTION_LATENCY.observe(time.time() - t0)
        DETECTION_BATCH_SIZE.observe(len(frames))
        results = []
        for dets, meta in zip(dets_list, metas):
            detections = [
                Detection(
                    bbox=d["bbox"],
                    confidence=d["confidence"],
                    keypoints=d["keypoints"],
                    keypoint_scores=d["keypoint_scores"],
                ) for d in dets
            ]
            results.append(FrameDetections(
                camera_id=meta["camera_id"],
                frame_id=meta["frame_id"],
                timestamp=meta["timestamp"],
                detections=detections,
            ))
        return results


# ---------------------------------------------------------------------
# Mock detector (CPU)
# ---------------------------------------------------------------------


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
            seed = int(hashlib.md5(f"{camera_id}_{frame_id}".encode()).hexdigest(), 16) & 0xFFFFFFFF
            rng = np.random.default_rng(seed)

            n_persons = int(rng.integers(2, 4))  # at least 2 for pairs
            detections: List[Detection] = []
            for i in range(n_persons):
                # Pair-friendly trajectories: persons (0,1) orbit close
                # together, persons (2,3) orbit close together, etc.
                # Within each pair the phase offset is small (0.15 rad)
                # so the two people stay near each other for sustained
                # periods — enough to trigger PairAnalyzer's sustained-
                # proximity requirement.
                pair_idx = i // 2           # which pair this person belongs to
                is_second = i % 2           # 0 or 1 within the pair
                base_phase = frame_id * 0.05 + pair_idx * 2.5
                phase = base_phase + is_second * 0.15

                cx = W * (0.5 + 0.25 * math.sin(phase))
                cy = H * (0.55 + 0.10 * math.cos(phase * 0.7))
                w = 90 + int(20 * math.sin(phase * 2))
                h = 180 + int(20 * math.cos(phase * 2))
                x1 = max(0.0, cx - w / 2)
                y1 = max(0.0, cy - h / 2)
                x2 = min(W, cx + w / 2)
                y2 = min(H, cy + h / 2)
                conf = float(rng.uniform(0.7, 0.97))

                # Deterministic face direction: even-indexed persons face
                # right, odd-indexed face left.  When an even/odd pair is
                # close, face_to_face_dot returns -1.0 (facing each other),
                # satisfying PairAnalyzer's face-to-face threshold.
                facing = 6.0 if (i % 2 == 0) else -6.0

                kps = self._synthetic_keypoints(x1, y1, x2, y2, rng, facing=facing)
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
    def _synthetic_keypoints(
        x1, y1, x2, y2, rng, facing: float = 0.0,
    ) -> List[List[float]]:
        """Place 17 COCO keypoints within bbox (rough anthropomorphic layout).

        Args:
            facing: horizontal bias in pixels for face-direction keypoints.
                    Positive = facing right, negative = facing left.  This
                    lets ``face_to_face_dot`` distinguish same-direction
                    vs face-to-face pairs.
        """
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
            # Face-direction bias for nose (0), eyes (1,2), ears (3,4)
            if rx <= 0.60 and ry < 0.15:
                kp_x += facing
            conf = float(rng.uniform(0.55, 0.95))
            kps.append([kp_x, kp_y, conf])
        return kps


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def Detector(use_mock: Optional[bool] = None) -> BaseDetector:
    """Factory — returns the best available detector based on settings.

    Priority:
      1. TensorRTDetector  (if engine file exists, tensorrt installed)
      2. ONNXDetector      (if onnx file exists, onnxruntime installed)
      3. MockDetector      (always available, default)
    """
    if use_mock is None:
        cfg = get_settings().layer1
        use_mock = cfg.use_mock or get_settings().mock_mode
    cfg = get_settings().layer1

    if not use_mock:
        engine_path = cfg.model_path
        if os.path.isfile(engine_path):
            try:
                logger.info(f"Loading TensorRTDetector from {engine_path}")
                return TensorRTDetector(
                    engine_path=engine_path,
                    imgsz=cfg.imgsz,
                    conf_threshold=cfg.conf_threshold,
                    iou_threshold=cfg.iou_threshold,
                    half=cfg.half_precision,
                    device_id=cfg.device_id,
                    warmup_iters=cfg.warmup_iters,
                )
            except Exception as e:
                logger.warning(f"TensorRTDetector failed ({e}), trying ONNXDetector")

        onnx_path = cfg.onnx_path
        if os.path.isfile(onnx_path):
            try:
                logger.info(f"Loading ONNXDetector from {onnx_path}")
                return ONNXDetector(
                    onnx_path=onnx_path,
                    imgsz=cfg.imgsz,
                    conf_threshold=cfg.conf_threshold,
                    iou_threshold=cfg.iou_threshold,
                    device_id=cfg.device_id,
                )
            except Exception as e:
                logger.warning(f"ONNXDetector failed ({e}), falling back to MockDetector")

        logger.warning("No model files found; falling back to MockDetector")

    return MockDetector(
        conf_threshold=cfg.conf_threshold,
        iou_threshold=cfg.iou_threshold,
    )


def warmup_detector(detector: BaseDetector) -> None:
    """Run a warmup inference if the detector supports it."""
    if isinstance(detector, (TensorRTDetector, ONNXDetector)):
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        metas = [{"camera_id": "warmup", "frame_id": 0, "timestamp": time.time()}]
        detector.detect_batch([dummy], metas)
        logger.info("Detector warmup complete")
