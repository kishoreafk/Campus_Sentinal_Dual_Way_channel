"""SlowFast RGB action recognition wrapper.

Real implementation uses a TensorRT-exported SlowFast model (e.g.
slowfast_r50_8xb8-8x8x256-256e_kinetics400). Mock returns deterministic
scores based on clip motion.
"""

from __future__ import annotations

import abc
from typing import Optional

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger

logger = get_logger()


class BaseSlowFast(abc.ABC):
    @abc.abstractmethod
    def infer_batch(self, clips: np.ndarray) -> np.ndarray:
        """Run inference.

        Args:
            clips: (B, 3, T, H, W) float32 normalised [0, 1]

        Returns:
            (B, num_classes) softmax probs
        """


class MockSlowFast(BaseSlowFast):
    """Mock SlowFast — uses clip motion + colour statistics."""

    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def infer_batch(self, clips: np.ndarray) -> np.ndarray:
        B = clips.shape[0]
        out = np.zeros((B, self.num_classes), dtype=np.float32)
        for b in range(B):
            out[b] = self._infer_one(clips[b])
        return out

    def _infer_one(self, clip: np.ndarray) -> np.ndarray:
        """Classify a pair ROI clip using motion and appearance features.

        The ROI clip is extracted from real video frames around the pair's
        union bounding box.  Inter-frame differences reflect BOTH the
        pair's physical motion AND the ROI window's translation, giving a
        meaningful (if coarse) motion signal.

        Motion thresholds are calibrated to agree with PoseConv3D's
        skeleton-based heuristic (CCTV fight-detection context):
          very high motion (>0.12) → push    (shoving at arm's length)
          high/moderate    (>0.02) → fight   (punching, grappling)
          low              (>0.005)→ handshake (greeting at medium dist)
          very low         (≤0.005)→ hug     (close embrace, static)
        """
        # clip shape (3, T, H, W)
        T = clip.shape[1]
        if T > 1:
            diffs = np.diff(clip, axis=1)
            # Overall motion magnitude
            motion = float(np.abs(diffs).mean())
            # Spatial concentration of motion — fight has localised
            # bursts while camera pan produces uniform motion
            per_frame_motion = np.abs(diffs).mean(axis=(0, 2, 3))  # (T-1,)
            motion_peak = float(per_frame_motion.max()) if per_frame_motion.size > 0 else 0.0
        else:
            motion = 0.0
            motion_peak = 0.0

        probs = np.zeros(self.num_classes, dtype=np.float32)
        # labels: [hug, kiss, fight, push, handshake, high-five, other, none]

        if motion > 0.12:
            # Very high motion — shoving / push at distance
            probs[3] = 0.85  # push
        elif motion > 0.02 or motion_peak > 0.08:
            # Moderate-to-high motion — fight (punching, grappling)
            probs[2] = 0.85  # fight
        elif motion > 0.005:
            # Low motion — gentle greeting interaction
            probs[4] = 0.85  # handshake
        else:
            # Near-static — close embrace
            probs[0] = 0.85  # hug

        probs = probs + 0.02
        probs = probs / probs.sum()
        return probs


# ---------------------------------------------------------------------
# TensorRT SlowFast (production)
# ---------------------------------------------------------------------
class TensorRTSlowFast(BaseSlowFast):
    """SlowFast via TensorRT engine.

    Input:  (B, 3, T, H, W) float32 — RGB clips normalised [0, 1]
    Output: (B, num_classes) float32 — softmax probabilities

    Requires: tensorrt, pycuda.
    """

    def __init__(
        self,
        engine_path: str,
        num_classes: int,
        clip_len: int = 32,
        img_size: int = 224,
        device_id: int = 0,
    ):
        self.num_classes = num_classes
        self.clip_len = clip_len
        self.img_size = img_size

        import tensorrt as trt
        import pycuda.driver as cuda

        cuda.init()
        self._cuda_ctx = cuda.Device(device_id).make_context()
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()

        self._allocate_buffers()

    def _allocate_buffers(self):
        import pycuda.driver as cuda
        import tensorrt as trt

        self._inputs = []
        self._outputs = []
        self._bindings = []
        for i in range(self._engine.num_bindings):
            dtype = trt.nptype(self._engine.get_binding_dtype(i))
            shape = self._engine.get_binding_shape(i)
            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, dtype)
            device = cuda.mem_alloc(host.nbytes)
            self._bindings.append(int(device))
            if self._engine.binding_is_input(i):
                self._inputs.append({"name": self._engine.get_binding_name(i), "host": host, "shape": shape})
            else:
                self._outputs.append({"name": self._engine.get_binding_name(i), "host": host, "shape": shape})

    def infer_batch(self, clips: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda
        B = clips.shape[0]
        self._cuda_ctx.push()
        try:
            inp = clips.astype(np.float32, copy=False)
            self._context.set_binding_shape(0, inp.shape)
            np.copyto(self._inputs[0]["host"], inp.ravel())
            cuda.memcpy_htod_async(self._bindings[0], self._inputs[0]["host"])
            self._context.execute_async_v2(self._bindings)
            cuda.memcpy_dtoh_async(self._outputs[0]["host"], self._bindings[1])
            out = np.copy(self._outputs[0]["host"]).reshape(B, self.num_classes)
            return out
        finally:
            self._cuda_ctx.pop()

    def close(self) -> None:
        self._cuda_ctx.pop()
        self._context = None
        self._engine = None


# ---------------------------------------------------------------------
# ONNX SlowFast (fallback)
# ---------------------------------------------------------------------
class ONNXSlowFast(BaseSlowFast):
    """SlowFast via ONNX Runtime."""

    def __init__(self, onnx_path: str, num_classes: int):
        self.num_classes = num_classes
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def infer_batch(self, clips: np.ndarray) -> np.ndarray:
        inp = clips.astype(np.float32, copy=False)
        out = self._session.run(None, {self._input_name: inp})[0]
        return out.reshape(clips.shape[0], self.num_classes)


def SlowFast(use_mock: Optional[bool] = None) -> BaseSlowFast:
    """Factory — returns TensorRTSlowFast, ONNXSlowFast, or MockSlowFast."""
    if use_mock is None:
        use_mock = get_settings().mock_mode
    cfg = get_settings().layer4a

    if not use_mock and not cfg.use_mock:
        engine_path = cfg.slowfast_engine_path
        onnx_path = engine_path.replace(".engine", ".onnx")

        import os as _os
        if _os.path.isfile(engine_path):
            try:
                logger.info(f"Loading TensorRTSlowFast from {engine_path}")
                return TensorRTSlowFast(
                    engine_path,
                    len(cfg.interaction_labels),
                    clip_len=cfg.slowfast_clip_len,
                    img_size=cfg.slowfast_img_size,
                )
            except Exception as e:
                logger.warning(f"TensorRTSlowFast failed ({e}), trying ONNX")

        if _os.path.isfile(onnx_path):
            try:
                logger.info(f"Loading ONNXSlowFast from {onnx_path}")
                return ONNXSlowFast(onnx_path, len(cfg.interaction_labels))
            except Exception as e:
                logger.warning(f"ONNXSlowFast failed ({e}), using mock")

        logger.warning("TensorRTSlowFast not available; using MockSlowFast")

    return MockSlowFast(len(cfg.interaction_labels))
