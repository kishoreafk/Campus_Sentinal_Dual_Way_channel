"""PoseConv3D wrapper for skeleton-based action recognition.

Two modes:
  - M=2 (pair)  : used as cascade filter in Layer 4A
  - M=1 (single): used for individual action recognition in Layer 4B

Mock implementation returns deterministic scores based on skeleton motion
magnitude — enough for end-to-end pipeline testing.
"""

from __future__ import annotations

import abc
from typing import List, Optional

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger

logger = get_logger()


class BasePoseConv3D(abc.ABC):
    """Abstract PoseConv3D interface."""

    # Whether input skeletons should be preprocessed (hip-centred,
    # height-normalised) before being fed to the model.  Real neural
    # network models expect preprocessed data; mock heuristic models
    # operate on raw pixel coordinates and must NOT be preprocessed.
    needs_preprocessing: bool = True

    @abc.abstractmethod
    def infer_batch(self, skeletons: np.ndarray) -> np.ndarray:
        """Run inference on a batch of skeleton clips.

        Args:
            skeletons: (B, 3, T, V, M) float32

        Returns:
            (B, num_classes) float32 — softmax probabilities
        """


class MockPoseConv3D(BasePoseConv3D):
    """Mock PoseConv3D — uses skeleton motion statistics as features.

    For pair (M=2): labels = [hug, kiss, fight, push, handshake, high-five, other, none]
    For single (M=1): labels = [walking, standing, running, sitting, waiting, other, none]

    The mock computes simple motion features (mean keypoint velocity, pairwise
    distance) and routes them to deterministic labels. This is NOT a real
    model, but it's deterministic and exercises the full pipeline.

    NOTE: The heuristics use pixel-scale thresholds and therefore require
    raw (un-preprocessed) skeleton input. ``needs_preprocessing`` is set
    to False so that the upstream recogniser skips hip-centring and
    height-normalisation which would destroy the absolute coordinate
    information the heuristics rely on.
    """

    needs_preprocessing = False  # mock heuristics need raw pixel coords

    def __init__(self, num_classes: int, mode: str = "pair"):
        if mode not in ("pair", "single"):
            raise ValueError(f"mode must be 'pair' or 'single', got {mode}")
        self.num_classes = num_classes
        self.mode = mode

    def infer_batch(self, skeletons: np.ndarray) -> np.ndarray:
        B = skeletons.shape[0]
        out = np.zeros((B, self.num_classes), dtype=np.float32)
        for b in range(B):
            out[b] = self._infer_one(skeletons[b])
        return out

    def _infer_one(self, skeleton: np.ndarray) -> np.ndarray:
        """Infer on a single (3, T, V, M) skeleton clip."""
        # skeleton shape: (3, T, V, M)
        T, V, M = skeleton.shape[1], skeleton.shape[2], skeleton.shape[3]
        # Compute per-frame mean keypoint position
        x = skeleton[0]  # (T, V, M)
        y = skeleton[1]  # (T, V, M)
        conf = skeleton[2]  # (T, V, M)

        # Identify real (non-zero-padded) frames.  Zero-padded frames
        # from short skeleton buffers have all-zero confidence.
        frame_conf_sum = conf.sum(axis=(1, 2))  # (T,)
        real_mask = frame_conf_sum > 0  # True for frames with real data

        # Overall motion magnitude — computed only over real frames
        if T > 1:
            dx = np.diff(x, axis=0)
            dy = np.diff(y, axis=0)
            motion_raw = np.sqrt(dx ** 2 + dy ** 2) * conf[:-1]
            # A transition between frames is real only if both frames are real
            pair_mask = real_mask[:-1] & real_mask[1:]
            if pair_mask.any():
                mean_motion = float(motion_raw[pair_mask].mean())
            else:
                mean_motion = 0.0
        else:
            mean_motion = 0.0

        if self.mode == "pair":
            return self._pair_probs(skeleton, mean_motion)
        return self._single_probs(skeleton, mean_motion)

    def _pair_probs(self, skeleton: np.ndarray, motion: float) -> np.ndarray:
        # labels = [hug, kiss, fight, push, handshake, high-five, other, none]
        # Compute inter-person distance (mean across wrists of person A to B)
        # wrists: kp 9 (left), 10 (right)
        if skeleton.shape[3] == 2:
            a_wrists = np.concatenate(
                [skeleton[0, :, 9, 0], skeleton[0, :, 10, 0]], axis=0
            )  # x of left/right wrist person A
            b_wrists = np.concatenate(
                [skeleton[0, :, 9, 1], skeleton[0, :, 10, 1]], axis=0
            )
            # Inter-person wrist distance — use first wrist only for simplicity
            a_w = skeleton[0, :, 9, 0]  # left wrist person A
            b_w = skeleton[0, :, 9, 1]  # left wrist person B
            wrist_dist = float(np.abs(a_w - b_w).mean())
            # Inter-person center distance
            a_center = skeleton[0, :, :, 0].mean(axis=1)  # mean x of all kps person A
            b_center = skeleton[0, :, :, 1].mean(axis=1)
            center_dist = float(np.abs(a_center - b_center).mean())
        else:
            wrist_dist = 50.0
            center_dist = 100.0

        # Heuristic: the feature space is (center_dist, motion).
        # In CCTV fight-detection context:
        #   close  + static       → hug (embrace, no movement)
        #   close  + any motion   → fight (punching, grappling, wrestling)
        #   medium + high motion  → push (shoving at arm's length)
        #   medium + low motion   → handshake / greeting
        #   far                   → none
        probs = np.zeros(self.num_classes, dtype=np.float32)
        if center_dist < 40:
            # Close proximity — physical contact
            if motion < 3:
                probs[0] = 0.85  # hug
            else:
                probs[2] = 0.85  # fight
        elif center_dist < 80:
            if motion > 10:
                probs[3] = 0.85  # push
            elif motion > 3:
                probs[4] = 0.85  # handshake
            else:
                probs[0] = 0.85  # hug
        elif center_dist < 150:
            if motion > 10:
                probs[3] = 0.85  # push
            elif motion > 3:
                probs[5] = 0.85  # high-five
            else:
                probs[6] = 0.85  # other
        else:
            probs[7] = 0.85  # none
        # Add small noise to others
        probs = probs + 0.02
        # Normalise
        probs = probs / probs.sum()
        return probs

    def _single_probs(self, skeleton: np.ndarray, motion: float) -> np.ndarray:
        # labels = [walking, standing, running, sitting, waiting, other, none]
        probs = np.zeros(self.num_classes, dtype=np.float32)
        # Vertical extent (height in image) — sit < stand
        y = skeleton[1]  # (T, V, M)
        conf = skeleton[2]
        if y.size > 0 and conf.size > 0:
            # height of body: max y - min y across shoulders/hips/knees/ankles
            # Only consider real (non-zero-padded) frames to avoid dilution.
            kps_indices = [5, 6, 11, 12, 13, 14, 15, 16]
            frame_conf_sum = conf.sum(axis=(1, 2))  # (T,)
            real_frames = frame_conf_sum > 0
            if real_frames.any():
                y_real = y[real_frames][:, kps_indices, 0]
                c_real = conf[real_frames][:, kps_indices, 0]
                valid = y_real * (c_real > 0.3)
                body_h = float(valid.max(axis=1).mean() - valid.min(axis=1).mean())
            else:
                body_h = 100.0
        else:
            body_h = 100.0

        if motion < 1.0:
            # Static
            if body_h < 80:
                probs[3] = 0.6  # sitting
            else:
                probs[1] = 0.6  # standing
                probs[4] = 0.2  # waiting
        elif motion < 10:
            probs[0] = 0.6  # walking
            probs[4] = 0.2  # waiting
        else:
            probs[2] = 0.7  # running
        probs = probs + 0.02
        probs = probs / probs.sum()
        return probs


# ---------------------------------------------------------------------
# TensorRT PoseConv3D (production)
# ---------------------------------------------------------------------
class TensorRTPoseConv3D(BasePoseConv3D):
    """PoseConv3D via TensorRT engine.

    Input:  (B, 3, T, V, M) float32 — skeleton clips
    Output: (B, num_classes) float32 — softmax probabilities

    Requires: tensorrt, pycuda.
    """

    def __init__(
        self,
        engine_path: str,
        num_classes: int,
        mode: str = "pair",
        device_id: int = 0,
    ):
        self.num_classes = num_classes
        self.mode = mode
        self.needs_preprocessing = True

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

    def infer_batch(self, skeletons: np.ndarray) -> np.ndarray:
        import pycuda.driver as cuda
        B = skeletons.shape[0]
        self._cuda_ctx.push()
        try:
            inp = skeletons.astype(np.float32, copy=False)
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
# ONNX PoseConv3D (fallback)
# ---------------------------------------------------------------------
class ONNXPoseConv3D(BasePoseConv3D):
    """PoseConv3D via ONNX Runtime."""

    def __init__(self, onnx_path: str, num_classes: int, mode: str = "pair"):
        self.num_classes = num_classes
        self.mode = mode
        self.needs_preprocessing = True

        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def infer_batch(self, skeletons: np.ndarray) -> np.ndarray:
        inp = skeletons.astype(np.float32, copy=False)
        out = self._session.run(None, {self._input_name: inp})[0]
        return out.reshape(skeletons.shape[0], self.num_classes)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def PoseConv3D(mode: str = "pair", use_mock: Optional[bool] = None) -> BasePoseConv3D:
    """Factory — returns TensorRTPoseConv3D, ONNXPoseConv3D, or MockPoseConv3D."""
    if mode not in ("pair", "single"):
        raise ValueError("mode must be 'pair' or 'single'")
    if use_mock is None:
        use_mock = get_settings().mock_mode

    if not use_mock:
        if mode == "pair":
            cfg = get_settings().layer4a
            engine_path = cfg.poseconv3d_engine_path
            onnx_path = engine_path.replace(".engine", ".onnx")
            num_classes = len(cfg.interaction_labels)
        else:
            cfg = get_settings().layer4b
            engine_path = cfg.poseconv3d_engine_path
            onnx_path = engine_path.replace(".engine", ".onnx")
            num_classes = len(cfg.individual_labels)

        import os as _os
        if _os.path.isfile(engine_path):
            try:
                logger.info(f"Loading TensorRTPoseConv3D ({mode}) from {engine_path}")
                return TensorRTPoseConv3D(engine_path, num_classes, mode=mode)
            except Exception as e:
                logger.warning(f"TensorRTPoseConv3D failed ({e}), trying ONNX")

        if _os.path.isfile(onnx_path):
            try:
                logger.info(f"Loading ONNXPoseConv3D ({mode}) from {onnx_path}")
                return ONNXPoseConv3D(onnx_path, num_classes, mode=mode)
            except Exception as e:
                logger.warning(f"ONNXPoseConv3D failed ({e}), using mock")

        logger.warning(f"TensorRTPoseConv3D not available; using MockPoseConv3D ({mode})")

    if mode == "pair":
        return MockPoseConv3D(len(get_settings().layer4a.interaction_labels), mode="pair")
    return MockPoseConv3D(len(get_settings().layer4b.individual_labels), mode="single")
