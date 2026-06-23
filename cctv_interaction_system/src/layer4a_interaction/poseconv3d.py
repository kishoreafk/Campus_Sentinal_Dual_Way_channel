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
    """

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

        # Overall motion magnitude (per person, averaged)
        if T > 1:
            dx = np.diff(x, axis=0)
            dy = np.diff(y, axis=0)
            motion = np.sqrt(dx ** 2 + dy ** 2) * conf[:-1]
            mean_motion = float(motion.mean())
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

        # Heuristic: small center_dist + small motion = hug/kiss
        # small center_dist + large motion = fight
        # large center_dist + medium motion = handshake / high-five
        probs = np.zeros(self.num_classes, dtype=np.float32)
        if center_dist < 40 and motion < 5:
            probs[0] = 0.7  # hug
            probs[1] = 0.2  # kiss
        elif center_dist < 60 and motion > 15:
            probs[2] = 0.7  # fight
            probs[3] = 0.15  # push
        elif center_dist >= 60 and center_dist < 150 and motion > 5:
            if motion < 15:
                probs[4] = 0.6  # handshake
            else:
                probs[5] = 0.6  # high-five
        else:
            probs[7] = 0.8  # none
        # Add small noise to others
        probs = probs + 0.02
        # Softmax
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
            kps_indices = [5, 6, 11, 12, 13, 14, 15, 16]
            valid = y[:, kps_indices, 0] * (conf[:, kps_indices, 0] > 0.3)
            if valid.size > 0:
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


class TensorRTPoseConv3D(BasePoseConv3D):
    """Real TensorRT PoseConv3D — stub.

    Requires:
      - mmaction2 / mmcv for model export
      - tensorrt engine file
      - pycuda for inference
    """

    def __init__(self, engine_path: str, num_classes: int, mode: str = "pair"):
        self.engine_path = engine_path
        self.num_classes = num_classes
        self.mode = mode
        # Lazy init omitted — would load trt engine here
        raise NotImplementedError(
            "TensorRTPoseConv3D is a stub. Use MockPoseConv3D for testing."
        )

    def infer_batch(self, skeletons: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def PoseConv3D(mode: str = "pair", use_mock: Optional[bool] = None) -> BasePoseConv3D:
    """Factory."""
    if mode not in ("pair", "single"):
        raise ValueError("mode must be 'pair' or 'single'")
    if use_mock is None:
        use_mock = get_settings().mock_mode
    if mode == "pair":
        cfg = get_settings().layer4a
        if use_mock or cfg.use_mock:
            return MockPoseConv3D(len(cfg.interaction_labels), mode="pair")
        return TensorRTPoseConv3D(cfg.poseconv3d_engine_path, len(cfg.interaction_labels), "pair")
    else:
        cfg = get_settings().layer4b
        if use_mock or cfg.use_mock:
            return MockPoseConv3D(len(cfg.individual_labels), mode="single")
        return TensorRTPoseConv3D(cfg.poseconv3d_engine_path, len(cfg.individual_labels), "single")
