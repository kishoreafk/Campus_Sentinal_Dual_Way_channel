"""OSNet Re-Identification wrapper.

Real OSNet requires `torchreid` and weights. We provide:
  - `OSNetReID`       — real implementation (lazy import torchreid + torch)
  - `ONNXReID`        — ONNX Runtime fallback for OSNet
  - `MockReID`        — deterministic mock that returns deterministic features
  - `ReID` factory    — picks based on settings

Cosine similarity is used for matching. The factory pattern means downstream
code is GPU-optional.
"""

from __future__ import annotations

import abc
import hashlib
import os
from typing import List, Optional

import numpy as np

from src.common.logger import get_logger

logger = get_logger()


class BaseReID(abc.ABC):
    """Re-ID interface."""

    @abc.abstractmethod
    def extract(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        """Extract appearance feature for a person crop."""

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < 1e-6 or nb < 1e-6:
            return 0.0
        return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------
# OSNetReID — real PyTorch implementation
# ---------------------------------------------------------------------
class OSNetReID(BaseReID):
    """OSNet ReID via torchreid.

    Requires: torch, torchreid.
    """

    def __init__(
        self,
        model_name: str = "osnet_x0_25",
        input_h: int = 256,
        input_w: int = 128,
        device: str = "cuda:0",
        feature_dim: int = 512,
    ):
        self.input_h = input_h
        self.input_w = input_w
        self.feature_dim = feature_dim

        import torch
        self._device = torch.device(device if torch.cuda.is_available() else "cpu")

        try:
            import torchreid
            self._model = torchreid.models.build_model(
                name=model_name,
                num_classes=1,
                pretrained=True,
            )
        except Exception:
            from src.common.schemas import DummyFeatureExtractor
            # Fallback: load OSNet from torch.hub
            import torchvision
            self._model = torch.hub.load(
                "bubbliiiing/osnet-pytorch", "osnet_x0_25", pretrained=True
            )

        self._model = self._model.to(self._device)
        self._model.eval()

    def _preprocess(self, crop: np.ndarray) -> "torch.Tensor":
        import cv2
        import torch
        resized = cv2.resize(crop, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        # Normalise to ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = resized.astype(np.float32) / 255.0
        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        return torch.from_numpy(img[None, ...]).to(self._device)

    def extract(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        import torch
        x1, y1, x2, y2 = [int(v) for v in bbox]
        H, W = frame.shape[:2]
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x1 >= x2 or y1 >= y2:
            return np.zeros(self.feature_dim, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        inp = self._preprocess(crop)
        with torch.no_grad():
            feat = self._model(inp).cpu().numpy().flatten()
        # L2-normalise
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 1e-6 else feat


# ---------------------------------------------------------------------
# ONNXReID — ONNX Runtime fallback (no torch needed)
# ---------------------------------------------------------------------
class ONNXReID(BaseReID):
    """OSNet via ONNX Runtime — works without PyTorch.

    Requires: onnxruntime.
    """

    def __init__(
        self,
        onnx_path: str = "models/osnet_x0_25.onnx",
        input_h: int = 256,
        input_w: int = 128,
        feature_dim: int = 512,
    ):
        self.input_h = input_h
        self.input_w = input_w
        self.feature_dim = feature_dim

        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(onnx_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        import cv2
        resized = cv2.resize(crop, (self.input_w, self.input_h), interpolation=cv2.INTER_LINEAR)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = resized.astype(np.float32) / 255.0
        img = (img - mean) / std
        return img.transpose(2, 0, 1)[None, ...]

    def extract(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        H, W = frame.shape[:2]
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x1 >= x2 or y1 >= y2:
            return np.zeros(self.feature_dim, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        inp = self._preprocess(crop)
        feat = self._session.run(None, {self._input_name: inp})[0].flatten()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 1e-6 else feat


class MockReID(BaseReID):
    """Deterministic mock — feature derived from crop pixel statistics.

    Two crops of the same person (same colour histogram) will produce
    similar feature vectors, which is enough for testing Re-ID logic.
    """

    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim

    def extract(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        H, W = frame.shape[:2]
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x1 >= x2 or y1 >= y2:
            return np.zeros(self.feature_dim, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        # Compute colour histogram (8 bins per channel = 24 dims) and hash-expand
        feat = []
        for c in range(3):
            hist, _ = np.histogram(crop[:, :, c].ravel(), bins=8, range=(0, 256))
            feat.extend(hist / max(1, hist.sum()))
        # Deterministic expansion to feature_dim
        seed = int(hashlib.md5(np.array(feat, dtype=np.float32).tobytes()).hexdigest(), 16) & 0xFFFFFFFF
        rng = np.random.default_rng(seed)
        expanded = rng.standard_normal(self.feature_dim).astype(np.float32) * 0.1
        expanded[: len(feat)] = feat
        # L2-normalise
        norm = np.linalg.norm(expanded)
        if norm > 1e-6:
            expanded = expanded / norm
        return expanded


def ReID(use_mock: Optional[bool] = None) -> BaseReID:
    """Factory — returns OSNetReID, ONNXReID, or MockReID based on settings."""
    if use_mock is None:
        from config.settings import get_settings
        use_mock = get_settings().layer2.use_mock or get_settings().mock_mode
    from config.settings import get_settings
    cfg = get_settings().layer2

    if not use_mock:
        # Try PyTorch OSNet first
        onnx_path = f"models/{cfg.reid_model}.onnx"
        try:
            logger.info(f"Loading OSNetReID ({cfg.reid_model})")
            return OSNetReID(
                model_name=cfg.reid_model,
                input_h=cfg.reid_input_h,
                input_w=cfg.reid_input_w,
                device=f"cuda:{cfg.get('device_id', 0)}" if hasattr(cfg, 'device_id') else "cuda:0",
                feature_dim=cfg.reid_feature_dim,
            )
        except Exception as e:
            logger.warning(f"OSNetReID failed ({e}), trying ONNXReID")
            if os.path.isfile(onnx_path):
                try:
                    return ONNXReID(
                        onnx_path=onnx_path,
                        input_h=cfg.reid_input_h,
                        input_w=cfg.reid_input_w,
                        feature_dim=cfg.reid_feature_dim,
                    )
                except Exception as e2:
                    logger.warning(f"ONNXReID failed ({e2}), falling back to MockReID")

    return MockReID(feature_dim=cfg.reid_feature_dim)
