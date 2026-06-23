"""OSNet Re-Identification wrapper.

Real OSNet requires `torchreid` and weights. We provide:
  - `OSNetReID`       — real implementation (lazy import torchreid)
  - `MockReID`        — deterministic mock that returns random features
  - `ReID` factory    — picks based on settings

Cosine similarity is used for matching. The factory pattern means downstream
code is GPU-optional.
"""

from __future__ import annotations

import abc
import hashlib
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


class OSNetReID(BaseReID):
    """Real OSNet Re-ID via torchreid. Lazy-imported."""

    def __init__(self, model_name: str = "osnet_x0_25", weights_path: Optional[str] = None):
        import torch
        import torchreid

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torchreid.models.build_model(
            name=model_name,
            num_classes=1000,
            pretrained=weights_path is None,
        )
        if weights_path:
            sd = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(sd)
        self.model.to(self.device).eval()
        self.input_h = 256
        self.input_w = 128

    def extract(self, frame: np.ndarray, bbox: tuple[float, float, float, float]) -> np.ndarray:
        import cv2
        x1, y1, x2, y2 = [int(v) for v in bbox]
        H, W = frame.shape[:2]
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x1 >= x2 or y1 >= y2:
            return np.zeros(512, dtype=np.float32)
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.input_w, self.input_h))
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        t = self.torch.from_numpy(crop).float().permute(2, 0, 1) / 255.0
        t = (t - t.mean()) / (t.std() + 1e-6)
        t = t.unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            feat = self.model(t)
        feat = feat.cpu().numpy().flatten()
        norm = np.linalg.norm(feat)
        return feat / norm if norm > 1e-6 else feat


def ReID(use_mock: Optional[bool] = None) -> BaseReID:
    """Factory."""
    if use_mock is None:
        from config.settings import get_settings
        use_mock = get_settings().layer2.use_mock or get_settings().mock_mode
    if use_mock:
        from config.settings import get_settings
        return MockReID(feature_dim=get_settings().layer2.reid_feature_dim)
    try:
        return OSNetReID()
    except Exception as e:
        logger.warning(f"OSNet init failed ({e}); falling back to MockReID")
        return MockReID()
