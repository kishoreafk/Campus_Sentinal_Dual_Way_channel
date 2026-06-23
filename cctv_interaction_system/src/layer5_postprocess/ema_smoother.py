"""EMA smoother — per-tracklet, per-action exponentially-weighted moving average.

Smooths noisy per-frame action probabilities so the downstream state machine
sees stable inputs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple

import numpy as np


class EMASmoother:
    """Per-(track_id, action_label) EMA smoother.

    score_t = alpha * score_{t-1} + (1 - alpha) * score_new
    """

    def __init__(self, alpha: float = 0.7):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.alpha = alpha
        # key: (camera_id, track_key, action_label) -> smoothed score
        self._scores: Dict[Tuple[str, str, str], float] = defaultdict(float)

    def update(
        self,
        camera_id: str,
        track_key: str,  # e.g. "tid_5" or "pair_5_7"
        action_label: str,
        raw_score: float,
    ) -> float:
        """Update with a new raw score and return the smoothed score."""
        key = (camera_id, track_key, action_label)
        prev = self._scores.get(key, 0.0)
        smoothed = self.alpha * prev + (1.0 - self.alpha) * raw_score
        self._scores[key] = smoothed
        return smoothed

    def update_probs(
        self,
        camera_id: str,
        track_key: str,
        labels: list[str],
        probs: np.ndarray,
    ) -> np.ndarray:
        """Smooth an entire probability vector.

        Returns a new probability vector (not re-normalised — caller decides).
        """
        if len(labels) != len(probs):
            raise ValueError(
                f"labels/probs length mismatch: {len(labels)} vs {len(probs)}"
            )
        out = np.zeros_like(probs, dtype=np.float32)
        for i, label in enumerate(labels):
            out[i] = self.update(camera_id, track_key, label, float(probs[i]))
        return out

    def get(self, camera_id: str, track_key: str, action_label: str) -> float:
        return self._scores.get((camera_id, track_key, action_label), 0.0)

    def reset(self, camera_id: str, track_key: str) -> None:
        """Reset all action scores for a (camera, track) pair."""
        keys_to_remove = [k for k in self._scores
                          if k[0] == camera_id and k[1] == track_key]
        for k in keys_to_remove:
            del self._scores[k]
