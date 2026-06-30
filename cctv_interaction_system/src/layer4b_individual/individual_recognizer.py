"""Individual action recognizer — wraps PoseConv3D (M=1) for single-person actions."""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger
from src.common.metrics import INDIVIDUAL_PREDICTIONS, RECOGNITION_LATENCY
from src.common.schemas import IndividualPrediction, Tracklet

from .skeleton_preprocess import preprocess_skeleton
from src.layer4a_interaction.poseconv3d import PoseConv3D

logger = get_logger()


class IndividualRecognizer:
    """Recognise single-person actions from skeleton clips."""

    def __init__(
        self,
        labels: List[str],
        conf_threshold: float = 0.6,
        pose_clip_len: int = 48,
    ):
        self.model = PoseConv3D(mode="single")
        self.labels = labels
        self.conf_threshold = conf_threshold
        self.pose_clip_len = pose_clip_len
        self._none_idx = labels.index("none") if "none" in labels else len(labels) - 1

    def recognize(
        self,
        tracklets: List[Tracklet],
        skeleton_buffers: dict[int, np.ndarray],  # track_id -> (T, 17, 3)
        frame_id: int,
        timestamp: float,
        camera_id: str,
    ) -> List[IndividualPrediction]:
        if not tracklets:
            return []

        # Build batch
        inputs: List[np.ndarray] = []
        valid_tracklets: List[Tracklet] = []
        for t in tracklets:
            skel = skeleton_buffers.get(t.track_id)
            if skel is None or skel.shape[0] == 0:
                continue
            if self.model.needs_preprocessing:
                inputs.append(preprocess_skeleton(skel))
            else:
                # Mock models operate on raw pixel coordinates — skip
                # hip-centring and height-normalisation so the heuristic
                # thresholds (which are in pixel space) remain valid.
                inputs.append(self._format_raw_skeleton(skel))
            valid_tracklets.append(t)

        if not inputs:
            return []

        batch = np.stack(inputs, axis=0)  # (B, 3, T, 17, 1)
        t0 = time.time()
        probs = self.model.infer_batch(batch)
        RECOGNITION_LATENCY.labels("individual").observe(time.time() - t0)

        results: List[IndividualPrediction] = []
        for t, p in zip(valid_tracklets, probs):
            top_idx = int(np.argmax(p))
            top_score = float(p[top_idx])
            # Apply confidence threshold -> "none"
            if top_score < self.conf_threshold:
                label = self.labels[self._none_idx]
                score = float(p[self._none_idx])
            else:
                label = self.labels[top_idx]
                score = top_score
            results.append(IndividualPrediction(
                camera_id=camera_id,
                frame_id=frame_id,
                timestamp=timestamp,
                track_id=t.track_id,
                label=label,
                confidence=score,
                probs=p.tolist(),
            ))
            INDIVIDUAL_PREDICTIONS.labels(camera_id, label).inc()
        return results

    def _format_raw_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """Format a raw skeleton clip for mock model input.

        Reshapes (T, 17, 3) to (3, T, 17, 1) with zero-padding to
        ``pose_clip_len`` frames.  Unlike ``preprocess_skeleton`` this
        keeps the original pixel coordinates intact (no hip-centring or
        height-normalisation).
        """
        T, V, _ = skeleton.shape
        skel = skeleton.copy().astype(np.float32)

        target_T = self.pose_clip_len
        if T < target_T:
            pad = np.zeros((target_T - T, V, 3), dtype=np.float32)
            skel = np.concatenate([pad, skel], axis=0)
        elif T > target_T:
            skel = skel[-target_T:]

        # Reshape to (3, T, V, M=1)
        out = skel.transpose(2, 0, 1)  # (3, T, V)
        out = out[:, :, :, None]        # (3, T, V, 1)
        return out


def make_individual_recognizer() -> IndividualRecognizer:
    cfg = get_settings().layer4b
    return IndividualRecognizer(
        labels=cfg.individual_labels,
        conf_threshold=cfg.conf_threshold,
        pose_clip_len=cfg.pose_clip_len,
    )
