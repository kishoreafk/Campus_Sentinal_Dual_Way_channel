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
            inputs.append(preprocess_skeleton(skel))
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


def make_individual_recognizer() -> IndividualRecognizer:
    cfg = get_settings().layer4b
    return IndividualRecognizer(
        labels=cfg.individual_labels,
        conf_threshold=cfg.conf_threshold,
        pose_clip_len=cfg.pose_clip_len,
    )
