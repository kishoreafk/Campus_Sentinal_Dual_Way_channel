"""Pair analyzer — finds candidate interacting pairs in a frame.

For each frame, compute pairwise distances between active tracklets, apply
filtering criteria, and produce PersonPair objects.

Filter criteria (ALL must pass):
  1. Distance < ratio * avg_height
  2. IoU > threshold
  3. Face-to-face dot < threshold
  4. Sustained proximity > N frames
  5. Both skeletons valid (avg KP confidence > threshold)
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger
from src.common.metrics import PAIRS_DETECTED
from src.common.schemas import PersonPair, Tracklet

from .distance_matrix import (
    bbox_center,
    bbox_iou,
    bbox_union,
    euclidean,
    face_to_face_dot,
)

logger = get_logger()


class PairAnalyzer:
    """Per-camera pair analyzer with sustained-proximity tracking."""

    def __init__(
        self,
        camera_id: str,
        distance_ratio_threshold: float = 0.8,
        iou_threshold: float = 0.15,
        face_to_face_dot_threshold: float = -0.3,
        sustained_proximity_frames: int = 15,
        min_keypoint_confidence: float = 0.5,
    ):
        self.camera_id = camera_id
        self.distance_ratio_threshold = distance_ratio_threshold
        self.iou_threshold = iou_threshold
        self.face_to_face_dot_threshold = face_to_face_dot_threshold
        self.sustained_proximity_frames = sustained_proximity_frames
        self.min_keypoint_confidence = min_keypoint_confidence
        # (id_a, id_b) -> sustained count
        self._proximity_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    def update(self, tracklets: List[Tracklet], frame_id: int, timestamp: float) -> List[PersonPair]:
        """Process one frame's tracklets and return qualifying pairs."""
        pairs: List[PersonPair] = []
        # Track which pairs are still "in proximity" this frame
        seen_pairs: set[Tuple[int, int]] = set()

        # O(n^2) pairwise
        for i in range(len(tracklets)):
            for j in range(i + 1, len(tracklets)):
                a, b = tracklets[i], tracklets[j]
                # Skip if either has poor keypoints
                if a.avg_kp_confidence < self.min_keypoint_confidence:
                    continue
                if b.avg_kp_confidence < self.min_keypoint_confidence:
                    continue

                # Distance check
                c_a = bbox_center(a.bbox)
                c_b = bbox_center(b.bbox)
                dist = euclidean(c_a, c_b)
                avg_h = (a.height + b.height) / 2.0
                if avg_h < 1e-3:
                    continue
                if dist >= self.distance_ratio_threshold * avg_h:
                    continue

                # IoU check
                iou = bbox_iou(a.bbox, b.bbox)
                if iou < self.iou_threshold:
                    # We allow IoU=0 if they're touching, but here we follow spec
                    # The spec says >0.15 — we follow it strictly
                    continue

                # Face-to-face check
                dot = face_to_face_dot(a.keypoints, b.keypoints)
                if dot is None:
                    # Cannot determine orientation — be conservative, skip
                    continue
                if dot >= self.face_to_face_dot_threshold:
                    continue

                # Sustained proximity
                key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                self._proximity_counts[key] += 1
                seen_pairs.add(key)
                sustained = self._proximity_counts[key]
                if sustained < self.sustained_proximity_frames:
                    continue

                union = bbox_union(a.bbox, b.bbox)
                pairs.append(PersonPair(
                    camera_id=self.camera_id,
                    frame_id=frame_id,
                    timestamp=timestamp,
                    track_id_a=a.track_id,
                    track_id_b=b.track_id,
                    bbox_a=a.bbox,
                    bbox_b=b.bbox,
                    distance=float(dist),
                    iou=float(iou),
                    face_to_face_dot=float(dot),
                    sustained_frames=sustained,
                    union_bbox=union,
                ))
                PAIRS_DETECTED.labels(self.camera_id).inc()

        # Decay proximity counts for pairs not seen this frame
        stale = [k for k in self._proximity_counts if k not in seen_pairs]
        for k in stale:
            self._proximity_counts[k] = max(0, self._proximity_counts[k] - 1)
            if self._proximity_counts[k] == 0:
                del self._proximity_counts[k]

        return pairs


def make_pair_analyzer(camera_id: str) -> PairAnalyzer:
    cfg = get_settings().layer3
    return PairAnalyzer(
        camera_id=camera_id,
        distance_ratio_threshold=cfg.distance_ratio_threshold,
        iou_threshold=cfg.iou_threshold,
        face_to_face_dot_threshold=cfg.face_to_face_dot_threshold,
        sustained_proximity_frames=cfg.sustained_proximity_frames,
        min_keypoint_confidence=cfg.min_keypoint_confidence,
    )
