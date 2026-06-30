"""Cascade filter for interaction recognition.

Stage 1: PoseConv3D (M=2) — lightweight skeleton-based inference.
  If max_score < threshold -> label "none", skip SlowFast (saves 70-80% of
  expensive RGB inference).

Stage 2: SlowFast RGB — only for cascade survivors.

Stage 3: Fusion — weighted average of PoseConv3D + SlowFast.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger
from src.common.metrics import (
    CASCADE_FILTER_RATE,
    INTERACTION_PREDICTIONS,
    RECOGNITION_LATENCY,
    SLOWFAST_SKIPPED,
)
from src.common.schemas import (
    InteractionPrediction,
    PersonPair,
    SkeletonBuffer,
)

from .poseconv3d import BasePoseConv3D
from .roi_extractor import extract_paired_skeletons, extract_roi_clip
from .slowfast import BaseSlowFast

logger = get_logger()


class CascadeFilter:
    """Cascade PoseConv3D -> SlowFast with fusion."""

    def __init__(
        self,
        poseconv3d: BasePoseConv3D,
        slowfast: Optional[BaseSlowFast],
        labels: List[str],
        cascade_score_threshold: float = 0.4,
        fusion_pose_weight: float = 0.6,
        fusion_rgb_weight: float = 0.4,
        slowfast_clip_len: int = 32,
        slowfast_img_size: int = 224,
        pose_clip_len: int = 48,
        roi_margin_ratio: float = 0.2,
    ):
        self.poseconv3d = poseconv3d
        self.slowfast = slowfast
        self.labels = labels
        self.cascade_score_threshold = cascade_score_threshold
        self.fusion_pose_weight = fusion_pose_weight
        self.fusion_rgb_weight = fusion_rgb_weight
        self.slowfast_clip_len = slowfast_clip_len
        self.slowfast_img_size = slowfast_img_size
        self.pose_clip_len = pose_clip_len
        self.roi_margin_ratio = roi_margin_ratio

        self._none_idx = labels.index("none") if "none" in labels else len(labels) - 1

    def recognize(
        self,
        pairs: List[PersonPair],
        skeleton_buffers: dict[int, np.ndarray],  # track_id -> (T, 17, 3)
        frame_buffer: Optional[List[np.ndarray]] = None,
    ) -> List[InteractionPrediction]:
        """Recognise interactions for a batch of pairs.

        Args:
            pairs: list of PersonPair
            skeleton_buffers: per-track padded skeleton arrays
            frame_buffer: recent frames for SlowFast RGB branch (most recent last)

        Returns:
            List of InteractionPrediction (one per pair)
        """
        if not pairs:
            return []

        # Stage 1: PoseConv3D on all pairs
        skel_inputs: List[np.ndarray] = []
        valid_pairs: List[PersonPair] = []
        for p in pairs:
            skel_a = skeleton_buffers.get(p.track_id_a)
            skel_b = skeleton_buffers.get(p.track_id_b)
            if skel_a is None or skel_b is None:
                # Cannot run pose model — emit "none" with low confidence
                continue
            skel_in = extract_paired_skeletons(skel_a, skel_b, self.pose_clip_len)
            skel_inputs.append(skel_in)
            valid_pairs.append(p)

        results: List[InteractionPrediction] = []
        if not valid_pairs:
            # All pairs missing skeletons — emit "none" predictions
            for p in pairs:
                results.append(self._none_prediction(p))
            return results

        t0 = time.time()
        skel_batch = np.stack(skel_inputs, axis=0)
        pose_probs = self.poseconv3d.infer_batch(skel_batch)
        RECOGNITION_LATENCY.labels("interaction").observe(time.time() - t0)

        # Stage 2: Cascade filter
        survivors: List[int] = []
        for i, p in enumerate(valid_pairs):
            top_idx = int(np.argmax(pose_probs[i]))
            top_score = float(pose_probs[i, top_idx])
            if top_score >= self.cascade_score_threshold and top_idx != self._none_idx:
                survivors.append(i)
            else:
                # Use pose-only prediction
                results.append(self._make_prediction(p, pose_probs[i], None, cascade_passed=False))
                INTERACTION_PREDICTIONS.labels(p.camera_id, self.labels[top_idx]).inc()

        # Update cascade filter rate metric
        total = len(valid_pairs)
        skipped = total - len(survivors)
        if total > 0:
            CASCADE_FILTER_RATE.set(skipped / total)

        # Stage 3: SlowFast for survivors
        if survivors and self.slowfast is not None and frame_buffer:
            rgb_inputs: List[np.ndarray] = []
            survivor_pairs: List[PersonPair] = []
            for i in survivors:
                p = valid_pairs[i]
                clip = extract_roi_clip(
                    frame_buffer,
                    p.union_bbox,
                    clip_len=self.slowfast_clip_len,
                    img_size=self.slowfast_img_size,
                    margin_ratio=self.roi_margin_ratio,
                )
                if clip is not None:
                    rgb_inputs.append(clip)
                    survivor_pairs.append(p)
            if rgb_inputs:
                t1 = time.time()
                rgb_batch = np.stack(rgb_inputs, axis=0)
                rgb_probs = self.slowfast.infer_batch(rgb_batch)
                RECOGNITION_LATENCY.labels("interaction").observe(time.time() - t1)
                for p, rgb_p in zip(survivor_pairs, rgb_probs):
                    # Find the corresponding pose probs
                    idx = valid_pairs.index(p)
                    fused = self._fuse(pose_probs[idx], rgb_p)
                    results.append(self._make_prediction(p, pose_probs[idx], rgb_p, fused, cascade_passed=True))
                    top_idx = int(np.argmax(fused))
                    INTERACTION_PREDICTIONS.labels(p.camera_id, self.labels[top_idx]).inc()
        elif survivors:
            SLOWFAST_SKIPPED.inc(len(survivors))
            logger.warning(f"SlowFast unavailable or frame_buffer empty — "
                           f"{len(survivors)} survivors falling back to pose-only")
            for i in survivors:
                p = valid_pairs[i]
                results.append(self._make_prediction(p, pose_probs[i], None, cascade_passed=False))

        return results

    def _fuse(self, pose_probs: np.ndarray, rgb_probs: np.ndarray) -> np.ndarray:
        """Weighted average fusion."""
        return (
            self.fusion_pose_weight * pose_probs +
            self.fusion_rgb_weight * rgb_probs
        )

    def _make_prediction(
        self,
        pair: PersonPair,
        pose_probs: np.ndarray,
        rgb_probs: Optional[np.ndarray],
        fused_probs: Optional[np.ndarray] = None,
        cascade_passed: bool = False,
    ) -> InteractionPrediction:
        probs = fused_probs if fused_probs is not None else pose_probs
        top_idx = int(np.argmax(probs))
        return InteractionPrediction(
            camera_id=pair.camera_id,
            frame_id=pair.frame_id,
            timestamp=pair.timestamp,
            track_id_a=pair.track_id_a,
            track_id_b=pair.track_id_b,
            label=self.labels[top_idx],
            confidence=float(probs[top_idx]),
            cascade_passed=cascade_passed,
            pose_score=float(pose_probs[top_idx]),
            rgb_score=float(rgb_probs[top_idx]) if rgb_probs is not None else 0.0,
            pose_probs=pose_probs.tolist(),
            rgb_probs=rgb_probs.tolist() if rgb_probs is not None else [],
        )

    def _none_prediction(self, pair: PersonPair) -> InteractionPrediction:
        """Generate a 'none' prediction when skeletons are missing."""
        none_idx = self._none_idx
        probs = np.zeros(len(self.labels), dtype=np.float32)
        probs[none_idx] = 1.0
        return InteractionPrediction(
            camera_id=pair.camera_id,
            frame_id=pair.frame_id,
            timestamp=pair.timestamp,
            track_id_a=pair.track_id_a,
            track_id_b=pair.track_id_b,
            label=self.labels[none_idx],
            confidence=1.0,
            cascade_passed=False,
            pose_score=1.0,
            rgb_score=0.0,
            pose_probs=probs.tolist(),
            rgb_probs=[],
        )


def make_cascade_filter() -> CascadeFilter:
    from .poseconv3d import PoseConv3D
    from .slowfast import SlowFast
    cfg = get_settings().layer4a
    return CascadeFilter(
        poseconv3d=PoseConv3D(mode="pair"),
        slowfast=SlowFast(),
        labels=cfg.interaction_labels,
        cascade_score_threshold=cfg.cascade_score_threshold,
        fusion_pose_weight=cfg.fusion_pose_weight,
        fusion_rgb_weight=cfg.fusion_rgb_weight,
        slowfast_clip_len=cfg.slowfast_clip_len,
        slowfast_img_size=cfg.slowfast_img_size,
        pose_clip_len=cfg.pose_clip_len,
        roi_margin_ratio=cfg.roi_margin_ratio,
    )
