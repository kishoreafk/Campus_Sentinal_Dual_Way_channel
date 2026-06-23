"""Track manager — ties ByteTrack + OSNet Re-ID + Kalman pose interpolation
+ skeleton buffers together.

This is the public entry point for Layer 2.
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from config.settings import get_settings
from src.common.logger import get_logger
from src.common.metrics import (
    ACTIVE_TRACKLETS,
    REID_MATCHES,
    TRACKING_LATENCY,
)
from src.common.schemas import Detection, Tracklet

from .bytetrack import ByteTrack, Track
from .kalman_pose import PoseInterpolator
from .osnet_reid import BaseReID, ReID
from .skeleton_buffer import SkeletonBuffer

logger = get_logger()


class TrackManager:
    """Per-camera track manager.

    Maintains:
      - ByteTrack for ID assignment
      - OSNet Re-ID for ID recovery after occlusion
      - PoseInterpolator per track (Kalman) for skipped frames
      - SkeletonBuffer per track (48-frame sliding window)
    """

    def __init__(
        self,
        camera_id: str,
        track_thresh: float = 0.5,
        track_high_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        reid_cosine_threshold: float = 0.6,
        skeleton_buffer_len: int = 48,
        reid: Optional[BaseReID] = None,
        process_noise: float = 0.01,
    ):
        self.camera_id = camera_id
        self.tracker = ByteTrack(
            track_thresh, track_high_thresh, match_thresh, track_buffer, frame_rate,
        )
        self.reid = reid or ReID()
        self.reid_cosine_threshold = reid_cosine_threshold
        self.skeleton_buffer_len = skeleton_buffer_len
        self.process_noise = process_noise

        # track_id -> Tracklet (post-processing view)
        self.tracklets: Dict[int, Tracklet] = {}
        # track_id -> SkeletonBuffer
        self.skeleton_buffers: Dict[int, SkeletonBuffer] = {}
        # track_id -> PoseInterpolator
        self.pose_interpolators: Dict[int, PoseInterpolator] = {}
        # track_id -> appearance feature (last seen)
        self.appearances: Dict[int, np.ndarray] = {}
        # Track ID aliasing (recovered IDs)
        self.id_aliases: Dict[int, int] = {}

    def update(self, detections: List[Detection], frame: Optional[np.ndarray] = None) -> List[Tracklet]:
        """Process one frame's detections.

        Args:
            detections: list of Detection objects (from Layer 1)
            frame: BGR frame for appearance extraction (optional in mock mode)

        Returns:
            List of active Tracklets (with assigned IDs, updated keypoints)
        """
        t0 = time.time()
        active_tracks: List[Track] = self.tracker.update(detections)

        active: List[Tracklet] = []
        for t in active_tracks:
            tid = t.track_id
            # Resolve aliases (if this ID was recovered to another)
            while tid in self.id_aliases:
                tid = self.id_aliases[tid]

            # Update or create tracklet
            if tid not in self.tracklets:
                self.tracklets[tid] = Tracklet(
                    track_id=tid,
                    camera_id=self.camera_id,
                    bbox=t.bbox,
                    confidence=t.score,
                    keypoints=t.keypoints,
                    keypoint_scores=t.keypoint_scores,
                    state="NEW",
                )
                self.skeleton_buffers[tid] = SkeletonBuffer(tid, self.skeleton_buffer_len)
                self.pose_interpolators[tid] = PoseInterpolator(
                    num_keypoints=17, process_noise=self.process_noise,
                )
            tracklet = self.tracklets[tid]
            tracklet.bbox = t.bbox
            tracklet.confidence = t.score
            tracklet.keypoints = t.keypoints
            tracklet.keypoint_scores = t.keypoint_scores
            tracklet.last_seen = time.time()
            tracklet.consecutive_seen += 1
            tracklet.consecutive_missed = 0

            # Update Kalman + skeleton buffer
            if t.keypoints:
                kp_arr = np.array(t.keypoints, dtype=np.float32)
                self.pose_interpolators[tid].update(kp_arr)
                self.skeleton_buffers[tid].push(kp_arr)

            # Extract appearance if frame available
            if frame is not None and self.reid is not None:
                try:
                    feat = self.reid.extract(frame, t.bbox)
                    self.appearances[tid] = feat
                except Exception as e:
                    logger.debug(f"ReID extract failed for track {tid}: {e}")

            # State machine
            if tracklet.consecutive_seen >= 3:
                tracklet.state = "ACTIVE"
            elif tracklet.consecutive_seen >= 1:
                tracklet.state = "CONFIRMED"

            active.append(tracklet)

        # Mark missing tracklets
        active_ids = {t.track_id for t in active_tracks}
        for tid, trk in self.tracklets.items():
            if tid not in active_ids:
                trk.consecutive_missed += 1
                trk.consecutive_seen = 0
                if trk.consecutive_missed >= 5:
                    trk.state = "OCCLUDED"
                if trk.consecutive_missed >= 30:
                    trk.state = "LOST"
                # Try Re-ID recovery against currently active tracks
                if trk.state == "OCCLUDED" and tid in self.appearances:
                    recovered = self._try_reid_recovery(tid)
                    if recovered is not None:
                        self.id_aliases[tid] = recovered
                        REID_MATCHES.labels(self.camera_id).inc()

        ACTIVE_TRACKLETS.labels(self.camera_id).set(len(active))
        TRACKING_LATENCY.observe(time.time() - t0)
        return active

    def _try_reid_recovery(self, lost_id: int) -> Optional[int]:
        """Look for an active track whose appearance matches the lost one."""
        lost_feat = self.appearances.get(lost_id)
        if lost_feat is None:
            return None
        best_id, best_sim = None, 0.0
        for tid, feat in self.appearances.items():
            if tid == lost_id:
                continue
            if tid not in self.tracklets:
                continue
            if self.tracklets[tid].state not in ("NEW", "CONFIRMED", "ACTIVE"):
                continue
            sim = self.reid.cosine_similarity(lost_feat, feat)
            if sim > best_sim:
                best_sim, best_id = sim, tid
        if best_id is not None and best_sim >= self.reid_cosine_threshold:
            logger.info(f"[{self.camera_id}] Re-ID recovery: {lost_id} -> {best_id} "
                        f"(sim={best_sim:.3f})")
            return best_id
        return None

    def get_skeleton(self, track_id: int) -> Optional[np.ndarray]:
        """Return padded skeleton (48, 17, 3) for a track."""
        buf = self.skeleton_buffers.get(track_id)
        if buf is None:
            return None
        return buf.to_padded(self.skeleton_buffer_len)

    def get_all_skeletons(self) -> Dict[int, np.ndarray]:
        return {tid: self.get_skeleton(tid) for tid in self.tracklets
                if self.tracklets[tid].state in ("NEW", "CONFIRMED", "ACTIVE")}


def make_track_manager(camera_id: str) -> TrackManager:
    """Factory using global settings."""
    cfg = get_settings().layer2
    return TrackManager(
        camera_id=camera_id,
        track_thresh=cfg.track_thresh,
        track_high_thresh=cfg.track_high_thresh,
        match_thresh=cfg.match_thresh,
        track_buffer=cfg.track_buffer,
        frame_rate=cfg.frame_rate,
        reid_cosine_threshold=cfg.reid_cosine_threshold,
        skeleton_buffer_len=cfg.skeleton_buffer_len,
        process_noise=cfg.kalman_process_noise,
    )
