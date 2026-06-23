"""ByteTrack-style multi-object tracker.

Pure-Python implementation that doesn't depend on external ByteTrack code.
Uses IoU-based bipartite matching with two-stage threshold (high + low),
matching the original ByteTrack paper.

References:
  - Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
    Detection Box", ECCV 2022.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.common.schemas import Detection


@dataclass
class Track:
    """Internal track representation used by ByteTrack."""

    track_id: int
    bbox: Tuple[float, float, float, float]
    score: float
    keypoints: List[List[float]] = field(default_factory=list)
    keypoint_scores: List[float] = field(default_factory=list)
    # Kalman state (x, y, aspect, h, vx, vy, va, vh)
    kalman_state: Optional[np.ndarray] = None
    frame_id: int = 0
    start_frame: int = 0
    tracklet_len: int = 0
    is_activated: bool = False
    state: str = "NEW"  # NEW | TRACKED | LOST

    @property
    def bbox_np(self) -> np.ndarray:
        return np.array(self.bbox, dtype=np.float32)


def bbox_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU matrix (Na, Nb)."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clip(min=0) * \
             (boxes_a[:, 3] - boxes_a[:, 1]).clip(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clip(min=0) * \
             (boxes_b[:, 3] - boxes_b[:, 1]).clip(min=0)
    inter_x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    inter_y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    inter_x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    inter_y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter_w = (inter_x2 - inter_x1).clip(min=0)
    inter_h = (inter_y2 - inter_y1).clip(min=0)
    inter = inter_w * inter_h
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / union, 0.0)


def linear_assignment(cost: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Greedy linear assignment (good enough for tests; replace with scipy
    linear_sum_assignment in production for true optimal matching).
    """
    from scipy.optimize import linear_sum_assignment
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    row_ind, col_ind = linear_sum_assignment(cost)
    matches = []
    used_rows, used_cols = set(), set()
    for r, c in zip(row_ind, col_ind):
        # Reject matches with cost >= 1 (IoU = 0)
        if cost[r, c] < 1.0:
            matches.append((int(r), int(c)))
            used_rows.add(int(r))
            used_cols.add(int(c))
    un_rows = [i for i in range(cost.shape[0]) if i not in used_rows]
    un_cols = [j for j in range(cost.shape[1]) if j not in used_cols]
    return matches, un_rows, un_cols


class ByteTrack:
    """Simplified ByteTrack implementation."""

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_high_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
    ):
        self.track_thresh = track_thresh
        self.track_high_thresh = track_high_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate

        self._next_id = 1
        self.tracked_tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.frame_id = 0

    def _allocate_id(self) -> int:
        tid = self._next_id
        self._next_id += 1
        return tid

    @staticmethod
    def _det_to_track(track_id: int, det: Detection, frame_id: int, is_new: bool = False) -> Track:
        return Track(
            track_id=track_id,
            bbox=det.bbox,
            score=det.confidence,
            keypoints=det.keypoints,
            keypoint_scores=det.keypoint_scores,
            frame_id=frame_id,
            start_frame=frame_id,
            tracklet_len=1 if is_new else 0,
            is_activated=True,
            state="NEW" if is_new else "TRACKED",
        )

    def update(self, detections: List[Detection]) -> List[Track]:
        """Process one frame's detections and return active tracks.

        Each returned Track has:
          - track_id assigned
          - bbox / keypoints updated
          - is_activated=True
        """
        self.frame_id += 1

        # Split detections by score
        det_boxes = np.array([d.bbox for d in detections], dtype=np.float32) \
            if detections else np.zeros((0, 4), dtype=np.float32)
        det_scores = np.array([d.confidence for d in detections], dtype=np.float32) \
            if detections else np.zeros((0,), dtype=np.float32)

        high_mask = det_scores >= self.track_high_thresh
        low_mask = (det_scores >= self.track_thresh) & (~high_mask)
        high_idx = np.where(high_mask)[0]
        low_idx = np.where(low_mask)[0]
        rem_idx = np.where(~high_mask & ~low_mask)[0]

        # Predict new positions of tracked tracks (Kalman predict step skipped —
        # we use last bbox directly, suitable at 30 FPS for nearby motion).
        # First match: high-confidence detections vs tracked
        active_boxes = np.array([t.bbox for t in self.tracked_tracks], dtype=np.float32) \
            if self.tracked_tracks else np.zeros((0, 4), dtype=np.float32)

        matches, un_high, un_tracked = self._match(
            active_boxes, det_boxes[high_idx]
        )

        # Update matched tracks
        new_tracked: List[Track] = []
        for r, c in matches:
            det = detections[int(high_idx[c])]
            t = self.tracked_tracks[r]
            t.bbox = det.bbox
            t.score = det.confidence
            t.keypoints = det.keypoints
            t.keypoint_scores = det.keypoint_scores
            t.frame_id = self.frame_id
            t.tracklet_len += 1
            t.is_activated = True
            t.state = "TRACKED"
            new_tracked.append(t)

        # Second match: low-confidence detections vs unmatched tracked
        unmatched_tracks = [self.tracked_tracks[r] for r in un_tracked]
        um_boxes = np.array([t.bbox for t in unmatched_tracks], dtype=np.float32) \
            if unmatched_tracks else np.zeros((0, 4), dtype=np.float32)
        matches2, un_low, un_um = self._match(um_boxes, det_boxes[low_idx])
        for r, c in matches2:
            det = detections[int(low_idx[c])]
            t = unmatched_tracks[r]
            t.bbox = det.bbox
            t.score = det.confidence
            t.keypoints = det.keypoints
            t.keypoint_scores = det.keypoint_scores
            t.frame_id = self.frame_id
            t.tracklet_len += 1
            t.is_activated = True
            t.state = "TRACKED"
            new_tracked.append(t)

        # Remaining unmatched tracked -> lost
        for r in un_um:
            t = unmatched_tracks[r]
            t.is_activated = False
            t.state = "LOST"
            self.lost_tracks.append(t)

        # Re-activate from lost using remaining high-confidence detections
        reactivated_lost: set[int] = set()
        if self.lost_tracks and len(un_high) > 0:
            lost_boxes = np.array([t.bbox for t in self.lost_tracks], dtype=np.float32)
            # _match returns (matches, unmatched_dets, unmatched_tracks)
            matches3, un_high2, un_lost = self._match(
                lost_boxes, det_boxes[high_idx][un_high]
            )
            for r, c in matches3:
                det = detections[int(high_idx[un_high[c]])]
                t = self.lost_tracks[r]
                t.bbox = det.bbox
                t.score = det.confidence
                t.keypoints = det.keypoints
                t.keypoint_scores = det.keypoint_scores
                t.frame_id = self.frame_id
                t.is_activated = True
                t.state = "TRACKED"
                t.tracklet_len += 1
                reactivated_lost.add(r)
                new_tracked.append(t)
            # Newly high dets — un_high2 are indices into det_boxes[high_idx][un_high]
            new_high_idx = [un_high[i] for i in un_high2]
        else:
            new_high_idx = list(un_high)

        # Initialise new tracks for remaining high-confidence detections
        for i in new_high_idx:
            det = detections[int(high_idx[i])]
            t = self._det_to_track(self._allocate_id(), det, self.frame_id, is_new=True)
            new_tracked.append(t)

        # Drop re-activated tracks from the lost pool and prune stale ones.
        self.lost_tracks = [
            t for i, t in enumerate(self.lost_tracks)
            if i not in reactivated_lost
            and self.frame_id - t.frame_id <= self.track_buffer
        ]

        self.tracked_tracks = new_tracked
        return [t for t in new_tracked if t.is_activated]

    def _match(self, tracks_boxes: np.ndarray, det_boxes: np.ndarray):
        """Returns (matches, unmatched_dets, unmatched_tracks).

        matches: list of (track_idx, det_idx) tuples
        unmatched_dets: list of indices into det_boxes (i.e. unmatched columns)
        unmatched_tracks: list of indices into tracks_boxes (i.e. unmatched rows)
        """
        if len(tracks_boxes) == 0 or len(det_boxes) == 0:
            return (
                [],
                list(range(len(det_boxes))),    # unmatched dets
                list(range(len(tracks_boxes))),  # unmatched tracks
            )
        iou_mat = bbox_iou_matrix(tracks_boxes, det_boxes)
        # Cost = 1 - IoU; we reject matches where IoU < (1 - match_thresh)
        cost = 1.0 - iou_mat
        matches, un_rows, un_cols = linear_assignment(cost)
        # Filter matches by match_thresh
        good_matches = []
        used_r, used_c = set(), set()
        for r, c in matches:
            if iou_mat[r, c] >= (1.0 - self.match_thresh):
                good_matches.append((r, c))
                used_r.add(r)
                used_c.add(c)
        un_r = [i for i in range(len(tracks_boxes)) if i not in used_r]
        un_c = [j for j in range(len(det_boxes)) if j not in used_c]
        return good_matches, un_c, un_r
