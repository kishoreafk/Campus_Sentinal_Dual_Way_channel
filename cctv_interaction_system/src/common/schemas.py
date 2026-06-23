"""Structured data models (pydantic) used across all layers.

These schemas are the *contract* between layers — every layer's output
must be a valid instance of the corresponding schema.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------
# Layer 0: Frame
# ---------------------------------------------------------------------
class Frame(BaseModel):
    """A decoded video frame."""

    camera_id: str
    frame_id: int
    timestamp: float = Field(default_factory=time.time)
    # Raw BGR frame (np.ndarray uint8 HxWx3) — kept out of pydantic validation
    data: Any = None  # np.ndarray
    width: int = 1280
    height: int = 720
    # Where this frame should be routed
    frame_type: str = "tracking"  # "detection" | "tracking"

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------
# Layer 1: Detection
# ---------------------------------------------------------------------
class Detection(BaseModel):
    """A single person detection with keypoints."""

    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    # 17 COCO keypoints, each [x, y, conf]
    keypoints: List[List[float]] = Field(default_factory=list)
    keypoint_scores: List[float] = Field(default_factory=list)
    # Assigned by Layer 2 (ByteTrack)
    track_id: Optional[int] = None

    model_config = {"arbitrary_types_allowed": True}

    @property
    def avg_kp_confidence(self) -> float:
        if not self.keypoint_scores:
            return 0.0
        return float(np.mean(self.keypoint_scores))

    @property
    def bbox_center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bbox_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, (x2 - x1) * (y2 - y1))

    @property
    def height(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, y2 - y1)


class FrameDetections(BaseModel):
    """All detections for a single frame."""

    camera_id: str
    frame_id: int
    timestamp: float
    detections: List[Detection] = Field(default_factory=list)

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------
# Layer 2: Tracklet
# ---------------------------------------------------------------------
class Tracklet(BaseModel):
    """A persistent track with appearance + skeleton history."""

    track_id: int
    camera_id: str
    bbox: Tuple[float, float, float, float]
    confidence: float = 0.0
    keypoints: List[List[float]] = Field(default_factory=list)
    keypoint_scores: List[float] = Field(default_factory=list)
    appearance_feature: Optional[List[float]] = None  # OSNet 512-d
    first_seen: float = Field(default_factory=time.time)
    last_seen: float = Field(default_factory=time.time)
    state: str = "ACTIVE"  # NEW | CONFIRMED | ACTIVE | OCCLUDED | RECOVERED | LOST | DELETED
    consecutive_seen: int = 0
    consecutive_missed: int = 0

    model_config = {"arbitrary_types_allowed": True}

    @property
    def avg_kp_confidence(self) -> float:
        if not self.keypoint_scores:
            return 0.0
        return float(np.mean(self.keypoint_scores))

    @property
    def height(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, y2 - y1)


# ---------------------------------------------------------------------
# Layer 3: Pair
# ---------------------------------------------------------------------
class PersonPair(BaseModel):
    """A candidate interacting pair."""

    camera_id: str
    frame_id: int
    timestamp: float
    track_id_a: int
    track_id_b: int
    bbox_a: Tuple[float, float, float, float]
    bbox_b: Tuple[float, float, float, float]
    distance: float
    iou: float
    face_to_face_dot: float
    sustained_frames: int
    # Union bbox of the pair (used as ROI for SlowFast)
    union_bbox: Tuple[float, float, float, float]

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------
# Layer 4: Predictions
# ---------------------------------------------------------------------
class InteractionPrediction(BaseModel):
    """Output of Layer 4A."""

    camera_id: str
    frame_id: int
    timestamp: float
    track_id_a: int
    track_id_b: int
    label: str
    confidence: float
    # Whether cascade filter accepted (SlowFast was run)
    cascade_passed: bool = False
    pose_score: float = 0.0
    rgb_score: float = 0.0
    pose_probs: List[float] = Field(default_factory=list)
    rgb_probs: List[float] = Field(default_factory=list)


class IndividualPrediction(BaseModel):
    """Output of Layer 4B."""

    camera_id: str
    frame_id: int
    timestamp: float
    track_id: int
    label: str
    confidence: float
    probs: List[float] = Field(default_factory=list)


# ---------------------------------------------------------------------
# Layer 5: Post-processed
# ---------------------------------------------------------------------
class ActionEvent(BaseModel):
    """Post-processed, state-machine-confirmed action event."""

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: str
    frame_id: int
    timestamp: float
    action_type: str  # e.g. "fight", "running"
    confidence: float
    # Either pair (interaction) or single (individual)
    track_ids: List[int] = Field(default_factory=list)
    bbox_coords: List[List[float]] = Field(default_factory=list)
    is_interaction: bool = False
    state: str = "candidate"  # candidate | confirmed | alert


# ---------------------------------------------------------------------
# Layer 6: Alert
# ---------------------------------------------------------------------
class Alert(BaseModel):
    """A confirmed alert ready for storage / dispatch."""

    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: str
    timestamp: float = Field(default_factory=time.time)
    action_type: str
    confidence: float
    track_ids: List[int] = Field(default_factory=list)
    bbox_coords: List[List[float]] = Field(default_factory=list)
    clip_path: Optional[str] = None
    processed: bool = False
    metadata: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------
# Skeleton buffer (used internally by Layer 2 and Layer 4)
# ---------------------------------------------------------------------
class SkeletonBuffer:
    """Sliding-window skeleton buffer (T, V, 3) per person.

    Stores the last `maxlen` pose frames for a tracklet. Older frames
    are evicted automatically.
    """

    __slots__ = ("track_id", "maxlen", "_buf")

    def __init__(self, track_id: int, maxlen: int = 48):
        self.track_id = track_id
        self.maxlen = maxlen
        self._buf: List[np.ndarray] = []

    def push(self, keypoints: np.ndarray) -> None:
        """Push a (17, 3) keypoint array (x, y, conf)."""
        kp = np.asarray(keypoints, dtype=np.float32).copy()
        if kp.shape != (17, 3):
            raise ValueError(f"keypoints must be (17, 3), got {kp.shape}")
        self._buf.append(kp)
        if len(self._buf) > self.maxlen:
            self._buf.pop(0)

    def to_array(self) -> np.ndarray:
        """Return (T, 17, 3) padded with zeros if shorter than maxlen."""
        if not self._buf:
            return np.zeros((0, 17, 3), dtype=np.float32)
        return np.stack(self._buf, axis=0)

    def to_padded(self, target_len: int | None = None) -> np.ndarray:
        """Return (target_len, 17, 3) — pad at front with zeros if short."""
        n = target_len or self.maxlen
        arr = self.to_array()
        T = arr.shape[0]
        if T >= n:
            return arr[-n:]
        pad = np.zeros((n - T, 17, 3), dtype=np.float32)
        return np.concatenate([pad, arr], axis=0)

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def is_full(self) -> bool:
        return len(self._buf) >= self.maxlen
