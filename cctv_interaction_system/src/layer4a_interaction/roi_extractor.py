"""ROI extractor for SlowFast RGB branch.

Given a PersonPair's union bbox and a recent frame buffer, extract an
AR-padded RGB clip centered on the pair.

AR padding: if the union bbox doesn't match the SlowFast input aspect
ratio (1:1 for 224x224), pad the short side with zeros to maintain aspect.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def ar_pad(box: tuple[int, int, int, int], target_ratio: float = 1.0) -> tuple[int, int, int, int]:
    """Aspect-ratio pad a bbox to target_ratio = w/h."""
    x1, y1, x2, y2 = box
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    cur_ratio = w / h
    if cur_ratio < target_ratio:
        # Need wider — pad x
        new_w = int(h * target_ratio)
        dx = (new_w - w) // 2
        x1 = max(0, x1 - dx)
        x2 = x1 + new_w
    elif cur_ratio > target_ratio:
        new_h = int(w / target_ratio)
        dy = (new_h - h) // 2
        y1 = max(0, y1 - dy)
        y2 = y1 + new_h
    return (x1, y1, x2, y2)


def extract_roi_clip(
    frame_buffer: List[np.ndarray],
    union_bbox: tuple[float, float, float, float],
    clip_len: int = 32,
    img_size: int = 224,
    margin_ratio: float = 0.2,
) -> Optional[np.ndarray]:
    """Extract an RGB clip for SlowFast.

    Args:
        frame_buffer: list of recent BGR frames (most recent last)
        union_bbox: (x1, y1, x2, y2) of the pair's union bbox
        clip_len: T dimension of SlowFast input
        img_size: H = W = img_size
        margin_ratio: extra margin to add around union bbox

    Returns:
        np.ndarray (clip_len, 3, img_size, img_size) float32, normalised to [0, 1]
        or None if frame buffer is empty.
    """
    if not frame_buffer:
        return None

    # Take last `clip_len` frames (pad with first if short)
    if len(frame_buffer) >= clip_len:
        clip = frame_buffer[-clip_len:]
    else:
        clip = [frame_buffer[0]] * (clip_len - len(frame_buffer)) + list(frame_buffer)

    x1, y1, x2, y2 = [int(v) for v in union_bbox]
    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    x1 = int(x1 - margin_ratio * w)
    y1 = int(y1 - margin_ratio * h)
    x2 = int(x2 + margin_ratio * w)
    y2 = int(y2 + margin_ratio * h)

    # AR pad to 1:1
    x1, y1, x2, y2 = ar_pad((x1, y1, x2, y2), target_ratio=1.0)

    out = np.zeros((clip_len, 3, img_size, img_size), dtype=np.float32)
    for i, frame in enumerate(clip):
        H, W = frame.shape[:2]
        cx1, cy1 = max(0, x1), max(0, y1)
        cx2, cy2 = min(W, x2), min(H, y2)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        # Resize to img_size x img_size (nearest for speed)
        try:
            import cv2
            resized = cv2.resize(crop, (img_size, img_size))
        except ImportError:
            # Naive resize via numpy
            sh = img_size / crop.shape[0]
            sw = img_size / crop.shape[1]
            y_idx = np.clip((np.arange(img_size) / sh).astype(int), 0, crop.shape[0] - 1)
            x_idx = np.clip((np.arange(img_size) / sw).astype(int), 0, crop.shape[1] - 1)
            resized = crop[y_idx[:, None], x_idx[None, :]]
        # BGR -> RGB, HWC -> CHW, normalise
        rgb = resized[:, :, ::-1]
        out[i] = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return out


def extract_paired_skeletons(
    skel_a: np.ndarray,  # (T, 17, 3)
    skel_b: np.ndarray,  # (T, 17, 3)
    clip_len: int = 48,
) -> np.ndarray:
    """Stack two skeletons into a (3, T, 17, 2) tensor for PoseConv3D M=2.

    Output shape: (3 channels = [x, y, conf], T=clip_len, V=17, M=2)
    """
    T = clip_len
    # Pad / trim to T
    def pad_to(arr: np.ndarray) -> np.ndarray:
        if arr.shape[0] >= T:
            return arr[-T:]
        pad = np.zeros((T - arr.shape[0], 17, 3), dtype=np.float32)
        return np.concatenate([pad, arr], axis=0)
    a = pad_to(skel_a)
    b = pad_to(skel_b)
    # Stack along M axis
    # (T, 17, 3, 2) -> (3, T, 17, 2)
    stacked = np.stack([a, b], axis=-1)  # (T, 17, 3, 2)
    out = stacked.transpose(2, 0, 1, 3)  # (3, T, 17, 2)
    return out.astype(np.float32)
