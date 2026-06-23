"""Tests for SkeletonBuffer."""

from __future__ import annotations

import numpy as np
import pytest

from src.common.schemas import SkeletonBuffer


def test_skeleton_buffer_push_and_to_array():
    buf = SkeletonBuffer(track_id=1, maxlen=48)
    for i in range(10):
        kps = np.full((17, 3), float(i), dtype=np.float32)
        buf.push(kps)
    arr = buf.to_array()
    assert arr.shape == (10, 17, 3)
    # Last frame should have value 9
    assert arr[-1, 0, 0] == 9.0


def test_skeleton_buffer_evicts_old():
    buf = SkeletonBuffer(track_id=1, maxlen=5)
    for i in range(10):
        buf.push(np.full((17, 3), float(i), dtype=np.float32))
    assert len(buf) == 5
    arr = buf.to_array()
    # Should keep frames 5..9
    assert arr[0, 0, 0] == 5.0
    assert arr[-1, 0, 0] == 9.0


def test_skeleton_buffer_to_padded_short():
    buf = SkeletonBuffer(track_id=1, maxlen=48)
    for i in range(5):
        buf.push(np.full((17, 3), float(i), dtype=np.float32))
    padded = buf.to_padded(48)
    assert padded.shape == (48, 17, 3)
    # First 43 frames should be zeros
    assert np.all(padded[:43] == 0.0)
    # Last 5 should be our data
    assert padded[-1, 0, 0] == 4.0


def test_skeleton_buffer_to_padded_full():
    buf = SkeletonBuffer(track_id=1, maxlen=48)
    for i in range(48):
        buf.push(np.full((17, 3), float(i), dtype=np.float32))
    padded = buf.to_padded(48)
    assert padded.shape == (48, 17, 3)
    assert padded[0, 0, 0] == 0.0
    assert padded[-1, 0, 0] == 47.0


def test_skeleton_buffer_to_padded_overlong():
    buf = SkeletonBuffer(track_id=1, maxlen=48)
    for i in range(60):
        buf.push(np.full((17, 3), float(i), dtype=np.float32))
    padded = buf.to_padded(48)
    assert padded.shape == (48, 17, 3)
    # Should keep last 48
    assert padded[0, 0, 0] == 12.0  # 60 - 48
    assert padded[-1, 0, 0] == 59.0


def test_skeleton_buffer_empty():
    buf = SkeletonBuffer(track_id=1, maxlen=48)
    arr = buf.to_array()
    assert arr.shape == (0, 17, 3)
    padded = buf.to_padded(48)
    assert padded.shape == (48, 17, 3)
    assert np.all(padded == 0.0)


def test_skeleton_buffer_invalid_shape():
    buf = SkeletonBuffer(track_id=1, maxlen=48)
    with pytest.raises(ValueError):
        buf.push(np.zeros((10, 3), dtype=np.float32))


def test_skeleton_buffer_is_full():
    buf = SkeletonBuffer(track_id=1, maxlen=3)
    assert not buf.is_full
    buf.push(np.zeros((17, 3), dtype=np.float32))
    buf.push(np.zeros((17, 3), dtype=np.float32))
    assert not buf.is_full
    buf.push(np.zeros((17, 3), dtype=np.float32))
    assert buf.is_full
