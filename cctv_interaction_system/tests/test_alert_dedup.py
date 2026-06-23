"""Tests for Layer 5 alert deduplication."""

from __future__ import annotations

from src.layer5_postprocess.alert_dedup import AlertDeduplicator


def test_dedup_first_alert_passes():
    d = AlertDeduplicator(window_s=3.0)
    assert d.should_alert("cam_1", (1, 2), "fight", now=100.0) is True


def test_dedup_duplicate_suppressed():
    d = AlertDeduplicator(window_s=3.0)
    assert d.should_alert("cam_1", (1, 2), "fight", now=100.0) is True
    assert d.should_alert("cam_1", (1, 2), "fight", now=101.0) is False
    assert d.should_alert("cam_1", (1, 2), "fight", now=102.0) is False


def test_dedup_after_window_passes():
    d = AlertDeduplicator(window_s=3.0)
    assert d.should_alert("cam_1", (1, 2), "fight", now=100.0) is True
    assert d.should_alert("cam_1", (1, 2), "fight", now=103.1) is True


def test_dedup_different_action_independent():
    d = AlertDeduplicator(window_s=3.0)
    assert d.should_alert("cam_1", (1, 2), "fight", now=100.0) is True
    assert d.should_alert("cam_1", (1, 2), "hug", now=100.5) is True


def test_dedup_different_camera_independent():
    d = AlertDeduplicator(window_s=3.0)
    assert d.should_alert("cam_1", (1, 2), "fight", now=100.0) is True
    assert d.should_alert("cam_2", (1, 2), "fight", now=100.5) is True


def test_dedup_track_id_order_independent():
    """(1, 2) and (2, 1) should dedup together."""
    d = AlertDeduplicator(window_s=3.0)
    assert d.should_alert("cam_1", (1, 2), "fight", now=100.0) is True
    assert d.should_alert("cam_1", (2, 1), "fight", now=100.5) is False


def test_dedup_cleanup():
    d = AlertDeduplicator(window_s=3.0)
    d.should_alert("cam_1", (1, 2), "fight", now=100.0)
    d.should_alert("cam_2", (3, 4), "hug", now=100.0)
    # Cleanup at t=200 should remove both
    d.cleanup(now=200.0)
    # After cleanup, first alert should pass
    assert d.should_alert("cam_1", (1, 2), "fight", now=200.0) is True
