"""Tests for Layer 7 FastAPI."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.layer7_monitoring.api import create_app


def test_health_endpoint():
    app = create_app(alert_manager=None, cameras=[])
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


def test_cameras_endpoint():
    cameras = [{"camera_id": "cam_001", "name": "Lobby"}]
    app = create_app(alert_manager=None, cameras=cameras)
    client = TestClient(app)
    r = client.get("/cameras")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["cameras"][0]["camera_id"] == "cam_001"


def test_metrics_endpoint():
    app = create_app(alert_manager=None, cameras=[])
    client = TestClient(app)
    r = client.get("/metrics")
    assert r.status_code == 200
    # Prometheus text format
    assert "cctv_" in r.text


def test_alerts_endpoint_no_manager():
    app = create_app(alert_manager=None, cameras=[])
    client = TestClient(app)
    r = client.get("/alerts")
    assert r.status_code == 200
    body = r.json()
    assert body["alerts"] == []
    assert body["count"] == 0


def test_alerts_endpoint_with_manager():
    from src.layer6_alerts.alert_manager import AlertManager
    from src.common.schemas import ActionEvent
    import time as t
    am = AlertManager(use_db=False)
    am.handle_event(ActionEvent(
        camera_id="cam_001",
        frame_id=1,
        timestamp=t.time(),
        action_type="fight",
        confidence=0.85,
        track_ids=[1, 2],
        state="alert",
    ))
    app = create_app(alert_manager=am, cameras=[])
    client = TestClient(app)
    r = client.get("/alerts")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 1
    assert body["alerts"][0]["action_type"] == "fight"


def test_alert_not_found():
    from src.layer6_alerts.alert_manager import AlertManager
    am = AlertManager(use_db=False)
    app = create_app(alert_manager=am, cameras=[])
    client = TestClient(app)
    r = client.get("/alerts/nonexistent")
    assert r.status_code == 404
