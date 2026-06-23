"""FastAPI service exposing:
  - GET /health
  - GET /metrics        (Prometheus)
  - GET /alerts         (list recent alerts)
  - GET /alerts/{id}    (alert detail)
  - GET /cameras        (camera status)
  - WS  /ws/alerts      (real-time alert push)
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from src.common.logger import get_logger
from src.common.metrics import metrics_text

logger = get_logger()


def create_app(alert_manager=None, cameras: Optional[list[dict]] = None) -> FastAPI:
    """Build the FastAPI app. `alert_manager` and `cameras` are injected so
    the API can be tested in isolation.
    """
    app = FastAPI(
        title="CCTV Interaction Recognition System",
        version="1.0.0",
        description="Real-time monitoring & alerting API",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _cameras = cameras or []
    _alert_manager = alert_manager
    _ws_clients: set[WebSocket] = set()

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "version": "1.0.0",
            "cameras": len(_cameras),
            "alert_manager": _alert_manager is not None,
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prometheus_metrics():
        return metrics_text()

    @app.get("/alerts")
    async def list_alerts(
        limit: int = Query(100, ge=1, le=1000),
        camera_id: Optional[str] = None,
        action_type: Optional[str] = None,
    ):
        if _alert_manager is None:
            return {"alerts": [], "count": 0}
        alerts = _alert_manager.list_alerts(
            limit=limit,
            **({"camera_id": camera_id} if camera_id else {}),
            **({"action_type": action_type} if action_type else {}),
        )
        return {
            "alerts": [a.model_dump() for a in alerts],
            "count": len(alerts),
        }

    @app.get("/alerts/{alert_id}")
    async def get_alert(alert_id: str):
        if _alert_manager is None:
            raise HTTPException(404, "alert manager not configured")
        alerts = _alert_manager.list_alerts(limit=1000)
        for a in alerts:
            if a.alert_id == alert_id:
                return a.model_dump()
        raise HTTPException(404, f"alert {alert_id} not found")

    @app.get("/cameras")
    async def list_cameras():
        return {"cameras": _cameras, "count": len(_cameras)}

    @app.websocket("/ws/alerts")
    async def ws_alerts(ws: WebSocket):
        await ws.accept()
        _ws_clients.add(ws)
        try:
            while True:
                # Heartbeat — clients receive alerts pushed via `push_alert`
                await ws.send_text(json.dumps({"type": "heartbeat"}))
                await asyncio.sleep(15)
        except WebSocketDisconnect:
            _ws_clients.discard(ws)
        except Exception:
            _ws_clients.discard(ws)

    async def push_alert(alert_dict: dict) -> None:
        """Broadcast an alert to all connected WS clients."""
        dead = set()
        for ws in _ws_clients:
            try:
                await ws.send_text(json.dumps({"type": "alert", "data": alert_dict}))
            except Exception:
                dead.add(ws)
        _ws_clients.difference_update(dead)

    app.state.push_alert = push_alert
    app.state.alert_manager = _alert_manager
    return app
