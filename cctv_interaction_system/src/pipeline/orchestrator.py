"""End-to-end pipeline orchestrator.

This module wires together all 7 layers into a single coherent pipeline.
Each per-camera worker:
  1. Reads frames from ingestion (Layer 0)
  2. Runs detection on every Nth frame (Layer 1)
  3. Updates tracklets via ByteTrack + Re-ID + Kalman + SkeletonBuffer (Layer 2)
  4. Finds pairs (Layer 3)
  5. Routes to interaction / individual recognizers (Layer 4A / 4B)
  6. Post-processes via EMA + state machine + dedup (Layer 5)
  7. Generates alerts via AlertManager (Layer 6)

The FastAPI server (Layer 7) runs in a separate thread.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from config.settings import Settings, get_settings
from src.common.logger import get_logger
from src.common.schemas import (
    ActionEvent,
    Detection,
    FrameDetections,
    IndividualPrediction,
    InteractionPrediction,
    Tracklet,
)
from src.layer1_detection.detector import Detector
from src.layer2_tracking.track_manager import TrackManager, make_track_manager
from src.layer3_pair_analysis.pair_analyzer import make_pair_analyzer
from src.layer3_pair_analysis.router import Router
from src.layer4a_interaction.cascade_filter import make_cascade_filter
from src.layer4b_individual.individual_recognizer import make_individual_recognizer
from src.layer5_postprocess.alert_dedup import AlertDeduplicator
from src.layer5_postprocess.ema_smoother import EMASmoother
from src.layer5_postprocess.state_machine import ActionStateMachine, State
from src.layer6_alerts.alert_manager import AlertManager
from src.layer0_ingestion.ingestion_worker import IngestionWorker

logger = get_logger()


@dataclass
class CameraPipeline:
    """Per-camera pipeline state."""

    camera_id: str
    track_manager: TrackManager
    pair_analyzer = None  # set in __post_init__
    cascade_filter = None
    individual_recognizer = None
    ema: EMASmoother = None
    state_machine: ActionStateMachine = None
    alert_dedup: AlertDeduplicator = None
    frame_buffer: deque = None  # for SlowFast RGB branch

    def __post_init__(self):
        self.pair_analyzer = make_pair_analyzer(self.camera_id)
        self.cascade_filter = make_cascade_filter()
        self.individual_recognizer = make_individual_recognizer()
        self.ema = EMASmoother(alpha=get_settings().layer5.ema_alpha)
        self.state_machine = ActionStateMachine(
            candidate_frames=get_settings().layer5.state_candidate_frames,
            confirmed_frames=get_settings().layer5.state_confirmed_frames,
            alert_frames=get_settings().layer5.state_alert_frames,
            candidate_score=get_settings().layer5.state_candidate_score,
            confirmed_score=get_settings().layer5.state_confirmed_score,
            alert_score=get_settings().layer5.state_alert_score,
            reset_score=get_settings().layer5.state_reset_score,
            reset_frames=get_settings().layer5.state_reset_frames,
        )
        self.alert_dedup = AlertDeduplicator(window_s=get_settings().layer5.dedup_window_s)
        self.frame_buffer = deque(maxlen=64)  # 2s @ 30 FPS

    def push_frame(self, frame: np.ndarray) -> None:
        """Cache a copy for SlowFast RGB branch (most recent N frames)."""
        self.frame_buffer.append(frame)


class Pipeline:
    """Top-level orchestrator that drives all per-camera pipelines."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        alert_manager: Optional[AlertManager] = None,
        cameras: Optional[List[dict]] = None,
        use_mock_ingestion: bool = True,
    ):
        self.settings = settings or get_settings()
        self.cameras = cameras or []
        self.detector = Detector(use_mock=self.settings.mock_mode)
        self.router = Router()
        self.alert_manager = alert_manager or AlertManager(use_db=False)
        self.use_mock_ingestion = use_mock_ingestion
        self._lock = threading.RLock()
        self._camera_pipelines: Dict[str, CameraPipeline] = {}
        self._ingestion_workers: Dict[str, IngestionWorker] = {}
        self._detection_queue: "queue.Queue" = queue.Queue(maxsize=512)
        self._tracking_queue: "queue.Queue" = queue.Queue(maxsize=2048)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        # Register cameras eagerly so start() has work to do.
        for cam in self.cameras:
            self.add_camera(cam)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def add_camera(self, camera: dict) -> None:
        with self._lock:
            cid = camera["camera_id"]
            if cid in self._camera_pipelines:
                logger.debug(f"camera {cid} already added — skipping")
                return
            self._camera_pipelines[cid] = CameraPipeline(
                camera_id=cid,
                track_manager=make_track_manager(cid),
            )
            if camera not in self.cameras:
                self.cameras.append(camera)

    def start(self) -> None:
        """Start ingestion workers + processing threads."""
        if not self._camera_pipelines:
            logger.warning("no cameras configured — pipeline idle")
            return

        for camera in self.cameras:
            cid = camera["camera_id"]
            worker = IngestionWorker(
                camera_id=cid,
                rtsp_url=camera.get("rtsp_url", ""),
                detection_queue=self._detection_queue,
                tracking_queue=self._tracking_queue,
                detection_stride=self.settings.layer0.detection_stride,
                width=self.settings.layer0.frame_width,
                height=self.settings.layer0.frame_height,
                fps=self.settings.layer0.target_fps,
                use_mock=self.use_mock_ingestion,
                use_hwaccel=not self.settings.mock_mode,
            )
            self._ingestion_workers[cid] = worker
            worker.start()

        self._threads.append(threading.Thread(
            target=self._detection_loop, name="detection-loop", daemon=True,
        ))
        self._threads.append(threading.Thread(
            target=self._tracking_loop, name="tracking-loop", daemon=True,
        ))
        for t in self._threads:
            t.start()
        logger.info(f"Pipeline started with {len(self.cameras)} cameras")

    def stop(self) -> None:
        self._stop.set()
        for w in self._ingestion_workers.values():
            w.stop()
        for t in self._threads:
            t.join(timeout=5)
        logger.info("Pipeline stopped")

    # ------------------------------------------------------------------
    # Detection loop — consumes detection frames, runs detector, publishes
    # FrameDetections back onto the tracking queue.
    # ------------------------------------------------------------------
    def _detection_loop(self) -> None:
        from src.layer1_detection.batch_collector import BatchCollector
        cfg = self.settings.layer1
        collector = BatchCollector(max_batch=cfg.max_batch, timeout_ms=cfg.batch_timeout_ms)
        while not self._stop.is_set():
            try:
                pkt = self._detection_queue.get(timeout=0.1)
                collector.add(pkt)
            except queue.Empty:
                pass
            batch = collector.try_get_batch()
            if batch:
                self._run_detection_batch(batch)

    def _run_detection_batch(self, batch: List[dict]) -> None:
        if not batch:
            return
        frames = [p["data"] for p in batch]
        metas = [{"camera_id": p["camera_id"], "frame_id": p["frame_id"],
                  "timestamp": p["timestamp"]} for p in batch]
        try:
            results: List[FrameDetections] = self.detector.detect_batch(frames, metas)
        except Exception as e:
            logger.error(f"detection batch failed: {e}")
            return
        for r in results:
            try:
                self._tracking_queue.put_nowait({
                    "type": "detection_result",
                    "data": r,
                })
            except queue.Full:
                logger.warning("tracking queue full, dropping detection result")

    # ------------------------------------------------------------------
    # Tracking loop — consumes both raw frames and detection results,
    # drives the per-camera pipeline.
    # ------------------------------------------------------------------
    def _tracking_loop(self) -> None:
        # Per-camera queue of pending detections (so we don't run track update
        # for a camera without a fresh detection).
        pending: Dict[str, list[FrameDetections]] = defaultdict(list)
        while not self._stop.is_set():
            try:
                pkt = self._tracking_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if pkt.get("type") == "detection_result":
                fd: FrameDetections = pkt["data"]
                pending[fd.camera_id].append(fd)
                # Process at most the latest 3 backlog entries per camera
                if len(pending[fd.camera_id]) > 3:
                    pending[fd.camera_id] = pending[fd.camera_id][-3:]
                continue

            # It's a raw frame (tracking packet)
            camera_id = pkt["camera_id"]
            frame = pkt["data"]
            timestamp = pkt["timestamp"]
            frame_id = pkt["frame_id"]

            pipeline = self._camera_pipelines.get(camera_id)
            if pipeline is None:
                continue

            pipeline.push_frame(frame)
            self.alert_manager.push_frame(camera_id, timestamp, frame)

            # If we have a pending detection for this camera, use the latest one
            if pending[camera_id]:
                fd = pending[camera_id].pop(0)
                # Convert FrameDetections to list[Detection]
                dets: List[Detection] = list(fd.detections)
                tracklets = pipeline.track_manager.update(dets, frame=frame)
            else:
                # No fresh detection — just predict poses via Kalman
                # (we still need tracklets to be returned)
                tracklets = [t for t in pipeline.track_manager.tracklets.values()
                             if t.state in ("NEW", "CONFIRMED", "ACTIVE")]

            if not tracklets:
                continue

            # Layer 3: Pair analysis
            pairs = pipeline.pair_analyzer.update(tracklets, frame_id, timestamp)

            # Layer 4A + 4B
            pair_tracklets, single_tracklets = self.router.route(tracklets, pairs)

            skeleton_buffers = pipeline.track_manager.get_all_skeletons()
            frame_buf = list(pipeline.frame_buffer)

            interaction_preds: List[InteractionPrediction] = []
            if pair_tracklets:
                interaction_preds = pipeline.cascade_filter.recognize(
                    pair_tracklets, skeleton_buffers, frame_buf,
                )

            individual_preds: List[IndividualPrediction] = []
            if single_tracklets:
                individual_preds = pipeline.individual_recognizer.recognize(
                    single_tracklets, skeleton_buffers, frame_id, timestamp, camera_id,
                )

            # Layer 5: Post-process
            self._post_process(pipeline, interaction_preds, individual_preds)

    # ------------------------------------------------------------------
    # Post-processing — EMA + state machine + alert
    # ------------------------------------------------------------------
    def _post_process(
        self,
        pipeline: CameraPipeline,
        interactions: List[InteractionPrediction],
        individuals: List[IndividualPrediction],
    ) -> None:
        for ip in interactions:
            if ip.label == "none":
                continue
            track_key = f"pair_{ip.track_id_a}_{ip.track_id_b}"
            # Update EMA on top-1 score for this label
            smoothed = pipeline.ema.update(
                ip.camera_id, track_key, ip.label, ip.confidence,
            )
            state = pipeline.state_machine.update(
                ip.camera_id, track_key, ip.label, smoothed,
            )
            if state == State.ALERT:
                track_ids = (ip.track_id_a, ip.track_id_b)
                if pipeline.alert_dedup.should_alert(
                    ip.camera_id, track_ids, ip.label, now=ip.timestamp,
                ):
                    event = ActionEvent(
                        camera_id=ip.camera_id,
                        frame_id=ip.frame_id,
                        timestamp=ip.timestamp,
                        action_type=ip.label,
                        confidence=smoothed,
                        track_ids=list(track_ids),
                        bbox_coords=[],  # could be filled from tracklets
                        is_interaction=True,
                        state="alert",
                    )
                    alert = self.alert_manager.handle_event(event)
                    if alert:
                        self._broadcast_alert(alert)

        for ind in individuals:
            if ind.label == "none":
                continue
            track_key = f"tid_{ind.track_id}"
            smoothed = pipeline.ema.update(
                ind.camera_id, track_key, ind.label, ind.confidence,
            )
            state = pipeline.state_machine.update(
                ind.camera_id, track_key, ind.label, smoothed,
            )
            if state == State.ALERT:
                if pipeline.alert_dedup.should_alert(
                    ind.camera_id, (ind.track_id,), ind.label, now=ind.timestamp,
                ):
                    event = ActionEvent(
                        camera_id=ind.camera_id,
                        frame_id=ind.frame_id,
                        timestamp=ind.timestamp,
                        action_type=ind.label,
                        confidence=smoothed,
                        track_ids=[ind.track_id],
                        bbox_coords=[],
                        is_interaction=False,
                        state="alert",
                    )
                    alert = self.alert_manager.handle_event(event)
                    if alert:
                        self._broadcast_alert(alert)

    def _broadcast_alert(self, alert) -> None:
        """Hook for pushing alerts to WS clients / Redis pub-sub."""
        # In production this would publish to Redis / WS broadcast
        logger.info(f"Broadcast alert: {alert.alert_id} {alert.action_type}")


# ---------------------------------------------------------------------
# Synchronous single-step API (used in tests + notebooks)
# ---------------------------------------------------------------------
class SyncPipeline:
    """Synchronous wrapper that processes one frame at a time.

    Useful for tests and offline evaluation — no threads, no queues.
    """

    def __init__(self, settings: Optional[Settings] = None, cameras: Optional[List[dict]] = None):
        self.settings = settings or get_settings()
        self.cameras = cameras or []
        self.detector = Detector(use_mock=self.settings.mock_mode)
        self.router = Router()
        self.alert_manager = AlertManager(use_db=False)
        self._pipelines: Dict[str, CameraPipeline] = {}
        for cam in self.cameras:
            self.add_camera(cam)

    def add_camera(self, camera: dict) -> None:
        cid = camera["camera_id"]
        if cid in self._pipelines:
            return
        self._pipelines[cid] = CameraPipeline(
            camera_id=cid,
            track_manager=make_track_manager(cid),
        )
        if camera not in self.cameras:
            self.cameras.append(camera)

    def process_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        frame_id: int,
        timestamp: Optional[float] = None,
        run_detection: bool = True,
    ) -> dict:
        """Process one frame synchronously.

        Args:
            run_detection: if False, skip detection (useful when caller has
                           pre-computed detections).

        Returns:
            dict with keys: tracklets, pairs, interactions, individuals, alerts
        """
        if camera_id not in self._pipelines:
            raise KeyError(f"camera {camera_id} not registered")
        if timestamp is None:
            timestamp = time.time()

        pipeline = self._pipelines[camera_id]
        pipeline.push_frame(frame)
        self.alert_manager.push_frame(camera_id, timestamp, frame)

        dets: List[Detection] = []
        if run_detection:
            metas = [{"camera_id": camera_id, "frame_id": frame_id, "timestamp": timestamp}]
            results = self.detector.detect_batch([frame], metas)
            if results:
                dets = list(results[0].detections)

        tracklets = pipeline.track_manager.update(dets, frame=frame)
        pairs = pipeline.pair_analyzer.update(tracklets, frame_id, timestamp)
        pair_pairs, single_tracklets = self.router.route(tracklets, pairs)

        skel_bufs = pipeline.track_manager.get_all_skeletons()
        frame_buf = list(pipeline.frame_buffer)

        interactions = pipeline.cascade_filter.recognize(
            pair_pairs, skel_bufs, frame_buf,
        ) if pair_pairs else []
        individuals = pipeline.individual_recognizer.recognize(
            single_tracklets, skel_bufs, frame_id, timestamp, camera_id,
        ) if single_tracklets else []

        # Post-process
        new_alerts = []
        for ip in interactions:
            if ip.label == "none":
                continue
            track_key = f"pair_{ip.track_id_a}_{ip.track_id_b}"
            smoothed = pipeline.ema.update(ip.camera_id, track_key, ip.label, ip.confidence)
            state = pipeline.state_machine.update(ip.camera_id, track_key, ip.label, smoothed)
            if state == State.ALERT:
                if pipeline.alert_dedup.should_alert(
                    ip.camera_id, (ip.track_id_a, ip.track_id_b), ip.label, now=ip.timestamp,
                ):
                    event = ActionEvent(
                        camera_id=ip.camera_id,
                        frame_id=ip.frame_id,
                        timestamp=ip.timestamp,
                        action_type=ip.label,
                        confidence=smoothed,
                        track_ids=[ip.track_id_a, ip.track_id_b],
                        bbox_coords=[],
                        is_interaction=True,
                        state="alert",
                    )
                    alert = self.alert_manager.handle_event(event)
                    if alert:
                        new_alerts.append(alert)

        for ind in individuals:
            if ind.label == "none":
                continue
            track_key = f"tid_{ind.track_id}"
            smoothed = pipeline.ema.update(ind.camera_id, track_key, ind.label, ind.confidence)
            state = pipeline.state_machine.update(ind.camera_id, track_key, ind.label, smoothed)
            if state == State.ALERT:
                if pipeline.alert_dedup.should_alert(
                    ind.camera_id, (ind.track_id,), ind.label, now=ind.timestamp,
                ):
                    event = ActionEvent(
                        camera_id=ind.camera_id,
                        frame_id=ind.frame_id,
                        timestamp=ind.timestamp,
                        action_type=ind.label,
                        confidence=smoothed,
                        track_ids=[ind.track_id],
                        bbox_coords=[],
                        is_interaction=False,
                        state="alert",
                    )
                    alert = self.alert_manager.handle_event(event)
                    if alert:
                        new_alerts.append(alert)

        return {
            "tracklets": tracklets,
            "pairs": pairs,
            "interactions": interactions,
            "individuals": individuals,
            "alerts": new_alerts,
        }
