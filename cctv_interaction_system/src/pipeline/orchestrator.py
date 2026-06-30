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

Production scaling features:
  - Configurable queue sizes via settings
  - Multi-threaded tracking loop (one thread per camera or shared pool)
  - TTL-based eviction for stale pending detections
  - Redis pub/sub for alert broadcasting
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
from src.common.metrics import QUEUE_FILL_RATIO
from src.common.schemas import (
    ActionEvent,
    Alert,
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
        s = self.settings
        self._detection_queue: "queue.Queue" = queue.Queue(maxsize=s.detection_queue_size)
        self._tracking_queue: "queue.Queue" = queue.Queue(maxsize=s.tracking_queue_size)
        self._stop = threading.Event()
        self._threads: List[threading.Thread] = []
        self._pending_ttl = s.pending_ttl_s
        self._tracking_threads = s.tracking_threads
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
                backpressure_sleep=self.settings.layer0.backpressure_sleep_s,
            )
            self._ingestion_workers[cid] = worker
            worker.start()

        self._threads.append(threading.Thread(
            target=self._detection_loop, name="detection-loop", daemon=True,
        ))
        n_track = max(1, self._tracking_threads)
        for i in range(n_track):
            self._threads.append(threading.Thread(
                target=self._tracking_loop, name=f"tracking-loop-{i}", daemon=True,
            ))
        for t in self._threads:
            t.start()
        logger.info(f"Pipeline started with {len(self.cameras)} cameras "
                     f"({n_track} tracking threads)")

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
        # Per-camera queue of pending detections
        pending: Dict[str, list[FrameDetections]] = defaultdict(list)
        # TTL for pending detections per camera (to evict stale entries)
        pending_ts: Dict[str, float] = {}

        while not self._stop.is_set():
            try:
                pkt = self._tracking_queue.get(timeout=0.05)
            except queue.Empty:
                self._evict_stale_pending(pending, pending_ts)
                continue

            if pkt.get("type") == "detection_result":
                fd: FrameDetections = pkt["data"]
                pending[fd.camera_id].append(fd)
                pending_ts[fd.camera_id] = time.time()
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
                dets: List[Detection] = list(fd.detections)
                tracklets = pipeline.track_manager.update(dets, frame=frame)
            else:
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
            alerts = _post_process(pipeline, interaction_preds, individual_preds, self.alert_manager)
            for alert in alerts:
                _broadcast_alert(alert)

            # Report queue fill ratios
            self._report_queue_metrics()

    def _evict_stale_pending(self, pending: dict, timestamps: dict) -> None:
        """Remove pending detections older than TTL."""
        now = time.time()
        stale = [cid for cid, ts in timestamps.items() if now - ts > self._pending_ttl]
        for cid in stale:
            pending.pop(cid, None)
            timestamps.pop(cid, None)

    def _report_queue_metrics(self) -> None:
        for name, q in [("detection", self._detection_queue), ("tracking", self._tracking_queue)]:
            try:
                size = q.qsize()
                maxsize = q.maxsize
                if maxsize > 0:
                    QUEUE_FILL_RATIO.labels(name, "all").set(size / maxsize)
            except Exception:
                pass

# ------------------------------------------------------------------
# Shared post-processing — EMA + state machine + dedup
# ------------------------------------------------------------------
def _process_prediction(
    pipeline: CameraPipeline,
    pred,
    track_key: str,
    track_ids: tuple,
    is_interaction: bool,
) -> Optional[ActionEvent]:
    if pred.label == "none":
        return None
    smoothed = pipeline.ema.update(pred.camera_id, track_key, pred.label, pred.confidence)
    state = pipeline.state_machine.update(pred.camera_id, track_key, pred.label, smoothed)
    if state != State.ALERT:
        return None
    if not pipeline.alert_dedup.should_alert(pred.camera_id, track_ids, pred.label, now=pred.timestamp):
        return None
    return ActionEvent(
        camera_id=pred.camera_id,
        frame_id=pred.frame_id,
        timestamp=pred.timestamp,
        action_type=pred.label,
        confidence=smoothed,
        track_ids=list(track_ids),
        bbox_coords=[],
        is_interaction=is_interaction,
        state="alert",
    )


def _post_process(
    pipeline: CameraPipeline,
    interactions: List[InteractionPrediction],
    individuals: List[IndividualPrediction],
    alert_manager: AlertManager,
) -> List[Alert]:
    alerts: List[Alert] = []
    for ip in interactions:
        event = _process_prediction(
            pipeline, ip, f"pair_{ip.track_id_a}_{ip.track_id_b}",
            (ip.track_id_a, ip.track_id_b), True,
        )
        if event:
            alert = alert_manager.handle_event(event)
            if alert:
                alerts.append(alert)
    for ind in individuals:
        event = _process_prediction(
            pipeline, ind, f"tid_{ind.track_id}",
            (ind.track_id,), False,
        )
        if event:
            alert = alert_manager.handle_event(event)
            if alert:
                alerts.append(alert)
    return alerts


# Redis client for alert broadcasting (lazy init)
_redis_client = None


def _get_redis():
    global _redis_client
    if _redis_client is None:
        try:
            from src.common.redis_client import RedisClient
            _redis_client = RedisClient()
            if _redis_client.ping():
                logger.info("Redis client connected for alert broadcasting")
            else:
                _redis_client = None
        except Exception as e:
            logger.warning(f"Redis not available ({e}); alerts logged only")
    return _redis_client


def _broadcast_alert(alert) -> None:
    """Push alert to Redis pub/sub and log."""
    logger.info(f"Broadcast alert: {alert.alert_id} {alert.action_type}")
    r = _get_redis()
    if r is not None:
        try:
            r.publish("alerts", {
                "alert_id": alert.alert_id,
                "camera_id": alert.camera_id,
                "action_type": alert.action_type,
                "confidence": alert.confidence,
                "timestamp": alert.timestamp,
            })
        except Exception as e:
            logger.debug(f"Redis publish failed: {e}")


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
        new_alerts = _post_process(pipeline, interactions, individuals, self.alert_manager)

        return {
            "tracklets": tracklets,
            "pairs": pairs,
            "interactions": interactions,
            "individuals": individuals,
            "alerts": new_alerts,
        }
